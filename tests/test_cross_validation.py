import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from rsna_knee.training.cross_validation import TARGETS, CachePaths, FoldCacheDataset
from rsna_knee.validation.masked_auc import binarize_soft_labels, masked_multilabel_auc


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _fixture(directory: Path, n_studies: int = 9, n_splits: int = 3) -> CachePaths:
    ids = [f"study-{i}" for i in range(n_studies)]
    fold_of = {study: i % n_splits for i, study in enumerate(ids)}
    records = [{"study_id": s, "soft_targets": {t: 0.9 for t in TARGETS},
                "observed_mask": {t: True for t in TARGETS}} for s in ids]
    (directory / "manifest.json").write_text(json.dumps({"records": records}))
    (directory / "folds.json").write_text(json.dumps({
        "configuration": {"n_splits": n_splits},
        "study_assignments": [{"study_id": s, "fold": f} for s, f in fold_of.items()]}))
    archives = directory / "archives"
    archives.mkdir()
    # Original export layout: fold-0 validation vs. the rest, split across two chunks.
    groups = {"validation-000": [s for s in ids if fold_of[s] == 0],
              "train-000": [s for s in ids if fold_of[s] != 0]}
    for chunk_id, members in groups.items():
        partition = chunk_id.split("-")[0]
        cache_records = []
        with zipfile.ZipFile(archives / f"local16-cache-{chunk_id}.zip", "w") as archive:
            for index, study in enumerate(members):
                buffer = io.BytesIO()
                np.savez(buffer, images_000=np.full((2, 1, 4, 4), float(ids.index(study)), np.float32),
                         masks_000=np.ones((2, 1, 4, 4), bool),
                         targets=np.full(12, 0.9, np.float32), observed=np.ones(12, bool))
                raw = buffer.getvalue()
                name = f"{partition}/sample-{index:06d}.npz"
                archive.writestr(name, raw)
                cache_records.append({"study_sha256": _sha(study), "file": name, "series_count": 1,
                                      "sha256": hashlib.sha256(raw).hexdigest()})
            archive.writestr("cache-manifest.json", json.dumps({"records": cache_records}))
    (directory / "plan.json").write_text(json.dumps({"chunks": [
        {"chunk_id": c, "partition": c.split("-")[0]} for c in groups]}))
    return CachePaths(directory / "manifest.json", directory / "folds.json",
                      directory / "plan.json", archives)


class FoldCacheDatasetTest(unittest.TestCase):
    def test_every_fold_partitions_the_cohort_without_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = _fixture(Path(temporary))
            validation_union = []
            for fold in range(3):
                train = FoldCacheDataset(paths, fold, "train")
                validation = FoldCacheDataset(paths, fold, "validation")
                self.assertFalse(set(train.study_ids) & set(validation.study_ids))
                self.assertEqual(len(train) + len(validation), 9)
                validation_union += validation.study_ids
            self.assertEqual(sorted(validation_union), sorted(f"study-{i}" for i in range(9)))

    def test_order_follows_manifest_and_payload_matches_study(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = _fixture(Path(temporary))
            train = FoldCacheDataset(paths, 1, "train")
            self.assertEqual(train.study_ids, ["study-0", "study-2", "study-3", "study-5", "study-6", "study-8"])
            images, masks, targets, observed, index = train[2]
            self.assertEqual(float(images[0][0, 0, 0, 0]), 3.0)
            self.assertEqual(index, 2)

    def test_external_labels_replace_targets_and_mark_every_cell_observed(self):
        from rsna_knee.training.cross_validation import load_label_table
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            paths = _fixture(directory)
            rows = ["StudyInstanceUID," + ",".join(TARGETS)]
            rows += [f"study-{i}," + ",".join([str(i / 10)] * 12) for i in range(9)]
            (directory / "labels.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
            labels = load_label_table(directory / "labels.csv")
            train = FoldCacheDataset(paths, 1, "train", labels)
            _, _, targets, observed, _ = train[2]  # study-3
            np.testing.assert_allclose(targets.numpy(), np.full(12, 0.3, np.float32))
            self.assertTrue(observed.numpy().all())
            self.assertEqual(str(targets.dtype), "torch.float32")
            _, _, cached, _, _ = FoldCacheDataset(paths, 1, "train")[2]
            np.testing.assert_allclose(cached.numpy(), np.full(12, 0.9, np.float32))
            del labels["study-3"]
            with self.assertRaises(ValueError):
                FoldCacheDataset(paths, 1, "train", labels)

    def test_label_table_rejects_bad_header_and_values(self):
        from rsna_knee.training.cross_validation import load_label_table
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "labels.csv"
            path.write_text("StudyInstanceUID," + ",".join(reversed(TARGETS)) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_label_table(path)
            path.write_text("StudyInstanceUID," + ",".join(TARGETS) + "\ns," + ",".join(["1.5"] * 12) + "\n",
                            encoding="utf-8")
            with self.assertRaises(ValueError):
                load_label_table(path)

    def test_rejects_out_of_range_fold(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = _fixture(Path(temporary))
            with self.assertRaises(ValueError):
                FoldCacheDataset(paths, 3, "train")


class MaskedAucTest(unittest.TestCase):
    def test_binarization_excludes_unobserved_and_threshold_cells(self):
        soft = np.array([[0.9, 0.5], [0.1, 0.8]])
        observed = np.array([[True, True], [True, False]])
        labels, valid = binarize_soft_labels(soft, observed)
        np.testing.assert_array_equal(labels, [[1, 0], [0, 1]])
        np.testing.assert_array_equal(valid, [[True, False], [True, False]])

    def test_auc_uses_only_valid_cells(self):
        soft = np.array([[0.9], [0.1], [0.9], [0.1]])
        observed = np.array([[True], [True], [False], [True]])
        scores = np.array([[0.8], [0.2], [0.0], [0.3]])
        result = masked_multilabel_auc(soft, observed, scores, ["A"])
        self.assertEqual(result["per_target"][0]["n"], 3)
        self.assertAlmostEqual(result["macro_auc"], 1.0)

    def test_single_class_target_is_excluded_from_macro(self):
        soft = np.array([[0.9, 0.9], [0.1, 0.9]])
        observed = np.ones((2, 2), bool)
        scores = np.array([[0.2, 0.5], [0.1, 0.4]])
        result = masked_multilabel_auc(soft, observed, scores, ["A", "B"])
        self.assertIsNone(result["per_target"][1]["auc"])
        self.assertEqual(result["targets_in_macro"], 1)
        self.assertAlmostEqual(result["macro_auc"], 1.0)


if __name__ == "__main__":
    unittest.main()


class ArmTest(unittest.TestCase):
    def test_known_arms_and_rejection(self):
        from rsna_knee.training.cross_validation import ARMS, _check_arm
        self.assertEqual(ARMS, ("target_attention", "equal_mean"))
        self.assertEqual(_check_arm("equal_mean"), "equal_mean")
        with self.assertRaises(ValueError):
            _check_arm("max_pool")


class Layer3Test(unittest.TestCase):
    def _model(self):
        import torch
        import torchvision
        from rsna_knee.models.attention_mil import RadImageNetAttentionStudyModel
        from rsna_knee.models.resnet50 import ResNet50StudyModel
        torch.manual_seed(0)
        base = ResNet50StudyModel(torchvision.models.resnet50(weights=None), arm="radimagenet")
        return RadImageNetAttentionStudyModel.from_base(base)

    def test_unfreeze_layer3_trains_layer3_but_not_batchnorm(self):
        import torch
        from rsna_knee.training.cross_validation import unfreeze_layer3
        model = self._model()
        baseline = model.optimizer_groups()
        self.assertFalse(any(p.requires_grad for p in model.backbone.layer3.parameters()))
        unfreeze_layer3(model)
        layer3 = model.backbone.layer3
        convs = [p for n, p in layer3.named_parameters() if "conv" in n or "downsample.0" in n]
        norms = [p for n, p in layer3.named_parameters() if "bn" in n or "downsample.1" in n]
        self.assertTrue(convs and all(p.requires_grad for p in convs))
        self.assertTrue(norms and not any(p.requires_grad for p in norms))
        self.assertFalse(any(p.requires_grad for p in model.backbone.layer2.parameters()))
        groups = model.optimizer_groups(head_lr=1e-3, backbone_lr=1e-5)
        self.assertEqual([g["name"] for g in groups], ["head", "layer4"])
        self.assertEqual(groups[1]["lr"], 1e-5)
        self.assertEqual(len(groups[1]["params"]), len(baseline[1]["params"]) + len(convs))
        optimizer = torch.optim.Adam(groups)  # rejects duplicates across groups
        self.assertEqual(len(optimizer.param_groups), 2)

        model.train()
        images = torch.rand(2, 1, 64, 64)
        masks = torch.ones_like(images, dtype=torch.bool)
        model([images], [masks]).sum().backward()
        self.assertIsNotNone(layer3[0].conv1.weight.grad)
        self.assertGreater(float(layer3[0].conv1.weight.grad.abs().sum()), 0.0)
        self.assertIsNone(model.backbone.layer2[0].conv1.weight.grad)

    def test_fingerprint_marks_layer3_and_keeps_baseline(self):
        from rsna_knee.training.cross_validation import fold_fingerprint
        with tempfile.TemporaryDirectory() as temp:
            paths = _fixture(Path(temp))
            weights = Path(temp) / "weights.pt"
            weights.write_bytes(b"w")
            base, base_config = fold_fingerprint(paths, 1, weights)
            same, _ = fold_fingerprint(paths, 1, weights, "target_attention", "layer4")
            layer3, config = fold_fingerprint(paths, 1, weights, "target_attention", "layer3")
            self.assertEqual(base, same)
            self.assertNotIn("trainable_backbone", base_config)
            self.assertEqual(config["trainable_backbone"], "layer3")
            self.assertNotEqual(base, layer3)
            with self.assertRaises(ValueError):
                fold_fingerprint(paths, 1, weights, "target_attention", "layer2")
            labels = Path(temp) / "labels.csv"
            labels.write_bytes(b"l")
            relabeled, relabeled_config = fold_fingerprint(paths, 1, weights, "target_attention", "layer3",
                                                           labels)
            self.assertNotIn("training_labels_sha256", config)
            self.assertEqual(relabeled_config["training_labels_sha256"], hashlib.sha256(b"l").hexdigest())
            self.assertNotEqual(relabeled, layer3)
