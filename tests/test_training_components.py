import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

try:
    import numpy as np
    import torch
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("optional NumPy/PyTorch training dependencies are not installed") from exc


class TrainingModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_mask_aware_study_model_runs_cpu_forward_backward(self):
        from rsna_knee.models.baseline import ScratchStudyCNN

        model = ScratchStudyCNN()
        images = [torch.rand(2, 1, 32, 32), torch.rand(1, 1, 32, 32)]
        masks = [torch.ones_like(images[0], dtype=torch.bool),
                 torch.ones_like(images[1], dtype=torch.bool)]
        masks[0][..., :4, :] = False
        images[0][~masks[0]] = 5.0
        logits = model(images, masks)
        self.assertEqual(tuple(logits.shape), (1, 12))
        comparison = [images[0].clone(), images[1].clone()]
        comparison[0][~masks[0]] = 0.0
        self.assertTrue(torch.allclose(logits, model(comparison, masks)))
        loss = torch.nn.BCEWithLogitsLoss()(logits, torch.zeros_like(logits))
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_rejects_invalid_series_and_masks(self):
        from rsna_knee.models.baseline import ScratchStudyCNN

        model = ScratchStudyCNN()
        image = torch.ones(1, 1, 16, 16)
        with self.assertRaises(ValueError):
            model([], [])
        with self.assertRaises(ValueError):
            model([image], [torch.zeros_like(image, dtype=torch.bool)])
        with self.assertRaises(ValueError):
            model([image.double()], [torch.ones_like(image, dtype=torch.bool)])


class TrainingDatasetTests(unittest.TestCase):
    def test_manifest_bytes_bind_recomputed_grouped_fold_artifact(self):
        from rsna_knee.data.training_dataset import TrainingDatasetError, _load_artifacts
        from rsna_knee.labels.subscription import LABEL_COLUMNS
        from rsna_knee.validation.cohort import build_grouped_training_folds

        manifest = {
            "schema_version": "baseline-training-cohort-v1",
            "configuration": {
                "cohort_size": 6,
                "report_group_normalization": "python-casefold-collapse-whitespace-retain-punctuation-v1",
            },
            "source_sha256": dict.fromkeys(
                ("train_csv", "labels_jsonl", "series_csv", "gold_exclusions", "prompt_journal"),
                "0" * 64,
            ),
            "prompt_provenance_counts": {
                "legacy": 6, "c2-literal-evidence-v1": 0,
                "c2-literal-evidence-v2": 0, "c2-literal-evidence-v3": 0,
            },
            "records": [],
        }
        for index in range(6):
            manifest["records"].append({
                "study_id": f"synthetic-{index}",
                "labels": {name: index % 2 for name in LABEL_COLUMNS},
                "series": [{
                    "series_id": f"series-{index}", "fluid_sensitive": 1,
                    "fat_suppression": 0, "anatomical_plane": "sagittal",
                }],
                "prompt_version": "legacy", "normalized_report_sha256": f"{index:064x}",
            })
        raw = json.dumps(manifest, separators=(",", ":")).encode()
        digest = hashlib.sha256(raw).hexdigest()
        folds = build_grouped_training_folds(manifest, manifest_sha256=digest)
        with TemporaryDirectory() as temp:
            manifest_path = Path(temp) / "manifest.json"
            folds_path = Path(temp) / "folds.json"
            manifest_path.write_bytes(raw)
            folds_path.write_text(json.dumps(folds), encoding="utf-8")
            self.assertEqual(_load_artifacts(manifest_path, folds_path)[2], digest)
            manifest_path.write_bytes(raw + b"\n")
            with self.assertRaises(TrainingDatasetError) as caught:
                _load_artifacts(manifest_path, folds_path)
            self.assertEqual(caught.exception.reason_code, "folds_mismatch")

    def test_loads_fold_bound_partition_lazily_with_safe_anonymous_errors(self):
        from rsna_knee.data.training_dataset import TrainingStudyDataset, TrainingDatasetError

        with unittest.mock.patch("rsna_knee.data.training_dataset._load_artifacts") as load:
            load.return_value = ([], [], [], "train")
            with self.assertRaises(TrainingDatasetError) as caught:
                TrainingStudyDataset("manifest", "folds", "images", partition="train", fold=0)
            self.assertNotIn("manifest", str(caught.exception))

    def test_injected_preprocessor_builds_one_study_without_materializing_partition(self):
        from rsna_knee.data.training_dataset import TrainingStudyDataset

        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "study" / "series").mkdir(parents=True)
            record = {
                "labels": {name: 0 for name in (
                    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
                    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture")},
                "study_id": "study", "series": [{"series_id": "series"}],
            }
            arrays = np.ones((1, 1, 8, 8), dtype=np.float32)
            masks = np.ones_like(arrays, dtype=np.bool_)
            loader = mock.patch(
                "rsna_knee.data.training_dataset._load_artifacts",
                return_value=([record], {"study": 1}, "digest"),
            )
            with loader:
                dataset = TrainingStudyDataset(
                    "manifest", "folds", root, partition="train", fold=0,
                    output_size=8, preprocessor=lambda *a, **kw: SimpleNamespace(
                        images=arrays, valid_mask=masks,
                    ),
                )
                sample = dataset[0]
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.indices, (0,))
            self.assertEqual(len(sample[0]), 1)
            self.assertEqual(tuple(sample[2].shape), (12,))
            self.assertEqual(sample[3], 0)
            self.assertEqual(len(dataset.bounded_training_subset(2)), 1)

    def test_rejects_preprocessor_intensities_outside_float32_image_range(self):
        from rsna_knee.data.training_dataset import TrainingStudyDataset, TrainingDatasetError

        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "study" / "series").mkdir(parents=True)
            record = {
                "labels": {name: 0 for name in (
                    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
                    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture")},
                "study_id": "study", "series": [{"series_id": "series"}],
            }
            with mock.patch(
                "rsna_knee.data.training_dataset._load_artifacts",
                return_value=([record], {"study": 1}, "digest"),
            ):
                for value, dtype in ((2.0, np.float32), (1e100, np.float64)):
                    arrays = np.full((1, 1, 8, 8), value, dtype=dtype)
                    masks = np.ones_like(arrays, dtype=np.bool_)
                    dataset = TrainingStudyDataset(
                        "manifest", "folds", root, partition="train", fold=0,
                        output_size=8, preprocessor=lambda *a, **kw: SimpleNamespace(
                            images=arrays, valid_mask=masks,
                        ),
                    )
                    with self.assertRaises(TrainingDatasetError):
                        dataset[0]


class TrainingMetricTests(unittest.TestCase):
    def test_auc_perfect_reversed_tied_and_missing_class(self):
        from rsna_knee.validation.metrics import multilabel_roc_auc

        y = np.zeros((4, 12), dtype=np.int64)
        y[:, 0] = [0, 0, 1, 1]
        y[:, 1] = [0, 0, 1, 1]
        scores = np.zeros((4, 12), dtype=np.float64)
        scores[:, 0] = [0.1, 0.2, 0.8, 0.9]
        scores[:, 1] = [0.9, 0.8, 0.2, 0.1]
        y[:, 2] = 0
        result = multilabel_roc_auc(y, scores)
        self.assertEqual(result.per_target[0], 1.0)
        self.assertEqual(result.per_target[1], 0.0)
        self.assertIsNone(result.per_target[2])
        self.assertIsNone(result.macro_auc)

        y[:, 0] = [0, 1, 0, 1]
        scores[:, 0] = [0.5, 0.5, 0.5, 0.5]
        result = multilabel_roc_auc(y, scores)
        self.assertEqual(result.per_target[0], 0.5)

    def test_rejects_malformed_or_nonfinite_matrices(self):
        from rsna_knee.validation.metrics import multilabel_roc_auc

        with self.assertRaises(ValueError):
            multilabel_roc_auc(np.zeros((2, 12)), np.zeros((2, 11)))
        scores = np.zeros((2, 12))
        scores[0, 0] = np.nan
        with self.assertRaises(ValueError):
            multilabel_roc_auc(np.zeros((2, 12)), scores)

    def test_rejects_complex_and_string_inputs_without_coercion(self):
        from rsna_knee.validation.metrics import multilabel_roc_auc

        labels = np.zeros((2, 12), dtype=np.complex128)
        scores = np.zeros((2, 12), dtype=np.float64)
        labels[0, 0] = 1 + 2j
        with self.assertRaises(ValueError):
            multilabel_roc_auc(labels, scores)
        labels = np.zeros((2, 12), dtype=np.int64)
        scores = np.zeros((2, 12), dtype=np.complex128)
        scores[0, 0] = 0.5 + 1j
        with self.assertRaises(ValueError):
            multilabel_roc_auc(labels, scores)
        with self.assertRaises(ValueError):
            multilabel_roc_auc(np.zeros((2, 12)), np.full((2, 12), "0.5"))


if __name__ == "__main__":
    unittest.main()
