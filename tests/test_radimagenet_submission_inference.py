"""Fail-closed RadImageNet checkpoint and inference contract tests."""
import copy
import contextlib
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import torch
from torch import nn

from rsna_knee.inference.radimagenet import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_REPORT_SHA256,
    NATIVE_FINGERPRINT,
    TARGETS,
    validate_radimagenet_report,
    load_radimagenet_model,
    RadImageNetTestStudyDataset,
    run_radimagenet_inference,
    _model,
)
from private_artifacts import requires


REPO = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO / "data/private/local-radimagenet-final-epochs/final-run01/training-report.json"
CHECKPOINT_PATH = REPO / "data/private/local-radimagenet-final-epochs/final-run01/checkpoint-epoch-05.pt"


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(12))

    def forward(self, images, masks):
        return self.weight.reshape(1, 12)


@requires('data/private/local-radimagenet-final-epochs/final-run01/training-report.json',
          'data/private/local-radimagenet-final-epochs/final-run01/checkpoint-epoch-05.pt')
class RadImageNetContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report_raw = REPORT_PATH.read_bytes()
        cls.report = json.loads(cls.report_raw)

    def test_actual_report_hash_and_native_training_contract_are_pinned(self):
        self.assertEqual(hashlib.sha256(self.report_raw).hexdigest(), EXPECTED_REPORT_SHA256)
        validate_radimagenet_report(self.report)
        self.assertEqual(self.report["fingerprint"], NATIVE_FINGERPRINT)
        self.assertEqual(self.report["runtime"]["python"], "3.11.9")

    def test_report_rejects_schema_history_and_training_provenance_drift(self):
        for key, value in (("arm", "imagenet"), ("output_size", 256), ("max_slices", 7),
                           ("normalization", "imagenet_mean_std"), ("fingerprint", "0" * 64),
                           ("completed_epochs", 4)):
            bad = copy.deepcopy(self.report)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_radimagenet_report(bad)
        bad = copy.deepcopy(self.report)
        bad["epochs"].pop()
        with self.assertRaises(ValueError):
            validate_radimagenet_report(bad)

    def test_report_hash_mismatch_rejects_before_checkpoint_load(self):
        with patch("torch.load") as load:
            with self.assertRaises(ValueError):
                load_radimagenet_model(CHECKPOINT_PATH, self.report, "0" * 64,
                                       expected_report_sha256=EXPECTED_REPORT_SHA256,
                                       model_factory=ToyModel)
        load.assert_not_called()

    def test_checkpoint_sha_mismatch_rejects_before_weights_only_load(self):
        with patch("torch.load") as load:
            with self.assertRaises(ValueError):
                load_radimagenet_model(CHECKPOINT_PATH, self.report, "0" * 64,
                                       expected_report_sha256=hashlib.sha256(self.report_raw).hexdigest(),
                                       model_factory=ToyModel)
        load.assert_not_called()

    def test_actual_trained_checkpoint_loads_strictly_and_has_finite_state(self):
        model = load_radimagenet_model(
            CHECKPOINT_PATH, self.report, self.report["checkpoint_sha256"],
            expected_report_sha256=EXPECTED_REPORT_SHA256,
        )
        self.assertTrue(all(torch.isfinite(value).all() for value in model.state_dict().values()))

    def test_native_checkpoint_schema_history_fingerprint_and_tensor_state_are_strict(self):
        model = ToyModel()
        value = {"schema": "resnet50-recipe-checkpoint-v1", "fingerprint": NATIVE_FINGERPRINT,
                 "completed_epochs": 5, "history": self.report["epochs"],
                 "model_state": model.state_dict(), "optimizer_state": {}, "rng_state": {},
                 "last_epoch_seconds": 1.0}
        with TemporaryDirectory() as td:
            path = Path(td) / "checkpoint.pt"
            torch.save(value, path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            # The production loader requires a known report and checkpoint hash; state validation
            # is exercised through its test-only model factory and checkpoint override seam.
            for key, replacement in (("fingerprint", "f" * 64), ("history", []),
                                     ("model_state", {"weight": torch.full((12,), float("nan"))})):
                bad = copy.deepcopy(value)
                bad[key] = replacement
                torch.save(bad, path)
                with patch("rsna_knee.inference.radimagenet._checkpoint_file", return_value=path), \
                     patch("rsna_knee.inference.radimagenet._checkpoint_sha256", return_value=EXPECTED_CHECKPOINT_SHA256):
                    with self.assertRaises(ValueError):
                        load_radimagenet_model(path, self.report, self.report["checkpoint_sha256"],
                                               expected_report_sha256=EXPECTED_REPORT_SHA256,
                                               model_factory=ToyModel, checkpoint_size=path.stat().st_size)

    def test_pp_preprocessor_receives_training_resolution_and_preserves_float_masks(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            (root / "images/study/series-a").mkdir(parents=True)
            (root / "images/study/series-b").mkdir(parents=True)
            (root / "test.csv").write_text("StudyInstanceUID\nstudy\n", encoding="utf-8")
            (root / "test_series.csv").write_text(
                "StudyInstanceUID,SeriesInstanceUID\nstudy,series-b\nstudy,series-a\n", encoding="utf-8")
            calls = []

            def preprocessor(path, **kwargs):
                calls.append((Path(path).name, kwargs))
                image = np.full((2, 1, 224, 224), .25, dtype=np.float32)
                mask = np.ones_like(image, dtype=np.bool_)
                return SimpleNamespace(images=image, valid_mask=mask, metadata={})

            data = RadImageNetTestStudyDataset(root / "test.csv", root / "test_series.csv",
                                                root / "images", preprocessor=preprocessor,
                                                recovery_policy_version=1)
            images, masks = data[0]
            self.assertEqual([kwargs for _, kwargs in calls],
                             [{"max_slices": 8, "output_size": 224, "recovery_policy_version": 1}] * 2)
            self.assertEqual(images[0].dtype, torch.float32)
            self.assertEqual(masks[0].dtype, torch.bool)
            self.assertEqual(tuple(images[0].shape), (2, 1, 224, 224))
            self.assertTrue(torch.equal(images[0], torch.full_like(images[0], .25)))
            self.assertEqual([name for name, _ in calls], ["series-b", "series-a"])

    def test_actual_checkpoint_and_cached_real_study_logits_match_training_architecture(self):
        archive_path = REPO / "data/private/local-radimagenet-epoch02/full-cache-archives/local-cache-validation-000.zip"
        if not archive_path.is_file():
            self.skipTest("private 8-slice cache archive not available")
        with ZipFile(archive_path) as archive:
            cache_manifest = json.loads(archive.read("cache-manifest.json"))
            record = cache_manifest["records"][0]
            sample_raw = archive.read(record["file"])
        self.assertEqual(hashlib.sha256(sample_raw).hexdigest(), record["sha256"])
        with np.load(io.BytesIO(sample_raw), allow_pickle=False) as sample:
            images = [torch.from_numpy(sample[f"images_{index:03d}"].copy())
                      for index in range(record["series_count"])]
            masks = [torch.from_numpy(sample[f"masks_{index:03d}"].copy())
                     for index in range(record["series_count"])]
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        training_model = _model().eval()
        training_model.load_state_dict(checkpoint["model_state"], strict=True)
        inference_model = load_radimagenet_model(
            CHECKPOINT_PATH, self.report, self.report["checkpoint_sha256"],
            expected_report_sha256=EXPECTED_REPORT_SHA256,
        )
        with torch.inference_mode():
            expected = training_model(images, masks)
            actual = inference_model(images, masks)
        self.assertTrue(torch.equal(expected, actual))

    def test_canary_writer_keeps_target_order_and_model_errors_never_publish_csv(self):
        class Data:
            study_ids = ("study-a", "study-b")
            series_count = 2
            last_recovery = {"series_skipped": 0, "slice_skipped": 0, "reason_counts": {}}

            def __init__(self, *args, **kwargs):
                pass

            def __len__(self):
                return len(self.study_ids)

            def __getitem__(self, index):
                image = torch.zeros((1, 1, 224, 224), dtype=torch.float32)
                mask = torch.ones_like(image, dtype=torch.bool)
                return [image], [mask]

        with TemporaryDirectory() as td:
            root = Path(td)
            with patch("rsna_knee.inference.radimagenet.load_radimagenet_model", return_value=ToyModel()), \
                 patch("rsna_knee.inference.radimagenet.torch.cuda.is_available", return_value=False), \
                 contextlib.redirect_stdout(io.StringIO()):
                result = run_radimagenet_inference(
                    root, REPORT_PATH.parent, EXPECTED_REPORT_SHA256, EXPECTED_CHECKPOINT_SHA256,
                    root / "canary", device="cpu", max_studies=1, dataset_factory=Data,
                )
            self.assertEqual(result["study_count"], 1)
            self.assertEqual(result["test_study_count"], 2)
            with (root / "canary/submission.csv").open(encoding="utf-8") as stream:
                self.assertEqual(list(next(iter(__import__("csv").DictReader(stream))).keys()),
                                 ["StudyInstanceUID", *TARGETS])

            class BrokenModel(ToyModel):
                def forward(self, images, masks):
                    raise RuntimeError("SECRET_REPORT_AND_PATH")

            captured = io.StringIO()
            with patch("rsna_knee.inference.radimagenet.load_radimagenet_model", return_value=BrokenModel()), \
                 patch("rsna_knee.inference.radimagenet.torch.cuda.is_available", return_value=False), \
                 contextlib.redirect_stdout(captured):
                with self.assertRaises(Exception) as failure:
                    run_radimagenet_inference(
                        root, REPORT_PATH.parent, EXPECTED_REPORT_SHA256, EXPECTED_CHECKPOINT_SHA256,
                        root / "failed", device="cpu", dataset_factory=Data,
                    )
            self.assertNotIn("SECRET", str(failure.exception))
            self.assertNotIn("SECRET", captured.getvalue())
            self.assertFalse((root / "failed/submission.csv").exists())


if __name__ == "__main__":
    unittest.main()
