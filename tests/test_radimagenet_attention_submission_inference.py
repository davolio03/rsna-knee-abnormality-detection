"""Strict inference and packaging contracts for the completed E048 attention arm."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import torch

from rsna_knee.inference.radimagenet_attention import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_REPORT_SHA256,
    export_attention_model,
    load_attention_model,
    validate_attention_report,
)
from rsna_knee.models.attention_mil import RadImageNetAttentionStudyModel
from rsna_knee.models.resnet50 import ResNet50StudyModel
from torchvision.models import resnet50


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "data/private/disease-attention-mil/e048-run01/training-report.json"
CHECKPOINT_PATH = ROOT / "data/private/disease-attention-mil/e048-run01/target_attention/checkpoint-epoch-05.pt"


class AttentionSubmissionInferenceTests(unittest.TestCase):
    def setUp(self):
        self.raw = REPORT_PATH.read_bytes()
        self.report = json.loads(self.raw)

    def test_frozen_report_pins_completed_attention_epoch_five(self):
        self.assertEqual(hashlib.sha256(self.raw).hexdigest(), EXPECTED_REPORT_SHA256)
        self.assertIsNone(validate_attention_report(self.report))
        self.assertEqual(self.report["arms"]["target_attention"]["completed_epochs"], 5)

    def test_wrong_arm_epoch_or_fingerprint_is_rejected(self):
        for path, value in ((("arms", "target_attention", "arm"), "equal_mean"),
                            (("arms", "target_attention", "completed_epochs"), 4),
                            (("fingerprint",), "0" * 64)):
            bad = json.loads(json.dumps(self.report))
            cursor = bad
            for key in path[:-1]:
                cursor = cursor[key]
            cursor[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_attention_report(bad)

    def test_hash_rejection_happens_before_deserialization(self):
        with patch("torch.load") as load:
            with self.assertRaises(ValueError):
                load_attention_model(CHECKPOINT_PATH, self.report, "0" * 64,
                                     expected_report_sha256=EXPECTED_REPORT_SHA256)
        load.assert_not_called()

    def test_actual_epoch_five_attention_checkpoint_loads_strictly(self):
        with __import__("tempfile").TemporaryDirectory() as temporary:
            export = Path(temporary) / "attention-model-epoch-05.pt"
            proof = export_attention_model(CHECKPOINT_PATH, REPORT_PATH, export)
            model = load_attention_model(
                export, self.report, proof["sha256"], expected_report_sha256=EXPECTED_REPORT_SHA256,
                checkpoint_size=proof["size"])
            self.assertEqual(model.__class__.__name__, "RadImageNetAttentionStudyModel")
            self.assertFalse(model.training)
            self.assertTrue(all(torch.isfinite(value).all() for value in model.state_dict().values()))

    def test_real_cached_study_logits_match_separately_loaded_trained_attention_model_exactly(self):
        archive_path = ROOT / "data/private/local-radimagenet-epoch02/full-cache-archives/local-cache-validation-000.zip"
        with ZipFile(archive_path) as archive:
            manifest = json.loads(archive.read("cache-manifest.json"))
            record = manifest["records"][0]
            raw = archive.read(record["file"])
        self.assertEqual(hashlib.sha256(raw).hexdigest(), record["sha256"])
        with np.load(io.BytesIO(raw), allow_pickle=False) as sample:
            images = [torch.from_numpy(sample[f"images_{index:03d}"].copy())
                      for index in range(record["series_count"])]
            masks = [torch.from_numpy(sample[f"masks_{index:03d}"].copy())
                     for index in range(record["series_count"])]
        trained = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        base = ResNet50StudyModel(resnet50(weights=None), arm="radimagenet", seed=42, slice_microbatch=2)
        reference = RadImageNetAttentionStudyModel.from_base(base).eval()
        reference.load_state_dict(trained["model_state"], strict=True)
        with __import__("tempfile").TemporaryDirectory() as temporary:
            export = Path(temporary) / "attention-model-epoch-05.pt"
            proof = export_attention_model(CHECKPOINT_PATH, REPORT_PATH, export)
            actual = load_attention_model(export, self.report, proof["sha256"],
                                          expected_report_sha256=EXPECTED_REPORT_SHA256,
                                          checkpoint_size=proof["size"])
            with torch.inference_mode():
                expected_logits = reference(images, masks)
                actual_logits = actual(images, masks)
        self.assertTrue(torch.equal(expected_logits, actual_logits))

    def test_model_export_header_and_tensor_state_are_strict(self):
        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))

        valid_header = {
            "schema": "radimagenet-attention-model-only-v1", "status": "complete",
            "arm": "target_attention", "fingerprint": "13658c21bba3f35bb0d9fd392034af852125f5f2639006e3937d6a82f85108cb",
            "completed_epochs": 5, "report_sha256": EXPECTED_REPORT_SHA256,
            "source_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "history_sha256": hashlib.sha256(json.dumps(
                self.report["arms"]["target_attention"]["epochs"], sort_keys=True,
                separators=(",", ":")).encode()).hexdigest(),
            "attention_model_source_sha256": "10367edd912daf18a8cfa2bcde2f60e103e7af224413123c362ce3558e3c01ce",
            "resnet_model_source_sha256": "f54085f53c640cb3cb33282f57cea10388b21438e659426405a53c39c3d3e3b0",
            "output_size": 224, "max_slices": 8, "normalization": "radimagenet_2x_minus_1",
            "slice_microbatch": 2,
            "pooling": "target_specific_slice_attention_then_series_attention_logits",
        }
        with __import__("tempfile").TemporaryDirectory() as temporary:
            path = Path(temporary) / "export.pt"

            def load(payload):
                torch.save(payload, path)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                return load_attention_model(path, self.report, digest,
                                            expected_report_sha256=EXPECTED_REPORT_SHA256,
                                            checkpoint_size=path.stat().st_size,
                                            model_factory=ToyModel)

            payload = {**valid_header, "model_state": ToyModel().state_dict()}
            self.assertFalse(load(payload).training)
            for key, bad_value in (("schema", "unknown"), ("arm", "equal_mean"),
                                   ("completed_epochs", 4), ("fingerprint", "0" * 64),
                                   ("history_sha256", "0" * 64)):
                bad = dict(payload, **{key: bad_value})
                with self.subTest(key=key), self.assertRaises(ValueError):
                    load(bad)
            for state in ({"other": torch.zeros(2)}, {"weight": torch.zeros(3)},
                          {"weight": torch.zeros(2, dtype=torch.float64)},
                          {"weight": torch.tensor([float("nan"), 0.0])}):
                with self.subTest(state=tuple(state)), self.assertRaises(ValueError):
                    load(dict(valid_header, model_state=state))


if __name__ == "__main__":
    unittest.main()
