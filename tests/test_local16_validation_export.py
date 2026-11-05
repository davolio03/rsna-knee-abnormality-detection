from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts import export_local16_validation as export


class _Dataset:
    target_names = export.TARGETS

    def __init__(self, manifest_path, plan_path, archive_dir, partition):
        self.rows = [(None, {"study_sha256": export._study_sha(study_id)})
                     for study_id in ("s1", "s2")]

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return ([torch.zeros((1, 1, 2, 2))], [torch.ones((1, 1, 2, 2), dtype=torch.bool)],
                torch.full((12,), 0.25 if index == 0 else 0.75),
                torch.tensor([True, False] + [True] * 10), index)


class _Model:
    def __init__(self):
        self.is_eval = False
        self.calls = 0

    def eval(self):
        self.is_eval = True
        return self

    def to(self, device, dtype=None):
        self.device = device
        self.dtype = dtype
        return self

    def __call__(self, images, masks):
        assert self.is_eval
        assert not torch.is_grad_enabled()
        self.calls += 1
        return torch.zeros((1, 12))


class Local16ValidationExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = self.root / "manifest.json"
        self.folds = self.root / "folds.json"
        self.plan = self.root / "plan.json"
        self.archive_dir = self.root / "archives"
        self.archive_dir.mkdir()
        self.model_path = self.root / "model.pt"
        self.report_path = self.root / "report.json"
        self.model_path.write_bytes(b"unit model")
        self.report_path.write_text(json.dumps({"validation": {"masked_bce": 0.0, "brier": 0.0}}), encoding="utf-8")
        model_hash = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        report_hash = hashlib.sha256(self.report_path.read_bytes()).hexdigest()
        manifest = {"schema_version": "full-cohort-soft-label-manifest-v1", "records": [
            {"study_id": "s1", "soft_targets": {name: 0.25 for name in export.TARGETS},
             "observed_mask": {name: name != "MCL" for name in export.TARGETS}},
            {"study_id": "s2", "soft_targets": {name: 0.75 for name in export.TARGETS},
             "observed_mask": {name: name != "MCL" for name in export.TARGETS}},
        ]}
        manifest_raw = (json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        self.manifest.write_bytes(manifest_raw.encode())
        folds = {"manifest_sha256": __import__("hashlib").sha256(manifest_raw.encode()).hexdigest(),
                 "study_assignments": [{"study_id": "s1", "fold": 0}, {"study_id": "s2", "fold": 0}]}
        folds_raw = json.dumps(folds, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        self.folds.write_bytes(folds_raw.encode())
        plan = {"schema_version": "local16-cache-chunk-plan-v1", "fold": 0,
                "cache_identity": "local16-attention-run01", "partition_counts": {"train": 0, "validation": 2},
                "max_slices_per_series": 16, "output_size": 224,
                "gold_included": False,
                "manifest_sha256": __import__("hashlib").sha256(manifest_raw.encode()).hexdigest(),
                "folds_sha256": __import__("hashlib").sha256(folds_raw.encode()).hexdigest(),
                "source_sha256": "a" * 64, "preprocessing_source_sha256": "b" * 64}
        self.plan.write_bytes(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode())
        self.model = _Model()
        self.output = self.root / "out"
        self.cache_verifier_patcher = patch.object(export, "verify_full_cache", return_value={"status": "verified",
                                                                                              "partition_study_counts": {"validation": 2},
                                                                                              "no_missing_or_extra_studies": True})
        self.cache_verifier = self.cache_verifier_patcher.start()
        self.patches = [
            patch.object(export, "MODEL_SHA256", model_hash),
            patch.object(export, "REPORT_SHA256", report_hash),
            patch.object(export, "Local16CacheDataset", _Dataset),
            self.cache_verifier_patcher,
            patch.object(export, "load_attention16_model", return_value=self.model),
        ]
        for p in self.patches:
            if p is not self.cache_verifier_patcher:
                p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.temp.cleanup()

    def args(self, **overrides):
        values = dict(manifest_path=self.manifest, folds_path=self.folds, plan_path=self.plan,
                      archive_dir=self.archive_dir, output_dir=self.output,
                      model_path=self.model_path, report_path=self.report_path,
                      model_sha256=hashlib.sha256(self.model_path.read_bytes()).hexdigest(),
                      report_sha256=hashlib.sha256(self.report_path.read_bytes()).hexdigest(),
                      cache_source=self.root / "cache-source.zip", run=True, max_studies=1,
                      resume=False, device="cpu")
        values.update(overrides)
        return values

    def test_default_plan_never_loads_model_or_cache(self):
        with patch.object(export, "load_attention16_model", side_effect=AssertionError):
            plan = export.export_validation(**self.args(run=False, max_studies=None))
        self.assertEqual(plan["status"], "planned")
        self.assertEqual(plan["validation_studies"], 2)

    def test_exports_private_masked_rows_with_pins_and_inference_only(self):
        result = export.export_validation(**self.args())
        self.cache_verifier.assert_not_called()
        row = json.loads((self.output / "study-000001.json").read_text(encoding="utf-8"))
        self.assertEqual(result["completed"], 1)
        self.assertTrue(self.model.is_eval)
        self.assertEqual(self.model.calls, 1)
        self.assertEqual(len(row["predictions"]), 12)
        self.assertEqual(len(row["logits"]), 12)
        self.assertEqual(row["observed_mask"][1], False)
        self.assertEqual(row["labels"][1], 0.25)
        self.assertEqual(row["study_id"], "s1")
        self.assertEqual(row["provenance"]["model_sha256"], hashlib.sha256(self.model_path.read_bytes()).hexdigest())
        self.assertEqual(row["provenance"]["device"], "cpu")
        self.assertEqual(row["provenance"]["precision"], "float32")
        self.assertFalse(row["provenance"]["tf32_enabled"])
        self.assertEqual(row["provenance"]["matmul_precision"], "highest")
        self.assertEqual(row["provenance"]["exporter_sha256"], export._sha256(Path(export.__file__)))
        self.assertEqual(torch.get_float32_matmul_precision(), "highest")
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.allow_tf32)
        self.assertIs(self.model.dtype, torch.float32)

    def test_resume_rejects_manifest_label_and_mask_tampering(self):
        for field, index, replacement in (("labels", 0, 0.5), ("observed_mask", 1, True)):
            with self.subTest(field=field):
                output = self.root / f"tampered-{field}"
                export.export_validation(**self.args(output_dir=output))
                row_path = output / "study-000001.json"
                row = json.loads(row_path.read_text(encoding="utf-8"))
                row[field][index] = replacement
                row_path.write_text(json.dumps(row), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "resume_row_invalid"):
                    export.export_validation(**self.args(output_dir=output, resume=True))

    def test_resume_rejects_logit_probability_mismatch(self):
        export.export_validation(**self.args())
        row_path = self.output / "study-000001.json"
        row = json.loads(row_path.read_text(encoding="utf-8"))
        row["logits"] = [2.0] * 12
        row_path.write_text(json.dumps(row), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "resume_row_invalid"):
            export.export_validation(**self.args(resume=True))

    def test_stable_logit_bce_handles_extreme_finite_values(self):
        metrics = export._masked_metrics(np.asarray([[1000.0, -1000.0]]),
                                         np.asarray([[1.0, 0.0]]),
                                         np.asarray([[0.0, 1.0]]),
                                         np.asarray([[True, True]]))
        self.assertTrue(np.isfinite(metrics["masked_bce"]))
        self.assertEqual(metrics["masked_bce"], 1000.0)

    def test_existing_output_requires_resume_and_resume_checks_pins(self):
        export.export_validation(**self.args())
        with self.assertRaisesRegex(ValueError, "output_exists"):
            export.export_validation(**self.args(max_studies=2))
        with self.assertRaisesRegex(ValueError, "model_or_report_pin_mismatch"):
            export.export_validation(**self.args(max_studies=2, resume=True,
                                                  model_sha256="0" * 64))

    def test_resume_continues_prefix_and_refuses_cache_order_mismatch(self):
        export.export_validation(**self.args())
        resumed = export.export_validation(**self.args(max_studies=2, resume=True))
        self.assertEqual(resumed["completed"], 2)
        self.assertEqual(self.model.calls, 2)
        self.assertTrue((self.output / "study-000002.json").is_file())
        with patch.object(export, "Local16CacheDataset", lambda *a, **k: _ReorderedDataset()):
            with self.assertRaisesRegex(ValueError, "cache_study_order_mismatch"):
                export.export_validation(**self.args(output_dir=self.root / "other"))

    def test_bad_logits_and_duplicate_manifest_ids_are_rejected(self):
        with patch.object(_Model, "__call__", lambda *args: torch.full((1, 12), float("nan"))):
            with self.assertRaisesRegex(ValueError, "prediction_invalid"):
                export.export_validation(**self.args())
        data = json.loads(self.manifest.read_text(encoding="utf-8"))
        data["records"][1]["study_id"] = "s1"
        self.manifest.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "validation_cohort_invalid"):
            export.export_validation(**self.args(output_dir=self.root / "bad"))


class _ReorderedDataset(_Dataset):
    def __init__(self, *args, **kwargs):
        self.rows = [(None, {"study_sha256": export._study_sha(study_id)}) for study_id in ("s2", "s1")]


if __name__ == "__main__":
    unittest.main()
