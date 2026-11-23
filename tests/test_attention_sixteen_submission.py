from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile, ZIP_DEFLATED

from scripts.build_kaggle_submission_package import build_submission_package
from rsna_knee.inference.attention_sixteen import (
    MAX_SLICES, SOURCE_CHECKPOINT_SHA256, SOURCE_REPORT_SHA256,
    Attention16TestStudyDataset, export_attention16_model, sanitized_attention16_provenance,
    run_attention16_inference, validate_attention16_source_report,
)
from scripts.build_kaggle_attention_sixteen_submission_package import (
    _extend_source_archive, _notebook_code, validate_private_artifacts,
)
from private_artifacts import requires

needs_artifacts = requires('data/private/attention-sixteen-slices/local16-attention-run01/training-report.json',
                           'data/private/attention-sixteen-slices/local16-attention-run01/target_attention/checkpoint-epoch-05.pt')


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data/private/attention-sixteen-slices/local16-attention-run01/training-report.json"


def archive_bytes(members):
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        for name, raw in members.items():
            archive.writestr(name, raw)
    return buffer.getvalue()


class AttentionSixteenSubmissionTests(unittest.TestCase):
    @needs_artifacts
    def test_pins_and_report_bind_complete_epoch_five_sixteen_slice_run(self):
        raw = REPORT.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), SOURCE_REPORT_SHA256)
        self.assertEqual(MAX_SLICES, 16)
        self.assertEqual(SOURCE_CHECKPOINT_SHA256,
                         "631536a30d0da51c0f6872fb45fac15b5bcb0f6185b5a53708eed1ef2be42f21")
        validate_attention16_source_report(__import__("json").loads(raw))

    @needs_artifacts
    def test_report_rejects_mutated_slice_count_or_incomplete_epoch_evidence(self):
        import json
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        report["configuration"]["max_slices"] = 8
        with self.assertRaisesRegex(ValueError, "source_report_provenance_mismatch"):
            validate_attention16_source_report(report)

    @needs_artifacts
    def test_private_report_is_sanitized_and_source_artifacts_are_pinned(self):
        report, raw, checkpoint, checkpoint_hash, sanitized = validate_private_artifacts(
            REPORT, ROOT / "data/private/attention-sixteen-slices/local16-attention-run01/target_attention/checkpoint-epoch-05.pt")
        safe = json.loads(sanitized)
        self.assertEqual(checkpoint_hash, SOURCE_CHECKPOINT_SHA256)
        self.assertNotIn("output_dir", safe)
        self.assertNotIn("C:\\Users", sanitized.decode("utf-8"))
        self.assertEqual(safe["configuration"]["max_slices"], 16)
        self.assertEqual(safe["completed_epochs"], 5)

    @needs_artifacts
    def test_model_only_export_drops_optimizer_rng_and_keeps_sixteen_slice_pin(self):
        checkpoint = ROOT / "data/private/attention-sixteen-slices/local16-attention-run01/target_attention/checkpoint-epoch-05.pt"
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pt"
            proof = export_attention16_model(checkpoint, REPORT, path)
            value = __import__("torch").load(path, map_location="cpu", weights_only=True)
            self.assertEqual(value["max_slices"], 16)
            self.assertEqual(value["completed_epochs"], 5)
            self.assertNotIn("optimizer_state", value)
            self.assertNotIn("rng_state", value)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), proof["sha256"])

    def test_dataset_requests_sixteen_slices_and_keeps_series_attention_inputs(self):
        dataset = object.__new__(Attention16TestStudyDataset)
        dataset.last_recovery = None
        dataset._series = [["series"]]
        dataset.study_ids = ["study"]
        dataset.image_root = Path("unused")
        dataset.recovery_policy_version = 1
        captured = {}
        def preprocess(path, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(images=__import__("numpy").ones((16, 1, 224, 224), dtype="float32"),
                                   valid_mask=__import__("numpy").ones((16, 1, 224, 224), dtype="uint8"),
                                   metadata={})
        dataset.preprocessor = preprocess
        with patch("rsna_knee.inference.attention_sixteen._contained", return_value=Path("series")):
            images, masks = dataset[0]
        self.assertEqual(captured["max_slices"], 16)
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].shape[0], 16)
        self.assertEqual(masks[0].shape[0], 16)

    @needs_artifacts
    def test_canary_runs_real_model_interface_but_never_writes_submission_csv(self):
        source_report = json.loads(REPORT.read_text(encoding="utf-8"))
        provenance = sanitized_attention16_provenance(source_report)
        report_raw = (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode()
        report_hash = hashlib.sha256(report_raw).hexdigest()
        class ToyModel:
            def to(self, device):
                return self
            def eval(self):
                return self
            def __call__(self, images, masks):
                return __import__("torch").zeros((1, 12))
        class ToyDataset:
            study_ids = ["opaque-study-id"]
            last_recovery = None
            def __len__(self):
                return 1
            def __getitem__(self, index):
                image = __import__("torch").zeros((16, 1, 224, 224))
                mask = __import__("torch").ones((16, 1, 224, 224), dtype=__import__("torch").bool)
                return [image], [mask]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "training-report.json").write_bytes(report_raw)
            output = root / "canary"
            with patch("rsna_knee.inference.attention_sixteen.load_attention16_model", return_value=ToyModel()):
                summary = run_attention16_inference(
                    root, source, report_hash, "a" * 64, output, checkpoint_size=1,
                    device="cpu", max_studies=1, dataset_factory=lambda *args, **kwargs: ToyDataset())
            self.assertFalse(summary["submission_written"])
            self.assertFalse((output / "submission.csv").exists())
            self.assertEqual(len(ToyDataset.study_ids), summary["study_count"])

    @needs_artifacts
    def test_source_archive_is_closed_over_sixteen_slice_adapter_and_native_model(self):
        base = archive_bytes({"rsna_knee/__init__.py": b"", "rsna_knee/inference/fold0.py": b"# codec\n"})
        result, members = _extend_source_archive(base, ROOT)
        self.assertIn("rsna_knee/inference/attention_sixteen.py", members)
        adapter = members["rsna_knee/inference/attention_sixteen.py"].decode("utf-8")
        self.assertIn("max_slices=MAX_SLICES", adapter)
        self.assertEqual(len(members), 6)
        with ZipFile(io.BytesIO(result)) as archive:
            self.assertEqual(set(archive.namelist()), set(members))

    @needs_artifacts
    def test_notebook_embeds_closed_source_and_keeps_canary_from_publishing_submission(self):
        export_sha = "a" * 64
        report_sha = "b" * 64
        with TemporaryDirectory() as temporary:
            base_dir = Path(temporary) / "base"
            base = build_submission_package(
                ROOT, base_dir, source_root="/kaggle/input/datasets/owner/model",
                checkpoint_sha256=export_sha, checkpoint_size=1000, report_sha256=report_sha,
                codec_manifest_sha256="c" * 64, run_inference=True, recovery_policy_version=1)
            base_archive = (base_dir / "fold0-inference-source.zip").read_bytes()
            notebook = json.loads(Path(base["notebook"]).read_text(encoding="utf-8"))
            archive, members = _extend_source_archive(base_archive, ROOT)
            digest = hashlib.sha256(archive).hexdigest()
            code = _notebook_code(notebook, base_archive, archive, digest,
                                  export_sha256=export_sha, export_size=1000,
                                  report_sha256=report_sha, max_studies=3,
                                  source_root="/kaggle/input/datasets/davidsulleiro/rsna-knee-attention16-epoch05")
        self.assertIn("run_attention16_inference", code)
        self.assertIn("max_slices=MAX_SLICES", members["rsna_knee/inference/attention_sixteen.py"].decode())
        self.assertIn("MAX_STUDIES = 3", code)
        self.assertIn("SOURCE_ROOT = '/kaggle/input/datasets/davidsulleiro/rsna-knee-attention16-epoch05'", code)
        self.assertIn("if MAX_STUDIES is None and summary['study_count'] == summary['test_study_count']:", code)
        self.assertNotIn("C:\\Users", code)
        self.assertIn("attach the private E050 attention16 model-only dataset", code)
        self.assertNotIn("execution 361175523", code)

    def test_builder_cli_exposes_source_output_root_override(self):
        completed = subprocess.run(
            [sys.executable, "scripts/build_kaggle_attention_sixteen_submission_package.py", "--help"],
            cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--source-output-root", completed.stdout)


if __name__ == "__main__":
    unittest.main()
