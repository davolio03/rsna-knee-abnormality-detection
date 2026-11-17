"""Source closure, exact training-payload and notebook row-guard tests."""
import base64
import hashlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile, ZIP_DEFLATED

from scripts.build_kaggle_radimagenet_submission_package import (
    CODEC_MANIFEST_SHA256,
    _extend_source_archive,
    _notebook_code,
    validate_private_artifacts,
)
from rsna_knee.inference.radimagenet import EXPECTED_CHECKPOINT_SHA256, EXPECTED_REPORT_SHA256


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data/private/local-radimagenet-final-epochs/final-run01/training-report.json"
CHECKPOINT = ROOT / "data/private/local-radimagenet-final-epochs/final-run01/checkpoint-epoch-05.pt"
TRAINING_SOURCE = ROOT / "data/private/resnet50-pretraining-preparation/run01/resnet50-recipe-source.zip"


def archive_bytes(members):
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        for name, raw in members.items():
            archive.writestr(name, raw)
    return buffer.getvalue()


class RadImageNetPackageTests(unittest.TestCase):
    def test_actual_private_pins_and_frozen_model_source_are_verified(self):
        report, raw, checkpoint_hash, size, training_raw = validate_private_artifacts(
            REPORT, CHECKPOINT, TRAINING_SOURCE)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), EXPECTED_REPORT_SHA256)
        self.assertEqual(checkpoint_hash, EXPECTED_CHECKPOINT_SHA256)
        self.assertEqual(size, CHECKPOINT.stat().st_size)
        self.assertEqual(report["fingerprint"], "5c08ab6be89773802ce4f18fa2209a7e85ec88f2703b5caaded9bd940025a920")
        base = archive_bytes({"rsna_knee/__init__.py": b"", "rsna_knee/inference/fold0.py": b"# frozen\n"})
        extended, members = _extend_source_archive(base, ROOT, training_raw)
        with ZipFile(io.BytesIO(extended)) as archive:
            self.assertEqual(archive.read("rsna_knee/inference/fold0.py"), b"# frozen\n")
            self.assertEqual(archive.read("rsna_knee/models/resnet50.py"),
                             (ROOT / "src/rsna_knee/models/resnet50.py").read_bytes())
            self.assertEqual(archive.read("rsna_knee/inference/radimagenet.py"),
                             (ROOT / "src/rsna_knee/inference/radimagenet.py").read_bytes())
        self.assertEqual(len(members), 4)

    def test_corrupted_report_checkpoint_or_training_source_fails_closed(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            report = root / "report.json"
            checkpoint = root / "checkpoint.pt"
            source = root / "source.zip"
            report.write_bytes(b"{}")
            checkpoint.write_bytes(b"checkpoint")
            source.write_bytes(b"source")
            with self.assertRaisesRegex(ValueError, "report_pin_mismatch"):
                validate_private_artifacts(report, checkpoint, source)

    def test_generated_code_embeds_exact_archive_and_publishes_canary_only_for_full_visible_rows(self):
        old = b"old archive bytes"
        new = b"new archive bytes"
        digest = hashlib.sha256(old).hexdigest()
        source = (
            "RUN_INFERENCE = True\n"
            "CHECKPOINT_SHA256 = None\nCHECKPOINT_SIZE = None\nREPORT_SHA256 = None\n"
            f"CODEC_MANIFEST_SHA256 = {CODEC_MANIFEST_SHA256!r}\n"
            "from rsna_knee.inference.fold0 import run_inference\n"
            f"PAYLOAD = '{base64.b64encode(old).decode()}'\n"
            f"if PAYLOAD and '{digest}':\n    pass\n"
            "if RUN_INFERENCE:\n"
            "    summary = run_inference()\n"
            "    destination_created = False\n"
            "    try:\n        pass\n    except Exception:\n        pass\n"
            "    print(json.dumps(summary, sort_keys=True))\n"
        )
        notebook = {"cells": [{"cell_type": "markdown"}, {"source": source.splitlines(keepends=True)}]}
        result = _notebook_code(notebook, old, new, hashlib.sha256(new).hexdigest(),
                                report_sha256="a" * 64, checkpoint_sha256="b" * 64,
                                checkpoint_size=214216115, fingerprint="c" * 64, max_studies=3)
        self.assertIn("MAX_STUDIES = 3", result)
        self.assertIn("if MAX_STUDIES is None or summary['study_count'] == summary['test_study_count']:", result)
        self.assertIn("run_radimagenet_inference", result)
        self.assertIn(base64.b64encode(new).decode(), result)
        self.assertIn(CODEC_MANIFEST_SHA256, result)
        self.assertIn("CHECKPOINT_SIZE = 214216115", result)


if __name__ == "__main__":
    unittest.main()
