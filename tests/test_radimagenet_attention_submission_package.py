"""Offline package privacy and frozen-source tests for E048 attention submission."""
from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile, ZIP_DEFLATED

from scripts.build_kaggle_radimagenet_attention_submission_package import (
    _extend_source_archive, _notebook_code, validate_private_artifacts,
)
from rsna_knee.inference.radimagenet_attention import (
    EXPECTED_CHECKPOINT_SHA256, EXPECTED_REPORT_SHA256, export_attention_model,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data/private/disease-attention-mil/e048-run01/training-report.json"
CHECKPOINT = ROOT / "data/private/disease-attention-mil/e048-run01/target_attention/checkpoint-epoch-05.pt"


def archive_bytes(members):
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        for name, raw in members.items():
            archive.writestr(name, raw)
    return buffer.getvalue()


class AttentionSubmissionPackageTests(unittest.TestCase):
    def test_private_source_hashes_and_model_only_export_are_pinned(self):
        report, raw, checkpoint, checkpoint_sha = validate_private_artifacts(REPORT, CHECKPOINT)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), EXPECTED_REPORT_SHA256)
        self.assertEqual(checkpoint_sha, EXPECTED_CHECKPOINT_SHA256)
        with TemporaryDirectory() as temporary:
            exported = Path(temporary) / "attention-model-epoch-05.pt"
            proof = export_attention_model(checkpoint, REPORT, exported)
            self.assertTrue(exported.is_file())
            self.assertEqual(hashlib.sha256(exported.read_bytes()).hexdigest(), proof["sha256"])
            self.assertLess(proof["size"], checkpoint.stat().st_size)
            self.assertEqual(proof["header"]["arm"], "target_attention")
            self.assertEqual(proof["header"]["completed_epochs"], 5)

    def test_corrupt_source_report_and_overwrite_fail_before_output_mutation(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            bad_report = root / "report.json"
            bad_report.write_bytes(b"{}")
            output = root / "existing.pt"
            output.write_bytes(b"keep")
            with self.assertRaises(ValueError):
                export_attention_model(CHECKPOINT, bad_report, root / "new.pt")
            with self.assertRaisesRegex(ValueError, "output_must_be_new"):
                export_attention_model(CHECKPOINT, REPORT, output)
            self.assertEqual(output.read_bytes(), b"keep")

    def test_source_archive_adds_only_attention_adapter_and_exact_model_sources(self):
        base = archive_bytes({"rsna_knee/__init__.py": b"", "rsna_knee/inference/fold0.py": b"# codec\n"})
        result, members = _extend_source_archive(base, ROOT)
        with ZipFile(io.BytesIO(result)) as archive:
            self.assertEqual(archive.read("rsna_knee/inference/fold0.py"), b"# codec\n")
            for name in ("rsna_knee/models/resnet50.py", "rsna_knee/models/attention_mil.py",
                         "rsna_knee/inference/radimagenet_attention.py"):
                self.assertEqual(archive.read(name), (ROOT / "src" / name).read_bytes())
            accepted_path = ROOT / "data/private/radimagenet-model-submission/run01/radimagenet-inference-source.zip"
            with ZipFile(accepted_path) as accepted:
                self.assertEqual(archive.read("rsna_knee/inference/radimagenet.py"),
                                 accepted.read("rsna_knee/inference/radimagenet.py"))
        self.assertEqual(len(members), 6)

    def test_full_frozen_closure_imports_every_member_from_zip_only_isolated_python(self):
        accepted = ROOT / "data/private/radimagenet-model-submission/run01/radimagenet-inference-source.zip"
        base = accepted.read_bytes()
        result, members = _extend_source_archive(base, ROOT)
        self.assertEqual(len(members), 24)
        self.assertIn("rsna_knee/inference/radimagenet.py", members)
        with ZipFile(accepted) as frozen:
            self.assertEqual(members["rsna_knee/inference/radimagenet.py"],
                             frozen.read("rsna_knee/inference/radimagenet.py"))
        with TemporaryDirectory() as temporary:
            archive = Path(temporary) / "attention-source.zip"
            archive.write_bytes(result)
            command = (
                "import importlib,sys,zipfile; "
                "z=sys.argv[1]; sys.path.insert(0,z); "
                "names=sorted({n[:-3].replace('/','.') for n in zipfile.ZipFile(z).namelist() "
                "if n.endswith('.py') and not n.endswith('/__init__.py')}); "
                "[importlib.import_module(n) for n in names]; print('IMPORTED',len(names))"
            )
            completed = subprocess.run(
                [sys.executable, "-I", "-c", command, str(archive)],
                cwd=temporary, capture_output=True, text=True, timeout=60, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("IMPORTED", completed.stdout)

    def test_notebook_binds_attention_adapter_and_preserves_canary_row_guard(self):
        old, new = b"old source", b"new source"
        digest = hashlib.sha256(old).hexdigest()
        code = ("RUN_INFERENCE = True\nSOURCE_ROOT = '/kaggle/input/datasets/x/y'\n"
                "CHECKPOINT_SHA256 = '" + EXPECTED_CHECKPOINT_SHA256 + "'\n"
                "CHECKPOINT_SIZE = 214808723\nREPORT_SHA256 = None\n"
                "CODEC_MANIFEST_SHA256 = '" + "a" * 64 + "'\n"
                "from rsna_knee.inference.fold0 import run_inference\n"
                f"PAYLOAD = '{base64.b64encode(old).decode()}'\nif PAYLOAD and '{digest}':\n    pass\n"
                "    summary = run_inference()\n    destination_created = False\n"
                "    try:\n        pass\n    except Exception:\n        pass\n"
                "    print(json.dumps(summary, sort_keys=True))\n")
        notebook = {"cells": [{"cell_type": "markdown"}, {"source": code.splitlines(keepends=True)}]}
        result = _notebook_code(notebook, old, new, hashlib.sha256(new).hexdigest(),
                                export_sha256="b" * 64, export_size=1234,
                                report_sha256=EXPECTED_REPORT_SHA256, max_studies=3)
        self.assertIn("run_attention_inference", result)
        self.assertIn(base64.b64encode(new).decode(), result)
        self.assertIn("MAX_STUDIES = 3", result)
        self.assertIn("if MAX_STUDIES is None or summary['study_count'] == summary['test_study_count']:", result)
        self.assertIn("CHECKPOINT_SHA256 = '" + "b" * 64 + "'", result)
        self.assertIn("CHECKPOINT_SIZE = 1234", result)


if __name__ == "__main__":
    unittest.main()
