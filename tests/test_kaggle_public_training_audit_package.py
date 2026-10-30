"""Private full-public-training inference audit package preparation checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

from scripts.build_kaggle_public_training_audit import build_public_training_audit_package
from rsna_knee.inference.public_training_audit import load_public_training_plan


class PublicTrainingAuditPackageTests(unittest.TestCase):
    def test_package_freezes_id_only_plan_and_real_runner_diagnostic_notebook(self):
        repository = Path(__file__).resolve().parents[1]
        with TemporaryDirectory(dir=repository) as temp:
            output = Path(temp) / "run10"
            result = build_public_training_audit_package(
                repository, output,
                source_root="/kaggle/input/notebooks/davidsulleiro/rsna-knee-resnet18-frozen-backbone-run01/resnet18-frozen-backbone/run01",
                checkpoint_sha256="d85f6531aeb8aa3035f05997ef3dfbb2f88a97c288c07b8786a13795fdf33447",
                checkpoint_size=44811979,
                report_sha256="5daef84f7ed8ad9aa7173989fe80fc168a900fc196025feeb98ee9c8c1792dc1",
                codec_manifest_sha256="50f39d3f627b3dc06a8080ba51dd6424898ef34fa4c4e21a84593dd326eb776d",
            )
            notebook_path = Path(result["notebook"])
            notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
            code = "".join(notebook["cells"][1]["source"])
            compile(code, str(notebook_path), "exec")
            self.assertIn("RUN_AUDIT = True", code)
            self.assertIn("run_public_training_inference_audit", code)
            self.assertIn(
                "/kaggle/input/datasets/davidsulleiro/rsna-knee-public-training-inference-plan/"
                "public-training-inference-plan/public-training-inference-plan.json",
                code,
            )
            self.assertNotIn("/kaggle/working/submission.csv", code)
            self.assertIn("Never submit this diagnostic notebook", " ".join(notebook["cells"][0]["source"]))
            embedded_source = __import__("re").search(
                r"payload = base64\.b64decode\('([^']+)'", code).group(1)
            embedded_bytes = __import__("base64").b64decode(embedded_source, validate=True)
            archive_bytes = Path(result["source_archive"]).read_bytes()
            expected_digest = __import__("re").search(
                r"hashlib\.sha256\(payload\)\.hexdigest\(\) != '([0-9a-f]{64})'", code
            ).group(1)
            self.assertEqual(hashlib.sha256(embedded_bytes).hexdigest(), expected_digest)
            self.assertEqual(embedded_bytes, archive_bytes)

            # Execute the generated bootstrap only through archive extraction, before codec setup.
            import tempfile
            prefix = code.split("        if CODEC_MANIFEST_SHA256:", 1)[0] + "    except Exception:\n        raise\n"
            with tempfile.TemporaryDirectory() as working_dir:
                safe_prefix = prefix.replace(
                    "Path('/kaggle/working/fold0-inference-source.zip')",
                    f"Path({str(Path(working_dir) / 'fold0-inference-source.zip')!r})",
                )
                safe_prefix = __import__("re").sub(
                    r"        if not source\.is_absolute\(\) or not source\.is_relative_to\('/kaggle/input'\):",
                    "        if False:", safe_prefix,
                )
                namespace = {}
                exec(compile(safe_prefix, "generated-bootstrap-prefix", "exec"), namespace)
                self.assertEqual(Path(working_dir, "fold0-inference-source.zip").read_bytes(), archive_bytes)

            plan_zip = Path(result["plan_archive"])
            self.assertEqual(result["study_count"], 100)
            self.assertEqual(result["series_count"], 546)
            self.assertEqual(result["plan_sha256"], hashlib.sha256(plan_zip.read_bytes()).hexdigest())
            with ZipFile(plan_zip) as archive:
                self.assertEqual(archive.namelist(), ["public-training-inference-plan.json"])
                raw_plan = archive.read(archive.namelist()[0])
            plan_payload_sha256 = hashlib.sha256(raw_plan).hexdigest()
            self.assertEqual(
                plan_payload_sha256,
                "7e1afed29ff7edcac646391cdeab382d66812cd748847d0deb794531dab85076",
            )
            extracted_plan = Path(temp) / "public-training-inference-plan.json"
            extracted_plan.write_bytes(raw_plan)
            loaded_extracted = load_public_training_plan(
                extracted_plan, plan_payload_sha256)
            self.assertEqual(loaded_extracted, json.loads(raw_plan))
            self.assertIn(
                f"expected_plan_sha256='{plan_payload_sha256}'", code)
            self.assertEqual(
                load_public_training_plan(plan_zip, hashlib.sha256(plan_zip.read_bytes()).hexdigest()),
                json.loads(raw_plan),
            )
            with self.assertRaisesRegex(ValueError, "^audit_plan_archive_pin_mismatch$"):
                load_public_training_plan(extracted_plan, "0" * 64)
            tampered_json = json.loads(raw_plan)
            tampered_json["seed"] = 43
            tampered_path = Path(temp) / "tampered-plan.json"
            tampered_bytes = json.dumps(tampered_json).encode("utf-8")
            tampered_path.write_bytes(tampered_bytes)
            with self.assertRaisesRegex(ValueError, "^audit_plan_pin_mismatch$"):
                load_public_training_plan(tampered_path, hashlib.sha256(tampered_bytes).hexdigest())
            plan = json.loads(raw_plan)
            self.assertEqual((plan["study_count"], plan["series_count"]), (100, 546))
            self.assertEqual(plan["plane_counts"], {"Axial": 131, "Coronal": 197, "Sagittal": 218})
            self.assertNotIn("gold_ids", plan)
            self.assertNotIn("prior_ids", plan)
            self.assertNotIn("labels", raw_plan.decode("utf-8").lower())
            self.assertNotIn("reports", raw_plan.decode("utf-8").lower())

            source_archive = Path(result["source_archive"])
            with ZipFile(source_archive) as archive:
                self.assertIn("rsna_knee/inference/fold0.py", archive.namelist())
                self.assertIn("rsna_knee/inference/public_training_audit.py", archive.namelist())
                self.assertEqual(archive.read("rsna_knee/inference/fold0.py"),
                                 (repository / "src/rsna_knee/inference/fold0.py").read_bytes())
                self.assertEqual(archive.read("smoke_submission.py"),
                                 (repository / "scripts/smoke_submission.py").read_bytes())
            proof = json.loads((output / "preparation-proof.json").read_text(encoding="utf-8"))
            self.assertFalse(proof["remote_executed"])
            self.assertFalse(proof["competition_submission"])
            self.assertEqual(proof["checkpoint_sha256"], "d85f6531aeb8aa3035f05997ef3dfbb2f88a97c288c07b8786a13795fdf33447")


if __name__ == "__main__":
    unittest.main()
