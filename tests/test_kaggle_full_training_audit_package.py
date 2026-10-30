"""Full non-Gold public-training audit preparation checks."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import base64
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

from scripts.build_kaggle_public_training_audit import build_full_public_training_audit_package
from rsna_knee.inference.public_training_audit import (
    load_public_training_plan,
    stage_training_metadata,
    validate_public_training_plan,
)


class FullTrainingAuditPackageTests(unittest.TestCase):
    def test_full_package_freezes_all_non_gold_studies_without_label_material(self):
        repository = Path(__file__).resolve().parents[1]
        with TemporaryDirectory(dir=repository) as temp:
            output = Path(temp) / "run11"
            result = build_full_public_training_audit_package(repository, output)
            plan_path = Path(result["plan_json"])
            plan_bytes = plan_path.read_bytes()
            plan = json.loads(plan_bytes)
            self.assertEqual(hashlib.sha256(plan_bytes).hexdigest(), result["plan_sha256"])
            self.assertEqual(load_public_training_plan(plan_path, result["plan_sha256"]), plan)
            with self.assertRaisesRegex(ValueError, "^audit_plan_archive_pin_mismatch$"):
                load_public_training_plan(plan_path, "0" * 64)
            self.assertEqual(validate_public_training_plan(plan), tuple(
                study["study_id"] for study in plan["studies"]))
            self.assertEqual(len(plan["studies"]), 4349)
            self.assertEqual(plan["study_count"], 4349)
            self.assertEqual(plan["series_count"], 24035)
            self.assertEqual(plan["plane_counts"], {
                "Axial": 5817, "Coronal": 8490, "Sagittal": 9728,
            })
            self.assertEqual(plan["gold_excluded_study_count"], 58)
            self.assertEqual(plan["gold_excluded_series_count"], 336)
            self.assertEqual(plan["prior_study_count_included"], 500)
            self.assertNotIn("gold_ids", plan)
            self.assertNotIn("prior_ids", plan)
            self.assertNotIn("labels", plan)
            self.assertNotIn("reports", plan)
            self.assertNotIn("label", plan_bytes.decode("utf-8").lower())
            self.assertNotIn("report", plan_bytes.decode("utf-8").lower())

            with (repository / "data/private/gold58_study_ids.csv").open(
                newline="", encoding="utf-8-sig"
            ) as stream:
                gold_ids = {row["StudyInstanceUID"] for row in csv.DictReader(stream)}
            plan_ids = {study["study_id"] for study in plan["studies"]}
            self.assertEqual(len(gold_ids), 58)
            self.assertFalse(gold_ids & plan_ids)
            with (repository / "data/private/training-compatibility-audit/run01-preparation-v2/audit-plan.json").open(
                encoding="utf-8"
            ) as stream:
                source_plan = json.load(stream)
            prior_ids = set(source_plan["prior_ids"])
            self.assertEqual(len(prior_ids), 500)
            self.assertTrue(prior_ids.issubset(plan_ids))
            expected_series = {}
            with (repository / "data/train_series.csv").open(newline="", encoding="utf-8-sig") as stream:
                for row in csv.DictReader(stream):
                    if row["StudyInstanceUID"] not in gold_ids:
                        expected_series.setdefault(row["StudyInstanceUID"], []).append(
                            (row["SeriesInstanceUID"], row["Anatomical_Plane"])
                        )
            actual_series = {
                study["study_id"]: [(row["series_id"], row["plane"]) for row in study["series"]]
                for study in plan["studies"]
            }
            self.assertEqual(actual_series, expected_series)
            staged = stage_training_metadata(
                plan, repository / "data/train_series.csv", Path(temp) / "staged"
            )
            with staged[0].open(newline="", encoding="utf-8") as stream:
                staged_study_rows = list(csv.DictReader(stream))
            with staged[1].open(newline="", encoding="utf-8") as stream:
                staged_series_rows = list(csv.DictReader(stream))
            self.assertEqual(len(staged_study_rows), 4349)
            self.assertEqual(len(staged_series_rows), 24035)
            self.assertEqual(set(staged_study_rows[0]), {"StudyInstanceUID"})
            self.assertEqual(set(staged_series_rows[0]), {"StudyInstanceUID", "SeriesInstanceUID"})

            duplicate_study = json.loads(plan_bytes)
            duplicate_study["studies"][1] = duplicate_study["studies"][0]
            with self.assertRaisesRegex(ValueError, "^audit_plan_invalid$"):
                validate_public_training_plan(duplicate_study)
            missing_study = json.loads(plan_bytes)
            missing_study["studies"].pop()
            with self.assertRaisesRegex(ValueError, "^audit_plan_pin_mismatch$"):
                validate_public_training_plan(missing_study)
            missing_series = json.loads(plan_bytes)
            missing_series["studies"][0]["series"].pop()
            with self.assertRaisesRegex(ValueError, "^audit_plan_counts_mismatch$"):
                validate_public_training_plan(missing_series)
            altered_gold_source = json.loads(plan_bytes)
            altered_gold_source["gold_source_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "^audit_plan_pin_mismatch$"):
                validate_public_training_plan(altered_gold_source)

            notebook_path = Path(result["notebook"])
            notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
            code = "".join(notebook["cells"][1]["source"])
            compile(code, str(notebook_path), "exec")
            self.assertIn("RUN_AUDIT = True", code)
            self.assertIn("full-training-inference-plan.json", code)
            self.assertIn(f"expected_plan_sha256='{result['plan_sha256']}'", code)
            self.assertNotIn("/kaggle/working/submission.csv", code)
            self.assertNotIn(plan_bytes.decode("utf-8"), code)
            self.assertFalse(result["competition_submission"])
            with ZipFile(result["source_archive"]) as archive:
                self.assertEqual(archive.read("rsna_knee/inference/fold0.py"),
                                 (repository / "src/rsna_knee/inference/fold0.py").read_bytes())
                self.assertEqual(archive.read("rsna_knee/data/dicom_preprocessing.py"),
                                 (repository / "src/rsna_knee/data/dicom_preprocessing.py").read_bytes())
                archive_members = archive.namelist()
                source_zip_bytes = Path(result["source_archive"]).read_bytes()
            self.assertNotIn("train_series.csv", archive_members)
            self.assertFalse(any("gold58" in name.lower() for name in archive_members))
            payload_literal = re.search(r"payload = base64\.b64decode\('([^']+)'", code).group(1)
            embedded_source = base64.b64decode(payload_literal, validate=True)
            expected_archive_sha = re.search(
                r"hashlib\.sha256\(payload\)\.hexdigest\(\) != '([0-9a-f]{64})'", code
            ).group(1)
            self.assertEqual(embedded_source, source_zip_bytes)
            self.assertEqual(hashlib.sha256(embedded_source).hexdigest(), expected_archive_sha)
            prefix = code.split("        if CODEC_MANIFEST_SHA256:", 1)[0] + "    except Exception:\n        raise\n"
            with TemporaryDirectory() as working_dir:
                safe_prefix = prefix.replace(
                    "Path('/kaggle/working/fold0-inference-source.zip')",
                    f"Path({str(Path(working_dir) / 'fold0-inference-source.zip')!r})",
                )
                safe_prefix = re.sub(
                    r"        if not source\.is_absolute\(\) or not source\.is_relative_to\('/kaggle/input'\):",
                    "        if False:", safe_prefix,
                )
                exec(compile(safe_prefix, "full-audit-bootstrap-prefix", "exec"), {})
                self.assertEqual(
                    Path(working_dir, "fold0-inference-source.zip").read_bytes(), source_zip_bytes
                )


if __name__ == "__main__":
    unittest.main()
