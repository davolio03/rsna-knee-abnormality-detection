import csv
import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsna_knee.data.cohort import build_training_cohort, write_private_manifest
from rsna_knee.labels.subscription import LABEL_COLUMNS
from scripts.prepare_training_cohort import main as prepare_main


class TrainingCohortTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.train = self.inputs / "train.csv"
        self.labels = self.inputs / "labels.jsonl"
        self.series = self.inputs / "train_series.csv"
        self.gold = self.inputs / "gold.csv"
        self.journal = self.inputs / "diagnostics.jsonl"
        self._write_inputs()

    def tearDown(self):
        self.temp.cleanup()

    def _write_inputs(self):
        with self.train.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["StudyInstanceUID", "Report", *LABEL_COLUMNS])
            writer.writeheader()
            writer.writerow({"StudyInstanceUID": "study-a", "Report": "  A   unique report. ",
                             **{column: "" for column in LABEL_COLUMNS}})
            writer.writerow({"StudyInstanceUID": "study-b", "Report": "A UNIQUE REPORT.",
                             **{column: "" for column in LABEL_COLUMNS}})
            writer.writerow({"StudyInstanceUID": "gold-0", "Report": "",
                             **{column: "" for column in LABEL_COLUMNS}})
            writer.writerow({"StudyInstanceUID": "later", "Report": "not in cohort",
                             **{column: "" for column in LABEL_COLUMNS}})
        records = []
        for study, positive, evidence in (("study-a", True, "unique"), ("study-b", False, "UNIQUE")):
            records.append({"StudyInstanceUID": study, "labels": {
                column: {"label": positive if column == LABEL_COLUMNS[0] else False,
                         "evidence": evidence, "uncertainty": 0.2}
                for column in LABEL_COLUMNS
            }})
        self.labels.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
        with self.series.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["StudyInstanceUID", "SeriesInstanceUID",
                                                         "Fluid_Sensitive", "Fat_Suppression",
                                                         "Anatomical_Plane"])
            writer.writeheader()
            writer.writerows([
                {"StudyInstanceUID": "study-a", "SeriesInstanceUID": "series-1",
                 "Fluid_Sensitive": "1", "Fat_Suppression": "1", "Anatomical_Plane": "Sagittal"},
                {"StudyInstanceUID": "study-b", "SeriesInstanceUID": "series-2",
                 "Fluid_Sensitive": "0", "Fat_Suppression": "0", "Anatomical_Plane": "Axial"},
            ])
        self.gold.write_text("StudyInstanceUID\n" + "\n".join(f"gold-{i}" for i in range(58)) + "\n",
                             encoding="utf-8")
        self.journal.write_text("\n".join([
            json.dumps({"event": "prompt_boundary", "prompt_version": "c2-literal-evidence-v1",
                        "start_offset": 1}),
            json.dumps({"event": "prompt_boundary", "prompt_version": "c2-literal-evidence-v2",
                        "start_offset": 2}),
            json.dumps({"event": "prompt_boundary", "prompt_version": "c2-literal-evidence-v3",
                        "start_offset": 2}),
        ]) + "\n", encoding="utf-8")

    def _build(self, **changes):
        args = {
            "train_csv": self.train, "labels_jsonl": self.labels,
            "series_csv": self.series, "gold_exclusions": self.gold,
            "prompt_journal": self.journal, "cohort_size": 2,
        }
        args.update(changes)
        return build_training_cohort(**args)

    def test_builds_private_binary_manifest_with_order_provenance_and_group_hash(self):
        manifest = self._build()
        self.assertEqual([record["study_id"] for record in manifest["records"]], ["study-a", "study-b"])
        self.assertEqual(manifest["prompt_provenance_counts"], {
            "legacy": 1, "c2-literal-evidence-v1": 1, "c2-literal-evidence-v2": 0,
            "c2-literal-evidence-v3": 0,
        })
        self.assertEqual(manifest["records"][0]["labels"][LABEL_COLUMNS[0]], 1)
        self.assertTrue(all(value in (0, 1) for value in manifest["records"][1]["labels"].values()))
        self.assertEqual([item["series_id"] for item in manifest["records"][0]["series"]],
                         ["series-1"])
        self.assertEqual(manifest["records"][0]["normalized_report_sha256"],
                         manifest["records"][1]["normalized_report_sha256"])
        serialized = json.dumps(manifest)
        self.assertNotIn("synthetic", serialized)
        self.assertNotIn("DO NOT ACCESS GOLD REPORT", serialized)
        self.assertIn("train_csv", manifest["source_sha256"])
        self.assertEqual(manifest["source_sha256"]["train_csv"],
                         hashlib.sha256(self.train.read_bytes()).hexdigest())

    def test_requires_exact_ordered_checkpoint_prefix_and_unique_ids(self):
        original = self.labels.read_text(encoding="utf-8")
        lines = original.splitlines()
        self.labels.write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            self._build()
        self.labels.write_text(original + lines[0] + "\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            self._build()

    def test_requires_all_series_and_valid_boolean_label_schema(self):
        self.series.write_text(self.series.read_text(encoding="utf-8").splitlines()[0] + "\n",
                                encoding="utf-8")
        with self.assertRaises(ValueError):
            self._build()
        self._write_inputs()
        raw = json.loads(self.labels.read_text(encoding="utf-8").splitlines()[0])
        raw["labels"][LABEL_COLUMNS[0]]["label"] = 1
        self.labels.write_text(json.dumps(raw) + "\n" + self.labels.read_text(encoding="utf-8").splitlines()[1] + "\n",
                               encoding="utf-8")
        with self.assertRaises(ValueError):
            self._build()

    def test_prompt_boundaries_must_cover_checkpoint_without_invalid_offsets(self):
        self.journal.write_text(json.dumps({"event": "prompt_boundary",
                                            "prompt_version": "c2-literal-evidence-v3",
                                            "start_offset": 3}) + "\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            self._build()

    def test_gold_rows_are_masked_before_report_validation(self):
        manifest = self._build()
        self.assertEqual([record["study_id"] for record in manifest["records"]], ["study-a", "study-b"])

    def test_writer_rejects_output_outside_private_directory(self):
        manifest = self._build()
        with self.assertRaises(ValueError):
            write_private_manifest(manifest, output_path=self.root / "elsewhere.json",
                                   project_root=self.root)
        with self.assertRaises(ValueError):
            write_private_manifest(manifest,
                                   output_path=self.root / "data" / "private" / ".." / "escape.json",
                                   project_root=self.root)

    def test_writer_refuses_existing_manifest_and_rejects_symlink_escape(self):
        manifest = self._build()
        output = self.root / "data" / "private" / "cohort.json"
        write_private_manifest(manifest, output_path=output, project_root=self.root)
        saved = output.read_bytes()
        with self.assertRaises(FileExistsError):
            write_private_manifest(manifest, output_path=output, project_root=self.root)
        self.assertEqual(output.read_bytes(), saved)
        outside = self.root / "outside.json"
        link = self.root / "data" / "private" / "linked.json"
        try:
            link.symlink_to(outside)
        except OSError:
            self.skipTest("File symlinks are unavailable in this environment.")
        with self.assertRaises(ValueError):
            write_private_manifest(manifest, output_path=link, project_root=self.root)
        self.assertFalse(outside.exists())

    def test_cli_reports_output_conflict_without_traceback_or_path(self):
        output = self.root / "data" / "private" / "existing.json"
        output.parent.mkdir(parents=True)
        output.write_text("do not overwrite", encoding="utf-8")
        stderr = io.StringIO()
        argv = [
            "--train-csv", str(self.train), "--labels", str(self.labels),
            "--series-csv", str(self.series), "--gold-exclusions", str(self.gold),
            "--prompt-journal", str(self.journal), "--output", str(output),
            "--cohort-size", "2",
        ]
        with patch("scripts.prepare_training_cohort.REPOSITORY_ROOT", self.root):
            with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                prepare_main(argv)
        self.assertEqual(raised.exception.code, 2)
        message = stderr.getvalue()
        self.assertNotIn("Traceback", message)
        self.assertNotIn(str(output), message)
        self.assertIn("already exists", message)
        self.assertEqual(output.read_text(encoding="utf-8"), "do not overwrite")


if __name__ == "__main__":
    unittest.main()
