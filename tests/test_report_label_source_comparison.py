import csv
import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsna_knee.labels.subscription import LABEL_COLUMNS
from rsna_knee.data.cohort import REPORT_NORMALIZATION_VERSION
from rsna_knee.validation.cohort import build_grouped_training_folds
from rsna_knee.labels.report_source_comparison import (
    SOURCE_ARCHIVE_SHA256,
    SOURCE_ARCHIVE_SIZE,
    SOURCE_CSV_SHA256,
    SOURCE_CSV_SIZE,
    build_source_comparison,
    stage_approved_source,
)


class ReportLabelSourceComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ids = tuple(f"synthetic-{letter}" for letter in "abcdef")
        self.gold_id = "synthetic-gold"
        self.official = self.root / "train.csv"
        with self.official.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["StudyInstanceUID", "Report", *LABEL_COLUMNS])
            writer.writeheader()
            for study in (*self.ids, self.gold_id):
                writer.writerow({"StudyInstanceUID": study, "Report": "SYNTHETIC_REPORT_SECRET",
                                 **{target: "" for target in LABEL_COLUMNS}})

        self.public = self.root / "llm_labels_full.csv"
        columns = ["StudyInstanceUID", *LABEL_COLUMNS]
        rows = []
        for index, study in enumerate((*self.ids, self.gold_id)):
            row = {"StudyInstanceUID": study}
            for target_index, target in enumerate(LABEL_COLUMNS):
                # One target has masked, positive and negative cases; remaining
                # targets exercise the all-masked, explicitly undefined metric path.
                row[target] = ("0.75" if index in (0, 3) else "0.25" if index in (1, 4) else "0.5")
                if target_index > 0:
                    row[target] = "0.5"
                if study == self.gold_id:
                    row[target] = "GOLD_VALUE_MUST_NOT_BE_PARSED"
            rows.append(row)
        with self.public.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader(); writer.writerows(rows)

        labels = {target: 0 for target in LABEL_COLUMNS}
        records = []
        for index, study in enumerate(self.ids):
            record_labels = {target: (1 if index % 2 == 0 else 0) for target in LABEL_COLUMNS}
            records.append({
                "study_id": study, "labels": record_labels,
                "series": [{"series_id": f"series-{index}", "fluid_sensitive": 1,
                            "fat_suppression": 1, "anatomical_plane": "Sagittal"}],
                "prompt_version": "legacy",
                "normalized_report_sha256": hashlib.sha256(f"report-{index}".encode()).hexdigest(),
            })
        self.manifest = {
            "schema_version": "baseline-training-cohort-v1",
            "configuration": {"report_group_normalization": REPORT_NORMALIZATION_VERSION, "cohort_size": 6},
            "source_sha256": {key: "a" * 64 for key in (
                "train_csv", "labels_jsonl", "series_csv", "gold_exclusions", "prompt_journal")},
            "prompt_provenance_counts": {"legacy": 6, "c2-literal-evidence-v1": 0,
                                         "c2-literal-evidence-v2": 0, "c2-literal-evidence-v3": 0},
            "records": records,
        }
        # Use the production fold builder for synthetic but structurally valid pins.
        manifest_bytes = (json.dumps(self.manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
        self.manifest_path = self.root / "cohort.json"; self.manifest_path.write_bytes(manifest_bytes)
        self.manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        folds = build_grouped_training_folds(self.manifest, manifest_sha256=self.manifest_sha)
        folds_bytes = (json.dumps(folds, sort_keys=True, separators=(",", ":")) + "\n").encode()
        self.folds_path = self.root / "folds.json"; self.folds_path.write_bytes(folds_bytes)
        self.folds_sha = hashlib.sha256(folds_bytes).hexdigest()
        self.gold = self.root / "gold.csv"
        self.gold.write_text("StudyInstanceUID\n" + self.gold_id + "\n", encoding="utf-8")
        self.gold_sha = hashlib.sha256(self.gold.read_bytes()).hexdigest()
        self.public_sha = hashlib.sha256(self.public.read_bytes()).hexdigest()
        self.official_count = 7
        self.output = self.root / "out"

    def tearDown(self):
        self.temp.cleanup()

    def _compare(self, **changes):
        args = {
            "source_csv": self.public, "official_train_csv": self.official,
            "cohort_path": self.manifest_path, "folds_path": self.folds_path,
            "gold_ids_path": self.gold, "output_dir": self.output,
        }
        args.update(changes)
        with patch.multiple(
            "rsna_knee.labels.report_source_comparison",
            SOURCE_CSV_SHA256=self.public_sha, SOURCE_CSV_SIZE=self.public.stat().st_size,
            COHORT_SHA256=self.manifest_sha, COHORT_SIZE=self.manifest_path.stat().st_size,
            FOLDS_SHA256=self.folds_sha, FOLDS_SIZE=self.folds_path.stat().st_size,
            GOLD_IDS_SHA256=self.gold_sha, GOLD_IDS_SIZE=self.gold.stat().st_size,
            EXPECTED_OFFICIAL_STUDIES=self.official_count,
            EXPECTED_NON_GOLD_STUDIES=6, EXPECTED_COHORT_STUDIES=6, EXPECTED_GOLD_STUDIES=1,
        ):
            return build_source_comparison(**args)

    def test_masks_exact_half_and_reports_descriptive_paired_metrics_by_fold(self):
        summary = self._compare()
        self.assertEqual(summary["source_inventory"]["non_gold_studies"], 6)
        self.assertEqual(summary["paired"]["study_count"], 6)
        target = summary["paired"]["targets"][LABEL_COLUMNS[0]]
        self.assertEqual(target["addressed_count"], 4)
        self.assertEqual(target["not_addressed_count"], 2)
        self.assertEqual(target["confusion"], {"tp": 1, "fp": 1, "fn": 1, "tn": 1})
        self.assertEqual(target["agreement"], 0.5)
        self.assertEqual(target["mean_absolute_probability_difference"], 0.5)
        self.assertEqual(target["c2_positive_prevalence"], 0.5)
        self.assertEqual(target["public_binary_positive_prevalence"], 0.5)
        for target_index, target_summary in enumerate(summary["paired"]["targets"].values()):
            if target_index == 0:
                continue
            self.assertIsNone(target_summary["agreement"])
            self.assertIsNone(target_summary["c2_positive_prevalence"])
            self.assertIsNone(target_summary["public_binary_positive_prevalence"])
            self.assertEqual(target_summary["addressed_count"], 0)
            self.assertTrue(all(
                fold["addressed_count"] + fold["not_addressed_count"] == fold["study_count"]
                for fold in target_summary["by_fold"].values()))
        overlay = json.loads((self.output / "soft_label_overlay.json").read_text(encoding="utf-8"))
        self.assertEqual([record["study_id"] for record in overlay["records"]], list(self.ids))
        self.assertEqual(overlay["records"][0]["observed_mask"][LABEL_COLUMNS[0]], True)
        self.assertEqual(overlay["records"][5]["observed_mask"][LABEL_COLUMNS[0]], False)

    def test_gold_values_are_excluded_before_probability_parsing_and_outputs_are_private_only(self):
        summary = self._compare()
        serialized = json.dumps(summary) + (self.output / "soft_label_overlay.json").read_text(encoding="utf-8")
        self.assertNotIn("GOLD_VALUE_MUST_NOT_BE_PARSED", serialized)
        self.assertNotIn("SYNTHETIC_REPORT_SECRET", serialized)
        self.assertEqual(summary["source_inventory"]["gold_studies_excluded"], 1)

    def test_rejects_duplicate_foreign_missing_schema_and_invalid_probabilities_without_outputs(self):
        original = self.public.read_text(encoding="utf-8")
        cases = [
            original + original.splitlines()[1] + "\n",
            original.replace("synthetic-c", "foreign-study"),
            "StudyInstanceUID,wrong-column\nsynthetic-a,0.5\n",
            original.replace("0.75", "NaN"),
            original.replace("0.75", "1.01"),
            original.replace("0.75", "1"),
        ]
        for index, content in enumerate(cases):
            with self.subTest(case=index):
                self.public.write_text(content, encoding="utf-8")
                self.output = self.root / f"failed-{index}"
                with self.assertRaises(ValueError): self._compare()
                self.assertFalse(self.output.exists())
                self.public.write_text(original, encoding="utf-8")

    def test_rejects_hash_or_fold_binding_mismatch_and_preserves_input_bytes(self):
        original_manifest = self.manifest_path.read_bytes()
        original_folds = self.folds_path.read_bytes()
        before = (self.manifest_path.read_bytes(), self.folds_path.read_bytes())
        self.folds_path.write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError): self._compare()
        self.assertFalse(self.output.exists())
        self.folds_path.write_bytes(original_folds)
        self.manifest_path.write_bytes(bytes([self.manifest_path.read_bytes()[0] ^ 1])
                                       + self.manifest_path.read_bytes()[1:])
        with self.assertRaises(ValueError): self._compare()
        self.manifest_path.write_bytes(original_manifest)
        self.assertEqual(before, (self.manifest_path.read_bytes(), self.folds_path.read_bytes()))
        self.assertEqual(original_manifest, before[0])

    def test_rejects_validly_pinned_but_manifest_unbound_fold_membership(self):
        folds = json.loads(self.folds_path.read_text(encoding="utf-8"))
        folds["study_assignments"][0]["fold"] = (folds["study_assignments"][0]["fold"] + 1) % 3
        raw = (json.dumps(folds, sort_keys=True, separators=(",", ":")) + "\n").encode()
        self.folds_path.write_bytes(raw)
        with patch.multiple(
            "rsna_knee.labels.report_source_comparison",
            FOLDS_SHA256=hashlib.sha256(raw).hexdigest(), FOLDS_SIZE=len(raw),
            SOURCE_CSV_SHA256=self.public_sha, SOURCE_CSV_SIZE=self.public.stat().st_size,
            COHORT_SHA256=self.manifest_sha, COHORT_SIZE=self.manifest_path.stat().st_size,
            GOLD_IDS_SHA256=self.gold_sha, GOLD_IDS_SIZE=self.gold.stat().st_size,
            EXPECTED_OFFICIAL_STUDIES=self.official_count,
            EXPECTED_NON_GOLD_STUDIES=6, EXPECTED_COHORT_STUDIES=6, EXPECTED_GOLD_STUDIES=1,
        ):
            with self.assertRaisesRegex(ValueError, "fold_binding_invalid"):
                build_source_comparison(
                    source_csv=self.public, official_train_csv=self.official,
                    cohort_path=self.manifest_path, folds_path=self.folds_path,
                    gold_ids_path=self.gold, output_dir=self.output)
        self.assertFalse(self.output.exists())

    def test_staging_extracts_only_pinned_original_csv_and_rejects_traversal(self):
        archive = self.root / "source.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("llm_labels_full.csv", self.public.read_bytes())
            bundle.writestr("llm_labels_v2.csv", b"DO_NOT_STAGE")
            bundle.writestr("llm_labels_v4_blend.csv", b"DO_NOT_STAGE")
        archive_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        destination = self.root / "stage"
        with patch.multiple(
            "rsna_knee.labels.report_source_comparison",
            SOURCE_ARCHIVE_SHA256=archive_digest, SOURCE_ARCHIVE_SIZE=archive.stat().st_size,
            SOURCE_CSV_SHA256=self.public_sha, SOURCE_CSV_SIZE=self.public.stat().st_size,
        ):
            result = stage_approved_source(archive, destination)
        self.assertEqual(Path(result["csv_path"]).read_bytes(), self.public.read_bytes())
        self.assertFalse((destination / "llm_labels_v2.csv").exists())
        self.assertFalse((destination / "llm_labels_v4_blend.csv").exists())
        unsafe = self.root / "unsafe.zip"
        with zipfile.ZipFile(unsafe, "w") as bundle:
            bundle.writestr("../llm_labels_full.csv", self.public.read_bytes())
        with patch.multiple(
            "rsna_knee.labels.report_source_comparison",
            SOURCE_ARCHIVE_SHA256=hashlib.sha256(unsafe.read_bytes()).hexdigest(),
            SOURCE_ARCHIVE_SIZE=unsafe.stat().st_size,
        ):
            with self.assertRaises(ValueError): stage_approved_source(unsafe, self.root / "unsafe-out")


if __name__ == "__main__":
    unittest.main()
