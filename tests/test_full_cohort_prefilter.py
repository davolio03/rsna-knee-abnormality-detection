import hashlib
import json
import sys
import tempfile
import csv
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsna_knee.labels.subscription import LABEL_COLUMNS
from rsna_knee.data.cohort import _normalized_report_hash
from rsna_knee.labels.full_cohort_prefilter import (
    build_prefilter_manifest,
    build_grouped_folds,
    build_training_weights,
    prepare_full_cohort,
)


class FullCohortPrefilterTests(unittest.TestCase):
    def _records(self, count=60):
        records = []
        for index in range(count):
            targets = {}
            masks = {}
            for target_index, target in enumerate(LABEL_COLUMNS):
                if target_index == 0:
                    positive = index < 12
                elif target_index == 1:
                    positive = index >= 12
                else:
                    positive = (index + target_index * 3) % 5 < 2
                observed = (index + target_index * 7) % 7 != 0
                targets[target] = (0.75 if positive else 0.25) if observed else 0.5
                masks[target] = observed
            records.append({
                "study_id": f"synthetic-{index}",
                "report_group_sha256": hashlib.sha256(f"group-{index // 2}".encode()).hexdigest(),
                "series": [{"series_id": f"series-{index}", "fluid_sensitive": 1,
                            "fat_suppression": 1, "anatomical_plane": "Sagittal"}],
                "soft_targets": targets,
                "observed_mask": masks,
            })
        return records

    def test_all_unknown_rows_are_reasoned_and_not_supervised(self):
        rows = self._records(60)
        rows[0]["soft_targets"] = {target: 0.5 for target in LABEL_COLUMNS}
        rows[0]["observed_mask"] = {target: False for target in LABEL_COLUMNS}
        manifest, audit = build_prefilter_manifest(rows, input_pins={"source_csv_sha256": "a" * 64})
        self.assertEqual(len(manifest["records"]), 51)
        self.assertEqual(audit["excluded_all_targets_unknown"], 9)
        self.assertEqual(audit["eligible_study_count"], 51)
        self.assertEqual(audit["target_addressed_counts"][LABEL_COLUMNS[0]],
                         sum(row["observed_mask"][LABEL_COLUMNS[0]] for row in rows))
        self.assertEqual(manifest["audit_rows"][0]["reason_code"], "all_targets_unknown")
        self.assertEqual(manifest["records"][0]["soft_targets"][LABEL_COLUMNS[0]], 0.75)

    def test_partial_probabilities_remain_masked_and_grouped_folds_are_deterministic(self):
        manifest, _ = build_prefilter_manifest(self._records())
        folds = build_grouped_folds(manifest)
        self.assertEqual(folds, build_grouped_folds(manifest))
        manifest_bytes = (json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        self.assertEqual(folds["manifest_sha256"], hashlib.sha256(manifest_bytes).hexdigest())
        assignments = {row["study_id"]: row["fold"] for row in folds["study_assignments"]}
        for record in manifest["records"]:
            self.assertEqual(record["observed_mask"][LABEL_COLUMNS[0]],
                             (int(record["study_id"].split("-")[-1]) + 0 * 7) % 7 != 0)
        for group in {row["report_group_sha256"] for row in manifest["records"]}:
            self.assertEqual(len({assignments[row["study_id"]] for row in manifest["records"]
                                  if row["report_group_sha256"] == group}), 1)
        for audit in folds["folds"]:
            self.assertTrue(all(value > 0 for value in audit["validation_support"]["positive"].values()))
            self.assertTrue(all(value > 0 for value in audit["validation_support"]["negative"].values()))
            self.assertTrue(all(value > 0 for value in audit["train_support"]["positive"].values()))
            self.assertTrue(all(value > 0 for value in audit["train_support"]["negative"].values()))

    def test_training_weights_are_train_only_capped_and_balance_both_classes(self):
        records = self._records()
        manifest, _ = build_prefilter_manifest(records)
        folds = build_grouped_folds(manifest)
        first = folds["folds"][0]
        train_ids = first["train_study_ids"]
        weights_before = build_training_weights(manifest["records"], train_ids)
        validation_ids = set(first["validation_study_ids"])
        mutated = json.loads(json.dumps(manifest["records"]))
        for record in mutated:
            if record["study_id"] in validation_ids:
                record["soft_targets"] = {target: 1.0 for target in LABEL_COLUMNS}
                record["observed_mask"] = {target: True for target in LABEL_COLUMNS}
        self.assertEqual(weights_before, build_training_weights(mutated, train_ids))
        for values in weights_before.values():
            self.assertGreaterEqual(values["positive"], 1.0)
            self.assertGreaterEqual(values["negative"], 1.0)
            self.assertLessEqual(values["positive"], 3.0)
            self.assertLessEqual(values["negative"], 3.0)
        self.assertTrue(any(v["positive"] > 1 or v["negative"] > 1
                            for v in weights_before.values()))
        self.assertGreater(weights_before[LABEL_COLUMNS[0]]["positive"], 1.0)
        self.assertGreater(weights_before[LABEL_COLUMNS[1]]["negative"], 1.0)

    def test_class_weight_ratio_is_capped_at_three(self):
        records = self._records()
        for index, record in enumerate(records):
            record["soft_targets"][LABEL_COLUMNS[0]] = 0.75 if index == 0 else 0.25
            record["observed_mask"][LABEL_COLUMNS[0]] = True
        weights = build_training_weights(records, [record["study_id"] for record in records])
        self.assertEqual(weights[LABEL_COLUMNS[0]]["positive"], 3.0)

    def test_rejects_conflicting_masks_bad_probabilities_and_duplicate_study_ids(self):
        records = self._records(60)
        records[0]["soft_targets"][LABEL_COLUMNS[0]] = 0.5
        records[0]["observed_mask"][LABEL_COLUMNS[0]] = True
        with self.assertRaisesRegex(ValueError, "unknown_probability_must_be_masked"):
            build_prefilter_manifest(records)
        records = self._records(60)
        records[0]["soft_targets"][LABEL_COLUMNS[0]] = 1.5
        with self.assertRaisesRegex(ValueError, "probability_out_of_range"):
            build_prefilter_manifest(records)
        records = self._records(60)
        records.append(dict(records[0]))
        with self.assertRaisesRegex(ValueError, "duplicate_study_id"):
            build_prefilter_manifest(records)

    def test_rejects_duplicate_or_malformed_series_metadata(self):
        records = self._records()
        records[1]["series"][0]["series_id"] = records[0]["series"][0]["series_id"]
        with self.assertRaisesRegex(ValueError, "series_duplicate_id"):
            build_prefilter_manifest(records)
        records = self._records()
        records[0]["series"][0]["unexpected"] = "value"
        with self.assertRaisesRegex(ValueError, "series_metadata_invalid"):
            build_prefilter_manifest(records)

    def test_gold_source_and_report_values_are_excluded_before_processing(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        study_ids = [f"synthetic-{index}" for index in range(6)]
        gold_id = "synthetic-gold"
        train_path, series_path = root / "train.csv", root / "series.csv"
        source_path, gold_path = root / "source.csv", root / "gold.csv"
        cohort_path, folds_path = root / "cohort.json", root / "folds.json"
        with train_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["StudyInstanceUID", "Report", *LABEL_COLUMNS])
            writer.writeheader()
            for index, study in enumerate([*study_ids, gold_id]):
                writer.writerow({"StudyInstanceUID": study,
                                 "Report": "GOLD_REPORT_POISON" if study == gold_id else f"report-{index}",
                                 **{target: "" for target in LABEL_COLUMNS}})
        series_fields = ["StudyInstanceUID", "SeriesInstanceUID", "Fluid_Sensitive",
                         "Fat_Suppression", "Anatomical_Plane"]
        with series_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=series_fields); writer.writeheader()
            for index, study in enumerate([*study_ids, gold_id]):
                writer.writerow({"StudyInstanceUID": study, "SeriesInstanceUID": f"series-{index}",
                                 "Fluid_Sensitive": "1", "Fat_Suppression": "1",
                                 "Anatomical_Plane": "Sagittal"})
        with source_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["StudyInstanceUID", *LABEL_COLUMNS]); writer.writeheader()
            for index, study in enumerate([*study_ids, gold_id]):
                probability = "NOT_A_PROBABILITY" if study == gold_id else ("0.75" if index % 2 == 0 else "0.25")
                writer.writerow({"StudyInstanceUID": study,
                                 **{target: probability for target in LABEL_COLUMNS}})
        gold_path.write_text("StudyInstanceUID\n" + gold_id + "\n", encoding="utf-8")
        cohort_path.write_text("{}", encoding="utf-8"); folds_path.write_text("{}", encoding="utf-8")
        sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        source_hash, gold_hash = sha(source_path), sha(gold_path)
        cohort_hash, folds_hash = sha(cohort_path), sha(folds_path)
        overrides = {
            "EXPECTED_OFFICIAL_STUDIES": 7, "EXPECTED_NON_GOLD_STUDIES": 6,
            "SOURCE_CSV_SHA256": source_hash,
            "SOURCE_CSV_SIZE": source_path.stat().st_size, "GOLD_IDS_SHA256": gold_hash,
            "GOLD_IDS_SIZE": gold_path.stat().st_size,
            "COHORT_SHA256": cohort_hash, "COHORT_SIZE": cohort_path.stat().st_size,
            "FOLDS_SHA256": folds_hash, "FOLDS_SIZE": folds_path.stat().st_size,
            "EXPECTED_ALL_UNKNOWN": 0, "EXPECTED_ELIGIBLE": 6, "EXPECTED_REPORT_GROUPS": 6,
            "EXPECTED_NON_GOLD_SERIES": 6,
        }
        with patch.multiple("rsna_knee.labels.report_source_comparison",
                            EXPECTED_OFFICIAL_STUDIES=7, EXPECTED_NON_GOLD_STUDIES=6,
                            EXPECTED_GOLD_STUDIES=1, SOURCE_CSV_SHA256=source_hash,
                            SOURCE_CSV_SIZE=source_path.stat().st_size, GOLD_IDS_SHA256=gold_hash,
                            GOLD_IDS_SIZE=gold_path.stat().st_size, EXPECTED_COHORT_STUDIES=6,
                            COHORT_SHA256=cohort_hash, COHORT_SIZE=cohort_path.stat().st_size,
                            FOLDS_SHA256=folds_hash, FOLDS_SIZE=folds_path.stat().st_size):
            with patch.multiple("rsna_knee.labels.full_cohort_prefilter",
                                PRIVATE_OUTPUT_ROOT=root, **overrides):
                def normalize_non_gold(report):
                    if report == "GOLD_REPORT_POISON":
                        self.fail("Gold report was normalized")
                    return _normalized_report_hash(report)
                with patch("rsna_knee.labels.full_cohort_prefilter._normalized_report_hash",
                           side_effect=normalize_non_gold):
                    summary = prepare_full_cohort(
                        source_csv=source_path, official_train_csv=train_path, series_csv=series_path,
                        gold_ids_path=gold_path, baseline_cohort_path=cohort_path,
                        baseline_folds_path=folds_path, output_dir=root / "prepared")
        self.assertEqual(summary["study_counts"]["eligible_study_count"], 6)
        aggregate = (root / "prepared" / "aggregate-summary.json").read_text(encoding="utf-8")
        self.assertNotIn(gold_id, aggregate)
        self.assertNotIn("GOLD_REPORT_POISON", aggregate)

    def test_private_cli_bundle_refuses_overwrite_and_reports_only_aggregates(self):
        # Full path ingestion is exercised with the real pinned artifacts by the CLI
        # acceptance run; this unit checks the fresh-directory safety boundary.
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        existing = root / "existing"
        existing.mkdir()
        with patch("rsna_knee.labels.full_cohort_prefilter.PRIVATE_OUTPUT_ROOT", root):
            with self.assertRaisesRegex(ValueError, "output_must_be_new"):
                prepare_full_cohort(
                    source_csv=root / "missing.csv", official_train_csv=root / "train.csv",
                    series_csv=root / "series.csv", gold_ids_path=root / "gold.csv",
                    baseline_cohort_path=root / "cohort.json", baseline_folds_path=root / "folds.json",
                    output_dir=existing,
                )
            with self.assertRaisesRegex(ValueError, "output_must_be_under_private_prefilter_root"):
                prepare_full_cohort(
                    source_csv=root / "missing.csv", official_train_csv=root / "train.csv",
                    series_csv=root / "series.csv", gold_ids_path=root / "gold.csv",
                    baseline_cohort_path=root / "cohort.json", baseline_folds_path=root / "folds.json",
                    output_dir=root.parent / "outside",
                )
            target = root / "target"
            target.mkdir()
            link = root / "linked"
            with patch.object(Path, "is_symlink", autospec=True,
                              side_effect=lambda path: path == link):
                with self.assertRaisesRegex(ValueError, "output_must_be_new"):
                    prepare_full_cohort(
                        source_csv=root / "missing.csv", official_train_csv=root / "train.csv",
                        series_csv=root / "series.csv", gold_ids_path=root / "gold.csv",
                        baseline_cohort_path=root / "cohort.json", baseline_folds_path=root / "folds.json",
                        output_dir=link / "nested",
                    )


if __name__ == "__main__":
    unittest.main()
