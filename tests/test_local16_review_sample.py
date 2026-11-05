"""Strict-TDD tests for the private, blinded Local16 review sample builder."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import prepare_local16_review_sample as builder
from scripts.prepare_local16_review_sample import (
    ALL_TARGETS,
    TARGETS,
    build_review_sample,
    load_reports,
    select_review_rows,
    validate_private_output,
)


def synthetic_rows(*, include_missing=False):
    rows = []
    ordinal = 1
    for target_index, target in enumerate(TARGETS):
        for group, label in (("high", 0.9), ("low", 0.1)):
            for item in range(30):
                # First ten are deliberately worst errors; final ten are controls.
                if item < 10:
                    prediction = 0.1 if group == "high" else 0.9
                elif item >= 20:
                    prediction = label
                else:
                    prediction = 0.5
                labels = [0.5] * 12
                predictions = [0.5] * 12
                observed = [False] * 12
                slot = ALL_TARGETS.index(target)
                labels[slot], predictions[slot], observed[slot] = label, prediction, True
                rows.append({
                    "ordinal": ordinal,
                    "study_id": f"study-{target_index}-{group}-{item}",
                    "predictions": predictions,
                    "labels": labels,
                    "observed_mask": observed,
                })
                ordinal += 1
    if include_missing:
        row = dict(rows[0])
        row["ordinal"] = ordinal
        row["study_id"] = "unobserved"
        row["observed_mask"] = list(row["observed_mask"])
        row["observed_mask"][TARGETS.index(TARGETS[0])] = False
        rows.append(row)
    return rows


class SelectionTests(unittest.TestCase):
    def test_selects_exact_protocol_per_target_and_deduplicates_studies(self):
        chosen = select_review_rows(synthetic_rows(include_missing=True))
        self.assertEqual(90, len(chosen))
        self.assertEqual(set(TARGETS), {row["target"] for row in chosen})
        self.assertEqual(90, len({row["case_id"] for row in chosen}))
        for target in TARGETS:
            target_rows = [row for row in chosen if row["target"] == target]
            self.assertEqual(30, len(target_rows))
            strata = [row["selection_stratum"] for row in target_rows]
            self.assertEqual({"high_error", "low_error", "high_control", "low_control"}, set(strata))
            self.assertEqual([10, 10, 5, 5], [strata.count(s) for s in
                             ("high_error", "low_error", "high_control", "low_control")])
        self.assertNotIn("unobserved", {row["study_id"] for row in chosen})

    def test_selection_is_deterministic_and_controls_are_disjoint_from_errors(self):
        rows = synthetic_rows()
        first = select_review_rows(rows)
        second = select_review_rows(rows)
        self.assertEqual(first, second)
        for target in TARGETS:
            target_rows = [row for row in first if row["target"] == target]
            study_ids = [row["study_id"] for row in target_rows]
            self.assertEqual(len(study_ids), len(set(study_ids)))
            for row in target_rows:
                if row["selection_stratum"].endswith("control"):
                    self.assertLessEqual(abs(row["prediction"] - row["soft_label"]), 0.1)
                else:
                    self.assertGreater(abs(row["prediction"] - row["soft_label"]), 0.1)

    def test_stable_ordinal_breaks_ties_before_shuffle(self):
        rows = synthetic_rows()
        target = TARGETS[0]
        slot = ALL_TARGETS.index(target)
        candidates = [row for row in rows if row["study_id"].startswith("study-0-high-")][:12]
        for row in candidates:
            row["predictions"][slot] = 0.0
        chosen = select_review_rows(rows)
        selected = {row["study_id"] for row in chosen
                    if row["target"] == target and row["selection_stratum"] == "high_error"}
        self.assertEqual({row["study_id"] for row in candidates[:10]}, selected)

    def test_insufficient_quota_fails_closed(self):
        rows = synthetic_rows()
        slot = ALL_TARGETS.index(TARGETS[0])
        rows = [row for row in rows if not (row["study_id"].startswith("study-0-high-")
                                             and int(row["study_id"].split("-")[-1]) >= 20)]
        with self.assertRaisesRegex(ValueError, "selection_quota_insufficient"):
            select_review_rows(rows)


class InputAndOutputTests(unittest.TestCase):
    def test_report_loader_rejects_duplicate_or_empty_ids_and_empty_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            path.write_text("StudyInstanceUID,Report\na,ok\na,duplicate\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "report_id_invalid"):
                load_reports(path)
            path.write_text("StudyInstanceUID,Report\n,ok\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "report_id_invalid"):
                load_reports(path)
            path.write_text("StudyInstanceUID,Report\na, \n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "report_missing"):
                load_reports(path, required_ids={"a"})

    def test_actual_headers_preserve_multiline_reports_and_filter_csv_superset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            expected = "First line, with comma\nSecond line stays exact."
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["StudyInstanceUID", "Report", "MCL"])
                writer.writerow(["validation-1", expected, "0.0"])
                writer.writerow(["train-only-1", "Must not be retained in the validation report map.", "1.0"])
            reports = load_reports(path, required_ids={"validation-1"})
            self.assertEqual({"validation-1": expected}, reports)
            self.assertNotIn("train-only-1", reports)

    def test_superset_join_requires_all_validation_reports_and_rejects_duplicate_anywhere(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["StudyInstanceUID", "Report"])
                writer.writerow(["validation-1", "report 1"])
                writer.writerow(["train-only-1", "extra"])
            with self.assertRaisesRegex(ValueError, "report_join_invalid"):
                load_reports(path, required_ids={"validation-1", "validation-2"})
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["StudyInstanceUID", "Report"])
                writer.writerow(["validation-1", "report 1"])
                writer.writerow(["train-only-1", "extra"])
                writer.writerow(["train-only-1", "duplicate extra"])
            with self.assertRaisesRegex(ValueError, "report_id_invalid"):
                load_reports(path, required_ids={"validation-1"})

    def test_output_path_must_be_private_and_not_symlinked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private = root / "data" / "private"
            private.mkdir(parents=True)
            self.assertEqual(private / "run01", validate_private_output(private / "run01", private))
            with self.assertRaisesRegex(ValueError, "output_path_invalid"):
                validate_private_output(root / "public", private)
            existing = private / "run01"
            existing.mkdir()
            with self.assertRaisesRegex(ValueError, "output_exists"):
                validate_private_output(existing, private)

    def test_blind_bundle_has_no_prediction_label_or_sampling_stratum(self):
        # Synthetic end-to-end build; no real private source data or model is opened.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private = root / "data" / "private"
            private.mkdir(parents=True)
            output = private / "sample"
            selected = select_review_rows(synthetic_rows())
            reports = {row["study_id"]: "Original <report> & text" for row in selected}
            preview_provider = lambda study_id, record: [{"series_ordinal": 1, "png_bytes": b"png"}]
            build_review_sample(
                selected_rows=selected,
                reports=reports,
                output_dir=output,
                private_root=private,
                input_provenance={"source": "synthetic"},
                preview_provider=preview_provider,
                code_path=Path(__file__),
            )
            html = (output / "index.html").read_text(encoding="utf-8")
            case_html = "\n".join(p.read_text(encoding="utf-8") for p in (output / "cases").glob("*.html"))
            blind = html + case_html
            self.assertNotIn("high_error", blind)
            self.assertNotIn("soft_label", blind)
            self.assertNotIn("prediction", blind)
            self.assertIn("Original &lt;report&gt; &amp; text", case_html)
            self.assertTrue((output / "unblinding.json").is_file())
            self.assertTrue((output / "review-template.csv").is_file())
            with (output / "review-template.csv").open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(["case_id", "target", "decision", "confidence", "note"], reader.fieldnames)
                self.assertTrue(all(not any(row[key] for key in ("decision", "confidence", "note"))
                                    for row in reader))
            unblind = json.loads((output / "unblinding.json").read_text(encoding="utf-8"))
            self.assertEqual(90, len(unblind["cases"]))
            self.assertTrue(all("prediction" in row and "soft_label" in row
                                and "selection_stratum" in row for row in unblind["cases"]))

    def test_output_is_no_overwrite_and_does_not_call_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private = root / "data" / "private"
            private.mkdir(parents=True)
            output = private / "sample"
            selected = select_review_rows(synthetic_rows())
            reports = {row["study_id"]: "report" for row in selected}
            kwargs = dict(selected_rows=selected, reports=reports, output_dir=output,
                          private_root=private, input_provenance={"source": "synthetic"},
                          preview_provider=lambda *_: [{"series_ordinal": 1, "png_bytes": b"png"}],
                          code_path=Path(__file__))
            build_review_sample(**kwargs)
            with self.assertRaisesRegex(ValueError, "output_exists"):
                build_review_sample(**kwargs)

    def test_dry_plan_validates_join_and_reports_actual_aggregate_counts_without_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private = root / "data" / "private"
            private.mkdir(parents=True)
            rows = synthetic_rows()
            shared = rows[0]
            second_slot = ALL_TARGETS.index("Lateral Meniscus")
            shared["observed_mask"][second_slot] = True
            shared["labels"][second_slot] = 0.9
            shared["predictions"][second_slot] = 0.1
            selected = select_review_rows(rows)
            reports = {row["study_id"]: "report" for row in selected}
            fake_info = {"ids": ["validation-1"], "by_id": {}, "provenance": {}}
            paths = {"output_dir": private / "run01"}
            with mock.patch.object(builder, "ROOT", root), \
                    mock.patch.object(builder, "_validate_sources", return_value=(rows, reports, None, fake_info)) as validate, \
                    mock.patch.object(builder, "build_review_sample") as build:
                result = builder.prepare(paths, build=False)
            validate.assert_called_once_with(paths)
            build.assert_not_called()
            self.assertEqual("planned", result["status"])
            self.assertEqual(90, result["expected_review_rows"])
            self.assertEqual(89, result["unique_study_count"])
            self.assertEqual({"Contusion": 30, "Lateral Meniscus": 30, "MCL": 30},
                             result["target_case_counts"])
            self.assertFalse(result["build_started"])


if __name__ == "__main__":
    unittest.main()
