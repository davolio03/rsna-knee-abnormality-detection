"""Synthetic-only tests for the local prompt-comparison command boundary."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from rsna_knee.labels.prompt_comparison_cli import (
    _read_comparison_csv,
    _read_study_manifest_csv,
    _validate_manifest_membership,
    main,
    run_prompt_comparison,
    write_comparison_summary,
)
from rsna_knee.labels.subscription import LABEL_COLUMNS, STUDY_COLUMN, REPORT_COLUMN
from rsna_knee.labels.subscription import LabelingError


class PromptComparisonCliTests(unittest.TestCase):
    def setUp(self):
        self.rows = []
        for index in range(58):
            row = {
                STUDY_COLUMN: f"synthetic-study-{index}",
                REPORT_COLUMN: f"synthetic report {index}",
            }
            for target_index, target in enumerate(LABEL_COLUMNS):
                row[target] = bool((index + target_index) % 2)
            self.rows.append(row)

    def test_live_evaluation_fails_closed_until_both_prerequisites_are_confirmed(self):
        calls = []
        for kwargs in (
            {},
            {"transfer_prerequisites_confirmed": True},
            {"subscription_oauth_confirmed": True},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "prerequisites"):
                    run_prompt_comparison(self.rows, infer=lambda *a, **k: calls.append(a), **kwargs)
        self.assertEqual(calls, [])

    def test_infeasible_fold_labels_are_rejected_before_inference(self):
        rows = [dict(row, **{target: True for target in LABEL_COLUMNS}) for row in self.rows]
        calls = []
        with self.assertRaisesRegex(ValueError, "positive and negative"):
            run_prompt_comparison(
                rows,
                infer=lambda *args, **kwargs: calls.append((args, kwargs)),
                transfer_prerequisites_confirmed=True,
                subscription_oauth_confirmed=True,
                study_manifest_keys=[row[STUDY_COLUMN] for row in self.rows],
            )
        self.assertEqual(calls, [])

    def test_cli_requires_explicit_confirmation_before_reading_input(self):
        with self.assertRaises(SystemExit) as raised:
            main(["--input", "missing-private-input.csv"])
        self.assertEqual(raised.exception.code, 2)

    def test_synthetic_run_writes_only_reproducible_redacted_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "data" / "outputs" / "gold58-prompt-comparison.json"
            target_scores = {target: 0.5 for target in LABEL_COLUMNS}
            candidate_metrics = {
                "macro_balanced_accuracy": 0.5,
                "per_target_balanced_accuracy": target_scores,
                "macro_brier_score": 0.25,
            }
            inner_fold_metrics = [
                {candidate: candidate_metrics for candidate in ("candidate_1", "candidate_2", "candidate_3")}
                for _ in range(3)
            ]
            summary = {
                "schema_version": 1,
                "protocol": "nested_study_level_3x3",
                "candidate_ids": ["candidate_1", "candidate_2", "candidate_3"],
                "selection_metric": "mean_inner_macro_balanced_accuracy",
                "tie_break_order": ["candidate_1", "candidate_2", "candidate_3"],
                "outer_fold_role": "evaluate_inner_selection_only",
                "model": "gpt-6-luna",
                "reasoning_effort": "high",
                "outer_folds": [
                    {
                        "selected_candidate_id": "candidate_1",
                        "inner_scores": {candidate: 0.5 for candidate in ("candidate_1", "candidate_2", "candidate_3")},
                        "outer_score": 0.5,
                        "inner_fold_metrics": inner_fold_metrics,
                        "outer_per_target_balanced_accuracy": target_scores,
                        "outer_macro_brier_score": 0.25,
                    }
                    for _ in range(3)
                ],
                "provider_usage": "unavailable_unless_explicitly_reported",
            }
            write_comparison_summary(output, summary, project_root=temp_dir)
            saved = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(summary, saved)
        self.assertEqual(len(saved["outer_folds"]), 3)
        self.assertTrue(all(set(fold) == {
            "selected_candidate_id", "inner_scores", "outer_score", "inner_fold_metrics",
            "outer_per_target_balanced_accuracy", "outer_macro_brier_score",
        }
                            for fold in saved["outer_folds"]))
        self.assertTrue(all(len(fold["inner_fold_metrics"]) == 3 for fold in saved["outer_folds"]))
        self.assertTrue(all(len(fold["outer_per_target_balanced_accuracy"]) == len(LABEL_COLUMNS)
                            for fold in saved["outer_folds"]))
        self.assertNotIn("synthetic-study-", json.dumps(saved))
        self.assertNotIn("synthetic report", json.dumps(saved))

    def test_manifest_mismatch_fails_with_redacted_message_before_inference(self):
        private_key = "SYNTHETIC_PRIVATE_KEY_CANARY"
        calls = []
        manifest = [row[STUDY_COLUMN] for row in self.rows]
        manifest[0] = private_key
        with self.assertRaisesRegex(ValueError, "manifest") as raised:
            run_prompt_comparison(
                self.rows,
                infer=lambda *args, **kwargs: calls.append(args),
                transfer_prerequisites_confirmed=True,
                subscription_oauth_confirmed=True,
                study_manifest_keys=manifest,
            )
        self.assertNotIn(private_key, str(raised.exception))
        self.assertEqual(calls, [])

    def test_manifest_exact_match_is_accepted_without_inference(self):
        keys = [row[STUDY_COLUMN] for row in self.rows]
        self.assertIsNone(_validate_manifest_membership(keys, list(keys)))

    def test_checkpoint_resumes_exact_run_without_repeating_inference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            calls = []

            def infer(_report, **_kwargs):
                calls.append(1)
                return {target: {"label": False, "uncertainty": 0.5} for target in LABEL_COLUMNS}

            kwargs = {
                "infer": infer,
                "output_path": Path(temp_dir) / "data" / "outputs" / "gold58-prompt-comparison.json",
                "project_root": temp_dir,
                "transfer_prerequisites_confirmed": True,
                "subscription_oauth_confirmed": True,
                "study_manifest_keys": [row[STUDY_COLUMN] for row in self.rows],
            }
            first = run_prompt_comparison(self.rows, **kwargs)
            first_call_count = len(calls)
            second = run_prompt_comparison(self.rows, **kwargs)
            self.assertGreater(first_call_count, 0)
            self.assertEqual(len(calls), first_call_count)
            self.assertEqual(first, second)
            checkpoint = Path(temp_dir) / "data" / "outputs" / ".gold58-prompt-comparison-checkpoint.json"
            contents = checkpoint.read_text(encoding="utf-8")
            self.assertNotIn("synthetic-study-", contents)
            self.assertNotIn("synthetic report", contents)

    def test_failure_aborts_once_and_persists_only_safe_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            calls = []
            sensitive = "SYNTHETIC_STUDY_REPORT_PROMPT_PAYLOAD"

            def infer(_report, **_kwargs):
                calls.append(1)
                if len(calls) == 2:
                    raise LabelingError(sensitive, stage="process_exit", cause_code="auth_error", exit_status=9)
                return {target: {"label": False, "uncertainty": 0.5} for target in LABEL_COLUMNS}

            with self.assertRaisesRegex(ValueError, "candidate evaluation failed"):
                run_prompt_comparison(
                    self.rows, infer=infer, project_root=temp_dir,
                    transfer_prerequisites_confirmed=True,
                    subscription_oauth_confirmed=True,
                    study_manifest_keys=[row[STUDY_COLUMN] for row in self.rows],
                )
            self.assertEqual(calls, [1, 1])
            checkpoint = Path(temp_dir) / "data" / "outputs" / ".gold58-prompt-comparison-checkpoint.json"
            serialized = checkpoint.read_text(encoding="utf-8")
            for token in (sensitive, "synthetic-study-", "synthetic report", "SYNTHETIC"):
                self.assertNotIn(token, serialized)
            saved = json.loads(serialized)
            self.assertEqual(len(saved["failures"]), 1)
            self.assertEqual(len(saved["completed"]), 1)
            self.assertEqual(next(iter(saved["failures"].values())), {
                "stage": "process_exit", "cause_code": "auth_error", "exit_status": 9,
            })
            retry_calls = []

            def retry_infer(_report, **_kwargs):
                retry_calls.append(1)
                return {target: {"label": False, "uncertainty": 0.5} for target in LABEL_COLUMNS}

            run_prompt_comparison(
                self.rows, infer=retry_infer, project_root=temp_dir,
                transfer_prerequisites_confirmed=True,
                subscription_oauth_confirmed=True,
                study_manifest_keys=[row[STUDY_COLUMN] for row in self.rows],
            )
            self.assertEqual(len(retry_calls), 405)

    def test_manifest_reader_rejects_noncanonical_manifest_path(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import rsna_knee.labels.prompt_comparison_cli as cli

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "repo"
            arbitrary = root / "data" / "private" / "other.csv"
            arbitrary.parent.mkdir(parents=True)
            arbitrary.write_text("synthetic\n", encoding="utf-8")
            with patch.object(cli, "REPOSITORY_ROOT", root):
                with self.assertRaisesRegex(ValueError, "canonical private study manifest"):
                    _read_study_manifest_csv(arbitrary)
    def test_manifest_reader_rejects_paths_outside_private_ignored_data(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import rsna_knee.labels.prompt_comparison_cli as cli

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "repo"
            root.mkdir()
            outside = Path(temp_dir) / "manifest.csv"
            with patch.object(cli, "REPOSITORY_ROOT", root):
                with self.assertRaisesRegex(ValueError, "canonical private study manifest"):
                    _read_study_manifest_csv(outside)

    def test_csv_duplicate_headers_are_rejected_before_dictreader(self):
        import csv
        import tempfile
        from pathlib import Path

        headers = [STUDY_COLUMN, STUDY_COLUMN, REPORT_COLUMN, *LABEL_COLUMNS]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "synthetic.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                csv.writer(stream).writerow(headers)
            with self.assertRaisesRegex(ValueError, "duplicate CSV header"):
                _read_comparison_csv(path)

    def test_cli_default_output_is_anchored_to_repository_root_not_current_directory(self):
        import csv
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import rsna_knee.labels.prompt_comparison_cli as cli

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            repo = base / "repo"
            elsewhere = base / "elsewhere"
            repo.mkdir()
            elsewhere.mkdir()
            source = base / "synthetic.csv"
            manifest = repo / "data" / "private" / "gold58-study-keys.csv"
            headers = [STUDY_COLUMN, REPORT_COLUMN, *LABEL_COLUMNS]
            with source.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=headers)
                writer.writeheader()
            manifest.parent.mkdir(parents=True)
            with manifest.open("w", encoding="utf-8", newline="") as stream:
                csv.writer(stream).writerow([STUDY_COLUMN])
            captured = {}

            def fake_run(rows, **kwargs):
                captured["output_path"] = kwargs["output_path"]
                return {"outer_folds": [1, 2, 3]}

            old_cwd = Path.cwd()
            try:
                os.chdir(elsewhere)
                with patch.object(cli, "REPOSITORY_ROOT", repo), patch.object(
                    cli, "run_prompt_comparison", side_effect=fake_run
                ):
                    main([
                        "--input", str(source), "--study-manifest", str(manifest),
                        "--confirm-transfer-prerequisites", "--subscription-oauth-confirmed",
                    ])
            finally:
                os.chdir(old_cwd)
            self.assertEqual(
                Path(captured["output_path"]),
                repo / "data" / "outputs" / "gold58-prompt-comparison.json",
            )

    def test_summary_writer_rejects_paths_outside_ignored_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = {
                "schema_version": 1,
                "protocol": "nested_study_level_3x3",
                "candidate_ids": list(("candidate_1", "candidate_2", "candidate_3")),
                "selection_metric": "mean_inner_macro_balanced_accuracy",
                "tie_break_order": list(("candidate_1", "candidate_2", "candidate_3")),
                "outer_fold_role": "evaluate_inner_selection_only",
                "model": "gpt-6-luna",
                "reasoning_effort": "high",
                "outer_folds": [],
                "provider_usage": "unavailable_unless_explicitly_reported",
            }
            with self.assertRaisesRegex(ValueError, "data/outputs"):
                write_comparison_summary(Path(temp_dir) / "summary.json", summary,
                                         project_root=temp_dir)

    def test_summary_writer_rejects_unrecognized_fields_that_could_leak_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "data" / "outputs" / "gold58-prompt-comparison.json"
            with self.assertRaisesRegex(ValueError, "summary schema"):
                write_comparison_summary(
                    destination,
                    {"report": "PRIVATE_REPORT_CANARY", "outer_folds": []},
                    project_root=temp_dir,
                )


if __name__ == "__main__":
    unittest.main()
