import json
import csv
import os
import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from label_reports import _read_rows

from rsna_knee.labels.subscription import (
    LABEL_COLUMNS,
    LabelingError,
    eligible_reports,
    run_pilot,
    validate_label_result,
)


def gold_fixture(*extra_ids):
    values = set(extra_ids)
    index = 0
    while len(values) < 58:
        values.add(f"synthetic-gold-{index}")
        index += 1
    return values


class SubscriptionLabelContractTests(unittest.TestCase):
    def test_csv_reader_rejects_rows_with_missing_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["StudyInstanceUID", "Report", *LABEL_COLUMNS])
                writer.writerow(["synthetic-study", "synthetic report", *([""] * (len(LABEL_COLUMNS) - 1))])
            with self.assertRaises(LabelingError):
                _read_rows(path)

    def test_target_schema_is_the_competition_target_order(self):
        self.assertEqual(
            LABEL_COLUMNS,
            (
                "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
                "Medial OA", "Lateral OA", "PF OA", "Effusion",
                "Synovitis", "Baker's", "Contusion", "Fracture",
            ),
        )

    def test_eligibility_requires_training_schema_and_all_targets_blank(self):
        rows = [
            {"StudyInstanceUID": "synthetic-a", "Report": "synthetic report", **{c: "" for c in LABEL_COLUMNS}},
            {"StudyInstanceUID": "synthetic-b", "Report": "synthetic report", **{c: "" for c in LABEL_COLUMNS}},
            {"StudyInstanceUID": "synthetic-c", "Report": "synthetic report", **{c: "" for c in LABEL_COLUMNS}},
        ]
        rows[1]["ACL"] = "0"
        selected = eligible_reports(rows, gold_ids=gold_fixture("synthetic-c"))
        self.assertEqual([row["StudyInstanceUID"] for row in selected], ["synthetic-a"])

    def test_selector_fails_closed_on_missing_schema_or_gold_exclusion(self):
        with self.assertRaises(LabelingError):
            eligible_reports([{"Report": "synthetic report"}], gold_ids=gold_fixture())
        row = {"StudyInstanceUID": "synthetic-a", "Report": "synthetic report"}
        row.update({column: "" for column in LABEL_COLUMNS[:-1]})
        with self.assertRaises(LabelingError):
            eligible_reports([row], gold_ids=None)
        with self.assertRaises(LabelingError):
            eligible_reports([row], gold_ids={"too-few"})
        with self.assertRaises(LabelingError):
            eligible_reports([row], gold_ids=["duplicate"] * 58)
        with self.assertRaises(LabelingError):
            eligible_reports([row], gold_ids=None)

    def test_response_requires_exact_boolean_targets_and_evidence(self):
        valid = {column: {"label": False, "evidence": "synthetic phrase", "uncertainty": 0.2}
                 for column in LABEL_COLUMNS}
        self.assertEqual(validate_label_result(valid), valid)
        invalid = dict(valid)
        invalid.pop(LABEL_COLUMNS[0])
        with self.assertRaises(LabelingError):
            validate_label_result(invalid)
        invalid = dict(valid)
        invalid[LABEL_COLUMNS[0]] = {"label": 1, "evidence": "x", "uncertainty": 0.2}
        with self.assertRaises(LabelingError):
            validate_label_result(invalid)

    def test_pilot_is_bounded_resumable_and_uses_only_codex_cli(self):
        rows = [{"StudyInstanceUID": f"synthetic-{i}", "Report": "synthetic report",
                 **{column: "" for column in LABEL_COLUMNS}} for i in range(3)]
        calls = []

        def fake_infer(report):
            calls.append(report)
            return {column: {"label": False, "evidence": "synthetic", "uncertainty": 0.1}
                    for column in LABEL_COLUMNS}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "data" / "outputs" / "labels.jsonl"
            with self.assertRaises(LabelingError):
                run_pilot(rows, gold_ids=gold_fixture(), max_records=0, output_path=output,
                          project_root=root, subscription_oauth_confirmed=True,
                          infer=fake_infer)
            result = run_pilot(rows, gold_ids=gold_fixture(), max_records=2, output_path=output,
                               project_root=root, subscription_oauth_confirmed=True,
                               infer=fake_infer)
            self.assertEqual(result, 2)
            self.assertEqual(len(calls), 2)
            self.assertEqual(run_pilot(rows, gold_ids=gold_fixture(), max_records=2,
                                       output_path=output, project_root=root,
                                       subscription_oauth_confirmed=True,
                                       infer=fake_infer), 2)
            self.assertEqual(len(calls), 2)
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 2)

    def test_cli_adapter_never_reads_or_invokes_api_key_auth(self):
        from rsna_knee.labels.subscription import codex_cli_infer

        # The adapter must construct a Codex CLI subscription flow and never expose prompts/results in errors.
        with self.assertRaises(LabelingError):
            codex_cli_infer("synthetic report", executable="openai")

    def test_cli_adapter_uses_codex_exec_stdin_and_does_not_leak_output(self):
        from rsna_knee.labels.subscription import codex_cli_infer

        response = {column: {"label": False, "evidence": "", "uncertainty": 0.5}
                    for column in LABEL_COLUMNS}
        stdout = json.dumps({"type": "item.completed", "item": {
            "type": "agent_message", "text": json.dumps(response)}})
        with patch("rsna_knee.labels.subscription.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = stdout
            result = codex_cli_infer("synthetic report", subscription_oauth_confirmed=True)
        self.assertEqual(result[LABEL_COLUMNS[0]]["label"], False)
        args, kwargs = run.call_args
        self.assertEqual(args[0], ["codex", "exec", "--json", "--sandbox", "read-only", "-"])
        self.assertIn("synthetic report", kwargs["input"])
        self.assertTrue(kwargs["capture_output"])
        self.assertNotIn("synthetic report", repr(result))


if __name__ == "__main__":
    unittest.main()
