import json
import csv
import ctypes
import os
import subprocess
import time
import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import Mock, patch

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
from rsna_knee.labels.siwc_responses import SIWCResponsesInfer


def gold_fixture(*extra_ids):
    values = set(extra_ids)
    index = 0
    while len(values) < 58:
        values.add(f"synthetic-gold-{index}")
        index += 1
    return values


class SubscriptionLabelContractTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows subprocess text encoding regression")
    def test_process_stdin_is_utf8_for_unicode_report_text(self):
        from rsna_knee.labels.subscription import _run_process

        payload = "Synthetic — report’s café"
        child = (
            "import sys\n"
            "data = sys.stdin.buffer.read()\n"
            "try:\n    text = data.decode('utf-8')\n"
            "except UnicodeDecodeError:\n    raise SystemExit(17)\n"
            f"raise SystemExit(0 if text == {payload!r} else 18)\n"
        )
        result = _run_process([sys.executable, "-c", child], payload, 5)
        self.assertEqual(result.returncode, 0)

    def test_diagnostics_journal_records_safe_cause_progress_and_offset(self):
        from rsna_knee.labels.subscription import codex_cli_infer

        rows = [{"StudyInstanceUID": f"synthetic-{i}", "Report": f"report-{i}",
                 **{column: "" for column in LABEL_COLUMNS}} for i in range(14)]
        valid = {column: {"label": False, "evidence": "synthetic", "uncertainty": 0.1}
                 for column in LABEL_COLUMNS}
        calls = 0

        def infer(report):
            nonlocal calls
            calls += 1
            if calls == 2:
                error = LabelingError("PRIVATE_REPORT SECRET_TOKEN PRIVATE_CANARY")
                error.cause_code = "auth_error"
                error.exit_status = 2
                raise error
            return valid

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            journal = root / "data" / "outputs" / "diagnostics.jsonl"
            output = root / "data" / "outputs" / "labels.jsonl"
            with self.assertRaises(LabelingError) as raised:
                run_pilot(rows, gold_ids=gold_fixture(), max_records=9, skip_records=5,
                          output_path=output, diagnostics_path=journal, project_root=root,
                          subscription_oauth_confirmed=True, infer=infer)
            self.assertEqual(calls, 2)
            self.assertIn("started=2", str(raised.exception))
            self.assertIn("completed=1", str(raised.exception))
            self.assertIn("failed=1", str(raised.exception))
            self.assertNotIn("PRIVATE_REPORT", str(raised.exception))
            self.assertNotIn("SECRET_TOKEN", str(raised.exception))
            self.assertNotIn("PRIVATE_CANARY", str(raised.exception))
            records = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([record["event"] for record in records],
                             ["attempt_started", "attempt_completed", "attempt_started", "attempt_failed"])
            self.assertEqual(records[2]["attempt_offset"], 6)
            self.assertEqual(records[3]["cause_code"], "auth_error")
            self.assertEqual(records[3]["stage"], "inference")
            self.assertGreaterEqual(records[3]["elapsed_ms"], 0)
            self.assertEqual(records[3]["exit_status"], 2)
            self.assertEqual(records[3]["progress"], {"started": 2, "completed": 1, "failed": 1})
            journal_text = journal.read_text(encoding="utf-8")
            self.assertNotIn("PRIVATE_REPORT", journal_text)
            self.assertNotIn("SECRET_TOKEN", journal_text)
            self.assertNotIn("PRIVATE_CANARY", journal_text)

    def test_invalid_label_json_type_is_allowlisted_in_journal_and_wrapper(self):
        rows = [{"StudyInstanceUID": "synthetic-study", "Report": "SYNTHETIC_REPORT_CANARY",
                 **{column: "" for column in LABEL_COLUMNS}}]
        invalid = {column: {"label": False, "evidence": "synthetic", "uncertainty": 0.2}
                   for column in LABEL_COLUMNS}
        invalid[LABEL_COLUMNS[0]]["label"] = {"secret": "PRIVATE_VALUE_CANARY"}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            journal = root / "data" / "outputs" / "diagnostics.jsonl"
            output = root / "data" / "outputs" / "labels.jsonl"
            with self.assertRaises(LabelingError) as raised:
                run_pilot(rows, gold_ids=gold_fixture(), max_records=1,
                          output_path=output, diagnostics_path=journal, project_root=root,
                          subscription_oauth_confirmed=True, infer=lambda _report: invalid)
            records = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]

        failed = next(record for record in records if record["event"] == "attempt_failed")
        self.assertEqual(failed["invalid_label_json_type"], "object")
        self.assertEqual(raised.exception.invalid_label_json_type, "object")
        self.assertIn("invalid_label_json_type=object", str(raised.exception))
        for text in (json.dumps(invalid), "PRIVATE_VALUE_CANARY", LABEL_COLUMNS[0]):
            self.assertNotIn(text, json.dumps(records))
            self.assertNotIn(text, str(raised.exception))

    def test_evidence_mismatch_kind_is_allowlisted_in_journal_and_wrapper(self):
        rows = [{"StudyInstanceUID": "synthetic-study", "Report": "SYNTHETIC_REPORT_CANARY",
                 **{target: "" for target in LABEL_COLUMNS}}]
        error = LabelingError("PRIVATE_VALUE_CANARY", cause_code="evidence_not_grounded",
                              evidence_mismatch_kind="case_only")
        error.evidence_mismatch_kind = "PRIVATE_MUTATED_CANARY"
        infer = Mock(side_effect=error)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "data" / "outputs" / "labels.jsonl"
            journal = root / "data" / "outputs" / "diagnostics.jsonl"
            with self.assertRaises(LabelingError) as raised:
                run_pilot(rows, gold_ids=gold_fixture(), max_records=1,
                          output_path=output, diagnostics_path=journal, project_root=root,
                          subscription_oauth_confirmed=True, infer=infer)
            journal_text = journal.read_text(encoding="utf-8")
            records = [json.loads(line) for line in journal_text.splitlines()]

        failed = next(record for record in records if record["event"] == "attempt_failed")
        self.assertNotIn("evidence_mismatch_kind", failed)
        self.assertIsNone(raised.exception.evidence_mismatch_kind)
        self.assertNotIn("PRIVATE_MUTATED_CANARY", journal_text)
        self.assertNotIn("PRIVATE_VALUE_CANARY", journal_text)
        self.assertNotIn("SYNTHETIC_REPORT_CANARY", str(raised.exception))

    def test_cached_whitespace_only_evidence_remains_strict(self):
        rows = [{"StudyInstanceUID": "synthetic-study", "Report": "ACL tear",
                 **{target: "" for target in LABEL_COLUMNS}}]
        cached = {target: {"label": False, "evidence": "", "uncertainty": 0.25}
                  for target in LABEL_COLUMNS}
        cached[LABEL_COLUMNS[0]]["evidence"] = "ACL\ntear"
        payload = json.dumps({"StudyInstanceUID": "synthetic-study", "labels": cached}) + "\n"
        infer = Mock()
        infer.validate_source_evidence = SIWCResponsesInfer.validate_source_evidence
        captured = []

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "data" / "outputs" / "labels.jsonl"
            output.parent.mkdir(parents=True)
            output.write_text(payload, encoding="utf-8")
            with self.assertRaises(LabelingError) as raised:
                run_pilot(rows, gold_ids=gold_fixture(), max_records=1,
                          output_path=output, diagnostics_path=root / "data" / "outputs" / "diagnostics.jsonl",
                          project_root=root, subscription_oauth_confirmed=True, infer=infer,
                          rejected_evidence_capture=lambda evidence, offset: captured.append((evidence, offset)))
            after = output.read_text(encoding="utf-8")

        infer.assert_not_called()
        self.assertEqual(after, payload)
        self.assertEqual(raised.exception.evidence_mismatch_kind, "whitespace_only")
        self.assertEqual(captured, [])

    def test_failure_classifier_uses_fixed_safe_causes_and_unknown_fallback(self):
        from rsna_knee.labels.subscription import _classify_cli_failure

        self.assertEqual(_classify_cli_failure("Error: 401 Unauthorized", ""), "auth_error")
        self.assertEqual(_classify_cli_failure("", "Error: rate limit exceeded"), "quota_error")
        self.assertEqual(_classify_cli_failure("", "Error: connection refused"), "network_error")
        self.assertEqual(_classify_cli_failure("", "Error: MCP server unavailable"), "mcp_error")
        self.assertEqual(_classify_cli_failure("", "Error: sandbox denied"), "sandbox_error")
        self.assertEqual(_classify_cli_failure("", "Error: unrecognized argument"), "arguments_error")
        self.assertEqual(_classify_cli_failure("", "Error: failed to parse config"), "configuration_error")
        self.assertEqual(_classify_cli_failure("", "Error: model not found"), "model_error")
        self.assertEqual(_classify_cli_failure("", "private report says unauthorized"), "unknown")
        self.assertEqual(_classify_cli_failure("", "PRIVATE_CANARY"), "unknown")

    def test_real_cli_failure_metadata_is_journaled_without_raw_cli_text(self):
        from rsna_knee.labels.subscription import codex_cli_infer

        rows = [{"StudyInstanceUID": "synthetic-a", "Report": "synthetic report",
                 **{column: "" for column in LABEL_COLUMNS}}]
        private_canary = "PRIVATE_REPORT_OR_TOKEN_CANARY"
        failure = subprocess.CompletedProcess(
            ["codex"], 1, stdout=private_canary,
            stderr=f"Error: 401 Unauthorized {private_canary}",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            journal = root / "data" / "outputs" / "diagnostics.jsonl"
            with patch("rsna_knee.labels.subscription._run_process", return_value=failure):
                with self.assertRaises(LabelingError) as raised:
                    run_pilot(rows, gold_ids=gold_fixture(), max_records=1,
                              output_path=root / "data" / "outputs" / "labels.jsonl",
                              diagnostics_path=journal, project_root=root,
                              subscription_oauth_confirmed=True)
            self.assertIn("authentication problem", str(raised.exception))
            self.assertEqual(raised.exception.exit_status, 1)
            journal_text = journal.read_text(encoding="utf-8")
            self.assertIn('"cause_code":"auth_error"', journal_text)
            self.assertIn('"exit_status":1', journal_text)
            self.assertNotIn(private_canary, journal_text)
            self.assertNotIn(private_canary, str(raised.exception))

    def test_diagnostics_path_must_be_private_outputs_and_not_alias_labels(self):
        rows = [{"StudyInstanceUID": "synthetic-a", "Report": "synthetic report",
                 **{column: "" for column in LABEL_COLUMNS}}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "data" / "outputs" / "labels.jsonl"
            with self.assertRaises(LabelingError):
                run_pilot(rows, gold_ids=gold_fixture(), max_records=1,
                          output_path=output, diagnostics_path=output, project_root=root,
                          subscription_oauth_confirmed=True,
                          infer=lambda report: {})

    def test_optin_capture_writer_is_one_slot_bounded_and_payload_minimal(self):
        from rsna_knee.labels.subscription import _RejectedEvidenceCapture

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = _RejectedEvidenceCapture(root)
            evidence = "synthetic evidence é"
            self.assertTrue(capture(evidence, 7))
            payload = json.loads(capture.path.read_text(encoding="utf-8"))
            self.assertEqual(payload, {"evidence": evidence, "eligible_offset": 7})
            original = capture.path.read_bytes()
            self.assertFalse(capture("SECOND_SYNTHETIC_EVIDENCE", 8))
            self.assertEqual(capture.path.read_bytes(), original)

    def test_optin_capture_writer_refuses_oversize_and_write_errors_without_payload(self):
        from rsna_knee.labels.subscription import _RejectedEvidenceCapture

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = _RejectedEvidenceCapture(root)
            self.assertFalse(capture("x" * (16 * 1024 + 1), 0))
            self.assertFalse(capture.path.exists())
            with patch("rsna_knee.labels.subscription.os.open", side_effect=OSError("PRIVATE_CANARY")):
                self.assertFalse(capture("safe synthetic evidence", 0))
            self.assertFalse(capture.path.exists())

        with tempfile.TemporaryDirectory() as directory:
            exact_limit = _RejectedEvidenceCapture(Path(directory))
            evidence = "é" * (16 * 1024 // len("é".encode("utf-8")))
            self.assertEqual(len(evidence.encode("utf-8")), 16 * 1024)
            self.assertTrue(exact_limit(evidence, 0))
        with tempfile.TemporaryDirectory() as directory:
            over_limit = _RejectedEvidenceCapture(Path(directory))
            self.assertFalse(over_limit("é" * (16 * 1024 // 2 + 1), 0))
            self.assertFalse(over_limit.path.exists())

    def test_optin_capture_writer_refuses_existing_slot_and_directory_alias(self):
        from rsna_knee.labels.subscription import _RejectedEvidenceCapture

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = _RejectedEvidenceCapture(root)
            self.assertTrue(capture.prepare())
            capture.path.write_text("SYNTHETIC_PREEXISTING", encoding="utf-8")
            self.assertFalse(capture.prepare())
            self.assertFalse(capture("replacement", 0))
            self.assertEqual(capture.path.read_text(encoding="utf-8"), "SYNTHETIC_PREEXISTING")

        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as target:
            root = Path(directory)
            (root / "data").mkdir()
            try:
                os.symlink(target, root / "data" / "private", target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("Directory symlinks are unavailable in this environment.")
            self.assertFalse(_RejectedEvidenceCapture(root).prepare())

    def test_optin_capture_writer_rejects_hardlink_at_fixed_slot(self):
        from rsna_knee.labels.subscription import _RejectedEvidenceCapture

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = _RejectedEvidenceCapture(root)
            capture.path.parent.mkdir(parents=True)
            original = root / "outside.json"
            original.write_text("SYNTHETIC_HARDLINK_TARGET", encoding="utf-8")
            try:
                os.link(original, capture.path)
            except (OSError, NotImplementedError):
                self.skipTest("Hard links are unavailable in this environment.")
            self.assertFalse(capture.prepare())
            self.assertFalse(capture("replacement", 0))
            self.assertEqual(original.read_text(encoding="utf-8"), "SYNTHETIC_HARDLINK_TARGET")

    def test_runner_optin_capture_receives_eligible_offset_and_preserves_checkpoint(self):
        rows = [{"StudyInstanceUID": f"synthetic-{i}", "Report": "synthetic report",
                 **{target: "" for target in LABEL_COLUMNS}} for i in range(4)]
        cached = {target: {"label": False, "evidence": "", "uncertainty": 0.2}
                  for target in LABEL_COLUMNS}
        calls = []

        def infer(_report, *, on_rejected_evidence):
            on_rejected_evidence("PRIVATE_EVIDENCE_CANARY")
            raise LabelingError("Responses evidence was not found in the source report.",
                                stage="response_validation", cause_code="evidence_not_grounded",
                                evidence_mismatch_kind="no_literal_match")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "data" / "outputs" / "labels.jsonl"
            output.parent.mkdir(parents=True)
            output.write_text(json.dumps({"StudyInstanceUID": "synthetic-0", "labels": cached}) + "\n",
                              encoding="utf-8")
            before = output.read_bytes()
            with self.assertRaises(LabelingError) as raised:
                run_pilot(
                    rows, gold_ids=gold_fixture(), max_records=2, skip_records=2,
                    output_path=output,
                    diagnostics_path=root / "data" / "outputs" / "diagnostics.jsonl",
                    project_root=root, subscription_oauth_confirmed=True, infer=infer,
                    rejected_evidence_capture=lambda evidence, offset: calls.append((evidence, offset)),
                )
            self.assertEqual(output.read_bytes(), before)
            journal_text = (root / "data" / "outputs" / "diagnostics.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("PRIVATE_EVIDENCE_CANARY", journal_text)
            self.assertNotIn("PRIVATE_EVIDENCE_CANARY", str(raised.exception))
        self.assertEqual(calls, [("PRIVATE_EVIDENCE_CANARY", 2)])

    def test_journal_start_failure_stops_before_inference(self):
        rows = [{"StudyInstanceUID": "synthetic-a", "Report": "synthetic report",
                 **{column: "" for column in LABEL_COLUMNS}}]
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("rsna_knee.labels.subscription._DiagnosticsJournal.append",
                       side_effect=LabelingError("safe journal failure", stage="journal_persist")):
                with self.assertRaises(LabelingError):
                    run_pilot(rows, gold_ids=gold_fixture(), max_records=1,
                              output_path=root / "data" / "outputs" / "labels.jsonl",
                              diagnostics_path=root / "data" / "outputs" / "diagnostics.jsonl",
                              project_root=root, subscription_oauth_confirmed=True,
                              infer=lambda report: calls.append(report))
        self.assertEqual(calls, [])
    def test_windows_timeout_cleanup_failure_is_surfaced(self):
        from rsna_knee.labels.subscription import _run_process

        process = Mock()
        process.pid = 1234
        process.communicate.side_effect = [
            subprocess.TimeoutExpired("synthetic", 0.01), ("", "")
        ]
        with patch("rsna_knee.labels.subscription.subprocess.Popen", return_value=process):
            with patch("rsna_knee.labels.subscription.os.name", "nt"):
                with patch("rsna_knee.labels.subscription.subprocess.run") as cleanup:
                    cleanup.return_value.returncode = 1
                    with self.assertRaisesRegex(LabelingError, "process tree cleanup failed"):
                        _run_process(["cmd.exe", "/c", "synthetic"], "synthetic report", 0.01)
        cleanup.assert_called_once()
        self.assertIn("/T", cleanup.call_args.args[0])
        self.assertNotIn("synthetic report", " ".join(cleanup.call_args.args[0]))
        process.kill.assert_called_once()

    @unittest.skipUnless(os.name == "nt", "Windows process-tree integration test")
    def test_windows_timeout_kills_synthetic_child_process(self):
        from rsna_knee.labels.subscription import _run_process

        with tempfile.TemporaryDirectory() as directory:
            child_pid_path = Path(directory) / "child.pid"
            child_code = (
                "import time; time.sleep(30)"
            )
            parent_code = (
                "import subprocess,sys,time; "
                "child=subprocess.Popen([sys.executable,'-c',sys.argv[1]]); "
                "open(sys.argv[2],'w').write(str(child.pid)); time.sleep(30)"
            )
            command = [sys.executable, "-c", parent_code, child_code, str(child_pid_path)]
            with self.assertRaisesRegex(LabelingError, "timed out"):
                _run_process(command, "synthetic input", 2)

            deadline = time.monotonic() + 5
            while not child_pid_path.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(child_pid_path.exists(), "synthetic child did not start")
            child_pid = int(child_pid_path.read_text(encoding="utf-8"))
            kernel32 = ctypes.windll.kernel32
            kernel32.OpenProcess.restype = ctypes.c_void_p
            kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                handle = kernel32.OpenProcess(0x00100000, False, child_pid)
                if not handle:
                    break
                try:
                    state = kernel32.WaitForSingleObject(handle, 0)
                finally:
                    kernel32.CloseHandle(handle)
                if state == 0:
                    break
                time.sleep(0.05)
            else:
                self.fail("synthetic child survived timeout cleanup")

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

    def test_output_schema_is_closed_and_contains_all_twelve_target_contracts(self):
        from pathlib import Path
        schema_path = Path(__file__).parents[1] / "src" / "rsna_knee" / "labels" / "codex_label_output.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(schema["type"], "object")
        self.assertIs(schema["additionalProperties"], False)
        self.assertEqual(set(schema["required"]), set(LABEL_COLUMNS))
        self.assertEqual(set(schema["properties"]), set(LABEL_COLUMNS))
        for target in LABEL_COLUMNS:
            with self.subTest(target=target):
                item = schema["properties"][target]
                self.assertEqual(item["type"], "object")
                self.assertIs(item["additionalProperties"], False)
                self.assertEqual(set(item["required"]), {"label", "evidence", "uncertainty"})
                self.assertEqual(item["properties"]["label"]["type"], "boolean")
                self.assertEqual(item["properties"]["evidence"]["type"], "string")
                self.assertEqual(item["properties"]["uncertainty"]["type"], "number")

    def test_validation_failures_have_granular_allowlisted_reason_codes(self):
        base = {column: {"label": False, "evidence": "synthetic", "uncertainty": 0.2}
                for column in LABEL_COLUMNS}
        cases = (
            (dict(list(base.items())[1:]), "target_set_invalid"),
            ({**base, LABEL_COLUMNS[0]: {"label": False, "evidence": "x", "uncertainty": 0.2,
                                         "extra": "PRIVATE_TARGET_CANARY"}}, "item_shape_invalid"),
            ({**base, LABEL_COLUMNS[0]: {"label": 1, "evidence": "x", "uncertainty": 0.2}}, "label_type_invalid"),
            ({**base, LABEL_COLUMNS[0]: {"label": False, "evidence": 7, "uncertainty": 0.2}}, "evidence_type_invalid"),
            ({**base, LABEL_COLUMNS[0]: {"label": False, "evidence": "x", "uncertainty": True}}, "uncertainty_type_invalid"),
            ({**base, LABEL_COLUMNS[0]: {"label": False, "evidence": "x", "uncertainty": 1.1}}, "uncertainty_range_invalid"),
        )
        for response, expected_reason in cases:
            with self.subTest(reason=expected_reason), self.assertRaises(LabelingError) as raised:
                validate_label_result(response)
            self.assertEqual(raised.exception.cause_code, expected_reason)
            self.assertNotIn("PRIVATE_TARGET_CANARY", str(raised.exception))

    def test_invalid_label_json_type_categories_are_fixed_and_payload_free(self):
        categories = (("PRIVATE_VALUE_CANARY", "string"), (2, "number"),
                      (None, "null"), (["PRIVATE_VALUE_CANARY"], "array"),
                      ({"private": "PRIVATE_VALUE_CANARY"}, "object"), (object(), "other"))
        for value, expected in categories:
            result = {column: {"label": False, "evidence": "synthetic", "uncertainty": 0.2}
                      for column in LABEL_COLUMNS}
            result[LABEL_COLUMNS[0]]["label"] = value
            with self.subTest(expected=expected), self.assertRaises(LabelingError) as raised:
                validate_label_result(result)
            self.assertEqual(raised.exception.invalid_label_json_type, expected)
            self.assertNotIn("PRIVATE_VALUE_CANARY", str(raised.exception))
            self.assertNotIn(LABEL_COLUMNS[0], str(raised.exception))

    def test_overlong_evidence_warns_without_truncation_at_201_characters(self):
        import warnings

        valid = {column: {"label": False, "evidence": "x", "uncertainty": 0.2}
                 for column in LABEL_COLUMNS}
        exact_limit = dict(valid)
        exact_limit[LABEL_COLUMNS[0]] = {
            "label": False, "evidence": "x" * 200, "uncertainty": 0.2,
        }
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            accepted_at_limit = validate_label_result(exact_limit)
        self.assertEqual(captured, [])
        self.assertEqual(accepted_at_limit[LABEL_COLUMNS[0]]["evidence"], "x" * 200)

        evidence = "SYNTHETIC_EVIDENCE_CANARY" + "x" * (201 - len("SYNTHETIC_EVIDENCE_CANARY"))
        over_limit = dict(valid)
        over_limit[LABEL_COLUMNS[0]] = {
            "label": False, "evidence": evidence, "uncertainty": 0.2,
        }
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            accepted_over_limit = validate_label_result(over_limit)
        self.assertEqual(len(captured), 1)
        self.assertEqual(accepted_over_limit[LABEL_COLUMNS[0]]["evidence"], evidence)
        warning_text = str(captured[0].message)
        self.assertNotIn(evidence, warning_text)
        self.assertNotIn("CANARY", warning_text)
        self.assertNotIn(LABEL_COLUMNS[0], warning_text)

    def test_malformed_final_json_has_safe_parse_reason(self):
        from rsna_knee.labels.subscription import codex_cli_infer
        canary = "PRIVATE_REPORT_CANARY"
        output = json.dumps({"type": "item.completed", "item": {
            "type": "agent_message", "text": "{" + f"not-json {canary}"}})
        process = Mock(returncode=0)
        process.communicate.return_value = (output, "")
        with patch("rsna_knee.labels.subscription.subprocess.Popen", return_value=process):
            with self.assertRaises(LabelingError) as raised:
                codex_cli_infer("synthetic report", subscription_oauth_confirmed=True)
        self.assertEqual(raised.exception.cause_code, "malformed_json")
        self.assertNotIn(canary, str(raised.exception))

    def test_codex_command_passes_output_schema_path_without_report_arguments(self):
        from rsna_knee.labels.subscription import _codex_command
        command = _codex_command()
        joined = command if isinstance(command, str) else " ".join(command)
        self.assertIn("--output-schema", joined)
        self.assertIn("codex_label_output.schema.json", joined)
        self.assertNotIn("synthetic report", joined)

    def test_windows_output_schema_argument_quotes_a_path_with_spaces(self):
        from pathlib import Path
        from rsna_knee.labels.subscription import _codex_command
        from rsna_knee.labels import subscription
        schema_path = Path("C:/synthetic project/schema with spaces.json")
        with patch("rsna_knee.labels.subscription.os.name", "nt"):
            with patch.object(subscription, "_OUTPUT_SCHEMA_PATH", schema_path):
                command = _codex_command()
        command_text = command if isinstance(command, str) else command[-1]
        self.assertIn(f'--output-schema "{schema_path}"', command_text)
        self.assertNotIn("synthetic report", command_text)

    @unittest.skipUnless(os.name == "nt", "Windows command-line regression test")
    def test_windows_launch_preserves_schema_path_as_one_child_argument(self):
        from rsna_knee.labels import subscription
        from rsna_knee.labels.subscription import _codex_command, _run_process

        payload = "SYNTHETIC_STDIN_CANARY"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "synthetic project"
            tools = root / "tools"
            tools.mkdir(parents=True)
            schema = root / "schema & (spaces)" / "label schema.json"
            schema.parent.mkdir()
            schema.write_text("{}", encoding="utf-8")
            capture = root / "capture argv.py"
            capture.write_text(
                "import json, sys\n"
                "print(json.dumps({'argv': sys.argv[1:], 'stdin': sys.stdin.read()}))\n",
                encoding="utf-8",
            )
            (tools / "codex.cmd").write_text(
                '@echo off\r\n"%PYTHON_EXE%" "%ARGV_CAPTURE_SCRIPT%" %*\r\n',
                encoding="ascii",
            )
            environment = {
                "PATH": str(tools) + os.pathsep + os.environ["PATH"],
                "PYTHON_EXE": sys.executable,
                "ARGV_CAPTURE_SCRIPT": str(capture),
            }
            with patch.object(subscription, "_OUTPUT_SCHEMA_PATH", schema):
                with patch.dict(os.environ, environment):
                    completed = _run_process(_codex_command(), payload, 20)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        captured = json.loads(completed.stdout)
        self.assertEqual(captured["argv"], [
            "exec", "--json", "--sandbox", "read-only", "--model", "gpt-6-luna",
            "--output-schema", str(schema), "--config", "model_reasoning_effort=high", "-",
        ])
        self.assertEqual(captured["stdin"], payload)

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
        from rsna_knee.labels.subscription import _codex_command, codex_cli_infer

        response = {column: {"label": False, "evidence": "", "uncertainty": 0.5}
                    for column in LABEL_COLUMNS}
        stdout = json.dumps({"type": "item.completed", "item": {
            "type": "agent_message", "text": json.dumps(response)}})
        process = Mock(returncode=0)
        process.communicate.return_value = (stdout, "")
        with patch("rsna_knee.labels.subscription.subprocess.Popen", return_value=process) as popen:
            result = codex_cli_infer("synthetic report", subscription_oauth_confirmed=True)
        self.assertEqual(result[LABEL_COLUMNS[0]]["label"], False)
        args, kwargs = popen.call_args
        self.assertEqual(args[0], _codex_command())
        self.assertEqual(process.communicate.call_args.kwargs["timeout"], 180)
        self.assertIn("synthetic report", process.communicate.call_args.kwargs["input"])
        self.assertNotIn("synthetic report", repr(result))

    def test_cli_adapter_preserves_exit_code_without_leaking_process_output(self):
        from rsna_knee.labels.subscription import _codex_command, codex_cli_infer

        canary = "PRIVATE_STDERR_CANARY"
        process = Mock(returncode=2)
        process.communicate.return_value = (f"stdout {canary}", f"stderr {canary}")
        with patch("rsna_knee.labels.subscription.subprocess.Popen", return_value=process) as popen:
            with self.assertRaises(LabelingError) as raised:
                codex_cli_infer("synthetic report", subscription_oauth_confirmed=True)

        self.assertIn("2", str(raised.exception))
        self.assertNotIn(canary, str(raised.exception))
        self.assertIn("details are suppressed for privacy", str(raised.exception))
        args, _ = popen.call_args
        self.assertEqual(args[0], _codex_command())
        self.assertIn("synthetic report", process.communicate.call_args.kwargs["input"])

    def test_cli_adapter_uses_windows_cmd_shim_without_passing_prompt_in_arguments(self):
        from rsna_knee.labels.subscription import _codex_command, codex_cli_infer

        response = {column: {"label": False, "evidence": "", "uncertainty": 0.5}
                    for column in LABEL_COLUMNS}
        stdout = json.dumps({"type": "item.completed", "item": {
            "type": "agent_message", "text": json.dumps(response)}})
        with patch("rsna_knee.labels.subscription.os.name", "nt"):
            expected_command = _codex_command()
            process = Mock(returncode=0)
            process.communicate.return_value = (stdout, "")
            with patch("rsna_knee.labels.subscription.subprocess.Popen", return_value=process) as popen:
                codex_cli_infer("synthetic report", subscription_oauth_confirmed=True)
        args, kwargs = popen.call_args
        self.assertEqual(args[0], expected_command)
        self.assertNotIn("synthetic report", str(args[0]))
        self.assertIn("synthetic report", process.communicate.call_args.kwargs["input"])
        self.assertFalse(kwargs.get("shell", False))

    @patch("rsna_knee.labels.subscription.os.name", "posix")
    def test_cli_command_pins_luna_and_high_effort_without_invoking_codex(self):
        from rsna_knee.labels.subscription import _OUTPUT_SCHEMA_PATH, _codex_command

        self.assertEqual(
            _codex_command(),
            ["codex", "exec", "--json", "--sandbox", "read-only",
             "--output-schema", str(_OUTPUT_SCHEMA_PATH),
             "--model", "gpt-6-luna", "--config",
             "model_reasoning_effort=high", "-"],
        )

    def test_pilot_skip_offset_selects_only_following_records_and_enforces_total_cap(self):
        rows = [{"StudyInstanceUID": f"synthetic-{i}", "Report": f"report-{i}",
                 **{column: "" for column in LABEL_COLUMNS}} for i in range(12)]
        calls = []

        def fake_infer(report):
            calls.append(report)
            return {column: {"label": False, "evidence": "synthetic", "uncertainty": 0.1}
                    for column in LABEL_COLUMNS}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "data" / "outputs" / "labels.jsonl"
            prior = {column: {"label": False, "evidence": "synthetic", "uncertainty": 0.1}
                     for column in LABEL_COLUMNS}
            output.parent.mkdir(parents=True)
            output.write_text(json.dumps({"StudyInstanceUID": "synthetic-0", "labels": prior}) + "\n",
                              encoding="utf-8")
            with self.assertRaises(LabelingError):
                run_pilot(rows, gold_ids=gold_fixture(), skip_records=5, max_records=10,
                          output_path=output, project_root=root,
                          subscription_oauth_confirmed=True, infer=fake_infer)
            self.assertEqual(calls, [])
            self.assertEqual(run_pilot(
                rows, gold_ids=gold_fixture(), skip_records=1, max_records=2,
                output_path=output, project_root=root,
                subscription_oauth_confirmed=True, infer=fake_infer,
            ), 3)
            self.assertEqual(calls, ["report-1", "report-2"])
            ids = [json.loads(line)["StudyInstanceUID"]
                   for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(ids, ["synthetic-0", "synthetic-1", "synthetic-2"])

    def test_pilot_allows_nine_record_batch_at_cumulative_limit_14(self):
        rows = [{"StudyInstanceUID": f"synthetic-{i}", "Report": f"report-{i}",
                 **{column: "" for column in LABEL_COLUMNS}} for i in range(14)]
        calls = []

        def fake_infer(report):
            calls.append(report)
            return {column: {"label": False, "evidence": "synthetic", "uncertainty": 0.1}
                    for column in LABEL_COLUMNS}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "data" / "outputs" / "labels.jsonl"
            prior = {column: {"label": False, "evidence": "synthetic", "uncertainty": 0.1}
                     for column in LABEL_COLUMNS}
            output.parent.mkdir(parents=True)
            output.write_text("".join(
                json.dumps({"StudyInstanceUID": f"synthetic-{i}", "labels": prior}) + "\n"
                for i in range(5)
            ), encoding="utf-8")

            self.assertEqual(run_pilot(
                rows, gold_ids=gold_fixture(), skip_records=5, max_records=9,
                output_path=output, project_root=root,
                subscription_oauth_confirmed=True, infer=fake_infer,
            ), 14)
            self.assertEqual(calls, [f"report-{i}" for i in range(5, 14)])

            with self.assertRaises(LabelingError):
                run_pilot(rows, gold_ids=gold_fixture(), skip_records=6, max_records=9,
                          output_path=output, project_root=root,
                          subscription_oauth_confirmed=True, infer=fake_infer)

    def test_full_cohort_resume_accepts_validated_records_beyond_current_14_record_window(self):
        rows = [{"StudyInstanceUID": f"synthetic-{index}", "Report": f"report-{index}",
                 **{column: "" for column in LABEL_COLUMNS}} for index in range(16)]
        prior = {column: {"label": False, "evidence": "synthetic", "uncertainty": 0.1}
                 for column in LABEL_COLUMNS}
        calls = []

        def fake_infer(report):
            calls.append(report)
            return prior

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "data" / "outputs" / "cohort.jsonl"
            output.parent.mkdir(parents=True)
            output.write_text("".join(
                json.dumps({"StudyInstanceUID": f"synthetic-{index}", "labels": prior}) + "\n"
                for index in range(16)
            ), encoding="utf-8")
            result = run_pilot(
                rows, gold_ids=gold_fixture(), skip_records=14, max_records=2,
                output_path=output, project_root=root, subscription_oauth_confirmed=True,
                allow_full_cohort_checkpoint=True, infer=fake_infer,
            )
            self.assertEqual(result, 16)
            self.assertEqual(calls, [])

    def test_inference_prompt_names_all_target_classes(self):
        from rsna_knee.labels.subscription import codex_cli_infer

        response = {column: {"label": False, "evidence": "", "uncertainty": 0.5}
                    for column in LABEL_COLUMNS}
        stdout = json.dumps({"type": "item.completed", "item": {
            "type": "agent_message", "text": json.dumps(response)}})
        process = Mock(returncode=0)
        process.communicate.return_value = (stdout, "")
        with patch("rsna_knee.labels.subscription.subprocess.Popen", return_value=process):
            codex_cli_infer("synthetic report", subscription_oauth_confirmed=True)
        prompt = process.communicate.call_args.kwargs["input"]
        expected_targets = "TARGETS:\n" + "\n".join(f"- {name}" for name in LABEL_COLUMNS)
        self.assertIn(expected_targets, prompt)

    def test_candidate_inference_requires_transfer_prerequisite_confirmation(self):
        from rsna_knee.labels.subscription import codex_cli_infer

        with patch("rsna_knee.labels.subscription.subprocess.Popen") as popen:
            with self.assertRaisesRegex(LabelingError, "transfer"):
                codex_cli_infer(
                    "synthetic report", candidate_id="candidate_2",
                    subscription_oauth_confirmed=True,
                )
        popen.assert_not_called()

    def test_candidate_inference_uses_fixed_candidate_prompt(self):
        from rsna_knee.labels.prompt_comparison import build_candidate_prompt
        from rsna_knee.labels.subscription import codex_cli_infer

        response = {column: {"label": False, "evidence": "", "uncertainty": 0.5}
                    for column in LABEL_COLUMNS}
        stdout = json.dumps({"type": "item.completed", "item": {
            "type": "agent_message", "text": json.dumps(response)}})
        process = Mock(returncode=0)
        process.communicate.return_value = (stdout, "")
        with patch("rsna_knee.labels.subscription.subprocess.Popen", return_value=process):
            codex_cli_infer(
                "synthetic report", candidate_id="candidate_2",
                transfer_prerequisites_confirmed=True,
                subscription_oauth_confirmed=True,
            )
        self.assertEqual(
            process.communicate.call_args.kwargs["input"],
            build_candidate_prompt("candidate_2", "synthetic report"),
        )


if __name__ == "__main__":
    unittest.main()
