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


def gold_fixture(*extra_ids):
    values = set(extra_ids)
    index = 0
    while len(values) < 58:
        values.add(f"synthetic-gold-{index}")
        index += 1
    return values


class SubscriptionLabelContractTests(unittest.TestCase):
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
        from rsna_knee.labels.subscription import codex_cli_infer

        response = {column: {"label": False, "evidence": "", "uncertainty": 0.5}
                    for column in LABEL_COLUMNS}
        stdout = json.dumps({"type": "item.completed", "item": {
            "type": "agent_message", "text": json.dumps(response)}})
        with patch("rsna_knee.labels.subscription.os.name", "nt"):
            process = Mock(returncode=0)
            process.communicate.return_value = (stdout, "")
            with patch("rsna_knee.labels.subscription.subprocess.Popen", return_value=process) as popen:
                codex_cli_infer("synthetic report", subscription_oauth_confirmed=True)
        args, kwargs = popen.call_args
        self.assertEqual(args[0], ["cmd.exe", "/d", "/s", "/c",
                                   "codex.cmd exec --json --sandbox read-only -"])
        self.assertNotIn("synthetic report", " ".join(args[0]))
        self.assertIn("synthetic report", process.communicate.call_args.kwargs["input"])
        self.assertFalse(kwargs.get("shell", False))

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
                run_pilot(rows, gold_ids=gold_fixture(), skip_records=1, max_records=10,
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


if __name__ == "__main__":
    unittest.main()
