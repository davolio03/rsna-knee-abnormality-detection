import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rsna_knee.labels.siwc_auth import AuthError
from rsna_knee.labels.subscription import LABEL_COLUMNS

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import siwc_label_reports as cli


class SIWCLabelReportsCliTests(unittest.TestCase):
    def test_stream_diagnostics_formatter_revalidates_mutable_metadata(self):
        safe = SimpleNamespace(
            stream_completed=False,
            stream_delta_chars=42,
            stream_final_chars=0,
            stream_text_source="none",
            stream_refusal=False,
            stream_format="json_invalid",
        )
        self.assertEqual(
            cli.safe_provider_diagnostics(safe),
            "stream_completed=False stream_delta_chars=42 stream_final_chars=0 "
            "stream_text_source=none stream_refusal=False stream_format=json_invalid",
        )
        malicious = SimpleNamespace(
            stream_completed="PRIVATE_CANARY", stream_delta_chars=True,
            stream_final_chars=-1, stream_text_source="PRIVATE_CANARY",
            stream_refusal="PRIVATE_CANARY", stream_format="PRIVATE_CANARY",
        )
        self.assertEqual(cli.safe_provider_diagnostics(malicious), "")

    def test_cli_requires_an_explicit_subcommand(self):
        with patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                cli.main([])

    def test_signin_keeps_callback_in_process_and_never_prints_code_or_url(self):
        code = "PRIVATE_AUTHORIZATION_CODE"
        state = "PRIVATE_STATE"
        callback = f"http://127.0.0.1:1455/auth/callback?code={code}&state={state}&client_id=issued-client"
        auth = Mock()
        auth.begin_sign_in.return_value = SimpleNamespace(
            authorization_url=f"https://auth.openai.com/authorize?state={state}",
        )
        auth.complete_sign_in.return_value = SimpleNamespace(
            profile_id="issued-client", plan_usage_enabled=True, email="private@example.test",
        )
        server = Mock()

        def handle_request():
            server.callback_url = callback

        server.handle_request.side_effect = handle_request
        output = io.StringIO()
        with patch.object(cli, "_create_authenticator", return_value=auth) as create_auth, \
                patch.object(cli, "HTTPServer", return_value=server) as server_factory, \
                patch.object(cli.webbrowser, "open", return_value=True) as open_browser:
            result = cli.sign_in(callback_port=1455, timeout_seconds=1, output=output)

        self.assertEqual(result, 0)
        create_auth.assert_called_once_with("http://127.0.0.1:1455/auth/callback")
        server_factory.assert_called_once()
        self.assertEqual(server_factory.call_args.args[0], ("127.0.0.1", 1455))
        self.assertTrue(server.handle_request.called)
        auth.complete_sign_in.assert_called_once_with(callback)
        open_browser.assert_called_once()
        self.assertNotIn(code, output.getvalue())
        self.assertNotIn(state, output.getvalue())
        self.assertNotIn("private@example.test", output.getvalue())
        self.assertIn("issued-client", output.getvalue())
        self.assertTrue(server.server_close.called)
        self.assertIsNone(server.callback_url)

    def test_signin_displays_only_sanitized_exchange_diagnostics(self):
        callback = "http://127.0.0.1:1455/auth/callback?code=PRIVATE_AUTH_CODE&state=PRIVATE_STATE&client_id=issued-client"
        auth = Mock()
        auth.begin_sign_in.return_value = SimpleNamespace(
            authorization_url="https://auth.openai.com/authorize?state=PRIVATE_STATE",
        )
        auth.complete_sign_in.side_effect = AuthError(
            "Authorization could not be completed.",
            _diagnostics={
                "http_status": 400,
                "provider_error_code": "invalid_grant",
                "request_id": "req_safe123",
                "response_body_shape": "json_object",
                "error_description": "PRIVATE_AUTH_CODE access-token private body",
                "response_body": "refresh-token private body",
            },
        )
        server = Mock()

        def handle_request():
            server.callback_url = callback

        server.handle_request.side_effect = handle_request
        error_output = io.StringIO()
        with patch.object(cli, "_create_authenticator", return_value=auth), \
                patch.object(cli, "HTTPServer", return_value=server), \
                patch.object(cli.webbrowser, "open", return_value=True), \
                patch("sys.stderr", error_output):
            with self.assertRaises(SystemExit):
                cli.main(["signin", "--callback-port", "1455", "--timeout-seconds", "1"])

        rendered = error_output.getvalue()
        for diagnostic in ("HTTP status: 400", "provider error code: invalid_grant",
                           "request ID: req_safe123", "response body shape: json_object"):
            self.assertIn(diagnostic, rendered)
        for sensitive in ("PRIVATE_AUTH_CODE", "PRIVATE_STATE", "access-token",
                          "refresh-token", "private body", "error_description",
                          "response_body"):
            self.assertNotIn(sensitive, rendered)

    def test_callback_handler_suppresses_request_line_logging(self):
        handler_type = cli._callback_handler_type(1455)
        handler = object.__new__(handler_type)
        error_output = io.StringIO()
        with patch("sys.stderr", error_output):
            handler.log_message("request contains code=%s", "PRIVATE_AUTHORIZATION_CODE")
        self.assertEqual(error_output.getvalue(), "")

    def test_models_command_displays_only_returned_visible_account_models(self):
        auth = object()
        adapter = Mock()
        adapter.list_available_models.return_value = [
            {"slug": "gpt-6-luna", "display_name": "Luna"},
            {"slug": "gpt-6-sol", "display_name": "Sol"},
        ]
        output = io.StringIO()
        with patch.object(cli, "_create_authenticator", return_value=auth), \
                patch.object(cli, "SIWCResponsesInfer", return_value=adapter) as adapter_factory:
            result = cli.main(["models", "--profile-id", "issued-client"], output=output)

        self.assertEqual(result, 0)
        adapter_factory.assert_called_once_with(
            authenticator=auth, profile_id="issued-client", model="gpt-6-luna",
        )
        self.assertIn("Luna (gpt-6-luna)", output.getvalue())
        self.assertIn("Sol (gpt-6-sol)", output.getvalue())
        self.assertNotIn("access_token", output.getvalue())

    def test_models_command_states_when_default_luna_is_not_account_listed(self):
        adapter = Mock()
        adapter.list_available_models.return_value = [
            {"slug": "gpt-6-sol", "display_name": "Sol"},
        ]
        output = io.StringIO()
        with patch.object(cli, "_create_authenticator", return_value=object()), \
                patch.object(cli, "SIWCResponsesInfer", return_value=adapter):
            cli.main(["models", "--profile-id", "issued-client"], output=output)
        self.assertIn("Default model gpt-6-luna is not listed", output.getvalue())
        self.assertNotIn("Luna (gpt-6-luna)", output.getvalue())

    def test_models_command_displays_allowlisted_catalog_diagnostics(self):
        error = cli.LabelingError(
            "The account model catalog could not be loaded.",
            stage="inference",
            cause_code="quota_error",
            http_status=429,
            provider_error_code="subscription_sharing_usage_limit_exceeded",
            provider_error_param="model",
            provider_body_shape="nested_error",
            request_id="req_catalog-123",
        )
        error.raw_error_text = "PRIVATE_EXCEPTION_CANARY"
        error.raw_response_body = "PRIVATE_BODY_CANARY"
        error.raw_headers = "PRIVATE_HEADER_CANARY"
        error.access_token = "PRIVATE_TOKEN_CANARY"
        adapter = Mock()
        adapter.list_available_models.side_effect = error
        error_output = io.StringIO()

        with patch.object(cli, "_create_authenticator", return_value=object()), \
                patch.object(cli, "SIWCResponsesInfer", return_value=adapter), \
                patch("sys.stderr", error_output):
            with self.assertRaises(SystemExit):
                cli.main(["models", "--profile-id", "issued-client"])

        rendered = error_output.getvalue()
        for safe_value in (
            "http_status=429",
            "provider_error_code=subscription_sharing_usage_limit_exceeded",
            "provider_error_param=model",
            "provider_body_shape=nested_error",
            "request_id=req_catalog-123",
        ):
            self.assertIn(safe_value, rendered)
        for canary in (
            "PRIVATE_EXCEPTION_CANARY", "PRIVATE_BODY_CANARY",
            "PRIVATE_HEADER_CANARY", "PRIVATE_TOKEN_CANARY",
        ):
            self.assertNotIn(canary, rendered)

    def test_models_command_revalidates_mutated_provider_diagnostics(self):
        error = cli.LabelingError(
            "The account model catalog could not be loaded.",
            http_status=429,
            provider_error_code="subscription_sharing_usage_limit_exceeded",
            provider_error_param="model",
            provider_body_shape="nested_error",
            request_id="req_catalog-123",
        )
        # Exception attributes are mutable; the final CLI boundary must still validate them.
        error.provider_error_param = "PRIVATE_PARAM_CANARY"
        error.provider_body_shape = "PRIVATE_SHAPE_CANARY"
        error.request_id = "PRIVATE_REQUEST_CANARY"
        adapter = Mock()
        adapter.list_available_models.side_effect = error
        error_output = io.StringIO()

        with patch.object(cli, "_create_authenticator", return_value=object()), \
                patch.object(cli, "SIWCResponsesInfer", return_value=adapter), \
                patch("sys.stderr", error_output):
            with self.assertRaises(SystemExit):
                cli.main(["models", "--profile-id", "issued-client"])

        rendered = error_output.getvalue()
        self.assertIn("http_status=429", rendered)
        self.assertIn("provider_error_code=subscription_sharing_usage_limit_exceeded", rendered)
        for canary in ("PRIVATE_PARAM_CANARY", "PRIVATE_SHAPE_CANARY", "PRIVATE_REQUEST_CANARY"):
            self.assertNotIn(canary, rendered)

    def test_label_runner_renders_sanitized_provider_diagnostics_once(self):
        rows = [{
            "StudyInstanceUID": "synthetic-study",
            "Report": "SYNTHETIC_REPORT_CANARY",
            **{target: "" for target in LABEL_COLUMNS},
        }]
        gold_ids = {f"synthetic-gold-{index}" for index in range(58)}

        def fail_with_safe_provider_error(_report):
            error = cli.LabelingError("The Responses provider request failed.", cause_code="quota_error")
            error.http_status = 429
            error.provider_error_code = "subscription_sharing_usage_limit_exceeded"
            error.provider_error_param = "model"
            error.provider_body_shape = "nested_error"
            error.request_id = "req_synthetic-123"
            error.stream_completed = False
            error.stream_delta_chars = 321
            error.stream_final_chars = 0
            error.stream_text_source = "none"
            error.stream_refusal = False
            error.stream_format = "empty"
            error.raw_detail = "PRIVATE_DETAIL_CANARY"
            error.raw_headers = "PRIVATE_HEADER_CANARY"
            error.access_token = "PRIVATE_TOKEN_CANARY"
            raise error

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_path = root / "data" / "outputs" / "labels.jsonl"
            diagnostics_path = root / "data" / "outputs" / "diagnostics.jsonl"
            args = [
                "label", "--train-csv", "synthetic-train.csv", "--gold-exclusions", "synthetic-gold.csv",
                "--profile-id", "synthetic-profile", "--model", "gpt-6-luna", "--max-records", "1",
                "--output", str(output_path), "--diagnostics", str(diagnostics_path),
                "--confirm-siwc-oauth", "--confirm-report-transfer",
            ]
            stderr = io.StringIO()
            with patch.object(cli, "REPOSITORY_ROOT", root), \
                    patch.object(cli, "_read_rows", return_value=rows), \
                    patch.object(cli, "_read_gold_ids", return_value=gold_ids), \
                    patch.object(cli, "_create_authenticator", return_value=object()), \
                    patch.object(cli, "SIWCResponsesInfer", return_value=fail_with_safe_provider_error), \
                    patch("sys.stderr", stderr), self.assertRaises(SystemExit):
                cli.main(args)

            rendered = stderr.getvalue()
            journal = [json.loads(line) for line in diagnostics_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(rendered.count("Provider diagnostics:"), 1)
        self.assertIn("http_status=429", rendered)
        self.assertIn("provider_error_code=subscription_sharing_usage_limit_exceeded", rendered)
        self.assertIn("stream_completed=False", rendered)
        self.assertIn("stream_delta_chars=321", rendered)
        self.assertIn("stream_final_chars=0", rendered)
        self.assertIn("stream_text_source=none", rendered)
        self.assertIn("stream_refusal=False", rendered)
        self.assertIn("stream_format=empty", rendered)
        self.assertNotIn("Codex CLI", rendered)
        self.assertNotIn("PRIVATE_DETAIL_CANARY", rendered)
        self.assertNotIn("PRIVATE_HEADER_CANARY", rendered)
        self.assertNotIn("PRIVATE_TOKEN_CANARY", rendered)
        failed = next(record for record in journal if record["event"] == "attempt_failed")
        self.assertEqual(failed["cause_explanation"], "The labeling provider reported a quota or rate-limit problem.")
        self.assertEqual(failed["http_status"], 429)
        self.assertIs(failed["stream_completed"], False)
        self.assertEqual(failed["stream_delta_chars"], 321)
        self.assertEqual(failed["stream_final_chars"], 0)
        self.assertEqual(failed["stream_text_source"], "none")
        self.assertIs(failed["stream_refusal"], False)
        self.assertEqual(failed["stream_format"], "empty")

    def test_label_requires_both_explicit_confirmation_flags(self):
        args = [
            "label", "--train-csv", "synthetic.csv", "--gold-exclusions", "synthetic.csv",
            "--profile-id", "issued-client", "--model", "gpt-6-luna", "--max-records", "1",
        ]
        for one_flag in ("--confirm-siwc-oauth", "--confirm-report-transfer"):
            with self.subTest(missing=one_flag):
                present_flag = ("--confirm-report-transfer" if one_flag == "--confirm-siwc-oauth"
                                else "--confirm-siwc-oauth")
                with patch.object(cli, "run_pilot") as run_pilot, \
                        patch("sys.stderr", new_callable=io.StringIO):
                    with self.assertRaises(SystemExit):
                        cli.main(args + [present_flag])
                    run_pilot.assert_not_called()

    def test_label_reuses_runner_and_helpers_with_selected_profile_and_model(self):
        rows = ({"Report": "synthetic only", "StudyInstanceUID": "synthetic-id"},)
        gold = {f"gold-{index}" for index in range(58)}
        auth = object()
        adapter = object()
        output = io.StringIO()
        args = [
            "label", "--train-csv", "synthetic-train.csv", "--gold-exclusions", "synthetic-gold.csv",
            "--profile-id", "issued-client", "--model", "gpt-6-luna", "--max-records", "2",
            "--skip-records", "1", "--confirm-siwc-oauth", "--confirm-report-transfer",
        ]
        with patch.object(cli, "_read_rows", return_value=rows) as read_rows, \
                patch.object(cli, "_read_gold_ids", return_value=gold) as read_gold, \
                patch.object(cli, "_create_authenticator", return_value=auth), \
                patch.object(cli, "SIWCResponsesInfer", return_value=adapter) as adapter_factory, \
                patch.object(cli, "run_pilot", return_value=1) as run_pilot:
            result = cli.main(args, output=output)

        self.assertEqual(result, 0)
        read_rows.assert_called_once_with(Path("synthetic-train.csv"))
        read_gold.assert_called_once_with(Path("synthetic-gold.csv"))
        adapter_factory.assert_called_once_with(
            authenticator=auth, profile_id="issued-client", model="gpt-6-luna",
        )
        kwargs = run_pilot.call_args.kwargs
        self.assertIs(run_pilot.call_args.args[0], rows)
        self.assertEqual(kwargs["gold_ids"], gold)
        self.assertEqual(kwargs["max_records"], 2)
        self.assertEqual(kwargs["skip_records"], 1)
        self.assertIs(kwargs["infer"], adapter)
        self.assertTrue(kwargs["subscription_oauth_confirmed"])
        self.assertEqual(kwargs["project_root"], cli.REPOSITORY_ROOT)
        self.assertTrue(str(kwargs["output_path"]).startswith(str(cli.REPOSITORY_ROOT / "data" / "outputs")))
        self.assertTrue(str(kwargs["diagnostics_path"]).startswith(str(cli.REPOSITORY_ROOT / "data" / "outputs")))
        self.assertIn("Local SIWC pilot output records: 1", output.getvalue())

    def test_label_rejects_batch_bounds_before_reading_inputs(self):
        with patch.object(cli, "_read_rows") as read_rows, \
                patch.object(cli, "_read_gold_ids") as read_gold:
            with self.assertRaisesRegex(cli.LabelingError, "between 1 and 14"):
                cli.run_label_pilot(
                    train_csv=Path("unused.csv"), gold_exclusions=Path("unused-gold.csv"),
                    profile_id="issued-client", model="gpt-6-luna", max_records=15,
                    confirm_siwc_oauth=True, confirm_report_transfer=True,
                )
        read_rows.assert_not_called()
        read_gold.assert_not_called()

    def test_label_function_fails_closed_without_both_confirmations(self):
        with patch.object(cli, "_read_rows") as read_rows, \
                patch.object(cli, "_read_gold_ids") as read_gold:
            with self.assertRaisesRegex(cli.LabelingError, "Both explicit SIWC OAuth"):
                cli.run_label_pilot(
                    train_csv=Path("unused.csv"), gold_exclusions=Path("unused-gold.csv"),
                    profile_id="issued-client", model="gpt-6-luna", max_records=1,
                )
        read_rows.assert_not_called()
        read_gold.assert_not_called()

    def test_label_rejects_token_cli_argument(self):
        error_output = io.StringIO()
        with patch("sys.stderr", error_output):
            with self.assertRaises(SystemExit):
                cli.main([
                    "label", "--train-csv", "synthetic.csv", "--gold-exclusions", "synthetic.csv",
                    "--profile-id", "issued-client", "--model", "gpt-6-luna", "--max-records", "1",
                    "--confirm-siwc-oauth", "--confirm-report-transfer",
                    "--access-token", "PRIVATE_TOKEN",
                ])
        self.assertNotIn("PRIVATE_TOKEN", error_output.getvalue())


if __name__ == "__main__":
    unittest.main()
