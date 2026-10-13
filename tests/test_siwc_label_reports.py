import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rsna_knee.labels.siwc_auth import AuthError

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import siwc_label_reports as cli


class SIWCLabelReportsCliTests(unittest.TestCase):
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
