import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rsna_knee.labels.siwc_responses import SIWCResponsesInfer
from rsna_knee.labels.siwc_auth import SIWCAuthenticator
from rsna_knee.labels.subscription import LABEL_COLUMNS, LabelingError, run_pilot


def valid_result():
    return {
        target: {"label": False, "evidence": "synthetic evidence", "uncertainty": 0.25}
        for target in LABEL_COLUMNS
    }


class FakeAuthorizedProfile:
    def __init__(self, token="synthetic-access-token", *, authorized=True):
        self.token = token
        self.authorized = authorized

    def with_plan_access_token(self, profile_id, operation):
        if not self.authorized:
            raise RuntimeError("ChatGPT plan usage permission was not granted.")
        return operation(self.token)


class MemoryCredentialStore:
    def __init__(self, record=None):
        self.record = record

    def load(self, _profile_id):
        return self.record

    def save(self, _profile_id, record):
        self.record = dict(record)


class FakeStream:
    def __init__(self, events):
        self.events = events

    def __enter__(self):
        return iter(self.events)

    def __exit__(self, *_args):
        return False


class SIWCResponsesTests(unittest.TestCase):
    def setUp(self):
        self.report_canary = "PRIVATE_REPORT_CANARY"
        self.result_text = json.dumps(valid_result())
        self.completed = SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(output_text=self.result_text),
        )
        self.client = Mock()
        self.client.get.return_value = {"models": [
            {"slug": "gpt-6-luna", "display_name": "Luna", "visibility": "list"},
            {"slug": "hidden-model", "display_name": "Hidden", "visibility": "hidden"},
        ]}
        self.client.responses.create.return_value = FakeStream([self.completed])
        self.client_factory = Mock(return_value=self.client)

    def make_infer(self, auth=None, model="gpt-6-luna"):
        return SIWCResponsesInfer(
            authenticator=auth or FakeAuthorizedProfile(),
            profile_id="synthetic-profile",
            model=model,
            client_factory=self.client_factory,
        )

    def test_request_uses_selected_available_model_and_nonstored_streaming(self):
        infer = self.make_infer()

        self.assertEqual(infer(self.report_canary), valid_result())
        self.client_factory.assert_called_once_with(
            api_key="synthetic-access-token",
            base_url="https://api.openai.com/v1",
            max_retries=0,
        )
        self.client.get.assert_called_once_with("/models", cast_to=dict)
        self.client.responses.create.assert_called_once()
        kwargs = self.client.responses.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "gpt-6-luna")
        self.assertFalse(kwargs["store"])
        self.assertTrue(kwargs["stream"])
        self.assertIn(self.report_canary, kwargs["input"])
        self.assertNotIn("reasoning_effort", kwargs)

    def test_requested_model_missing_from_account_fails_before_inference(self):
        self.client.get.return_value = {"models": [
            {"slug": "other-model", "visibility": "list"},
        ]}
        with self.assertRaisesRegex(LabelingError, "not available"):
            self.make_infer()(self.report_canary)
        self.client.responses.create.assert_not_called()

    def test_only_response_completed_returns_validated_labels(self):
        self.client.responses.create.return_value = FakeStream([
            SimpleNamespace(type="response.output_text.delta", delta=self.result_text),
            self.completed,
        ])
        self.assertEqual(self.make_infer()(self.report_canary), valid_result())

    def test_failed_incomplete_and_uncompleted_streams_fail_without_payload_leak(self):
        failure_cases = (
            [SimpleNamespace(type="response.failed", response=SimpleNamespace(
                error=SimpleNamespace(code="private-provider-detail")))],
            [SimpleNamespace(type="response.incomplete", response=SimpleNamespace(
                output_text=self.report_canary))],
            [SimpleNamespace(type="response.output_text.delta", delta=self.report_canary)],
        )
        for events in failure_cases:
            with self.subTest(events=events):
                self.client.responses.create.return_value = FakeStream(events)
                with self.assertRaises(LabelingError) as raised:
                    self.make_infer()(self.report_canary)
                self.assertNotIn(self.report_canary, str(raised.exception))
                self.assertNotIn("private-provider-detail", str(raised.exception))

    def test_interrupted_stream_is_redacted(self):
        self.client.responses.create.return_value = FakeStream(
            [SimpleNamespace(type="response.output_text.delta", delta="partial")]
        )
        # Iteration failures may contain SDK or transport details; they must not escape.
        class InterruptedStream(FakeStream):
            def __enter__(self):
                def events():
                    yield SimpleNamespace(type="response.output_text.delta", delta="partial")
                    raise RuntimeError("transport leak " + self.report_canary)
                return events()

        self.client.responses.create.return_value = InterruptedStream([])
        with self.assertRaises(LabelingError) as raised:
            self.make_infer()(self.report_canary)
        self.assertNotIn(self.report_canary, str(raised.exception))
        self.assertNotIn("transport leak", str(raised.exception))

    def test_http_error_preserves_only_safe_status_code_param_and_request_id(self):
        provider_exception = RuntimeError("PRIVATE_REPORT_CANARY and raw body")
        provider_exception.status_code = 429
        provider_exception.request_id = "req_synthetic-123"
        provider_exception.body = {"error": {
            "code": "subscription_sharing_usage_limit_exceeded",
            "param": "model",
            "message": self.report_canary,
        }}
        self.client.responses.create.side_effect = provider_exception

        with self.assertRaises(LabelingError) as raised:
            self.make_infer()(self.report_canary)

        self.assertEqual(raised.exception.http_status, 429)
        self.assertEqual(raised.exception.provider_error_code,
                         "subscription_sharing_usage_limit_exceeded")
        self.assertEqual(raised.exception.provider_error_param, "model")
        self.assertEqual(raised.exception.provider_body_shape, "nested_error")
        self.assertEqual(raised.exception.request_id, "req_synthetic-123")
        self.assertNotIn(self.report_canary, str(raised.exception))
        self.assertNotIn("raw body", str(raised.exception))

    def test_untrusted_error_code_param_and_request_id_are_discarded(self):
        provider_exception = RuntimeError("private error")
        provider_exception.status_code = 400
        provider_exception.request_id = self.report_canary
        provider_exception.body = {"error": {
            "code": self.report_canary,
            "param": self.report_canary,
        }}
        self.client.get.side_effect = provider_exception

        with self.assertRaises(LabelingError) as raised:
            self.make_infer()(self.report_canary)

        self.assertEqual(raised.exception.http_status, 400)
        self.assertIsNone(raised.exception.provider_error_code)
        self.assertIsNone(raised.exception.provider_error_param)
        self.assertIsNone(raised.exception.request_id)
        self.assertNotIn(self.report_canary, str(raised.exception))

    def test_admission_detail_shape_and_request_id_are_preserved_without_text(self):
        provider_exception = RuntimeError("private admission details")
        provider_exception.status_code = 403
        provider_exception.response = SimpleNamespace(
            headers={"x-request-id": "req_admission-901"}
        )
        provider_exception.body = {"detail": self.report_canary}
        self.client.get.side_effect = provider_exception

        with self.assertRaises(LabelingError) as raised:
            self.make_infer()(self.report_canary)

        self.assertEqual(raised.exception.http_status, 403)
        self.assertEqual(raised.exception.provider_body_shape, "detail_object")
        self.assertEqual(raised.exception.request_id, "req_admission-901")
        self.assertIsNone(raised.exception.provider_error_code)
        self.assertNotIn(self.report_canary, str(raised.exception))
        self.assertNotIn("private admission details", str(raised.exception))

    def test_failure_event_preserves_allowlisted_recovery_code(self):
        self.client.responses.create.return_value = FakeStream([
            SimpleNamespace(type="response.failed", response=SimpleNamespace(
                error=SimpleNamespace(
                    code="subscription_sharing_unsupported_capability",
                    param="service_tier",
                ),
                _request_id="req_stream-456",
            )),
        ])
        with self.assertRaises(LabelingError) as raised:
            self.make_infer()(self.report_canary)
        self.assertEqual(raised.exception.provider_error_code,
                         "subscription_sharing_unsupported_capability")
        self.assertEqual(raised.exception.provider_error_param, "service_tier")
        self.assertEqual(raised.exception.request_id, "req_stream-456")
        self.assertNotIn(self.report_canary, str(raised.exception))

    def test_usage_unavailable_keeps_retry_later_recovery_classification(self):
        self.client.responses.create.return_value = FakeStream([
            SimpleNamespace(type="response.failed", response=SimpleNamespace(
                error=SimpleNamespace(code="subscription_sharing_usage_unavailable"),
            )),
        ])
        with self.assertRaises(LabelingError) as raised:
            self.make_infer()(self.report_canary)
        self.assertEqual(raised.exception.cause_code, "network_error")
        self.assertEqual(raised.exception.provider_error_code,
                         "subscription_sharing_usage_unavailable")

    def test_default_openai_client_construction_and_pinned_dependency(self):
        sdk_constructor = Mock(return_value=self.client)
        with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=sdk_constructor)}):
            created = SIWCResponsesInfer._default_client_factory(
                api_key="synthetic-access-token",
                base_url="https://api.openai.com/v1",
                max_retries=0,
            )
        sdk_constructor.assert_called_once_with(
            api_key="synthetic-access-token",
            base_url="https://api.openai.com/v1",
            max_retries=0,
        )
        self.assertIs(created, self.client)
        requirements = (Path(__file__).resolve().parents[1] / "requirements-siwc.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("openai==2.11.0", requirements)

    def test_run_pilot_persists_only_sanitized_provider_recovery_metadata(self):
        rows = [{
            "StudyInstanceUID": "synthetic-study",
            "Report": self.report_canary,
            **{target: "" for target in LABEL_COLUMNS},
        }]

        def failed_infer(_report):
            raise LabelingError(
                "safe provider failure",
                cause_code="quota_error",
                http_status=429,
                provider_error_code="subscription_sharing_usage_limit_exceeded",
                provider_error_param="model",
                provider_body_shape="nested_error",
                request_id="req_synthetic-789",
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            journal = root / "data" / "outputs" / "diagnostics.jsonl"
            with self.assertRaises(LabelingError) as raised:
                run_pilot(
                    rows,
                    gold_ids={f"synthetic-gold-{index}" for index in range(58)},
                    max_records=1,
                    output_path=root / "data" / "outputs" / "labels.jsonl",
                    diagnostics_path=journal,
                    project_root=root,
                    subscription_oauth_confirmed=True,
                    infer=failed_infer,
                )
            diagnostic_text = journal.read_text(encoding="utf-8")
        self.assertIn("http_status=429", str(raised.exception))
        self.assertIn("subscription_sharing_usage_limit_exceeded", str(raised.exception))
        self.assertIn('"request_id":"req_synthetic-789"', diagnostic_text)
        self.assertIn('"provider_error_param":"model"', diagnostic_text)
        self.assertIn('"provider_body_shape":"nested_error"', diagnostic_text)
        self.assertNotIn(self.report_canary, str(raised.exception))
        self.assertNotIn(self.report_canary, diagnostic_text)

    def test_no_authorization_or_plan_scope_fails_before_any_client_call(self):
        auth_cases = (
            SIWCAuthenticator(
                store=MemoryCredentialStore(),
                host_id="urn:uuid:00000000-0000-0000-0000-000000000001",
                redirect_uri="http://127.0.0.1:1455/auth/callback",
            ),
            SIWCAuthenticator(
                store=MemoryCredentialStore({
                    "client_id": "synthetic-client",
                    "subject": "synthetic-subject",
                    "scopes": [],
                    "expires_at": 4_000_000_000,
                    "access_token": "synthetic-access-token",
                }),
                host_id="urn:uuid:00000000-0000-0000-0000-000000000001",
                redirect_uri="http://127.0.0.1:1455/auth/callback",
            ),
        )
        for index, auth in enumerate(auth_cases):
            with self.subTest(index=index):
                with self.assertRaises(LabelingError) as raised:
                    self.make_infer(auth=auth)(self.report_canary)
                self.assertEqual(raised.exception.cause_code, "auth_error")
                self.assertNotIn(self.report_canary, str(raised.exception))
                self.client_factory.assert_not_called()

    def test_adapter_is_injected_at_run_pilot_infer_boundary(self):
        infer = self.make_infer()
        rows = [{
            "StudyInstanceUID": "synthetic-study",
            "Report": self.report_canary,
            **{target: "" for target in LABEL_COLUMNS},
        }]
        gold_ids = {f"synthetic-gold-{index}" for index in range(58)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "data" / "outputs" / "labels.jsonl"
            count = run_pilot(
                rows,
                gold_ids=gold_ids,
                max_records=1,
                output_path=output,
                project_root=root,
                subscription_oauth_confirmed=True,
                infer=infer,
            )
        self.assertEqual(count, 1)
        self.assertEqual(self.client.responses.create.call_count, 1)


if __name__ == "__main__":
    unittest.main()
