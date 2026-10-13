import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

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
