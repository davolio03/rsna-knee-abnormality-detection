import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import httpx
from openai import OpenAI

from rsna_knee.labels.siwc_responses import (
    SIWCResponsesInfer,
    _recover_fresh_boundary_aware_evidence,
)
from rsna_knee.labels.siwc_auth import AuthError, SIWCAuthenticator
from rsna_knee.labels.subscription import LABEL_COLUMNS, LabelingError, run_pilot
from rsna_knee.labels.prompt_comparison import build_candidate_prompt


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
        self.report_canary = "PRIVATE_REPORT_CANARY synthetic evidence"
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

    def make_sdk_infer(self, handler, *, model="gpt-6-luna"):
        def client_factory(**kwargs):
            return OpenAI(
                **kwargs,
                http_client=httpx.Client(transport=httpx.MockTransport(handler)),
            )

        return SIWCResponsesInfer(
            authenticator=FakeAuthorizedProfile(),
            profile_id="synthetic-profile",
            model=model,
            client_factory=client_factory,
        )

    def test_real_sdk_model_listing_parses_synthetic_http_response(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={"models": [
                {"slug": "gpt-6-luna", "display_name": "Luna", "visibility": "list"},
                {"slug": "hidden-model", "display_name": "Hidden", "visibility": "hidden"},
            ]})

        self.assertEqual(
            self.make_sdk_infer(handler).list_available_models(),
            [{"slug": "gpt-6-luna", "display_name": "Luna"}],
        )
        self.assertEqual([request.url.path for request in requests], ["/v1/models"])

    def test_model_catalog_surfaces_only_validated_auth_diagnostics(self):
        class FailedAuth:
            def with_plan_access_token(self, _profile_id, _operation):
                raise AuthError(
                    "The saved authorization could not be refreshed.",
                    _diagnostics={
                        "failure_category": "identity_claims_invalid",
                        "provider_error_code": "PRIVATE-CODE-CANARY",
                        "refresh_token": "PRIVATE-TOKEN-CANARY",
                    },
                )

        with self.assertRaises(LabelingError) as raised:
            self.make_infer(auth=FailedAuth()).list_available_models()

        self.assertEqual(raised.exception.cause_code, "auth_error")
        self.assertIn("failure category: identity_claims_invalid", str(raised.exception))
        self.assertNotIn("PRIVATE-CODE-CANARY", str(raised.exception))
        self.assertNotIn("PRIVATE-TOKEN-CANARY", str(raised.exception))
        self.client_factory.assert_not_called()

    def test_real_sdk_inference_preflight_parses_catalog_before_model_rejection(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={"models": [
                {"slug": "another-model", "display_name": "Other", "visibility": "list"},
            ]})

        with self.assertRaisesRegex(LabelingError, "not available") as raised:
            self.make_sdk_infer(handler)(self.report_canary)

        self.assertEqual(raised.exception.cause_code, "model_error")
        self.assertEqual([request.url.path for request in requests], ["/v1/models"])

    def test_real_sdk_serializes_documented_message_array_for_inference(self):
        requests = []
        expected_prompt = build_candidate_prompt("candidate_2", self.report_canary)
        result_text = json.dumps(valid_result())

        def handler(request):
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"models": [
                    {"slug": "gpt-6-luna", "display_name": "Luna", "visibility": "list"},
                ]})
            requests.append(request)
            event = {
                "type": "response.completed",
                "response": {
                    "id": "resp_synthetic",
                    "object": "response",
                    "created_at": 0,
                    "status": "completed",
                    "output": [{
                        "id": "msg_synthetic",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": result_text, "annotations": []}],
                    }],
                    "output_text": result_text,
                },
            }
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=f"event: response.completed\ndata: {json.dumps(event)}\n\n",
            )

        self.assertEqual(self.make_sdk_infer(handler)(self.report_canary), valid_result())
        self.assertEqual(len(requests), 1)
        payload = json.loads(requests[0].content)
        self.assertEqual(payload["input"], [{"role": "user", "content": expected_prompt}])
        self.assertIsInstance(payload["input"], list)
        self.assertFalse(payload["store"])
        self.assertTrue(payload["stream"])
        self.assertNotIn("system", [item.get("role") for item in payload["input"]])

    def _sse_handler(self, events):
        def handler(request):
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"models": [
                    {"slug": "gpt-6-luna", "display_name": "Luna", "visibility": "list"},
                ]})
            body = "".join(
                f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                for event in events
            )
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=body,
            )
        return handler

    def _completed_sse_event(self, *, text="", status="completed", output=None):
        if output is None:
            output = ([{
                "id": "msg_synthetic", "type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }] if text else [])
        return {
            "type": "response.completed" if status == "completed" else f"response.{status}",
            "response": {
                "id": "resp_synthetic", "object": "response", "created_at": 0,
                "status": status, "output": output,
            },
        }

    def test_real_sdk_uses_delta_text_only_as_fallback_after_completion(self):
        events = [
            {"type": "response.output_text.delta", "sequence_number": 0,
             "item_id": "msg_synthetic", "output_index": 0,
             "content_index": 0, "delta": self.result_text},
            self._completed_sse_event(),
        ]
        self.assertEqual(
            self.make_sdk_infer(self._sse_handler(events))(self.report_canary),
            valid_result(),
        )

    def test_real_sdk_completed_text_is_authoritative_without_delta_duplication(self):
        events = [
            {"type": "response.output_text.delta", "sequence_number": 0,
             "item_id": "msg_synthetic", "output_index": 0,
             "content_index": 0, "delta": self.result_text},
            self._completed_sse_event(text=self.result_text),
        ]
        self.assertEqual(
            self.make_sdk_infer(self._sse_handler(events))(self.report_canary),
            valid_result(),
        )

    def test_real_sdk_refusal_with_valid_json_delta_is_rejected_and_redacted(self):
        events = [
            {"type": "response.output_text.delta", "sequence_number": 0,
             "item_id": "msg_synthetic", "output_index": 0,
             "content_index": 0, "delta": self.result_text},
            self._completed_sse_event(output=[{
                "id": "msg_synthetic", "type": "message", "role": "assistant",
                "content": [{"type": "refusal", "refusal": "PRIVATE_REFUSAL_CANARY"}],
            }]),
        ]
        with self.assertRaises(LabelingError) as raised:
            self.make_sdk_infer(self._sse_handler(events))(self.report_canary)
        self.assertEqual(raised.exception.cause_code, "model_error")
        self.assertTrue(raised.exception.stream_refusal)
        self.assertEqual(raised.exception.stream_format, "refusal")
        for canary in (self.report_canary, "PRIVATE_REFUSAL_CANARY", self.result_text[:30]):
            self.assertNotIn(canary, str(raised.exception))

    def test_real_sdk_rejects_valid_delta_without_successful_completion(self):
        events = [{
            "type": "response.output_text.delta", "sequence_number": 0,
            "item_id": "msg_synthetic", "output_index": 0,
            "content_index": 0, "delta": self.result_text,
        }]
        with self.assertRaises(LabelingError) as raised:
            self.make_sdk_infer(self._sse_handler(events))(self.report_canary)
        self.assertFalse(raised.exception.stream_completed)
        self.assertEqual(raised.exception.stream_text_source, "none")
        self.assertEqual(raised.exception.stream_delta_chars, len(self.result_text))
        self.assertEqual(raised.exception.stream_format, "json_valid")

    def test_real_sdk_incomplete_and_failed_streams_reject_valid_deltas(self):
        delta = {
            "type": "response.output_text.delta", "sequence_number": 0,
            "item_id": "msg_synthetic", "output_index": 0,
            "content_index": 0, "delta": self.result_text,
        }
        events = (
            [delta, self._completed_sse_event(status="incomplete")],
            [delta, {"type": "response.failed", "response": {
                "id": "resp_synthetic", "object": "response", "created_at": 0,
                "status": "failed", "output": [],
                "error": {"code": "server_error", "message": "PRIVATE_FAILURE_CANARY"},
            }}],
        )
        for sequence in events:
            with self.subTest(event_types=[event["type"] for event in sequence]):
                with self.assertRaises(LabelingError) as raised:
                    self.make_sdk_infer(self._sse_handler(sequence))(self.report_canary)
                self.assertFalse(raised.exception.stream_completed)
                self.assertNotIn("PRIVATE_FAILURE_CANARY", str(raised.exception))

    def test_empty_and_fenced_stream_text_are_rejected_with_safe_classes(self):
        cases = (
            ("", "empty"),
            (f"```json\n{self.result_text}\n```", "fenced"),
            ("not-json", "json_invalid"),
        )
        for text, expected_class in cases:
            events = [self._completed_sse_event(text=text)]
            with self.subTest(expected_class=expected_class), self.assertRaises(LabelingError) as raised:
                self.make_sdk_infer(self._sse_handler(events))(self.report_canary)
            self.assertEqual(raised.exception.stream_completed, True)
            self.assertEqual(raised.exception.stream_format, expected_class)
            self.assertNotIn(self.result_text[:30], str(raised.exception))

    def test_valid_json_with_invalid_label_schema_is_classified_without_text(self):
        private_json = json.dumps({"PRIVATE_TARGET_CANARY": True})
        events = [self._completed_sse_event(text=private_json)]
        with self.assertRaises(LabelingError) as raised:
            self.make_sdk_infer(self._sse_handler(events))(self.report_canary)
        self.assertEqual(raised.exception.cause_code, "target_set_invalid")
        self.assertEqual(raised.exception.stream_format, "json_valid")
        self.assertNotIn("PRIVATE_TARGET_CANARY", str(raised.exception))

    def test_invalid_label_types_expose_only_allowlisted_json_type(self):
        for invalid, expected in (("PRIVATE_VALUE_CANARY", "string"), (7, "number"),
                                  (None, "null"), (["PRIVATE_VALUE_CANARY"], "array"),
                                  ({"secret": "PRIVATE_VALUE_CANARY"}, "object")):
            with self.subTest(expected=expected):
                payload = valid_result()
                payload[LABEL_COLUMNS[0]]["label"] = invalid
                events = [self._completed_sse_event(text=json.dumps(payload))]
                with self.assertRaises(LabelingError) as raised:
                    self.make_sdk_infer(self._sse_handler(events))(self.report_canary)
                self.assertEqual(raised.exception.cause_code, "label_type_invalid")
                self.assertEqual(raised.exception.invalid_label_json_type, expected)
                self.assertEqual(raised.exception.stream_completed, True)
                self.assertEqual(raised.exception.stream_format, "json_valid")
                self.assertNotIn("PRIVATE_VALUE_CANARY", str(raised.exception))
                self.assertNotIn(LABEL_COLUMNS[0], str(raised.exception))

    def test_real_sdk_refusal_delta_is_rejected_even_when_final_has_valid_json(self):
        events = [
            {"type": "response.refusal.delta", "sequence_number": 0,
             "item_id": "msg_synthetic", "output_index": 0,
             "content_index": 0, "delta": "PRIVATE_REFUSAL_CANARY"},
            self._completed_sse_event(text=self.result_text),
        ]
        with self.assertRaises(LabelingError) as raised:
            self.make_sdk_infer(self._sse_handler(events))(self.report_canary)
        self.assertTrue(raised.exception.stream_refusal)
        self.assertEqual(raised.exception.stream_format, "refusal")
        self.assertNotIn("PRIVATE_REFUSAL_CANARY", str(raised.exception))

    def test_request_uses_selected_available_model_and_nonstored_streaming(self):
        infer = self.make_infer()

        self.assertEqual(infer(self.report_canary), valid_result())
        self.client_factory.assert_called_once_with(
            api_key="synthetic-access-token",
            base_url="https://api.openai.com/v1",
            max_retries=0,
        )
        self.client.get.assert_called_once_with("/models", cast_to=dict[str, Any])
        self.client.responses.create.assert_called_once()
        kwargs = self.client.responses.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "gpt-6-luna")
        self.assertFalse(kwargs["store"])
        self.assertTrue(kwargs["stream"])
        self.assertEqual(kwargs["input"], [{
            "role": "user", "content": build_candidate_prompt("candidate_2", self.report_canary),
        }])
        self.assertNotIn("reasoning_effort", kwargs)

    def test_requested_model_missing_from_account_fails_before_inference(self):
        self.client.get.return_value = {"models": [
            {"slug": "other-model", "visibility": "list"},
        ]}
        with self.assertRaisesRegex(LabelingError, "not available"):
            self.make_infer()(self.report_canary)
        self.client.responses.create.assert_not_called()

    def test_list_available_models_returns_only_visible_catalog_entries(self):
        self.client.get.return_value = {"models": [
            {"slug": "gpt-6-luna", "display_name": "Luna", "visibility": "list"},
            {"slug": "hidden-model", "display_name": "Hidden", "visibility": "hidden"},
            {"slug": "missing-name", "visibility": "list"},
        ]}
        self.assertEqual(
            self.make_infer().list_available_models(),
            [{"slug": "gpt-6-luna", "display_name": "Luna"}],
        )
        self.client.get.assert_called_once_with("/models", cast_to=dict[str, Any])
        self.assertTrue(self.client.close.called)

    def test_list_available_models_preserves_allowlisted_provider_diagnostics_only(self):
        provider_exception = RuntimeError("PRIVATE_CATALOG_CANARY raw response body")
        provider_exception.status_code = 429
        provider_exception.request_id = "req_catalog-123"
        provider_exception.body = {"error": {
            "code": "subscription_sharing_usage_limit_exceeded",
            "param": "model",
            "message": "PRIVATE_CATALOG_CANARY",
        }}
        provider_exception.response = SimpleNamespace(
            headers={"x-request-id": "PRIVATE_HEADER_CANARY"},
        )
        self.client.get.side_effect = provider_exception

        with self.assertRaises(LabelingError) as raised:
            self.make_infer().list_available_models()

        error = raised.exception
        self.assertEqual(error.http_status, 429)
        self.assertEqual(error.provider_error_code,
                         "subscription_sharing_usage_limit_exceeded")
        self.assertEqual(error.provider_error_param, "model")
        self.assertEqual(error.provider_body_shape, "nested_error")
        self.assertEqual(error.request_id, "req_catalog-123")
        for canary in ("PRIVATE_CATALOG_CANARY", "PRIVATE_HEADER_CANARY", "raw response body"):
            self.assertNotIn(canary, str(error))

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
        self.assertFalse(raised.exception.stream_completed)
        self.assertEqual(raised.exception.stream_delta_chars, len("partial"))
        self.assertEqual(raised.exception.stream_text_source, "none")

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
                stream_completed=False,
                stream_delta_chars=124,
                stream_final_chars=0,
                stream_text_source="none",
                stream_refusal=False,
                stream_format="empty",
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
        self.assertIn('"stream_completed":false', diagnostic_text)
        self.assertIn('"stream_delta_chars":124', diagnostic_text)
        self.assertIn('"stream_final_chars":0', diagnostic_text)
        self.assertIn('"stream_text_source":"none"', diagnostic_text)
        self.assertIn('"stream_refusal":false', diagnostic_text)
        self.assertIn('"stream_format":"empty"', diagnostic_text)
        self.assertNotIn(self.report_canary, str(raised.exception))
        self.assertNotIn(self.report_canary, diagnostic_text)

    def test_run_pilot_never_renders_raw_injected_exception_diagnostics(self):
        rows = [{
            "StudyInstanceUID": "synthetic-study",
            "Report": self.report_canary,
            **{target: "" for target in LABEL_COLUMNS},
        }]
        private_canary = "PRIVATE_REPORT_DIAGNOSTIC_CANARY"

        def failed_infer(_report):
            error = LabelingError("safe provider failure", cause_code="quota_error")
            error.http_status = private_canary
            error.provider_error_code = private_canary
            error.provider_error_param = private_canary
            error.provider_body_shape = private_canary
            error.request_id = private_canary
            error.stream_completed = private_canary
            error.stream_delta_chars = private_canary
            error.stream_final_chars = private_canary
            error.stream_text_source = private_canary
            error.stream_refusal = private_canary
            error.stream_format = private_canary
            raise error

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
        self.assertNotIn(private_canary, str(raised.exception))
        self.assertNotIn(private_canary, diagnostic_text)
        self.assertIn('"cause_code":"quota_error"', diagnostic_text)
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

    def test_nonempty_evidence_must_be_literal_source_substring(self):
        invalid = valid_result()
        invalid["ACL"]["evidence"] = "PRIVATE_UNGROUNDED_EVIDENCE"
        self.client.responses.create.return_value = FakeStream([SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(output_text=json.dumps(invalid)),
        )])

        with self.assertRaises(LabelingError) as raised:
            self.make_infer()(self.report_canary)

        self.assertEqual(raised.exception.cause_code, "evidence_not_grounded")
        self.assertEqual(raised.exception.evidence_mismatch_kind, "no_literal_match")
        self.assertNotIn("PRIVATE_UNGROUNDED_EVIDENCE", str(raised.exception))
        self.assertNotIn(self.report_canary, str(raised.exception))

    def test_optin_capture_receives_only_rejected_fresh_evidence_and_never_leaks(self):
        evidence = "PRIVATE_REJECTED_EVIDENCE_CANARY"
        invalid = valid_result()
        for target in invalid.values():
            target["evidence"] = ""
        invalid["ACL"]["evidence"] = evidence
        invalid["MCL"]["evidence"] = "PRIVATE_SECOND_REJECTED_CANARY"
        self.client.responses.create.return_value = FakeStream([SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(output_text=json.dumps(invalid)),
        )])
        captured = []

        with self.assertRaises(LabelingError) as raised:
            self.make_infer()(self.report_canary, on_rejected_evidence=captured.append)

        self.assertEqual(captured, [evidence])
        self.assertEqual(raised.exception.cause_code, "evidence_not_grounded")
        for rendered in (str(raised.exception), repr(raised.exception), repr(raised.exception.__dict__)):
            self.assertNotIn(evidence, rendered)
            self.assertNotIn("PRIVATE_SECOND_REJECTED_CANARY", rendered)
            self.assertNotIn(self.report_canary, rendered)

    def test_capture_callback_failure_does_not_replace_safe_validation_error(self):
        evidence = "PRIVATE_CALLBACK_FAILURE_CANARY"
        invalid = valid_result()
        for target in invalid.values():
            target["evidence"] = ""
        invalid["ACL"]["evidence"] = evidence
        self.client.responses.create.return_value = FakeStream([SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(output_text=json.dumps(invalid)),
        )])

        def failing_capture(value):
            raise RuntimeError(f"PRIVATE_WRITE_EXCEPTION:{value}")

        with self.assertRaises(LabelingError) as raised:
            self.make_infer()(self.report_canary, on_rejected_evidence=failing_capture)

        self.assertEqual(raised.exception.cause_code, "evidence_not_grounded")
        for rendered in (str(raised.exception), repr(raised.exception), repr(raised.exception.__dict__)):
            self.assertNotIn(evidence, rendered)
            self.assertNotIn("PRIVATE_WRITE_EXCEPTION", rendered)

    def test_optin_capture_callback_is_not_called_for_accepted_fresh_output(self):
        captured = []
        result = self.make_infer()(self.report_canary,
                                   on_rejected_evidence=captured.append)
        self.assertEqual(result, valid_result())
        self.assertEqual(captured, [])

    def test_optin_capture_callback_is_not_called_for_schema_invalid_output(self):
        self.client.responses.create.return_value = FakeStream([SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(output_text=json.dumps({"ACL": "invalid"})),
        )])
        captured = []
        with self.assertRaises(LabelingError):
            self.make_infer()(self.report_canary, on_rejected_evidence=captured.append)
        self.assertEqual(captured, [])

    def test_literal_evidence_mismatch_reports_only_static_category(self):
        cases = (
            ("prefix\n ACL tear", "prefix ACL tear", "whitespace_only"),
            ("ACL TEAR", "acl tear", "case_only"),
            ("cafe\u0301", "caf\u00e9", "unicode_only"),
            ("ACL fracture", "ACL tear", "no_literal_match"),
        )
        for evidence, report, expected in cases:
            with self.subTest(expected=expected):
                result = valid_result()
                result["ACL"]["evidence"] = evidence
                with self.assertRaises(LabelingError) as raised:
                    self.make_infer().validate_source_evidence(result, report)
                self.assertEqual(raised.exception.evidence_mismatch_kind, expected)
                self.assertNotIn(evidence, str(raised.exception))
                self.assertNotIn(report, str(raised.exception))

    def test_blank_normalization_does_not_match_but_empty_and_exact_evidence_do(self):
        result = valid_result()
        result["ACL"]["evidence"] = " \n\t "
        with self.assertRaises(LabelingError) as raised:
            self.make_infer().validate_source_evidence(result, " \t ")
        self.assertEqual(raised.exception.evidence_mismatch_kind, "no_literal_match")
        result["ACL"]["evidence"] = ""
        for target in result.values():
            target["evidence"] = ""
        self.make_infer().validate_source_evidence(result, "")
        result["ACL"]["evidence"] = "ACL tear"
        self.make_infer().validate_source_evidence(result, "Report: ACL tear.")

    def test_fresh_inference_recovers_whitespace_span_as_source_literal(self):
        raw_result = valid_result()
        raw_result["ACL"]["evidence"] = "ACL\n\ttear;"
        serialized = json.dumps(raw_result)
        self.client.responses.create.return_value = FakeStream([SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(output_text=serialized),
        )])

        result = self.make_infer()("Report says ACL  tear; synthetic evidence")

        self.assertEqual(result["ACL"]["evidence"], "ACL  tear;")
        original = json.loads(serialized)
        self.assertEqual(original["ACL"]["evidence"], "ACL\n\ttear;")
        for target in result:
            self.assertEqual(result[target]["label"], original[target]["label"])
            self.assertEqual(result[target]["uncertainty"], original[target]["uncertainty"])

    def test_fresh_inference_selects_earliest_duplicate_or_overlapping_span(self):
        cases = (
            ("ACL\ntear;", "prefix ACL  tear; then ACL\ttear;", "ACL  tear;"),
            ("A\nA", "A  A A  A", "A  A"),
        )
        for evidence, report, expected in cases:
            with self.subTest(expected=expected):
                raw_result = valid_result()
                for target in raw_result.values():
                    target["evidence"] = ""
                raw_result["ACL"]["evidence"] = evidence
                original = json.loads(json.dumps(raw_result))
                self.client.responses.create.return_value = FakeStream([SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(output_text=json.dumps(raw_result)),
                )])

                result = self.make_infer()(report)

                self.assertEqual(result["ACL"]["evidence"], expected)
                self.assertEqual(raw_result, original)
                for target in result:
                    self.assertEqual(result[target]["label"], original[target]["label"])
                    self.assertEqual(result[target]["uncertainty"], original[target]["uncertainty"])

    def test_fresh_inference_recovers_case_and_whitespace_to_earliest_source_span(self):
        cases = (
            ("acl TEAR", "prefix ACL tear then AcL TeaR", "ACL tear"),
            ("\u00c1CL\nTear;", "\u00e1cl  TEAR; second \u00c1CL\tTear;", "\u00e1cl  TEAR;"),
        )
        for evidence, report, expected in cases:
            with self.subTest(expected=expected):
                raw_result = valid_result()
                for target in raw_result.values():
                    target["evidence"] = ""
                raw_result["ACL"]["evidence"] = evidence
                original = json.loads(json.dumps(raw_result))
                self.client.responses.create.return_value = FakeStream([SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(output_text=json.dumps(raw_result)),
                )])
                captured = []

                result = self.make_infer()(report, on_rejected_evidence=captured.append)

                self.assertEqual(result["ACL"]["evidence"], expected)
                self.assertEqual(captured, [])
                self.assertEqual(raw_result, original)
                for target in result:
                    self.assertEqual(result[target]["label"], original[target]["label"])
                    self.assertEqual(result[target]["uncertainty"], original[target]["uncertainty"])

    def test_case_recovery_rejects_lowercase_expansion_and_accent_changes(self):
        cases = (
            ("Stra\u00dfe", "STRASSE", "no_literal_match"),
            ("caf\u00e9", "cafe", "no_literal_match"),
        )
        for evidence, report, mismatch_kind in cases:
            with self.subTest(mismatch_kind=mismatch_kind):
                raw_result = valid_result()
                for target in raw_result.values():
                    target["evidence"] = ""
                raw_result["ACL"]["evidence"] = evidence
                self.client.responses.create.return_value = FakeStream([SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(output_text=json.dumps(raw_result)),
                )])

                with self.assertRaises(LabelingError) as raised:
                    self.make_infer()(report)

                self.assertEqual(raised.exception.evidence_mismatch_kind, mismatch_kind)

    def test_fresh_inference_rejects_partial_token_and_nonwhitespace_changes(self):
        cases = (
            ("ACL injury", "ACL tear", "no_literal_match"),
            ("ACL-tear", "ACL tear", "no_literal_match"),
            ("ACL\ntear", "ACL  tears", "no_literal_match"),
            ("cafe\u0301", "caf\u00e9", "unicode_only"),
        )
        for evidence, report, mismatch_kind in cases:
            with self.subTest(mismatch_kind=mismatch_kind, report=report):
                raw_result = valid_result()
                for target in raw_result.values():
                    target["evidence"] = ""
                raw_result["ACL"]["evidence"] = evidence
                self.client.responses.create.return_value = FakeStream([SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(output_text=json.dumps(raw_result)),
                )])
                with self.assertRaises(LabelingError) as raised:
                    self.make_infer()(report)
                self.assertEqual(raised.exception.evidence_mismatch_kind, mismatch_kind)
                self.assertNotIn(evidence, str(raised.exception))

    def test_fresh_inference_keeps_exact_duplicate_substrings_accepted(self):
        result = valid_result()
        for target in result.values():
            target["evidence"] = ""
        result["ACL"]["evidence"] = "ACL tear"
        self.client.responses.create.return_value = FakeStream([SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(output_text=json.dumps(result)),
        )])

        accepted = self.make_infer()("ACL tear; then ACL tear")

        self.assertEqual(accepted["ACL"]["evidence"], "ACL tear")

    def test_boundary_aware_recovery_copies_each_match_without_mutating_input(self):
        result = valid_result()
        result["ACL"]["evidence"] = "ACL\ntear;"
        recovered = _recover_fresh_boundary_aware_evidence(result, "ACL  tear; synthetic evidence")
        self.assertIsNot(recovered, result)
        self.assertIsNot(recovered["ACL"], result["ACL"])
        self.assertEqual(recovered["ACL"]["evidence"], "ACL  tear;")
        self.assertEqual(result["ACL"]["evidence"], "ACL\ntear;")

        result["MCL"]["evidence"] = "missing\ttext"
        original = json.loads(json.dumps(result))
        rejected = _recover_fresh_boundary_aware_evidence(result, "ACL  tear; synthetic evidence")
        self.assertIsNot(rejected, result)
        self.assertEqual(rejected["ACL"]["evidence"], "ACL  tear;")
        self.assertEqual(rejected["MCL"]["evidence"], "missing\ttext")
        self.assertEqual(result, original)
        self.assertEqual(result["ACL"]["evidence"], "ACL\ntear;")
        self.assertEqual(result["MCL"]["evidence"], "missing\ttext")

    def test_runner_preserves_cached_prefix_and_saves_recovered_source_span(self):
        rows = [{
            "StudyInstanceUID": f"synthetic-study-{index}",
            "Report": "ACL  tear with synthetic evidence",
            **{target: "" for target in LABEL_COLUMNS},
        } for index in range(3)]
        previous = [valid_result(), valid_result()]
        source_result = valid_result()
        source_result["ACL"]["evidence"] = "ACL\ntear"
        serialized = json.dumps(source_result)
        self.client.responses.create.return_value = FakeStream([SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(output_text=serialized),
        )])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "data" / "outputs" / "labels.jsonl"
            journal = root / "data" / "outputs" / "diagnostics.jsonl"
            original_records = [
                {"StudyInstanceUID": f"synthetic-study-{index}", "labels": labels}
                for index, labels in enumerate(previous)
            ]
            output.parent.mkdir(parents=True)
            output.write_text("".join(json.dumps(record) + "\n" for record in original_records),
                              encoding="utf-8")

            count = run_pilot(
                rows,
                gold_ids={f"synthetic-gold-{index}" for index in range(58)},
                max_records=1,
                skip_records=2,
                output_path=output,
                diagnostics_path=journal,
                project_root=root,
                subscription_oauth_confirmed=True,
                infer=self.make_infer(),
            )
            saved = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(count, 3)
        self.assertEqual(saved[:2], original_records)
        self.assertEqual(saved[2]["labels"]["ACL"]["evidence"], "ACL  tear")
        expected = json.loads(serialized)
        expected["ACL"]["evidence"] = "ACL  tear"
        self.assertEqual(saved[2]["labels"], expected)

    def test_empty_evidence_remains_allowed(self):
        empty = valid_result()
        empty["ACL"]["evidence"] = ""
        self.client.responses.create.return_value = FakeStream([SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(output_text=json.dumps(empty)),
        )])

        result = self.make_infer()(self.report_canary)

        self.assertEqual(result["ACL"]["evidence"], "")

    def test_cached_ungrounded_evidence_fails_before_inference_and_preserves_output(self):
        rows = [{
            "StudyInstanceUID": "synthetic-study",
            "Report": self.report_canary,
            **{target: "" for target in LABEL_COLUMNS},
        }]
        cached = valid_result()
        cached["ACL"]["evidence"] = "PRIVATE_CACHED_EVIDENCE"
        payload = json.dumps({"StudyInstanceUID": "synthetic-study", "labels": cached}) + "\n"
        infer = Mock()
        infer.validate_source_evidence = self.make_infer().validate_source_evidence

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "data" / "outputs" / "labels.jsonl"
            output.parent.mkdir(parents=True)
            output.write_text(payload, encoding="utf-8")
            with self.assertRaises(LabelingError) as raised:
                run_pilot(
                    rows,
                    gold_ids={f"synthetic-gold-{index}" for index in range(58)},
                    max_records=1,
                    output_path=output,
                    diagnostics_path=root / "data" / "outputs" / "diagnostics.jsonl",
                    project_root=root,
                    subscription_oauth_confirmed=True,
                    infer=infer,
                )
            after = output.read_text(encoding="utf-8")

        infer.assert_not_called()
        self.assertEqual(after, payload)
        self.assertEqual(raised.exception.cause_code, "evidence_not_grounded")
        self.assertNotIn("PRIVATE_CACHED_EVIDENCE", str(raised.exception))
        self.assertNotIn("synthetic-study", str(raised.exception))

    def test_fresh_ungrounded_evidence_is_rejected_before_persist_with_safe_diagnostic(self):
        rows = [{
            "StudyInstanceUID": "synthetic-study",
            "Report": self.report_canary,
            **{target: "" for target in LABEL_COLUMNS},
        }]
        invalid = valid_result()
        invalid["ACL"]["evidence"] = "PRIVATE_FRESH_EVIDENCE"
        infer = Mock(return_value=invalid)
        infer.validate_source_evidence = self.make_infer().validate_source_evidence

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "data" / "outputs" / "labels.jsonl"
            journal = root / "data" / "outputs" / "diagnostics.jsonl"
            with self.assertRaises(LabelingError) as raised:
                run_pilot(
                    rows,
                    gold_ids={f"synthetic-gold-{index}" for index in range(58)},
                    max_records=1,
                    output_path=output,
                    diagnostics_path=journal,
                    project_root=root,
                    subscription_oauth_confirmed=True,
                    infer=infer,
                )
            diagnostic_text = journal.read_text(encoding="utf-8")

        self.assertFalse(output.exists())
        self.assertEqual(raised.exception.cause_code, "evidence_not_grounded")
        self.assertIn('"cause_code":"evidence_not_grounded"', diagnostic_text)
        failed_record = next(json.loads(line) for line in diagnostic_text.splitlines()
                             if '"event":"attempt_failed"' in line)
        self.assertEqual(failed_record["evidence_mismatch_kind"], "no_literal_match")
        for private_value in ("PRIVATE_FRESH_EVIDENCE", self.report_canary, "synthetic-study"):
            self.assertNotIn(private_value, str(raised.exception))
            self.assertNotIn(private_value, diagnostic_text)

    def test_valid_cached_nonselected_prefix_is_grounded_before_resume(self):
        rows = [
            {
                "StudyInstanceUID": f"synthetic-study-{index}",
                "Report": self.report_canary,
                **{target: "" for target in LABEL_COLUMNS},
            }
            for index in range(2)
        ]
        labels = valid_result()
        infer_calls = []
        def infer(_report):
            infer_calls.append(True)
            return valid_result()

        infer.validate_source_evidence = self.make_infer().validate_source_evidence

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "data" / "outputs" / "labels.jsonl"
            output.parent.mkdir(parents=True)
            output.write_text(
                json.dumps({"StudyInstanceUID": "synthetic-study-0", "labels": labels}) + "\n",
                encoding="utf-8",
            )
            count = run_pilot(
                rows,
                gold_ids={f"synthetic-gold-{index}" for index in range(58)},
                max_records=1,
                skip_records=1,
                output_path=output,
                diagnostics_path=root / "data" / "outputs" / "diagnostics.jsonl",
                project_root=root,
                subscription_oauth_confirmed=True,
                infer=infer,
            )

        self.assertEqual(count, 2)
        self.assertEqual(infer_calls, [True])

    def test_generic_infer_path_does_not_require_source_grounding_hook(self):
        rows = [{
            "StudyInstanceUID": "synthetic-study",
            "Report": self.report_canary,
            **{target: "" for target in LABEL_COLUMNS},
        }]
        def infer(_report):
            return valid_result()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            count = run_pilot(
                rows,
                gold_ids={f"synthetic-gold-{index}" for index in range(58)},
                max_records=1,
                output_path=root / "data" / "outputs" / "labels.jsonl",
                diagnostics_path=root / "data" / "outputs" / "diagnostics.jsonl",
                project_root=root,
                subscription_oauth_confirmed=True,
                infer=infer,
            )

        self.assertEqual(count, 1)

    def test_fresh_recovery_uses_boundary_aware_case_and_whitespace_matching(self):
        cases = (
            ("acl tear", "prefix: ACL tear. suffix", "ACL tear"),
            ("acl tear.", "(ACL tear.)", "ACL tear."),
            ("ACL-TEAR", "acl-tear", "acl-tear"),
            ("\u00e1cl\ntear", "\u00c1CL  TEAR.", "\u00c1CL  TEAR"),
            ("\u03bf\u03c2 tear", "\u039f\u03a3  tear.", "\u039f\u03a3  tear"),
        )
        for evidence, report, expected in cases:
            with self.subTest(evidence=evidence):
                raw_result = valid_result()
                for target in raw_result.values():
                    target["evidence"] = ""
                raw_result["ACL"]["evidence"] = evidence
                self.client.responses.create.return_value = FakeStream([SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(output_text=json.dumps(raw_result)),
                )])

                result = self.make_infer()(report)

                self.assertEqual(result["ACL"]["evidence"], expected)
                self.assertEqual(raw_result["ACL"]["evidence"], evidence)

    def test_fresh_recovery_and_case_classifier_reject_partial_word_and_expansion_matches(self):
        rejected_reports = (
            "xACL tear", "1ACL tear", "_ACL tear", "\u03b1ACL tear",
            "ACL tear2", "ACL tear_", "ACL tear\u0301", "ACL-tear",
            "\u0130",
        )
        for report in rejected_reports:
            with self.subTest(report=report):
                evidence = "i" if report == "\u0130" else "acl tear"
                raw_result = valid_result()
                for target in raw_result.values():
                    target["evidence"] = ""
                raw_result["ACL"]["evidence"] = evidence
                self.client.responses.create.return_value = FakeStream([SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(output_text=json.dumps(raw_result)),
                )])

                with self.assertRaises(LabelingError) as raised:
                    self.make_infer()(report)

                self.assertEqual(raised.exception.evidence_mismatch_kind, "no_literal_match")
                if len(evidence) > 4:
                    self.assertNotIn(evidence, str(raised.exception))

    def test_later_target_failure_is_attributed_after_recovering_earlier_target(self):
        report = "ACL tear."
        raw_result = valid_result()
        for target in raw_result.values():
            target["evidence"] = ""
        raw_result["ACL"]["evidence"] = "acl tear"
        raw_result["MCL"]["evidence"] = "unmatched synthetic evidence"
        original = json.loads(json.dumps(raw_result))
        self.client.responses.create.return_value = FakeStream([SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(output_text=json.dumps(raw_result)),
        )])
        captured = []

        rows = [{
            "StudyInstanceUID": "synthetic-study",
            "Report": report,
            **{target: "" for target in LABEL_COLUMNS},
        }]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "data" / "outputs" / "labels.jsonl"
            journal = root / "data" / "outputs" / "diagnostics.jsonl"
            with self.assertRaises(LabelingError) as raised:
                run_pilot(
                    rows,
                    gold_ids={f"synthetic-gold-{index}" for index in range(58)},
                    max_records=1,
                    output_path=output,
                    diagnostics_path=journal,
                    project_root=root,
                    subscription_oauth_confirmed=True,
                    infer=self.make_infer(),
                    rejected_evidence_capture=lambda evidence, offset: captured.append(
                        (evidence, offset)
                    ),
                )
            journal_text = journal.read_text(encoding="utf-8")

        self.assertEqual(captured, [("unmatched synthetic evidence", 0)])
        self.assertEqual(raised.exception.evidence_mismatch_kind, "no_literal_match")
        self.assertFalse(output.exists())
        self.assertEqual(raw_result, original)
        self.assertNotIn(report, str(raised.exception))
        self.assertNotIn("unmatched synthetic evidence", str(raised.exception))
        self.assertNotIn("unmatched synthetic evidence", journal_text)

    def test_case_and_whitespace_diagnostics_use_shared_boundaries(self):
        cases = (
            ("acl tear", "ACL tear.", "case_only"),
            ("ACL\ntear", "ACL  tear", "whitespace_only"),
            ("ACL\ntear", "ACL  tears", "no_literal_match"),
            ("i", "\u0130", "no_literal_match"),
        )
        for evidence, report, expected in cases:
            with self.subTest(expected=expected):
                result = valid_result()
                for target in result.values():
                    target["evidence"] = ""
                result["ACL"]["evidence"] = evidence
                with self.assertRaises(LabelingError) as raised:
                    self.make_infer().validate_source_evidence(result, report)
                self.assertEqual(raised.exception.evidence_mismatch_kind, expected)


if __name__ == "__main__":
    unittest.main()
