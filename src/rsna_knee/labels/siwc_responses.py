"""Responses API inference through a locally authorized SIWC profile."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from rsna_knee.labels.siwc_auth import AuthError, SIWCAuthenticator
from rsna_knee.labels.subscription import LABEL_COLUMNS, LabelingError, validate_label_result


RESPONSES_BASE_URL = "https://api.openai.com/v1"
DEFAULT_SIWC_MODEL = "gpt-6-luna"


class SIWCResponsesInfer:
    """Callable `run_pilot` adapter using an authorized account profile.

    The requested model is checked against the account-specific list endpoint
    before any report text is sent. The account's catalog, not a hard-coded
    global catalog, is authoritative for availability.
    """

    def __init__(self, *, authenticator: SIWCAuthenticator,
                 profile_id: str, model: str = DEFAULT_SIWC_MODEL,
                 client_factory: Callable[..., Any] | None = None):
        if not isinstance(profile_id, str) or not profile_id.strip():
            raise LabelingError("A selected SIWC account profile is required.",
                                cause_code="auth_error")
        if not isinstance(model, str) or not model.strip():
            raise LabelingError("A requested account model is required.",
                                cause_code="model_error")
        self._authenticator = authenticator
        self._profile_id = profile_id
        self._model = model.strip()
        self._client_factory = client_factory or self._default_client_factory

    @staticmethod
    def _default_client_factory(**kwargs):
        try:
            from openai import OpenAI
        except ImportError:
            raise LabelingError("Responses API support is unavailable.",
                                cause_code="network_error") from None
        return OpenAI(**kwargs)

    def __call__(self, report: str) -> dict[str, dict[str, object]]:
        if not isinstance(report, str) or not report.strip():
            raise LabelingError("Report input is blank or invalid.")
        try:
            return self._authenticator.with_plan_access_token(
                self._profile_id,
                lambda access_token: self._infer_authorized(access_token, report),
            )
        except LabelingError:
            raise
        except AuthError:
            raise LabelingError("A valid ChatGPT plan-use authorization is required.",
                                stage="inference", cause_code="auth_error") from None
        except Exception:
            # Provider exceptions can include request or response bodies.
            raise LabelingError("Responses inference could not be completed.",
                                stage="inference", cause_code="network_error") from None

    def _infer_authorized(self, access_token: str,
                          report: str) -> dict[str, dict[str, object]]:
        client = None
        try:
            client = self._client_factory(
                api_key=access_token,
                base_url=RESPONSES_BASE_URL,
                max_retries=0,
            )
            catalog = client.get("/models", cast_to=dict)
            if not self._model_is_available(catalog):
                raise LabelingError(
                    "The requested model is not available to the selected account.",
                    stage="inference", cause_code="model_error",
                )
            prompt = _build_label_prompt(report)
            stream = client.responses.create(
                model=self._model,
                input=prompt,
                store=False,
                stream=True,
            )
            completed_text = None
            with stream as events:
                for event in events:
                    event_type = getattr(event, "type", None)
                    if event_type in {"response.failed", "response.incomplete", "error"}:
                        raise LabelingError(
                            "Responses inference did not complete successfully.",
                            stage="inference", cause_code="model_error",
                        )
                    if event_type == "response.completed":
                        response = getattr(event, "response", None)
                        text = getattr(response, "output_text", None)
                        if isinstance(text, str):
                            completed_text = text
            if completed_text is None:
                raise LabelingError(
                    "Responses stream ended without response.completed.",
                    stage="inference", cause_code="network_error",
                )
            try:
                parsed = json.loads(completed_text)
            except (TypeError, ValueError, json.JSONDecodeError):
                raise LabelingError("Responses output was not valid structured label data.",
                                    stage="response_validation",
                                    cause_code="malformed_json") from None
            try:
                return validate_label_result(parsed)
            except LabelingError as exc:
                raise LabelingError("Responses output failed local label validation.",
                                    stage="response_validation",
                                    cause_code=exc.cause_code) from None
        except LabelingError:
            raise
        except Exception:
            raise LabelingError("Responses inference could not be completed.",
                                stage="inference", cause_code="network_error") from None
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass

    def _model_is_available(self, catalog: object) -> bool:
        if not isinstance(catalog, Mapping):
            return False
        models = catalog.get("models")
        if not isinstance(models, list):
            return False
        return any(
            isinstance(item, Mapping)
            and item.get("visibility") == "list"
            and item.get("slug") == self._model
            for item in models
        )


def _build_label_prompt(report: str) -> str:
    targets = "\n".join(f"- {column}" for column in LABEL_COLUMNS)
    schema = {
        column: {"label": "boolean", "evidence": "string", "uncertainty": "number from 0 to 1"}
        for column in LABEL_COLUMNS
    }
    return (
        "Extract provisional binary findings for these targets:\n"
        f"TARGETS:\n{targets}\n\n"
        "Return only one JSON object matching this target schema: "
        f"{json.dumps(schema, separators=(',', ':'))}. Evidence must be a short exact report "
        "excerpt (maximum 200 characters); use an empty excerpt when unsupported. Do not add keys.\n\n"
        f"REPORT:\n{report}"
    )
