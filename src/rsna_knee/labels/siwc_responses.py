"""Responses API inference through a locally authorized SIWC profile."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from rsna_knee.labels.siwc_auth import AuthError, SIWCAuthenticator
from rsna_knee.labels.subscription import (
    LABEL_COLUMNS,
    LabelingError,
    _SAFE_REQUEST_ID,
    _SAFE_SIWC_BODY_SHAPES,
    _SAFE_SIWC_ERROR_CODES,
    _SAFE_SIWC_ERROR_PARAMS,
    validate_label_result,
)


RESPONSES_BASE_URL = "https://api.openai.com/v1"
DEFAULT_SIWC_MODEL = "gpt-6-luna"


def safe_provider_diagnostics(error: object) -> str:
    """Format only metadata already covered by the SIWC provider allowlists."""
    values = (
        ("http_status", getattr(error, "http_status", None),
         lambda value: type(value) is int and 100 <= value <= 599),
        ("provider_error_code", getattr(error, "provider_error_code", None),
         lambda value: type(value) is str and value in _SAFE_SIWC_ERROR_CODES),
        ("provider_error_param", getattr(error, "provider_error_param", None),
         lambda value: type(value) is str and value in _SAFE_SIWC_ERROR_PARAMS),
        ("provider_body_shape", getattr(error, "provider_body_shape", None),
         lambda value: type(value) is str and value in _SAFE_SIWC_BODY_SHAPES),
        ("request_id", getattr(error, "request_id", None),
         lambda value: type(value) is str and _SAFE_REQUEST_ID.fullmatch(value)),
    )
    return " ".join(f"{name}={value}" for name, value, is_safe in values if is_safe(value))


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

    def list_available_models(self) -> list[dict[str, str]]:
        """Return only displayable models for this authorized ChatGPT account."""
        try:
            return self._authenticator.with_plan_access_token(
                self._profile_id,
                self._list_models_authorized,
            )
        except LabelingError:
            raise
        except AuthError:
            raise LabelingError("A valid ChatGPT plan-use authorization is required.",
                                stage="inference", cause_code="auth_error") from None
        except Exception:
            raise LabelingError("The account model catalog could not be loaded.",
                                stage="inference", cause_code="network_error") from None

    def _list_models_authorized(self, access_token: str) -> list[dict[str, str]]:
        client = None
        try:
            client = self._client_factory(
                api_key=access_token,
                base_url=RESPONSES_BASE_URL,
                max_retries=0,
            )
            catalog = client.get("/models", cast_to=dict)
            if not isinstance(catalog, Mapping) or not isinstance(catalog.get("models"), list):
                raise LabelingError("The account model catalog has an invalid structure.",
                                    stage="inference", cause_code="model_error")
            return [
                {"slug": item["slug"], "display_name": item["display_name"]}
                for item in catalog["models"]
                if isinstance(item, Mapping)
                and item.get("visibility") == "list"
                and isinstance(item.get("slug"), str) and item["slug"].strip()
                and isinstance(item.get("display_name"), str) and item["display_name"].strip()
            ]
        except LabelingError:
            raise
        except Exception as exc:
            raise _provider_error("The account model catalog could not be loaded.", exc) from None
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass

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
                        response = getattr(event, "response", None)
                        error = getattr(response, "error", None)
                        if error is None:
                            error = getattr(event, "error", None)
                        raise _provider_error(
                            "Responses inference did not complete successfully.",
                            error,
                            response=response,
                            default_cause="model_error",
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
        except Exception as exc:
            raise _provider_error("Responses inference could not be completed.", exc) from None
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


def _provider_error(message: str, error: object, *, response: object = None,
                    default_cause: str = "network_error") -> LabelingError:
    """Preserve only allowlisted provider fields; never retain body text."""
    status = _safe_int(getattr(error, "status_code", None))
    if status is None:
        status = _safe_int(getattr(response, "status_code", None))
    code = getattr(error, "code", None)
    param = getattr(error, "param", None)
    body = getattr(error, "body", None)
    body_shape = None
    if isinstance(body, Mapping):
        nested = body.get("error")
        if isinstance(nested, Mapping):
            body_shape = "nested_error"
            code = code or nested.get("code")
            param = param or nested.get("param")
        elif isinstance(body.get("detail"), str):
            body_shape = "detail_object"
        elif isinstance(body.get("error"), str):
            body_shape = "error_text"
        elif "code" in body or "param" in body:
            body_shape = "top_level_error"
            code = code or body.get("code")
            param = param or body.get("param")
        else:
            body_shape = "object"
    elif body is not None:
        body_shape = "non_object"
    safe_code = (code if isinstance(code, str)
                 and code in _SAFE_SIWC_ERROR_CODES else None)
    safe_param = (param if isinstance(param, str)
                  and param in _SAFE_SIWC_ERROR_PARAMS else None)
    request_id = getattr(error, "request_id", None)
    if not isinstance(request_id, str):
        request_id = getattr(response, "_request_id", None)
    if not isinstance(request_id, str):
        response_obj = getattr(error, "response", None)
        headers = getattr(response_obj, "headers", None)
        if headers is not None:
            try:
                request_id = headers.get("x-request-id")
            except Exception:
                request_id = None
    safe_request_id = (request_id if isinstance(request_id, str)
                       and _SAFE_REQUEST_ID.fullmatch(request_id) else None)

    cause = default_cause
    if status == 401 or safe_code == "subscription_sharing_invalid_user":
        cause = "auth_error"
    elif status == 429 or safe_code == "subscription_sharing_usage_limit_exceeded":
        cause = "quota_error"
    elif (status == 503 or safe_code in {
            "subscription_sharing_usage_unavailable",
            "subscription_sharing_user_unavailable",
        }):
        cause = "network_error"
    elif safe_code is not None or status in {400, 403}:
        cause = "model_error"
    return LabelingError(
        message,
        stage="inference",
        cause_code=cause,
        http_status=status,
        provider_error_code=safe_code,
        provider_error_param=safe_param,
        provider_body_shape=(body_shape if body_shape in _SAFE_SIWC_BODY_SHAPES else None),
        request_id=safe_request_id,
    )


def _safe_int(value: object) -> int | None:
    if type(value) is int and 100 <= value <= 599:
        return value
    return None
