"""Responses API inference through a locally authorized SIWC profile."""

from __future__ import annotations

import json
import unicodedata
from bisect import bisect_right
from collections.abc import Callable, Mapping
from typing import Any

from rsna_knee.labels.siwc_auth import (
    AuthError,
    SIWCAuthenticator,
    _format_safe_auth_diagnostics,
)
from rsna_knee.labels.prompt_comparison import build_candidate_prompt
from rsna_knee.labels.subscription import (
    LabelingError,
    LABEL_COLUMNS,
    _SAFE_REQUEST_ID,
    _SAFE_SIWC_BODY_SHAPES,
    _SAFE_SIWC_ERROR_CODES,
    _SAFE_SIWC_ERROR_PARAMS,
    _SAFE_STREAM_FORMATS,
    _SAFE_STREAM_TEXT_SOURCES,
    _SAFE_JSON_VALUE_TYPES,
    _SAFE_EVIDENCE_MISMATCH_KINDS,
    validate_label_result,
)


RESPONSES_BASE_URL = "https://api.openai.com/v1"
DEFAULT_SIWC_MODEL = "gpt-6-luna"

SIWC_PROMPT_VERSION = "c2-literal-evidence-v3"
_FRAGMENT_INSTRUCTIONS = (
    "\n\nSource-fragment selection (" + SIWC_PROMPT_VERSION + "):\n"
    "The following WIRE output instructions override the earlier evidence:string "
    "and 200-character maximum instructions; target definitions, binary-label "
    "decisions and the probability-of-error uncertainty definition are unchanged. "
    "Return only one JSON object with exactly the listed targets, each mapped to "
    "{label:boolean, evidence_id:string|null, uncertainty:number from 0 to 1}. "
    "Do not return an evidence field or free-text quotation. First locate the "
    "relevant supporting or contradicting passage in the complete original REPORT, "
    "then select its ID from SOURCE_FRAGMENTS_JSON. Select one entire original "
    "nonblank source line; never invent an ID. Use null only when no relevant "
    "supporting or contradicting passage exists, never to avoid a difficult quote. "
    "The code copies the selected original line unchanged, including source typos, "
    "punctuation, numbers, whitespace and line endings; do not correct spelling, "
    "grammar or encoding. There is no hard evidence-length cap: a one-line REPORT "
    "has one whole-report fragment, which may be long. Fragment selection does not "
    "change the clinical labeling rules. REPORT remains the complete context."
)


def _source_line_catalogue(report: str) -> dict[str, str]:
    """Index nonblank original line slices without normalizing their contents."""
    return {f"line_{index:06d}": line
            for index, line in enumerate(
                (line for line in report.splitlines(keepends=True) if line.strip()), 1)}


def _fragment_request(report: str) -> tuple[str, dict[str, str]]:
    base = build_candidate_prompt("candidate_2", report)
    instructions, marker, copied_report = base.partition("\n\nREPORT:\n")
    if not marker or copied_report != report:
        raise ValueError("SIWC prompt template is incompatible")
    catalogue = _source_line_catalogue(report)
    prompt = (instructions + _FRAGMENT_INSTRUCTIONS + "\n\nSOURCE_FRAGMENTS_JSON:\n"
              + json.dumps(catalogue, ensure_ascii=True) + marker + copied_report)
    return prompt, catalogue


def build_siwc_prompt(report: str) -> str:
    """Build the SIWC-only request; the frozen E007 registry remains unchanged."""
    return _fragment_request(report)[0]


def _resolve_fragment_result(value: object, catalogue: Mapping[str, str]) -> dict[str, dict[str, object]]:
    """Resolve only this request's source IDs into the unchanged public schema."""
    if not isinstance(value, dict) or set(value) != set(LABEL_COLUMNS):
        raise LabelingError("Model result must contain exactly the 12 target fields.",
                            cause_code="target_set_invalid")
    resolved = {}
    for column in LABEL_COLUMNS:
        item = value[column]
        if not isinstance(item, dict) or set(item) != {"label", "evidence_id", "uncertainty"}:
            raise LabelingError("Model target has an invalid fragment structure.",
                                cause_code="item_shape_invalid")
        evidence_id = item["evidence_id"]
        if evidence_id is None:
            evidence = ""
        elif type(evidence_id) is not str:
            raise LabelingError("Model evidence ID must be a string or null.",
                                cause_code="evidence_type_invalid")
        elif evidence_id not in catalogue:
            raise LabelingError("Model evidence ID is not in the current source catalogue.",
                                cause_code="evidence_not_grounded")
        else:
            evidence = catalogue[evidence_id]
        resolved[column] = {"label": item["label"], "evidence": evidence,
                            "uncertainty": item["uncertainty"]}
    return validate_label_result(resolved)


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
        ("stream_completed", getattr(error, "stream_completed", None),
         lambda value: type(value) is bool),
        ("stream_delta_chars", getattr(error, "stream_delta_chars", None),
         lambda value: type(value) is int and 0 <= value <= 2_147_483_647),
        ("stream_final_chars", getattr(error, "stream_final_chars", None),
         lambda value: type(value) is int and 0 <= value <= 2_147_483_647),
        ("stream_text_source", getattr(error, "stream_text_source", None),
         lambda value: type(value) is str and value in _SAFE_STREAM_TEXT_SOURCES),
        ("stream_refusal", getattr(error, "stream_refusal", None),
         lambda value: type(value) is bool),
        ("stream_format", getattr(error, "stream_format", None),
         lambda value: type(value) is str and value in _SAFE_STREAM_FORMATS),
        ("invalid_label_json_type", getattr(error, "invalid_label_json_type", None),
         lambda value: type(value) is str and value in _SAFE_JSON_VALUE_TYPES),
        ("evidence_mismatch_kind", getattr(error, "evidence_mismatch_kind", None),
         lambda value: type(value) is str and value in _SAFE_EVIDENCE_MISMATCH_KINDS),
    )
    return " ".join(f"{name}={value}" for name, value, is_safe in values if is_safe(value))


def _evidence_case_key(character: str) -> str:
    """Use single-codepoint caseless equivalences without fold expansions."""
    folded = character.casefold()
    return folded if len(folded) == 1 else character.lower()


def _normalized_evidence_with_offsets(
    text: str, *, ignore_case: bool,
) -> tuple[str, list[int], list[int]]:
    """Normalize case and whitespace while mapping every output character to source offsets."""
    normalized: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    index = 0
    while index < len(text):
        if text[index].isspace():
            run_start = index
            while index < len(text) and text[index].isspace():
                index += 1
            if normalized and index < len(text):
                normalized.append(" ")
                starts.append(run_start)
                ends.append(index)
            continue

        token_start = index
        while index < len(text) and not text[index].isspace():
            index += 1
        token = text[token_start:index]
        normalized_characters = [
            _evidence_case_key(character) if ignore_case else character
            for character in token
        ]
        lowered = "".join(normalized_characters)
        source_widths = [len(character) for character in normalized_characters]
        if sum(source_widths) == len(lowered):
            output_start = 0
            for offset, width in enumerate(source_widths):
                for character in lowered[output_start:output_start + width]:
                    normalized.append(character)
                    starts.append(token_start + offset)
                    ends.append(token_start + offset + 1)
                output_start += width
        else:
            # Keep contextual mappings conservative: a partial token match will
            # fail source-candidate equality instead of producing a bad span.
            for character in lowered:
                normalized.append(character)
                starts.append(token_start)
                ends.append(index)
    return "".join(normalized), starts, ends


def _is_evidence_word_character(character: str) -> bool:
    category = unicodedata.category(character)
    return character == "_" or category[0] in {"L", "N", "M"}


def _evidence_word_spans(text: str, *, offset: int = 0) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    start: int | None = None
    for index, character in enumerate(text):
        if _is_evidence_word_character(character):
            if start is None:
                start = index
        elif start is not None:
            spans.append((text[start:index], offset + start, offset + index))
            start = None
    if start is not None:
        spans.append((text[start:], offset + start, offset + len(text)))
    return spans


def _unique_minimal_context_source_span(evidence: str, report: str) -> str | None:
    """Recover a unique minimal ordered word window within one sentence/line."""
    query_terms = [
        "".join(_evidence_case_key(character) for character in term)
        for term, _, _ in _evidence_word_spans(evidence)
    ]
    if len(query_terms) < 2:
        return None

    candidate_spans: dict[tuple[int, int], int] = {}
    matching_segments = 0
    segment_start = 0
    boundaries = {".", "!", "?", "\r", "\n"}
    segment_ranges = []
    for index, character in enumerate(report):
        if character in boundaries:
            if segment_start < index:
                segment_ranges.append((segment_start, index))
            segment_start = index + 1
    if segment_start < len(report):
        segment_ranges.append((segment_start, len(report)))

    for segment_start, segment_end in segment_ranges:
        segment = report[segment_start:segment_end]
        source_words = _evidence_word_spans(segment, offset=segment_start)
        keys = [
            "".join(_evidence_case_key(character) for character in word)
            for word, _, _ in source_words
        ]
        positions: dict[str, list[int]] = {}
        for word_index, key in enumerate(keys):
            positions.setdefault(key, []).append(word_index)

        segment_spans: dict[tuple[int, int], int] = {}
        for first_index in positions.get(query_terms[0], []):
            last_index = first_index
            for query_term in query_terms[1:]:
                term_positions = positions.get(query_term, [])
                next_position = bisect_right(term_positions, last_index)
                if next_position == len(term_positions):
                    break
                last_index = term_positions[next_position]
            else:
                # Earliest completion is the shortest window for this start.
                # Keep zero-gap candidates: filtering them first could admit
                # a longer, misleading window instead of rejecting the minimum.
                source_start = source_words[first_index][1]
                source_end = source_words[last_index][2]
                segment_spans[(source_start, source_end)] = last_index - first_index + 1

        if segment_spans:
            matching_segments += 1
            if matching_segments > 1:
                return None
            candidate_spans.update(segment_spans)

    if not candidate_spans:
        return None
    minimum_word_count = min(candidate_spans.values())
    minimal_spans = [span for span, count in candidate_spans.items()
                     if count == minimum_word_count]
    if len(minimal_spans) != 1 or minimum_word_count <= len(query_terms):
        return None
    source_start, source_end = minimal_spans[0]
    return report[source_start:source_end]


def _earliest_boundary_aware_source_span(
    evidence: str, report: str, *, ignore_case: bool,
) -> str | None:
    """Find the earliest normalized substring span without splitting source characters or words."""
    if not isinstance(evidence, str) or not isinstance(report, str):
        return None
    needle, _, _ = _normalized_evidence_with_offsets(evidence, ignore_case=ignore_case)
    haystack, source_starts, source_ends = _normalized_evidence_with_offsets(
        report, ignore_case=ignore_case,
    )
    if not needle:
        return None

    edge = 0
    while True:
        found = haystack.find(needle, edge)
        if found < 0:
            return None
        match_end = found + len(needle)
        source_start = source_starts[found]
        source_end = source_ends[match_end - 1]

        # A lowercased source character can expand (for example, U+0130).
        # Never return a span that captures only part of that source character.
        starts_inside_expansion = (
            found > 0
            and source_starts[found - 1] == source_start
            and source_ends[found - 1] == source_ends[found]
        )
        ends_inside_expansion = (
            match_end < len(haystack)
            and source_starts[match_end] == source_starts[match_end - 1]
            and source_ends[match_end] == source_end
        )

        before_is_word = source_start > 0 and _is_evidence_word_character(report[source_start - 1])
        after_is_word = source_end < len(report) and _is_evidence_word_character(report[source_end])
        starts_with_word = _is_evidence_word_character(evidence.lstrip()[0])
        ends_with_word = _is_evidence_word_character(evidence.rstrip()[-1])
        if (not starts_inside_expansion and not ends_inside_expansion
                and not (starts_with_word and before_is_word)
                and not (ends_with_word and after_is_word)):
            source_span = report[source_start:source_end]
            candidate, _, _ = _normalized_evidence_with_offsets(
                source_span, ignore_case=ignore_case,
            )
            if candidate == needle:
                return source_span
        edge = found + 1


def _classify_evidence_mismatch(evidence: str, report: object) -> str:
    """Classify only a boundary-valid case/whitespace match after literal rejection."""
    if not isinstance(report, str):
        return "no_literal_match"
    if _earliest_boundary_aware_source_span(evidence, report, ignore_case=False) is not None:
        return "whitespace_only"
    if _earliest_boundary_aware_source_span(evidence, report, ignore_case=True) is not None:
        return "case_only"
    if (canonical := unicodedata.normalize("NFC", evidence)) and canonical in unicodedata.normalize("NFC", report):
        return "unicode_only"
    return "no_literal_match"


def _recover_fresh_boundary_aware_evidence(
    result: Mapping[str, Mapping[str, object]], report: str,
) -> Mapping[str, Mapping[str, object]]:
    """Copy a result with each recoverable fresh evidence span mapped to exact report text."""
    updates: dict[str, str] = {}
    for target_name, target in result.items():
        evidence = target["evidence"]
        if not evidence or evidence in report:
            continue
        source_span = _earliest_boundary_aware_source_span(evidence, report, ignore_case=True)
        if source_span is None:
            source_span = _unique_minimal_context_source_span(evidence, report)
        if source_span is not None:
            updates[target_name] = source_span
    if not updates:
        return result
    recovered: dict[str, dict[str, object]] = {
        target_name: dict(target) for target_name, target in result.items()
    }
    for target_name, evidence in updates.items():
        recovered[target_name]["evidence"] = evidence
    return recovered


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

    def __call__(self, report: str, *,
                 on_rejected_evidence: Callable[[str], object] | None = None
                 ) -> dict[str, dict[str, object]]:
        if not isinstance(report, str) or not report.strip():
            raise LabelingError("Report input is blank or invalid.")
        try:
            return self._authenticator.with_plan_access_token(
                self._profile_id,
                lambda access_token: self._infer_authorized(
                    access_token, report, on_rejected_evidence=on_rejected_evidence,
                ),
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
        except AuthError as exc:
            message = "A valid ChatGPT plan-use authorization is required."
            diagnostics = _format_safe_auth_diagnostics(exc._diagnostics)
            if diagnostics:
                message += f" Diagnostics: {diagnostics}"
            raise LabelingError(message, stage="inference",
                                cause_code="auth_error") from None
        except Exception:
            raise LabelingError("The account model catalog could not be loaded.",
                                stage="inference", cause_code="network_error") from None

    @staticmethod
    def validate_source_evidence(
        result: Mapping[str, Mapping[str, object]], report: str, *,
        on_rejected_evidence: Callable[[str], object] | None = None,
    ) -> None:
        """Reject non-empty evidence not literally present in the exact report text."""
        if not isinstance(report, str):
            raise LabelingError("Responses evidence could not be checked against the source report.",
                                stage="response_validation", cause_code="evidence_not_grounded")
        for target in result.values():
            evidence = target["evidence"]
            if evidence and evidence not in report:
                mismatch_kind = _classify_evidence_mismatch(evidence, report)
                if callable(on_rejected_evidence):
                    try:
                        on_rejected_evidence(evidence)
                    except Exception:
                        # Never let capture failure replace the safe validation error.
                        pass
                raise LabelingError("Responses evidence was not found in the source report.",
                                    stage="response_validation", cause_code="evidence_not_grounded",
                                    evidence_mismatch_kind=mismatch_kind)

    def _list_models_authorized(self, access_token: str) -> list[dict[str, str]]:
        client = None
        try:
            client = self._client_factory(
                api_key=access_token,
                base_url=RESPONSES_BASE_URL,
                max_retries=0,
            )
            catalog = client.get("/models", cast_to=dict[str, Any])
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

    def _infer_authorized(
        self, access_token: str, report: str, *,
        on_rejected_evidence: Callable[[str], object] | None = None,
    ) -> dict[str, dict[str, object]]:
        client = None
        try:
            client = self._client_factory(
                api_key=access_token,
                base_url=RESPONSES_BASE_URL,
                max_retries=0,
            )
            catalog = client.get("/models", cast_to=dict[str, Any])
            if not self._model_is_available(catalog):
                raise LabelingError(
                    "The requested model is not available to the selected account.",
                    stage="inference", cause_code="model_error",
                )
            prompt, catalogue = _fragment_request(report)
            stream = client.responses.create(
                model=self._model,
                input=[{"role": "user", "content": prompt}],
                store=False,
                stream=True,
            )
            delta_parts = []
            final_text = None
            completed = False
            refusal = False
            try:
                with stream as events:
                    for event in events:
                        event_type = getattr(event, "type", None)
                        if event_type in {"response.refusal.delta", "response.refusal.done"}:
                            refusal = True
                        if event_type == "response.output_text.delta":
                            delta = getattr(event, "delta", None)
                            if isinstance(delta, str):
                                delta_parts.append(delta)
                        if event_type in {"response.failed", "response.incomplete", "error"}:
                            response = getattr(event, "response", None)
                            error = getattr(response, "error", None)
                            if error is None:
                                error = getattr(event, "error", None)
                            final_text = _response_text(response)
                            refusal = refusal or _response_has_refusal(response)
                            observed_text = final_text or "".join(delta_parts)
                            raise _provider_error(
                                "Responses inference did not complete successfully.",
                                error,
                                response=response,
                                default_cause="model_error",
                                stream_completed=False,
                                stream_delta_chars=sum(map(len, delta_parts)),
                                stream_final_chars=len(final_text) if final_text is not None else None,
                                stream_text_source="none",
                                stream_refusal=refusal,
                                stream_format=_classify_response_format(observed_text, refusal=refusal),
                            )
                        if event_type == "response.completed":
                            response = getattr(event, "response", None)
                            final_text = _response_text(response)
                            refusal = refusal or _response_has_refusal(response)
                            completed = True
            except LabelingError:
                raise
            except Exception as exc:
                delta_text = "".join(delta_parts)
                raise _provider_error(
                    "Responses inference could not be completed.", exc,
                    stream_completed=False,
                    stream_delta_chars=len(delta_text),
                    stream_final_chars=None,
                    stream_text_source="none",
                    stream_refusal=refusal,
                    stream_format=_classify_response_format(delta_text, refusal=refusal),
                ) from None
            delta_text = "".join(delta_parts)
            delta_chars = len(delta_text)
            final_chars = len(final_text) if final_text is not None else None
            selected_source = ("completed" if isinstance(final_text, str) and final_text
                               else "deltas" if delta_text else "none")
            selected_text = final_text if selected_source == "completed" else delta_text
            format_class = _classify_response_format(
                selected_text, refusal=refusal,
            )
            if not completed:
                raise LabelingError(
                    "Responses stream ended without response.completed.",
                    stage="inference", cause_code="network_error",
                    stream_completed=False, stream_delta_chars=delta_chars,
                    stream_final_chars=final_chars, stream_text_source="none",
                    stream_refusal=refusal, stream_format=format_class,
                )
            if refusal:
                raise LabelingError(
                    "Responses returned a refusal instead of label data.",
                    stage="response_validation", cause_code="model_error",
                    stream_completed=True, stream_delta_chars=delta_chars,
                    stream_final_chars=final_chars, stream_text_source="none",
                    stream_refusal=True, stream_format="refusal",
                )
            if format_class in {"empty", "fenced"}:
                raise LabelingError(
                    "Responses output was not valid structured label data.",
                    stage="response_validation", cause_code="malformed_json",
                    stream_completed=True, stream_delta_chars=delta_chars,
                    stream_final_chars=final_chars, stream_text_source=selected_source,
                    stream_refusal=False, stream_format=format_class,
                )
            try:
                parsed = json.loads(selected_text)
            except (TypeError, ValueError, json.JSONDecodeError):
                raise LabelingError("Responses output was not valid structured label data.",
                                    stage="response_validation",
                                    cause_code="malformed_json",
                                    stream_completed=True, stream_delta_chars=delta_chars,
                                    stream_final_chars=final_chars, stream_text_source=selected_source,
                                    stream_refusal=False, stream_format=format_class) from None
            try:
                result = _resolve_fragment_result(parsed, catalogue)
                self.validate_source_evidence(
                    result, report, on_rejected_evidence=on_rejected_evidence,
                )
                return result
            except LabelingError as exc:
                raise LabelingError("Responses output failed local label validation.",
                                    stage="response_validation",
                                    cause_code=exc.cause_code,
                                    stream_completed=True, stream_delta_chars=delta_chars,
                                    stream_final_chars=final_chars, stream_text_source=selected_source,
                                    stream_refusal=False, stream_format="json_valid",
                                    invalid_label_json_type=exc.invalid_label_json_type,
                                    evidence_mismatch_kind=exc.evidence_mismatch_kind) from None
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


def _provider_error(message: str, error: object, *, response: object = None,
                    default_cause: str = "network_error",
                    stream_completed: bool | None = None,
                    stream_delta_chars: int | None = None,
                    stream_final_chars: int | None = None,
                    stream_text_source: str | None = None,
                    stream_refusal: bool | None = None,
                    stream_format: str | None = None) -> LabelingError:
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
        stream_completed=stream_completed,
        stream_delta_chars=stream_delta_chars,
        stream_final_chars=stream_final_chars,
        stream_text_source=stream_text_source,
        stream_refusal=stream_refusal,
        stream_format=stream_format,
    )


def _safe_int(value: object) -> int | None:
    if type(value) is int and 100 <= value <= 599:
        return value
    return None


def _response_text(response: object) -> str | None:
    text = getattr(response, "output_text", None)
    return text if isinstance(text, str) else None


def _response_has_refusal(response: object) -> bool:
    output = getattr(response, "output", None)
    if not isinstance(output, (list, tuple)):
        return False
    for item in output:
        content = getattr(item, "content", None)
        if isinstance(item, Mapping):
            content = item.get("content")
        if not isinstance(content, (list, tuple)):
            continue
        for block in content:
            block_type = block.get("type") if isinstance(block, Mapping) else getattr(block, "type", None)
            if block_type == "refusal":
                return True
    return False


def _classify_response_format(text: str, *, refusal: bool) -> str:
    if refusal:
        return "refusal"
    if not text.strip():
        return "empty"
    if text.lstrip().startswith("```"):
        return "fenced"
    try:
        json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return "json_invalid"
    return "json_valid"
