"""Fail-closed boundaries for synthetic and nested prompt comparison."""

from __future__ import annotations

import math
import hashlib
import hmac
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Any
from pathlib import Path

from rsna_knee.labels.subscription import (
    CODEX_EXEC_TRANSPORT_VERSION,
    CODEX_OUTPUT_SCHEMA_VERSION,
    LABEL_COLUMNS,
    LabelingError,
)
from rsna_knee.validation.multilabel import NestedSplit


CANDIDATE_IDS = ("candidate_1", "candidate_2", "candidate_3")
COMPARISON_MODEL = "gpt-6-luna"
COMPARISON_REASONING_EFFORT = "high"

_SAFE_FAILURE_STAGES = {
    "process_spawn", "process_wait", "process_cleanup", "process_exit",
    "response_parse", "response_validation", "inference", "output_persist",
}
_SAFE_FAILURE_CAUSES = {
    "configuration_error", "arguments_error", "auth_error", "quota_error",
    "network_error", "model_error", "mcp_error", "sandbox_error", "timeout",
    "spawn_error", "invalid_response", "unknown",
    "malformed_json", "target_set_invalid", "item_shape_invalid",
    "label_type_invalid", "evidence_type_invalid", "evidence_length_invalid",
    "uncertainty_type_invalid", "uncertainty_range_invalid",
}
_SAFE_FAILURE_INTEGER_RANGE = (-65535, 65535)


class ComparisonEvaluationError(ValueError):
    """Fixed-text failure carrying only validated, allowlisted metadata."""

    def __init__(self, safe_metadata: Mapping[str, Any] | None = None):
        candidate = safe_metadata if isinstance(safe_metadata, Mapping) else {}
        reason = candidate.get("cause_code")
        metadata: dict[str, Any] = {
            "cause_code": reason if isinstance(reason, str) and reason in _SAFE_FAILURE_CAUSES else "unknown"
        }
        stage = candidate.get("stage")
        if isinstance(stage, str) and stage in _SAFE_FAILURE_STAGES:
            metadata["stage"] = stage
        for key in ("exit_status", "errno"):
            value = candidate.get(key)
            if type(value) is int and _SAFE_FAILURE_INTEGER_RANGE[0] <= value <= _SAFE_FAILURE_INTEGER_RANGE[1]:
                metadata[key] = value
        if metadata["cause_code"] == "unknown":
            metadata.pop("stage", None)
        super().__init__(f"candidate evaluation failed; reason_code={metadata['cause_code']}; details suppressed")
        self.safe_metadata = metadata


def _safe_failure_metadata(exc: Exception) -> dict[str, Any]:
    if type(exc) is not LabelingError:
        return {}
    stage = getattr(exc, "stage", None)
    cause = getattr(exc, "cause_code", None)
    result: dict[str, Any] = {}
    if isinstance(stage, str) and stage in _SAFE_FAILURE_STAGES:
        result["stage"] = stage
    if isinstance(cause, str) and cause in _SAFE_FAILURE_CAUSES:
        result["cause_code"] = cause
    for attr, key in (("exit_status", "exit_status"), ("errno_value", "errno")):
        value = getattr(exc, attr, None)
        if type(value) is int and -65535 <= value <= 65535:
            result[key] = value
    return result


def make_private_fingerprint(source: bytes, secret: bytes) -> str:
    """Return an HMAC fingerprint for private checkpoint binding."""
    if not isinstance(source, bytes) or not isinstance(secret, bytes) or len(secret) < 32:
        raise ValueError("checkpoint fingerprint inputs are invalid")
    return hmac.new(secret, source, hashlib.sha256).hexdigest()


class CheckpointStore:
    """Atomic private checkpoint containing only validated predictions and safe errors."""

    def __init__(self, path: str | Path, fingerprint: str, *, expected_coordinates: Sequence[str]):
        self.path = Path(path)
        self.fingerprint = fingerprint
        self.expected = set(expected_coordinates)
        if not self.expected or any(not _opaque_coordinate(x) for x in self.expected):
            raise ValueError("checkpoint coordinates are invalid")
        self.data = {"schema_version": 1, "fingerprint": fingerprint,
                     "completed": {}, "failures": {}}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if (not isinstance(loaded, dict) or set(loaded) != set(self.data)
                        or type(loaded["schema_version"]) is not int or loaded["schema_version"] != 1
                        or not isinstance(loaded["fingerprint"], str)
                        or loaded["fingerprint"] != fingerprint
                        or not isinstance(loaded["completed"], dict)
                        or not isinstance(loaded["failures"], dict)):
                    raise ValueError
                completed, failures = loaded["completed"], loaded["failures"]
                if (set(completed) - self.expected or set(failures) - self.expected
                        or set(completed) & set(failures)):
                    raise ValueError
                for coordinate, prediction in completed.items():
                    if not _valid_prediction_record(prediction):
                        raise ValueError
                    if not _opaque_coordinate(coordinate):
                        raise ValueError
                for coordinate, failure in failures.items():
                    if not _opaque_coordinate(coordinate) or not _valid_safe_failure(failure):
                        raise ValueError
                self.data = loaded
            except (OSError, UnicodeError, ValueError, TypeError):
                raise ValueError("comparison checkpoint is corrupt or incompatible") from None

    def success(self, coordinate: str) -> Mapping[str, Any] | None:
        return self.data["completed"].get(coordinate)

    def failure(self, coordinate: str) -> Mapping[str, Any] | None:
        return self.data["failures"].get(coordinate)

    def save_success(self, coordinate: str, prediction: Mapping[str, Any]) -> None:
        self._check_coordinate(coordinate)
        if not _valid_prediction_record(prediction):
            raise ValueError("checkpoint prediction is invalid")
        self.data["failures"].pop(coordinate, None)
        self.data["completed"][coordinate] = prediction
        self._write()

    def save_failure(self, coordinate: str, exc: Exception) -> None:
        self._check_coordinate(coordinate)
        metadata = _safe_failure_metadata(exc)
        self.data["failures"][coordinate] = metadata
        self._write()

    def _check_coordinate(self, coordinate: str) -> None:
        if coordinate not in self.expected or not _opaque_coordinate(coordinate):
            raise ValueError("checkpoint coordinate is invalid")

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(self.data, sort_keys=True, separators=(",", ":"), allow_nan=False)
        fd, temp_name = tempfile.mkstemp(prefix=".comparison-checkpoint-", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(serialized)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, self.path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)


def _opaque_coordinate(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _valid_prediction_record(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != set(LABEL_COLUMNS):
        return False
    return all(isinstance(item, Mapping) and set(item) == {"label", "uncertainty"}
               and type(item["label"]) is bool and type(item["uncertainty"]) in {int, float}
               and math.isfinite(item["uncertainty"]) and 0 <= item["uncertainty"] <= 1
               for item in value.values())


def _valid_safe_failure(value: Any) -> bool:
    return (isinstance(value, Mapping) and set(value).issubset(
        {"stage", "cause_code", "exit_status", "errno"})
        and ("stage" not in value or value["stage"] in _SAFE_FAILURE_STAGES)
        and ("cause_code" not in value or value["cause_code"] in _SAFE_FAILURE_CAUSES)
        and all(type(value[k]) is int and -65535 <= value[k] <= 65535
                for k in ("exit_status", "errno") if k in value))


@dataclass(frozen=True)
class PromptCandidateDefinition:
    """Immutable candidate-specific strategy text appended to shared instructions."""

    candidate_id: str
    strategy_block: str


PROMPT_REGISTRY: Mapping[str, PromptCandidateDefinition] = MappingProxyType(
    {
        "candidate_1": PromptCandidateDefinition("candidate_1", ""),
        "candidate_2": PromptCandidateDefinition(
            "candidate_2",
            "\n\nEvidence workflow:\n"
            "For each target, first locate the exact report passage that may support "
            "or contradict that target, then decide the binary label from the report. "
            "Do not infer a finding from unrelated text. If no supporting or "
            "contradicting passage is present, use an empty evidence string and do not "
            "invent evidence. Keep the required output schema unchanged.",
        ),
        "candidate_3": PromptCandidateDefinition(
            "candidate_3",
            "\n\nReport interpretation policy (illustrative hypothesis; requires "
            "approval):\n"
            "Interpret statements in context. Distinguish explicit negation from a "
            "positive finding; do not treat a negated finding as present. Do not "
            "convert tentative wording into a positive finding without explicit "
            "support. Do not treat a historical or resolved finding as a current "
            "finding unless the report explicitly establishes that it remains current. "
            "When the report contains no relevant evidence, return the required "
            "binary label as false and use empty evidence; the schema has no abstention "
            "value. Use the shared probability-of-error definition for the uncertainty "
            "field. Do not invent clinical criteria or target definitions. Keep the "
            "required output schema unchanged.",
        ),
    }
)
if tuple(PROMPT_REGISTRY) != CANDIDATE_IDS:
    raise RuntimeError("prompt registry order must match fixed candidate selection order")

_SHARED_PROMPT_TEMPLATE = (
    "Extract provisional binary findings for these targets:\n"
    "TARGETS:\n{target_list}\n\n"
    "Return only one JSON object with each target mapped to {label:boolean, "
    "evidence:string, uncertainty:number from 0 to 1}. Evidence must be a short "
    "exact report excerpt (maximum 200 characters); use an empty excerpt when "
    "unsupported. Do not add keys.\n\n"
    "For each target, set uncertainty to your estimated probability that its binary "
    "label is incorrect: 0 means you estimate the label is correct, and 1 means you "
    "estimate the label is incorrect."
)


def build_candidate_prompt(candidate_id: str, report: str) -> str:
    """Build one fixed E007 prompt without exposing report text in errors or logs."""
    if not isinstance(candidate_id, str) or candidate_id not in PROMPT_REGISTRY:
        raise ValueError("unknown prompt candidate")
    if not isinstance(report, str) or not report.strip():
        raise ValueError("report must be a non-empty string")
    definition = PROMPT_REGISTRY[candidate_id]
    target_list = "\n".join(f"- {target}" for target in LABEL_COLUMNS)
    return (
        _SHARED_PROMPT_TEMPLATE.replace("{target_list}", target_list)
        + definition.strategy_block
        + "\n\nREPORT:\n"
        + report
    )


@dataclass(frozen=True)
class NestedComparisonResult:
    """Safe comparison summary; raw evaluator output is intentionally omitted."""

    selected_candidate_id: str
    inner_scores: Mapping[str, float]
    outer_score: float


def select_candidate(inner_scores: Mapping[str, float]) -> str:
    """Select the highest-scoring predeclared candidate, using fixed tie order."""
    if not isinstance(inner_scores, Mapping):
        raise ValueError("inner candidate scores must be a mapping")
    if set(inner_scores) != set(CANDIDATE_IDS):
        raise ValueError(
            "candidate scores must contain exactly the predeclared candidate IDs"
        )

    scores = {}
    for candidate_id in CANDIDATE_IDS:
        score = inner_scores[candidate_id]
        if isinstance(score, bool) or not isinstance(score, Real):
            raise ValueError("candidate scores must be finite numbers in [0, 1]")
        score = float(score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("candidate scores must be finite numbers in [0, 1]")
        scores[candidate_id] = score
    return max(CANDIDATE_IDS, key=lambda candidate_id: scores[candidate_id])


def run_nested_prompt_comparison(
    outer_split: NestedSplit[Any],
    *,
    payload_factory: Callable[[Sequence[Any], Sequence[Any]], Mapping[str, Any]],
    evaluator: Callable[[str, Mapping[str, Any], Mapping[str, type]], Mapping[str, Any]],
    scorer: Callable[[Mapping[str, Any], Sequence[Any]], Real],
    output_schema: Mapping[str, type],
) -> NestedComparisonResult:
    """Evaluate candidates only on inner splits, then score the selected one outside.

    The inner callbacks receive only each inner training/validation partition.
    The outer test keys are passed to callbacks only after selection is complete,
    so outer results cannot affect candidate selection through this interface.
    Callback failures are replaced with fixed messages to avoid leaking payloads.
    """
    if not callable(payload_factory) or not callable(evaluator) or not callable(scorer):
        raise ValueError("payload factory, evaluator, and scorer callbacks are required")
    schema = _validate_schema(output_schema)
    _validate_nested_splits(outer_split)

    candidate_scores: dict[str, list[float]] = {
        candidate_id: [] for candidate_id in CANDIDATE_IDS
    }
    for inner_split in outer_split.inner_splits:
        for candidate_id in CANDIDATE_IDS:
            payload = _make_payload(
                payload_factory, inner_split.train, inner_split.validation
            )
            output = _evaluate(evaluator, candidate_id, payload, schema)
            candidate_scores[candidate_id].append(
                _score(scorer, output, inner_split.validation)
            )

    mean_scores = {
        candidate_id: sum(scores) / len(scores)
        for candidate_id, scores in candidate_scores.items()
    }
    selected = select_candidate(mean_scores)

    # This is the first point at which the outer test partition is exposed.
    outer_payload = _make_payload(payload_factory, outer_split.train, outer_split.test)
    outer_output = _evaluate(evaluator, selected, outer_payload, schema)
    outer_score = _score(scorer, outer_output, outer_split.test)
    return NestedComparisonResult(selected, dict(mean_scores), outer_score)


def _validate_schema(output_schema: Mapping[str, type]) -> dict[str, type]:
    if not isinstance(output_schema, Mapping) or not output_schema:
        raise ValueError("a non-empty output schema is required")
    if any(
        not isinstance(key, str) or not key or not isinstance(value, type)
        for key, value in output_schema.items()
    ):
        raise ValueError("output schema must map field names to types")
    return dict(output_schema)


def _validate_outer_split(outer_split: NestedSplit[Any]) -> None:
    if not isinstance(outer_split, NestedSplit):
        raise ValueError("a valid outer split is required")
    train = _unique_key_set(outer_split.train, "outer training")
    test = _unique_key_set(outer_split.test, "outer test")
    if not train or not test or train & test or not outer_split.inner_splits:
        raise ValueError("outer split partitions are invalid")


def _validate_inner_split(outer_split: NestedSplit[Any], inner_split: Any) -> None:
    if not hasattr(inner_split, "train") or not hasattr(inner_split, "validation"):
        raise ValueError("inner split partitions are invalid")
    train = _unique_key_set(outer_split.train, "outer training")
    test = _unique_key_set(outer_split.test, "outer test")
    inner_train = _unique_key_set(inner_split.train, "inner training")
    validation = _unique_key_set(inner_split.validation, "inner validation")
    if (
        not inner_train
        or not validation
        or inner_train & validation
        or inner_train | validation != train
        or (inner_train | validation) & test
    ):
        raise ValueError("inner split overlaps or escapes the outer training partition")


def _validate_nested_splits(outer_split: NestedSplit[Any]) -> None:
    _validate_outer_split(outer_split)
    validation_keys = []
    for inner_split in outer_split.inner_splits:
        _validate_inner_split(outer_split, inner_split)
        validation_keys.extend(inner_split.validation)
    validation_set = _unique_key_set(validation_keys, "inner validation folds")
    if validation_set != set(outer_split.train):
        raise ValueError("inner validation folds must partition the outer training keys")


def _unique_key_set(keys: Sequence[Any], partition_name: str) -> set[Any]:
    try:
        values = tuple(keys)
        key_set = set(values)
    except TypeError:
        raise ValueError(f"{partition_name} study keys must be hashable") from None
    if len(key_set) != len(values):
        raise ValueError(f"{partition_name} partition contains duplicate study keys")
    return key_set


def _make_payload(
    factory: Callable[..., Any], train: Sequence[Any], evaluation: Sequence[Any]
) -> Mapping[str, Any]:
    try:
        payload = factory(train, evaluation)
    except Exception:
        raise ValueError("payload construction failed; details suppressed") from None
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("payload must be a non-empty mapping")
    return payload


def _evaluate(
    evaluator: Callable[..., Any],
    candidate_id: str,
    payload: Mapping[str, Any],
    schema: Mapping[str, type],
) -> Mapping[str, Any]:
    try:
        output = evaluator(candidate_id, payload, schema)
    except Exception as exc:
        raise ComparisonEvaluationError(_safe_failure_metadata(exc)) from None
    if not isinstance(output, Mapping) or set(output) != set(schema):
        raise ValueError("candidate output does not match the required schema")
    if any(type(output[key]) is not expected for key, expected in schema.items()):
        raise ValueError("candidate output does not match the required schema")
    return output


def _score(
    scorer: Callable[..., Any],
    output: Mapping[str, Any],
    evaluation_keys: Sequence[Any],
) -> float:
    try:
        score = scorer(output, evaluation_keys)
    except Exception:
        raise ValueError("candidate scoring failed; details suppressed") from None
    if isinstance(score, bool) or not isinstance(score, Real):
        raise ValueError("scores must be finite numbers in [0, 1]")
    score = float(score)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("scores must be finite numbers in [0, 1]")
    return score
