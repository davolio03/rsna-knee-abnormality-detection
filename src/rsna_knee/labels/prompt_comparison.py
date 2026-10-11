"""Fail-closed boundaries for synthetic and nested prompt comparison."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

from rsna_knee.validation.multilabel import NestedSplit


CANDIDATE_IDS = ("candidate_1", "candidate_2", "candidate_3")


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
    except Exception:
        raise ValueError("candidate evaluation failed; details suppressed") from None
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
