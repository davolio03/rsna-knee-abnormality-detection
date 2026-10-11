"""Local Gold-58 prompt comparison orchestration and sanitized output."""

from __future__ import annotations

import json
import os
import tempfile
import argparse
import csv
from pathlib import Path
from collections.abc import Callable, Mapping, Sequence
from numbers import Real
from typing import Any

from rsna_knee.labels.prompt_comparison import (
    CANDIDATE_IDS,
    COMPARISON_MODEL,
    COMPARISON_REASONING_EFFORT,
    run_nested_prompt_comparison,
)
from rsna_knee.labels.subscription import LABEL_COLUMNS, REPORT_COLUMN, STUDY_COLUMN, codex_cli_infer
from rsna_knee.validation.multilabel import nested_multilabel_splits


_SUMMARY_NAME = "gold58-prompt-comparison.json"
_OUTPUT_SCHEMA = {"predictions": dict}
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; both transfer and subscription confirmations are mandatory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--study-manifest", type=Path, required=True)
    parser.add_argument(
        "--confirm-transfer-prerequisites", action="store_true", required=True,
        help="Assert that project authorization, applicable terms, and competition constraints were independently confirmed.",
    )
    parser.add_argument(
        "--subscription-oauth-confirmed", action="store_true", required=True,
        help="Assert Codex CLI uses the intended ChatGPT subscription OAuth account.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        rows = _read_comparison_csv(args.input)
        manifest_keys = _read_study_manifest_csv(args.study_manifest)
        summary = run_prompt_comparison(
            rows,
            output_path=args.output or REPOSITORY_ROOT / "data" / "outputs" / _SUMMARY_NAME,
            project_root=REPOSITORY_ROOT,
            transfer_prerequisites_confirmed=args.confirm_transfer_prerequisites,
            subscription_oauth_confirmed=args.subscription_oauth_confirmed,
            study_manifest_keys=manifest_keys,
        )
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        # Error messages are fixed and never include row contents or provider payloads.
        parser.error(str(exc) if isinstance(exc, ValueError) else "comparison input could not be read")
    print(f"Local comparison summary written for {len(summary['outer_folds'])} outer folds.")
    return 0


def _read_comparison_csv(path: Path) -> tuple[dict[str, Any], ...]:
    required = {STUDY_COLUMN, REPORT_COLUMN, *LABEL_COLUMNS}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            headers = next(reader, None)
            if headers is None:
                raise ValueError("comparison input schema is invalid")
            if len(set(headers)) != len(headers):
                raise ValueError("duplicate CSV header names are not allowed")
            if set(headers) != required:
                raise ValueError("comparison input schema is invalid")
            rows = []
            for values in reader:
                if len(values) != len(headers):
                    raise ValueError("comparison input row schema is invalid")
                converted = dict(zip(headers, values))
                for target in LABEL_COLUMNS:
                    value = converted[target].strip()
                    if value not in {"0", "1"}:
                        raise ValueError("comparison targets must be encoded as 0 or 1")
                    converted[target] = value == "1"
                rows.append(converted)
    except (OSError, UnicodeError, csv.Error):
        raise ValueError("comparison input could not be read") from None
    return tuple(rows)


def _read_study_manifest_csv(path: Path) -> tuple[str, ...]:
    root = REPOSITORY_ROOT.resolve()
    private_root = root / "data"
    canonical_path = private_root / "private" / "gold58-study-keys.csv"
    manifest_path = Path(path)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    if manifest_path.resolve(strict=False) != canonical_path.resolve(strict=False):
        raise ValueError("canonical private study manifest path is required")
    manifest_path = manifest_path.absolute()
    try:
        manifest_path.relative_to(private_root.absolute())
    except ValueError:
        raise ValueError("private study manifest must be under ignored private data") from None
    if private_root.exists() and private_root.resolve() != private_root:
        raise ValueError("private study manifest path is unsafe")
    if manifest_path.is_symlink():
        raise ValueError("private study manifest path is unsafe")
    try:
        manifest_path.resolve().relative_to(private_root.resolve())
    except (OSError, ValueError):
        raise ValueError("private study manifest path is unsafe") from None
    try:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            if next(reader, None) != [STUDY_COLUMN]:
                raise ValueError("private study manifest schema is invalid")
            keys = []
            for row in reader:
                if len(row) != 1:
                    raise ValueError("private study manifest schema is invalid")
                keys.append(row[0])
    except (OSError, UnicodeError, csv.Error):
        raise ValueError("private study manifest could not be read") from None
    return tuple(keys)


def run_prompt_comparison(
    rows: Sequence[Mapping[str, Any]],
    *,
    infer: Callable[..., Mapping[str, Any]] = codex_cli_infer,
    output_path: str | Path | None = None,
    project_root: str | Path | None = None,
    transfer_prerequisites_confirmed: bool = False,
    subscription_oauth_confirmed: bool = False,
    study_manifest_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run fixed nested evaluation only after explicit operator confirmations.

    Confirmations are assertions by the caller, not automated legal/provider
    clearance. This function never discovers authorization or checks terms.
    Returned and persisted results contain only fold-level aggregate scores.
    """
    if transfer_prerequisites_confirmed is not True or subscription_oauth_confirmed is not True:
        raise ValueError("transfer and subscription prerequisites must be confirmed before execution")
    if not callable(infer):
        raise ValueError("inference callback is required")
    validated = _validate_rows(rows)
    keys = tuple(row[STUDY_COLUMN] for row in validated)
    _validate_manifest_membership(keys, study_manifest_keys)
    targets = tuple(tuple(int(row[target]) for target in LABEL_COLUMNS) for row in validated)
    reports = {row[STUDY_COLUMN]: row[REPORT_COLUMN] for row in validated}
    labels = {
        row[STUDY_COLUMN]: {target: bool(row[target]) for target in LABEL_COLUMNS}
        for row in validated
    }
    splits = nested_multilabel_splits(keys, targets, outer_n_splits=3, inner_n_splits=3, seed=42)
    _validate_fold_feasibility(splits, labels)
    folds = []

    for outer_split in splits:
        outer_metrics: dict[str, Any] = {}
        inner_metrics: dict[int, dict[str, Any]] = {}
        validation_fold_index = {
            frozenset(split.validation): index
            for index, split in enumerate(outer_split.inner_splits)
        }
        active_candidate: dict[str, str | None] = {"candidate_id": None}

        def payload_factory(_train_keys: Sequence[str], evaluation_keys: Sequence[str]) -> Mapping[str, Any]:
            return {"reports": {key: reports[key] for key in evaluation_keys}}

        def evaluator(candidate_id: str, payload: Mapping[str, Any], _schema: Mapping[str, type]) -> Mapping[str, Any]:
            active_candidate["candidate_id"] = candidate_id
            predictions = {}
            for key, report in payload["reports"].items():
                result = infer(
                    report,
                    candidate_id=candidate_id,
                    transfer_prerequisites_confirmed=True,
                    subscription_oauth_confirmed=True,
                )
                predictions[key] = _validate_prediction(result)
            return {"predictions": predictions}

        def scorer(output: Mapping[str, Any], evaluation_keys: Sequence[str]) -> float:
            metrics = _fold_metrics(output["predictions"], labels, evaluation_keys)
            if set(evaluation_keys) == set(outer_split.test):
                outer_metrics.update(metrics)
            else:
                index = validation_fold_index[frozenset(evaluation_keys)]
                candidate_id = active_candidate["candidate_id"]
                inner_metrics.setdefault(index, {})[candidate_id] = metrics
            return metrics["macro_balanced_accuracy"]

        result = run_nested_prompt_comparison(
            outer_split,
            payload_factory=payload_factory,
            evaluator=evaluator,
            scorer=scorer,
            output_schema=_OUTPUT_SCHEMA,
        )
        folds.append(
            {
                "selected_candidate_id": result.selected_candidate_id,
                "inner_scores": dict(result.inner_scores),
                "outer_score": result.outer_score,
                "inner_fold_metrics": [inner_metrics[index] for index in range(3)],
                "outer_per_target_balanced_accuracy": outer_metrics["per_target_balanced_accuracy"],
                "outer_macro_brier_score": outer_metrics["macro_brier_score"],
            }
        )

    summary = {
        "schema_version": 1,
        "protocol": "nested_study_level_3x3",
        "candidate_ids": list(CANDIDATE_IDS),
        "selection_metric": "mean_inner_macro_balanced_accuracy",
        "tie_break_order": list(CANDIDATE_IDS),
        "outer_fold_role": "evaluate_inner_selection_only",
        "model": COMPARISON_MODEL,
        "reasoning_effort": COMPARISON_REASONING_EFFORT,
        "outer_folds": folds,
        "provider_usage": "unavailable_unless_explicitly_reported",
    }
    if output_path is not None:
        write_comparison_summary(
            output_path, summary, project_root=project_root
        )
    return summary


def write_comparison_summary(
    path: str | Path,
    summary: Mapping[str, Any],
    *,
    project_root: str | Path | None = None,
) -> None:
    """Atomically write the aggregate-only summary under ignored data/outputs."""
    root = Path(project_root or REPOSITORY_ROOT).resolve()
    outputs = root / "data" / "outputs"
    destination = Path(path)
    if not destination.is_absolute():
        destination = root / destination
    destination = destination.absolute()
    try:
        destination.relative_to(outputs.absolute())
    except ValueError:
        raise ValueError("comparison output must be stored under ignored data/outputs") from None
    if destination.name != _SUMMARY_NAME:
        raise ValueError("comparison output filename must be gold58-prompt-comparison.json")
    if outputs.exists() and outputs.resolve() != outputs:
        raise ValueError("comparison output directory is unsafe")
    if destination.is_symlink():
        raise ValueError("comparison output path is unsafe")
    _validate_summary(summary)
    outputs.mkdir(parents=True, exist_ok=True)
    try:
        resolved = destination.resolve()
        resolved.relative_to(outputs.resolve())
    except (OSError, ValueError):
        raise ValueError("comparison output path is unsafe") from None
    serialized = json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n"
    fd, temp_name = tempfile.mkstemp(prefix=".gold58-comparison-", suffix=".tmp", dir=outputs)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, destination)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _validate_summary(summary: Mapping[str, Any]) -> None:
    expected = {
        "schema_version", "protocol", "candidate_ids", "selection_metric",
        "tie_break_order", "outer_fold_role", "model", "reasoning_effort",
        "outer_folds", "provider_usage",
    }
    if not isinstance(summary, Mapping) or set(summary) != expected:
        raise ValueError("comparison summary schema is invalid")
    constants = {
        "schema_version": 1,
        "protocol": "nested_study_level_3x3",
        "candidate_ids": list(CANDIDATE_IDS),
        "selection_metric": "mean_inner_macro_balanced_accuracy",
        "tie_break_order": list(CANDIDATE_IDS),
        "outer_fold_role": "evaluate_inner_selection_only",
        "model": COMPARISON_MODEL,
        "reasoning_effort": COMPARISON_REASONING_EFFORT,
        "provider_usage": "unavailable_unless_explicitly_reported",
    }
    if any(summary[key] != value for key, value in constants.items()):
        raise ValueError("comparison summary schema is invalid")
    folds = summary["outer_folds"]
    if not isinstance(folds, (list, tuple)) or len(folds) != 3:
        raise ValueError("comparison summary schema is invalid")
    for fold in folds:
        if not isinstance(fold, Mapping) or set(fold) != {
            "selected_candidate_id", "inner_scores", "outer_score", "inner_fold_metrics",
            "outer_per_target_balanced_accuracy", "outer_macro_brier_score",
        }:
            raise ValueError("comparison summary schema is invalid")
        if fold["selected_candidate_id"] not in CANDIDATE_IDS:
            raise ValueError("comparison summary schema is invalid")
        scores = fold["inner_scores"]
        if not isinstance(scores, Mapping) or set(scores) != set(CANDIDATE_IDS):
            raise ValueError("comparison summary schema is invalid")
        values = [*scores.values(), fold["outer_score"]]
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not 0 <= float(value) <= 1
            for value in values
        ):
            raise ValueError("comparison summary schema is invalid")
        target_metrics = fold["outer_per_target_balanced_accuracy"]
        if not isinstance(target_metrics, Mapping) or set(target_metrics) != set(LABEL_COLUMNS):
            raise ValueError("comparison summary schema is invalid")
        metric_values = [*target_metrics.values(), fold["outer_macro_brier_score"]]
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not 0 <= float(value) <= 1
            for value in metric_values
        ):
            raise ValueError("comparison summary schema is invalid")
        inner_folds = fold["inner_fold_metrics"]
        if not isinstance(inner_folds, (list, tuple)) or len(inner_folds) != 3:
            raise ValueError("comparison summary schema is invalid")
        for inner_fold in inner_folds:
            if not isinstance(inner_fold, Mapping) or set(inner_fold) != set(CANDIDATE_IDS):
                raise ValueError("comparison summary schema is invalid")
            for candidate_metrics in inner_fold.values():
                if not isinstance(candidate_metrics, Mapping) or set(candidate_metrics) != {
                    "macro_balanced_accuracy", "per_target_balanced_accuracy", "macro_brier_score"
                }:
                    raise ValueError("comparison summary schema is invalid")
                candidate_targets = candidate_metrics["per_target_balanced_accuracy"]
                if not isinstance(candidate_targets, Mapping) or set(candidate_targets) != set(LABEL_COLUMNS):
                    raise ValueError("comparison summary schema is invalid")
                numeric = [candidate_metrics["macro_balanced_accuracy"],
                           candidate_metrics["macro_brier_score"], *candidate_targets.values()]
                if any(isinstance(value, bool) or not isinstance(value, Real)
                       or not 0 <= float(value) <= 1 for value in numeric):
                    raise ValueError("comparison summary schema is invalid")


def _validate_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or len(rows) != 58:
        raise ValueError("comparison input must contain exactly 58 study rows")
    required = {STUDY_COLUMN, REPORT_COLUMN, *LABEL_COLUMNS}
    normalized = []
    seen = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != required:
            raise ValueError("comparison input row schema is invalid")
        key = row[STUDY_COLUMN]
        report = row[REPORT_COLUMN]
        if not isinstance(key, str) or not key.strip() or key in seen:
            raise ValueError("comparison study keys must be non-empty and unique")
        if not isinstance(report, str) or not report.strip():
            raise ValueError("comparison report text is missing")
        if any(type(row[target]) is not bool for target in LABEL_COLUMNS):
            raise ValueError("comparison targets must be boolean values")
        seen.add(key)
        normalized.append(row)
    return tuple(normalized)


def _validate_manifest_membership(
    input_keys: Sequence[str], manifest_keys: Sequence[str] | None
) -> None:
    if (
        manifest_keys is None
        or isinstance(manifest_keys, (str, bytes))
        or not isinstance(manifest_keys, Sequence)
        or len(manifest_keys) != 58
    ):
        raise ValueError("comparison cohort does not match private study manifest")
    if any(not isinstance(key, str) or not key.strip() for key in manifest_keys):
        raise ValueError("comparison cohort does not match private study manifest")
    try:
        manifest_set = set(manifest_keys)
        input_set = set(input_keys)
    except TypeError:
        raise ValueError("comparison cohort does not match private study manifest") from None
    if len(manifest_set) != 58 or len(input_set) != 58 or manifest_set != input_set:
        raise ValueError("comparison cohort does not match private study manifest")


def _validate_prediction(result: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if not isinstance(result, Mapping) or set(result) != set(LABEL_COLUMNS):
        raise ValueError("candidate output does not match the required label schema")
    clean = {}
    for target in LABEL_COLUMNS:
        value = result[target]
        if not isinstance(value, Mapping) or type(value.get("label")) is not bool:
            raise ValueError("candidate output does not match the required label schema")
        uncertainty = value.get("uncertainty")
        if isinstance(uncertainty, bool) or not isinstance(uncertainty, Real):
            raise ValueError("candidate output does not match the required label schema")
        uncertainty = float(uncertainty)
        if not 0 <= uncertainty <= 1:
            raise ValueError("candidate output does not match the required label schema")
        clean[target] = {"label": value["label"], "uncertainty": uncertainty}
    return clean


def _fold_metrics(
    predictions: Mapping[str, Mapping[str, Mapping[str, Any]]],
    labels: Mapping[str, Mapping[str, bool]],
    evaluation_keys: Sequence[str],
) -> dict[str, Any]:
    target_scores = []
    target_brier_scores = []
    for target in LABEL_COLUMNS:
        actual = [labels[key][target] for key in evaluation_keys]
        predicted = [predictions[key][target]["label"] for key in evaluation_keys]
        positives = sum(actual)
        negatives = len(actual) - positives
        if positives == 0 or negatives == 0:
            raise ValueError("every evaluation fold must contain positive and negative targets")
        sensitivity = sum(p and y for p, y in zip(predicted, actual)) / positives
        specificity = sum((not p) and (not y) for p, y in zip(predicted, actual)) / negatives
        target_scores.append((sensitivity + specificity) / 2)
        brier = sum(
            (predictions[key][target]["uncertainty"] - float(predictions[key][target]["label"] != labels[key][target])) ** 2
            for key in evaluation_keys
        ) / len(evaluation_keys)
        target_brier_scores.append(brier)
    return {
        "macro_balanced_accuracy": sum(target_scores) / len(target_scores),
        "per_target_balanced_accuracy": dict(zip(LABEL_COLUMNS, target_scores)),
        "macro_brier_score": sum(target_brier_scores) / len(target_brier_scores),
    }


def _validate_fold_feasibility(splits: Sequence[Any], labels: Mapping[str, Mapping[str, bool]]) -> None:
    for outer_split in splits:
        evaluation_partitions = [outer_split.test]
        evaluation_partitions.extend(inner.validation for inner in outer_split.inner_splits)
        for keys in evaluation_partitions:
            for target in LABEL_COLUMNS:
                observed = {labels[key][target] for key in keys}
                if observed != {False, True}:
                    raise ValueError(
                        "every inner and outer evaluation fold must contain positive and negative targets"
                    )
