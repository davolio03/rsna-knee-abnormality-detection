"""NumPy-only checked multilabel ROC-AUC metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rsna_knee.labels.subscription import LABEL_COLUMNS


@dataclass(frozen=True)
class MultilabelRocAuc:
    """ROC-AUC values in target order; undefined targets remain explicit."""

    target_names: tuple[str, ...]
    per_target: tuple[float | None, ...]
    macro_auc: float | None
    missing_class_targets: tuple[str, ...]


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    negatives = len(labels) - positives
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = float(ranks[labels == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def multilabel_roc_auc(labels: object, scores: object) -> MultilabelRocAuc:
    """Calculate binary ROC-AUC per target; macro is undefined if any is undefined."""
    try:
        y = np.asarray(labels)
        raw_scores = np.asarray(scores)
    except (TypeError, ValueError):
        raise ValueError("labels and scores must be numeric matrices") from None
    expected = (len(y), len(LABEL_COLUMNS)) if y.ndim == 2 else None
    if (y.ndim != 2 or y.shape[0] == 0 or y.shape[1] != len(LABEL_COLUMNS)
            or raw_scores.shape != expected):
        raise ValueError("labels and scores must be nonempty N-by-12 matrices")
    if y.dtype.kind not in "biuf" or raw_scores.dtype.kind not in "biuf":
        raise ValueError("labels and scores must use real numeric values")
    try:
        prediction = raw_scores.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("scores must be finite real numeric values") from None
    if not np.isfinite(prediction).all():
        raise ValueError("scores must be finite")
    if not np.isin(y, (0, 1, False, True)).all():
        raise ValueError("labels must contain only binary values")
    y = y.astype(np.uint8, copy=False)
    values: list[float | None] = []
    missing: list[str] = []
    for column, name in enumerate(LABEL_COLUMNS):
        target = y[:, column]
        if target.sum() == 0 or target.sum() == len(target):
            values.append(None)
            missing.append(name)
        else:
            values.append(float(_auc(target, prediction[:, column])))
    macro = None if missing else float(np.mean(values))
    return MultilabelRocAuc(tuple(LABEL_COLUMNS), tuple(values), macro, tuple(missing))
