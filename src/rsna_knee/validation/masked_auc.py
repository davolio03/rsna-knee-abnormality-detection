"""Competition-style ROC-AUC against partially observed soft report labels.

The competition metric is the mean of per-target ROC-AUC values. Local training
labels are soft probabilities extracted from radiology reports, and a target is
only *observed* when the report mentions it. For local evaluation each observed
soft label is binarized at ``threshold``; unobserved cells and labels exactly
equal to the threshold are excluded from that target's AUC.

This is a noisy proxy for the hidden expert labels, not an estimate of them.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from rsna_knee.validation.metrics import _auc


def binarize_soft_labels(soft: np.ndarray, observed: np.ndarray, threshold: float = 0.5):
    """Return ``(labels, valid)`` with labels in {0, 1} and ``valid`` marking usable cells."""
    soft = np.asarray(soft, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    if soft.shape != observed.shape or soft.ndim != 2:
        raise ValueError("soft and observed must be matching N-by-T matrices")
    valid = observed & (soft != threshold) & np.isfinite(soft)
    labels = (soft > threshold).astype(np.uint8)
    return labels, valid


def masked_multilabel_auc(soft: np.ndarray, observed: np.ndarray, scores: np.ndarray,
                          target_names: Sequence[str], threshold: float = 0.5) -> dict:
    """Per-target and macro ROC-AUC over observed, binarized soft labels.

    A target whose valid cells lack either class gets ``auc=None`` and is left
    out of the macro mean, which is reported together with the number of
    targets it averages.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels, valid = binarize_soft_labels(soft, observed, threshold)
    if scores.shape != labels.shape or scores.shape[1] != len(target_names):
        raise ValueError("scores must match labels and target names")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    per_target = []
    for column, name in enumerate(target_names):
        mask = valid[:, column]
        y, s = labels[mask, column], scores[mask, column]
        positives = int(y.sum())
        negatives = int(len(y) - positives)
        auc = float(_auc(y, s)) if positives and negatives else None
        per_target.append({"target": name, "auc": auc, "n": int(len(y)),
                           "positives": positives, "negatives": negatives})
    defined = [row["auc"] for row in per_target if row["auc"] is not None]
    return {"macro_auc": float(np.mean(defined)) if defined else None,
            "targets_in_macro": len(defined), "threshold": threshold,
            "per_target": per_target}
