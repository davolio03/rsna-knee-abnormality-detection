"""Training-only constant comparison on pinned, observed soft-label cells."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from rsna_knee.labels.full_cohort_prefilter import TARGET_COLUMNS


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("metric_invalid")
    return float(value)


def compare_constant_baseline(records, assignment, report, *, fold=0):
    """Fit each constant on observed training soft targets, never validation."""
    if type(fold) is not int or fold not in (0, 1, 2):
        raise ValueError("fold_invalid")
    ids = [row["study_id"] for row in records]
    if len(ids) != len(set(ids)) or set(ids) != set(assignment):
        raise ValueError("membership_invalid")
    if any(type(value) is not int or value not in (0, 1, 2) for value in assignment.values()):
        raise ValueError("membership_invalid")
    training = [row for row in records if assignment[row["study_id"]] != fold]
    validation = [row for row in records if assignment[row["study_id"]] == fold]
    if not training or not validation:
        raise ValueError("partition_empty")
    for row in records:
        if set(row["soft_targets"]) != set(TARGET_COLUMNS) or set(row["observed_mask"]) != set(TARGET_COLUMNS):
            raise ValueError("labels_invalid")
        for name in TARGET_COLUMNS:
            value = _number(row["soft_targets"][name])
            observed = row["observed_mask"][name]
            if type(observed) is not bool or not 0 <= value <= 1 or (not observed and value != .5):
                raise ValueError("labels_invalid")
    metrics = report["epochs"][-1]["validation"]
    if [entry["target"] for entry in metrics["per_target"]] != list(TARGET_COLUMNS):
        raise ValueError("target_order_invalid")
    details = []
    for index, name in enumerate(TARGET_COLUMNS):
        train = [row["soft_targets"][name] for row in training if row["observed_mask"][name]]
        valid = [row["soft_targets"][name] for row in validation if row["observed_mask"][name]]
        if not train or not valid:
            raise ValueError("target_support_missing")
        # Fixed numerical safeguard only, not validation-fitted calibration.
        probability = min(1 - 1e-12, max(1e-12, math.fsum(train) / len(train)))
        bce = math.fsum(-value * math.log(probability) - (1-value) * math.log1p(-probability)
                        for value in valid) / len(valid)
        brier = math.fsum((probability-value)**2 for value in valid) / len(valid)
        model = metrics["per_target"][index]
        if type(model["observed_count"]) is not int or model["observed_count"] != len(valid):
            raise ValueError("observed_count_mismatch")
        model_bce, model_brier = _number(model["masked_bce"]), _number(model["brier"])
        if model_bce < 0 or not 0 <= model_brier <= 1:
            raise ValueError("metric_invalid")
        details.append({"target": name, "training_observed_count": len(train),
                        "validation_observed_count": len(valid), "constant_probability": probability,
                        "constant_bce": bce, "constant_brier": brier,
                        "model_bce": model_bce, "model_brier": model_brier,
                        "model_minus_constant_bce": model_bce-bce,
                        "model_minus_constant_brier": model_brier-brier})
    total = sum(item["validation_observed_count"] for item in details)
    def pooled(key):
        return math.fsum(item[key] * item["validation_observed_count"] for item in details) / total
    if (type(metrics["observed_cell_count"]) is not int or metrics["observed_cell_count"] != total
            or not math.isclose(_number(metrics["masked_bce"]), pooled("model_bce"), abs_tol=1e-12)
            or not math.isclose(_number(metrics["brier"]), pooled("model_brier"), abs_tol=1e-12)):
        raise ValueError("aggregate_mismatch")
    return {"schema_version": "masked-constant-comparison-v1", "fold": fold,
            "training_study_count": len(training), "validation_study_count": len(validation),
            "constant_fit": "training_observed_soft_mean_clipped_1e-12",
            "constant": {"observed_cell_count": total, "masked_bce": pooled("constant_bce"),
                         "brier": pooled("constant_brier")},
            "model": {"observed_cell_count": total, "masked_bce": pooled("model_bce"),
                      "brier": pooled("model_brier")}, "per_target": details,
            "bce_targets_better_than_constant": sum(item["model_minus_constant_bce"] < 0 for item in details),
            "brier_targets_better_than_constant": sum(item["model_minus_constant_brier"] < 0 for item in details),
            "interpretation": "soft_report_label_fit_not_clinical_ground_truth_or_auc"}


def evaluate_masked_report(manifest_path, folds_path, report_path, expected_report_sha256):
    """Verify exact metadata/report provenance before an image-free comparison."""
    from rsna_knee.data.masked_soft_dataset import load_verified_soft_artifacts
    from rsna_knee.inference.masked import validate_masked_report
    raw = Path(report_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_report_sha256:
        raise ValueError("report_pin_mismatch")
    report = json.loads(raw)
    validate_masked_report(report)
    records, assignments, manifest_sha = load_verified_soft_artifacts(manifest_path, folds_path)
    result = compare_constant_baseline(records, assignments, report)
    result.update({"report_sha256": expected_report_sha256, "manifest_sha256": manifest_sha,
                   "folds_sha256": hashlib.sha256(Path(folds_path).read_bytes()).hexdigest()})
    return result
