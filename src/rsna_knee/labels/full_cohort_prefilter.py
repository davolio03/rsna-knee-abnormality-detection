"""Prepare a pinned, private soft-label inventory for the full non-Gold cohort."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from rsna_knee.data.cohort import (
    REPORT_NORMALIZATION_VERSION,
    _normalized_report_hash,
    _read_csv_rows,
    _series_by_study,
    _binary_flag,
    _TRAIN_COLUMNS,
    _SERIES_COLUMNS,
)
from rsna_knee.labels.report_source_comparison import (
    COHORT_SHA256,
    COHORT_SIZE,
    EXPECTED_NON_GOLD_STUDIES,
    EXPECTED_OFFICIAL_STUDIES,
    FOLDS_SHA256,
    FOLDS_SIZE,
    GOLD_IDS_SHA256,
    GOLD_IDS_SIZE,
    SOURCE_CSV_NAME,
    SOURCE_CSV_SHA256,
    SOURCE_CSV_SIZE,
    DATASET_CARD_URL,
    DATASET_VERSION,
    TARGET_COLUMNS,
    _identifier,
    _read_gold_ids,
    _read_official_study_ids,
    _read_pinned,
    _read_public_rows,
)
from rsna_knee.validation.multilabel import iterative_multilabel_folds


PREFILTER_POLICY_VERSION = "full-cohort-soft-label-prefilter-v1"
FOLD_SCHEMA_VERSION = "full-cohort-grouped-folds-v1"
N_SPLITS = 3
SEED = 42
WEIGHT_CAP = 3.0
EXPECTED_ALL_UNKNOWN = 51
EXPECTED_ELIGIBLE = EXPECTED_NON_GOLD_STUDIES - EXPECTED_ALL_UNKNOWN
EXPECTED_REPORT_GROUPS = 4175
EXPECTED_NON_GOLD_SERIES = 24035
_SHA256_TEXT = set("0123456789abcdef")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRIVATE_OUTPUT_ROOT = PROJECT_ROOT / "data" / "private" / "full-cohort-prefilter"


def _fail(code: str) -> None:
    raise ValueError(code)


def _validate_probabilities(targets: Mapping[str, object], masks: Mapping[str, object]) -> None:
    if set(targets) != set(TARGET_COLUMNS) or set(masks) != set(TARGET_COLUMNS):
        _fail("target_schema_invalid")
    for target in TARGET_COLUMNS:
        probability = targets[target]
        observed = masks[target]
        if type(observed) is not bool:
            _fail("observation_mask_invalid")
        if type(probability) is not float:
            _fail("probability_invalid")
        if not math.isfinite(float(probability)) or not 0.0 <= float(probability) <= 1.0:
            _fail("probability_out_of_range")
        if (float(probability) == 0.5) == observed:
            _fail("unknown_probability_must_be_masked")


def build_prefilter_manifest(rows: Sequence[Mapping[str, object]], *,
                             input_pins: Mapping[str, str] | None = None):
    """Build the private row inventory; unknown-only rows are audited, not supervised."""
    pins = dict(input_pins or {})
    for key, value in pins.items():
        if (not isinstance(key, str) or not isinstance(value, str) or len(value) != 64
                or any(char not in _SHA256_TEXT for char in value)):
            _fail("input_pin_invalid")
    seen: set[str] = set()
    seen_series: set[str] = set()
    records: list[dict[str, object]] = []
    audit_rows: list[dict[str, str]] = []
    for row in rows:
        study_id = _identifier(row.get("study_id"))
        if study_id in seen:
            _fail("duplicate_study_id")
        seen.add(study_id)
        report_hash = row.get("report_group_sha256")
        if (not isinstance(report_hash, str) or len(report_hash) != 64
                or any(char not in _SHA256_TEXT for char in report_hash)):
            _fail("report_group_hash_invalid")
        series = row.get("series")
        if not isinstance(series, list) or not series:
            _fail("series_metadata_incomplete")
        for item in series:
            expected_series_keys = {"series_id", "fluid_sensitive", "fat_suppression", "anatomical_plane"}
            if (not isinstance(item, Mapping) or set(item) != expected_series_keys
                    or not isinstance(item.get("series_id"), str)
                    or not item["series_id"] or item.get("anatomical_plane") not in {"Axial", "Coronal", "Sagittal"}
                    or type(item.get("fluid_sensitive")) is not int or item["fluid_sensitive"] not in (0, 1)
                    or type(item.get("fat_suppression")) is not int or item["fat_suppression"] not in (0, 1)):
                _fail("series_metadata_invalid")
            if item["series_id"] in seen_series:
                _fail("series_duplicate_id")
            seen_series.add(item["series_id"])
        targets, masks = row.get("soft_targets"), row.get("observed_mask")
        if not isinstance(targets, Mapping) or not isinstance(masks, Mapping):
            _fail("target_schema_invalid")
        _validate_probabilities(targets, masks)
        observed_count = sum(masks[target] for target in TARGET_COLUMNS)
        if observed_count == 0:
            audit_rows.append({"study_id": study_id, "reason_code": "all_targets_unknown"})
            continue
        record = {
            "study_id": study_id,
            "report_group_sha256": report_hash,
            "series": [dict(item) for item in series],
            "soft_targets": {target: float(targets[target]) for target in TARGET_COLUMNS},
            "observed_mask": {target: bool(masks[target]) for target in TARGET_COLUMNS},
            "observed_target_count": observed_count,
        }
        records.append(record)
        audit_rows.append({"study_id": study_id, "reason_code": "eligible_at_least_one_observed_target"})

    if not records:
        _fail("no_eligible_studies")
    manifest = {
        "schema_version": "full-cohort-soft-label-manifest-v1",
        "configuration": {
            "policy_version": PREFILTER_POLICY_VERSION,
            "source_dataset_card_url": DATASET_CARD_URL,
            "source_dataset_version": DATASET_VERSION,
            "source_file": SOURCE_CSV_NAME,
            "report_group_normalization": REPORT_NORMALIZATION_VERSION,
            "unknown_probability": 0.5,
            "eligible_rule": "at least one observed target and valid series metadata",
            "image_quality_status": "not_inspected",
        },
        "input_pins": pins,
        "target_order": list(TARGET_COLUMNS),
        "records": records,
        "audit_rows": audit_rows,
    }
    target_coverage = {
        target: sum(bool(row["observed_mask"][target]) for row in rows)
        for target in TARGET_COLUMNS
    }
    return manifest, {
        "non_gold_study_count": len(rows),
        "eligible_study_count": len(records),
        "excluded_all_targets_unknown": len(rows) - len(records),
        "report_group_count": len({record["report_group_sha256"] for record in records}),
        "series_count": sum(len(row["series"]) for row in rows),
        "target_addressed_counts": target_coverage,
    }


def _support(records: Sequence[Mapping[str, object]], ids: set[str]) -> dict[str, dict[str, int]]:
    by_id = {str(record["study_id"]): record for record in records}
    positive = Counter()
    negative = Counter()
    for study_id in ids:
        record = by_id[study_id]
        for target in TARGET_COLUMNS:
            if record["observed_mask"][target]:
                if record["soft_targets"][target] > 0.5:
                    positive[target] += 1
                elif record["soft_targets"][target] < 0.5:
                    negative[target] += 1
    return {"positive": {target: positive[target] for target in TARGET_COLUMNS},
            "negative": {target: negative[target] for target in TARGET_COLUMNS}}


def build_training_weights(records: Sequence[Mapping[str, object]],
                           training_study_ids: Sequence[str], *, cap: float = WEIGHT_CAP
                           ) -> dict[str, dict[str, float]]:
    """Compute capped per-class weights from observed training targets only."""
    if type(cap) not in (float, int) or isinstance(cap, bool) or not math.isfinite(float(cap)) or cap < 1:
        _fail("weight_cap_invalid")
    by_id = {str(record["study_id"]): record for record in records}
    ids = list(training_study_ids)
    if len(ids) != len(set(ids)) or any(study_id not in by_id for study_id in ids):
        _fail("training_membership_invalid")
    result: dict[str, dict[str, float]] = {}
    for target in TARGET_COLUMNS:
        positive = negative = 0
        for study_id in ids:
            record = by_id[study_id]
            if not record["observed_mask"][target]:
                continue
            if record["soft_targets"][target] > 0.5:
                positive += 1
            elif record["soft_targets"][target] < 0.5:
                negative += 1
        if not positive or not negative:
            _fail("training_class_support_missing")
        majority = max(positive, negative)
        result[target] = {
            "positive": max(1.0, min(float(cap), math.sqrt(majority / positive))),
            "negative": max(1.0, min(float(cap), math.sqrt(majority / negative))),
            "positive_count": positive,
            "negative_count": negative,
        }
    return result


def build_grouped_folds(manifest: Mapping[str, object]) -> dict[str, object]:
    """Assign normalized-report groups with 24 binary incidence features."""
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        _fail("manifest_records_invalid")
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        _validate_probabilities(record["soft_targets"], record["observed_mask"])
        grouped[str(record["report_group_sha256"])].append(record)
    group_ids = sorted(grouped)
    features = []
    for group_id in group_ids:
        members = grouped[group_id]
        row = []
        for target in TARGET_COLUMNS:
            row.append(int(any(member["observed_mask"][target]
                               and member["soft_targets"][target] > 0.5 for member in members)))
            row.append(int(any(member["observed_mask"][target]
                               and member["soft_targets"][target] < 0.5 for member in members)))
        features.append(row)
    try:
        group_folds = iterative_multilabel_folds(group_ids, features, N_SPLITS, SEED)
    except ValueError:
        _fail("group_fold_allocation_failed")
    group_to_fold = {group_id: fold for fold, values in enumerate(group_folds) for group_id in values}
    study_assignments = [{"study_id": str(record["study_id"]),
                          "report_group_sha256": str(record["report_group_sha256"]),
                          "fold": group_to_fold[str(record["report_group_sha256"])]}
                         for record in records]
    folds = []
    all_ids = {str(record["study_id"]) for record in records}
    for fold in range(N_SPLITS):
        validation_ids = {item["study_id"] for item in study_assignments if item["fold"] == fold}
        training_ids = all_ids - validation_ids
        validation_support = _support(records, validation_ids)
        training_support = _support(records, training_ids)
        if any(value <= 0 for side in (validation_support, training_support)
               for counts in side.values() for value in counts.values()):
            _fail("fold_class_support_missing")
        weights = build_training_weights(records, sorted(training_ids))
        folds.append({
            "fold": fold,
            "validation_study_count": len(validation_ids),
            "validation_group_count": len({item["report_group_sha256"] for item in study_assignments
                                            if item["fold"] == fold}),
            "training_study_count": len(training_ids),
            "training_group_count": len(group_ids) - len({item["report_group_sha256"]
                                                            for item in study_assignments
                                                            if item["fold"] == fold}),
            "validation_support": validation_support,
            "train_support": training_support,
            "training_only_class_weights": weights,
            "validation_study_ids": sorted(validation_ids),
            "train_study_ids": sorted(training_ids),
        })
    return {
        "schema_version": FOLD_SCHEMA_VERSION,
        "configuration": {"n_splits": N_SPLITS, "seed": SEED,
                          "group_key": "normalized_report_sha256",
                          "features": "24 observed positive/negative incidence columns",
                          "allocator": "iterative_multilabel_folds-v1"},
        "manifest_sha256": hashlib.sha256(_canonical_json(manifest)).hexdigest(),
        "study_assignments": study_assignments,
        "folds": folds,
        "intended_loss": {
            "formula": "mean over observed cells of mask * (w_pos*y*softplus(-logit) + w_neg*(1-y)*softplus(logit))",
            "weights": "per-target class weights computed from that fold's training partition only",
            "cap": WEIGHT_CAP,
            "validation": "unweighted, never resampled; no validation labels enter training weights",
        },
    }


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        _fail("input_unreadable")
    return digest.hexdigest()


def _read_reports(train_csv: Path, official_ids: set[str], gold_ids: set[str]) -> dict[str, str]:
    try:
        rows = _read_csv_rows(train_csv, _TRAIN_COLUMNS)
    except ValueError:
        _fail("official_train_schema_invalid")
    found: set[str] = set()
    reports: dict[str, str] = {}
    for row in rows:
        study_id = _identifier(row.get("StudyInstanceUID", "").strip())
        if study_id in found:
            _fail("official_train_duplicate_id")
        found.add(study_id)
        if study_id not in official_ids:
            _fail("official_train_foreign_id")
        # Do not normalize or otherwise inspect report text for excluded Gold studies.
        if study_id in gold_ids:
            continue
        report = row.get("Report")
        if not isinstance(report, str) or not report.strip():
            _fail("non_gold_report_missing")
        reports[study_id] = _normalized_report_hash(report)
    if found != official_ids or len(found) != EXPECTED_OFFICIAL_STUDIES:
        _fail("official_train_coverage_mismatch")
    return reports


def _read_series(train_series_csv: Path, official_ids: set[str], gold_ids: set[str]):
    try:
        rows = _read_csv_rows(train_series_csv, _SERIES_COLUMNS)
    except ValueError:
        _fail("series_metadata_invalid")
    seen_series: set[str] = set()
    for row in rows:
        study_id = _identifier(row.get("StudyInstanceUID", "").strip())
        series_id = _identifier(row.get("SeriesInstanceUID", "").strip())
        if study_id not in official_ids:
            _fail("series_foreign_study")
        if series_id in seen_series:
            _fail("series_duplicate_id")
        seen_series.add(series_id)
        plane = row.get("Anatomical_Plane", "").strip()
        if plane not in {"Axial", "Coronal", "Sagittal"}:
            _fail("series_plane_invalid")
        try:
            _binary_flag(row.get("Fluid_Sensitive", ""))
            _binary_flag(row.get("Fat_Suppression", ""))
        except ValueError:
            _fail("series_flag_invalid")
    non_gold = official_ids - gold_ids
    try:
        series_by_study = _series_by_study(train_series_csv, non_gold)
    except ValueError:
        _fail("series_metadata_incomplete")
    return series_by_study


def _make_input_pins(source_csv: Path, train_csv: Path, series_csv: Path,
                     gold_path: Path, baseline_cohort_path: Path,
                     baseline_folds_path: Path) -> dict[str, str]:
    _read_pinned(source_csv, SOURCE_CSV_SIZE, SOURCE_CSV_SHA256, "source_csv_pin_mismatch")
    _read_pinned(gold_path, GOLD_IDS_SIZE, GOLD_IDS_SHA256, "gold_pin_mismatch")
    _read_pinned(baseline_cohort_path, COHORT_SIZE, COHORT_SHA256, "baseline_cohort_pin_mismatch")
    _read_pinned(baseline_folds_path, FOLDS_SIZE, FOLDS_SHA256, "baseline_folds_pin_mismatch")
    return {
        "source_csv_sha256": SOURCE_CSV_SHA256,
        "official_train_csv_sha256": _sha256_file(train_csv),
        "train_series_csv_sha256": _sha256_file(series_csv),
        "gold_ids_sha256": GOLD_IDS_SHA256,
        "baseline_cohort_sha256": COHORT_SHA256,
        "baseline_folds_sha256": FOLDS_SHA256,
        "policy_module_sha256": _sha256_file(Path(__file__)),
    }


def _ensure_new_safe_directory(path: Path) -> None:
    if path.exists() or path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        _fail("output_must_be_new")
    if ".." in path.parts:
        _fail("output_path_invalid")


def _remove_owned_output(path: Path) -> None:
    if not path.is_dir() or path.is_symlink():
        return
    if path.resolve().parent == path.parent.resolve():
        shutil.rmtree(path)


def prepare_full_cohort(*, source_csv: str | Path, official_train_csv: str | Path,
                        series_csv: str | Path, gold_ids_path: str | Path,
                        baseline_cohort_path: str | Path, baseline_folds_path: str | Path,
                        output_dir: str | Path) -> dict[str, object]:
    """Prepare a complete, new private artifact bundle without training or image reads."""
    paths = [Path(value) for value in (source_csv, official_train_csv, series_csv,
                                       gold_ids_path, baseline_cohort_path, baseline_folds_path)]
    source_path, train_path, series_path, gold_path, cohort_path, folds_path = paths
    output_path = Path(output_dir).absolute()
    private_root = PRIVATE_OUTPUT_ROOT.absolute()
    try:
        relative_output = output_path.relative_to(private_root)
    except ValueError:
        _fail("output_must_be_under_private_prefilter_root")
    if not relative_output.parts:
        _fail("output_must_be_new")
    _ensure_new_safe_directory(output_path)
    pins = _make_input_pins(source_path, train_path, series_path, gold_path, cohort_path, folds_path)
    gold_ids = _read_gold_ids(gold_path)
    official_ids = _read_official_study_ids(train_path)
    if not gold_ids.issubset(official_ids):
        _fail("gold_inventory_mismatch")
    source_rows = _read_public_rows(source_path, gold_ids, official_ids)
    reports = _read_reports(train_path, official_ids, gold_ids)
    series_by_study = _read_series(series_path, official_ids, gold_ids)
    if set(source_rows) != set(reports) or set(source_rows) != set(series_by_study):
        _fail("non_gold_membership_mismatch")
    row_inputs = []
    for study_id in (study for study in source_rows):
        probabilities = source_rows[study_id]
        row_inputs.append({
            "study_id": study_id,
            "report_group_sha256": reports[study_id],
            "series": series_by_study[study_id],
            "soft_targets": dict(zip(TARGET_COLUMNS, probabilities)),
            "observed_mask": {target: value != 0.5 for target, value in zip(TARGET_COLUMNS, probabilities)},
        })
    manifest, counts = build_prefilter_manifest(row_inputs, input_pins=pins)
    if (counts["non_gold_study_count"] != EXPECTED_NON_GOLD_STUDIES
            or counts["eligible_study_count"] != EXPECTED_ELIGIBLE
            or counts["excluded_all_targets_unknown"] != EXPECTED_ALL_UNKNOWN
            or counts["report_group_count"] != EXPECTED_REPORT_GROUPS
            or counts["series_count"] != EXPECTED_NON_GOLD_SERIES):
        _fail("pinned_cohort_counts_mismatch")
    folds_artifact = build_grouped_folds(manifest)
    summary = _aggregate_summary(manifest, counts, folds_artifact, gold_excluded_count=len(gold_ids))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_new_safe_directory(output_path)
    created = False
    payloads = {
        "full-cohort-soft-label-manifest.json": _canonical_json(manifest),
        "full-cohort-grouped-folds.json": _canonical_json(folds_artifact),
        "aggregate-summary.json": _canonical_json(summary),
    }
    try:
        with tempfile.TemporaryDirectory(prefix=f".{output_path.name}-", dir=output_path.parent) as temp_name:
            temp = Path(temp_name)
            for name, payload in payloads.items():
                (temp / name).write_bytes(payload)
            try:
                output_path.mkdir(parents=False, exist_ok=False)
            except FileExistsError:
                _fail("output_must_be_new")
            created = True
            if output_path.is_symlink() or output_path.resolve().parent != output_path.parent.resolve():
                _fail("output_path_invalid")
            for name in payloads:
                with (output_path / name).open("xb") as stream:
                    stream.write((temp / name).read_bytes())
            if (output_path.is_symlink() or output_path.resolve().parent != output_path.parent.resolve()
                    or set(path.name for path in output_path.iterdir()) != set(payloads)
                    or any(path.is_symlink() for path in output_path.iterdir())
                    or any((output_path / name).read_bytes() != payload for name, payload in payloads.items())):
                _fail("output_readback_mismatch")
    except (OSError, ValueError) as exc:
        if created:
            try:
                _remove_owned_output(output_path)
            except OSError as cleanup_error:
                raise ValueError("output_cleanup_failed") from exc
        if isinstance(exc, ValueError) and str(exc) in {"output_must_be_new", "output_path_invalid",
                                                        "output_readback_mismatch"}:
            raise
        raise ValueError("output_write_failed") from exc
    try:
        _read_pinned(source_path, SOURCE_CSV_SIZE, SOURCE_CSV_SHA256, "source_csv_pin_mismatch")
        _read_pinned(gold_path, GOLD_IDS_SIZE, GOLD_IDS_SHA256, "gold_pin_mismatch")
        _read_pinned(cohort_path, COHORT_SIZE, COHORT_SHA256, "baseline_cohort_pin_mismatch")
        _read_pinned(folds_path, FOLDS_SIZE, FOLDS_SHA256, "baseline_folds_pin_mismatch")
        if (_sha256_file(train_path) != pins["official_train_csv_sha256"]
                or _sha256_file(series_path) != pins["train_series_csv_sha256"]
                or _sha256_file(Path(__file__)) != pins["policy_module_sha256"]):
            _fail("local_input_changed_during_preparation")
    except ValueError as exc:
        try:
            _remove_owned_output(output_path)
        except OSError:
            raise ValueError("immutable_input_cleanup_failed") from exc
        raise ValueError("immutable_input_changed_during_preparation") from exc
    return summary


def _aggregate_summary(manifest: Mapping[str, object], counts: Mapping[str, int],
                       folds: Mapping[str, object], *, gold_excluded_count: int) -> dict[str, object]:
    target_counts = {}
    for target in TARGET_COLUMNS:
        addressed = counts["target_addressed_counts"][target]
        target_counts[target] = {"non_gold_study_count": counts["non_gold_study_count"],
                                 "addressed_count": addressed,
                                 "unknown_count": counts["non_gold_study_count"] - addressed}
    return {
        "schema_version": "full-cohort-prefilter-summary-v1",
        "study_counts": dict(counts),
        "gold_excluded_count": gold_excluded_count,
        "target_coverage": target_counts,
        "fold_counts": [{"fold": fold["fold"], "training_study_count": fold["training_study_count"],
                         "validation_study_count": fold["validation_study_count"],
                         "training_group_count": fold["training_group_count"],
                         "validation_group_count": fold["validation_group_count"],
                         "train_support": fold["train_support"],
                         "validation_support": fold["validation_support"]}
                        for fold in folds["folds"]],
        "grouping_limit": "normalized report hash is a dependency proxy, not patient/site identity",
        "image_quality_status": "not inspected; metadata eligibility only",
        "training_status": "preparation_only_not_trained",
        "gold_metrics_used": False,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path,
                        default=Path("data/private/report-label-source-comparison/steven-v6") / SOURCE_CSV_NAME)
    parser.add_argument("--official-train-csv", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--series-csv", type=Path, default=Path("data/train_series.csv"))
    parser.add_argument("--gold-ids", type=Path, default=Path("data/private/gold58_study_ids.csv"))
    parser.add_argument("--baseline-cohort", type=Path, default=Path("data/private/500-study-training-cohort.json"))
    parser.add_argument("--baseline-folds", type=Path, default=Path("data/private/500-study-training-folds.json"))
    parser.add_argument("--output", type=Path, default=Path("data/private/full-cohort-prefilter/run01"))
    args = parser.parse_args(argv)
    try:
        summary = prepare_full_cohort(
            source_csv=args.source_csv, official_train_csv=args.official_train_csv,
            series_csv=args.series_csv, gold_ids_path=args.gold_ids,
            baseline_cohort_path=args.baseline_cohort, baseline_folds_path=args.baseline_folds,
            output_dir=args.output)
    except ValueError as error:
        print(json.dumps({"status": "failed", "reason_code": str(error)}, sort_keys=True))
        return 2
    print(json.dumps({"status": "complete", "study_counts": summary["study_counts"],
                      "fold_counts": summary["fold_counts"],
                      "training_status": summary["training_status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
