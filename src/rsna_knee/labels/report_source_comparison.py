"""Pinned, aggregate-only comparison of external report labels with frozen C2 labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import stat
import zipfile
from pathlib import Path
from zipfile import ZipFile

from rsna_knee.labels.subscription import LABEL_COLUMNS
from rsna_knee.validation.cohort import _validate_manifest, build_grouped_training_folds


DATASET_CARD_URL = "https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels"
DATASET_VERSION = 6
SOURCE_ARCHIVE_SHA256 = "573f11c299a30b15c13f0f19368543f2d0ee81810809541b188ed83b359a9aa0"
SOURCE_ARCHIVE_SIZE = 426025
SOURCE_CSV_NAME = "llm_labels_full.csv"
SOURCE_CSV_SHA256 = "96444cd3c7d396bae4a71454317d9cbe0689759b5278b30939781e04d0774537"
SOURCE_CSV_SIZE = 534442
COHORT_SHA256 = "e88c61c22a463737f16951b7c76db03834034aecd1b31dc61b4ce495080b2be2"
COHORT_SIZE = 893129
FOLDS_SHA256 = "b82b376b0086201e3364b11a0203b62890e628442075a52ea4a9e09ee0e778ac"
FOLDS_SIZE = 179820
GOLD_IDS_SHA256 = "e0e584d475a7e8b0749ffcce70267ac19a2e2549338f98f50ed515f8fabce3a1"
GOLD_IDS_SIZE = 3846
EXPECTED_OFFICIAL_STUDIES = 4407
EXPECTED_NON_GOLD_STUDIES = 4349
EXPECTED_COHORT_STUDIES = 500
EXPECTED_GOLD_STUDIES = 58
TARGET_COLUMNS = tuple(LABEL_COLUMNS)
_PROBABILITY_TEXT = re.compile(r"(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?\Z")
_FUTURE_METRICS = (
    "fixed validation memberships from pinned fold artifact",
    "C2 binary reference: per-target ROC-AUC",
    "Steven source: masked soft-target binary cross-entropy and Brier score",
)


def _fail(code: str) -> None:
    raise ValueError(code)


def _read_pinned(path: Path, expected_size: int, expected_sha256: str, code: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        _fail(code)
    try:
        raw = path.read_bytes()
    except OSError:
        _fail(code)
    if len(raw) != expected_size or hashlib.sha256(raw).hexdigest() != expected_sha256:
        _fail(code)
    return raw


def _write_new(path: Path, data: bytes) -> None:
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        _fail("output_path_invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(data)
    except OSError:
        _fail("output_must_be_new")


def _write_json_new(path: Path, value: object) -> None:
    _write_new(path, (json.dumps(value, sort_keys=True, separators=(",", ":"),
                               allow_nan=False) + "\n").encode("utf-8"))


def _csv_reader(path: Path):
    try:
        stream = path.open("r", encoding="utf-8-sig", newline="")
    except OSError:
        _fail("csv_unavailable")
    try:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
            _fail("csv_schema_invalid")
        return stream, reader
    except Exception:
        stream.close()
        raise


def _identifier(value: object) -> str:
    if type(value) is not str or not value or value != value.strip() or len(value) > 128:
        _fail("study_identifier_invalid")
    return value


def _read_gold_ids(path: Path) -> set[str]:
    raw = _read_pinned(path, GOLD_IDS_SIZE, GOLD_IDS_SHA256, "gold_pin_mismatch")
    try:
        reader = csv.DictReader(raw.decode("utf-8-sig").splitlines())
        if reader.fieldnames != ["StudyInstanceUID"]:
            _fail("gold_schema_invalid")
        ids = [_identifier(row.get("StudyInstanceUID")) for row in reader]
    except (UnicodeError, csv.Error):
        _fail("gold_schema_invalid")
    if len(ids) != EXPECTED_GOLD_STUDIES or len(set(ids)) != len(ids):
        _fail("gold_inventory_invalid")
    return set(ids)


def _read_official_study_ids(path: Path) -> set[str]:
    """Read only the study-key column from official train.csv."""
    if path.is_symlink() or not path.is_file():
        _fail("official_inventory_unavailable")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            header = reader.fieldnames
            if (header is None or len(header) != len(set(header))
                    or header.count("StudyInstanceUID") != 1):
                _fail("official_inventory_schema_invalid")
            ids = []
            for row in reader:
                ids.append(_identifier(row.get("StudyInstanceUID")))
    except (OSError, UnicodeError, csv.Error, StopIteration):
        _fail("official_inventory_invalid")
    if len(ids) != EXPECTED_OFFICIAL_STUDIES or len(set(ids)) != len(ids):
        _fail("official_inventory_invalid")
    return set(ids)


def _load_frozen_cohort(cohort_path: Path, folds_path: Path, gold_ids: set[str]):
    cohort_raw = _read_pinned(cohort_path, COHORT_SIZE, COHORT_SHA256, "cohort_pin_mismatch")
    folds_raw = _read_pinned(folds_path, FOLDS_SIZE, FOLDS_SHA256, "folds_pin_mismatch")
    try:
        manifest = json.loads(cohort_raw.decode("utf-8"))
        folds = json.loads(folds_raw.decode("utf-8"))
        records = _validate_manifest(manifest)
        expected = build_grouped_training_folds(
            manifest, manifest_sha256=hashlib.sha256(cohort_raw).hexdigest())
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        _fail("cohort_or_folds_invalid")
    if folds != expected or len(records) != EXPECTED_COHORT_STUDIES:
        _fail("fold_binding_invalid")
    cohort_ids = [_identifier(record["study_id"]) for record in records]
    if len(set(cohort_ids)) != len(cohort_ids) or set(cohort_ids) & gold_ids:
        _fail("cohort_membership_invalid")
    assignments = {item["study_id"]: item["fold"] for item in folds["study_assignments"]}
    if set(assignments) != set(cohort_ids) or any(type(value) is not int or value not in (0, 1, 2)
                                                   for value in assignments.values()):
        _fail("fold_binding_invalid")
    return records, assignments, cohort_raw, folds_raw, folds


def _parse_probability(value: object) -> float:
    # CSV lexemes must be explicitly floating-point; integer-looking labels are
    # rejected rather than ambiguously interpreted as binary classes.
    if type(value) is not str or not _PROBABILITY_TEXT.fullmatch(value):
        _fail("probability_format_invalid")
    try:
        probability = float(value)
    except ValueError:
        _fail("probability_invalid")
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        _fail("probability_out_of_range")
    return probability


def _read_public_rows(path: Path, gold_ids: set[str], official_ids: set[str]):
    stream, reader = _csv_reader(path)
    try:
        if reader.fieldnames != ["StudyInstanceUID", *TARGET_COLUMNS]:
            _fail("source_schema_invalid")
        rows: dict[str, tuple[float, ...]] = {}
        all_source_ids: set[str] = set()
        for row in reader:
            study_id = _identifier(row.get("StudyInstanceUID"))
            if study_id in all_source_ids:
                _fail("source_duplicate_id")
            all_source_ids.add(study_id)
            if study_id not in official_ids:
                _fail("source_foreign_id")
            # Gold rows are excluded before any probability token is converted.
            if study_id in gold_ids:
                continue
            rows[study_id] = tuple(_parse_probability(row.get(target)) for target in TARGET_COLUMNS)
    except (csv.Error, UnicodeError):
        _fail("source_csv_invalid")
    finally:
        stream.close()
    if all_source_ids != official_ids or len(all_source_ids) != EXPECTED_OFFICIAL_STUDIES:
        _fail("source_coverage_mismatch")
    if len(rows) != EXPECTED_NON_GOLD_STUDIES:
        _fail("source_non_gold_count_invalid")
    return rows


def stage_approved_source(archive_path: str | Path, destination_root: str | Path) -> dict[str, object]:
    """Copy the pinned source archive and extract only the approved original v1 CSV."""
    archive_path = Path(archive_path)
    raw = _read_pinned(archive_path, SOURCE_ARCHIVE_SIZE, SOURCE_ARCHIVE_SHA256,
                       "source_archive_pin_mismatch")
    destination_root = Path(destination_root)
    if destination_root.exists() or destination_root.is_symlink():
        _fail("stage_output_must_be_new")
    try:
        destination_root.mkdir(parents=True, exist_ok=False)
        archive_copy = destination_root / "source-archive.zip"
        _write_new(archive_copy, raw)
        with ZipFile(archive_copy) as archive:
            infos = archive.infolist()
            expected_names = {SOURCE_CSV_NAME, "llm_labels_v2.csv", "llm_labels_v4_blend.csv"}
            if (len(infos) != 3 or {info.filename for info in infos} != expected_names
                    or any(info.is_dir() or "/" in info.filename or "\\" in info.filename
                           or info.filename in {".", ".."}
                           or stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK
                           for info in infos)):
                _fail("source_archive_inventory_invalid")
            selected = archive.getinfo(SOURCE_CSV_NAME)
            selected_bytes = archive.read(selected)
    except (OSError, zipfile.BadZipFile, KeyError):
        _fail("source_archive_invalid")
    if len(selected_bytes) != SOURCE_CSV_SIZE or hashlib.sha256(selected_bytes).hexdigest() != SOURCE_CSV_SHA256:
        _fail("source_csv_pin_mismatch")
    selected_path = destination_root / SOURCE_CSV_NAME
    _write_new(selected_path, selected_bytes)
    provenance = {
        "schema_version": "steven-report-label-source-provenance-v1",
        "dataset_card_url": DATASET_CARD_URL, "dataset_version": DATASET_VERSION,
        "selected_file": SOURCE_CSV_NAME, "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "source_archive_size": SOURCE_ARCHIVE_SIZE, "selected_csv_sha256": SOURCE_CSV_SHA256,
        "selected_csv_size": SOURCE_CSV_SIZE, "expected_columns": ["StudyInstanceUID", *TARGET_COLUMNS],
        "author_probability_semantics": "0.5 means not addressed; independently unverified",
        "author_llm_model_and_prompt_disclosed": False,
        "author_reported_gold_auc": "0.878 (unverified; not computed or used)",
        "card_license_metadata": "CC0; not a legal determination and does not override competition terms",
        "secondary_files_extracted": False,
    }
    _write_json_new(destination_root / "source_provenance.json", provenance)
    return {"csv_path": str(selected_path), "provenance_path": str(destination_root / "source_provenance.json"),
            "source_csv_sha256": SOURCE_CSV_SHA256, "source_archive_sha256": SOURCE_ARCHIVE_SHA256}


def _comparison_metrics(values: list[tuple[int, float]], *, study_count: int) -> dict[str, object]:
    addressed = [(label, probability) for label, probability in values if probability != 0.5]
    tp = sum(1 for label, prob in addressed if label == 1 and prob > 0.5)
    fp = sum(1 for label, prob in addressed if label == 0 and prob > 0.5)
    fn = sum(1 for label, prob in addressed if label == 1 and prob < 0.5)
    tn = sum(1 for label, prob in addressed if label == 0 and prob < 0.5)
    count = len(addressed)
    return {
        "study_count": study_count,
        "addressed_count": count,
        "not_addressed_count": study_count - count,
        "c2_positive_count": sum(label for label, _ in addressed),
        "public_binary_positive_count": sum(prob > 0.5 for _, prob in addressed),
        "c2_positive_prevalence": (
            sum(label for label, _ in addressed) / count if count else None),
        "public_binary_positive_prevalence": (
            sum(prob > 0.5 for _, prob in addressed) / count if count else None),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "agreement": (tp + tn) / count if count else None,
        "mean_absolute_probability_difference": (
            sum(abs(prob - label) for label, prob in addressed) / count if count else None),
    }


def build_source_comparison(*, source_csv: str | Path, official_train_csv: str | Path,
                            cohort_path: str | Path, folds_path: str | Path,
                            gold_ids_path: str | Path, output_dir: str | Path) -> dict[str, object]:
    """Validate pinned inputs and write aggregate-only comparison plus private overlay."""
    source_raw = _read_pinned(Path(source_csv), SOURCE_CSV_SIZE, SOURCE_CSV_SHA256,
                              "source_csv_pin_mismatch")
    gold_ids = _read_gold_ids(Path(gold_ids_path))
    official_ids = _read_official_study_ids(Path(official_train_csv))
    records, assignments, cohort_raw, folds_raw, folds = _load_frozen_cohort(
        Path(cohort_path), Path(folds_path), gold_ids)
    if not set(gold_ids).issubset(official_ids):
        _fail("gold_inventory_mismatch")
    public = _read_public_rows(Path(source_csv), gold_ids, official_ids)
    cohort_ids = [record["study_id"] for record in records]
    if any(study not in public for study in cohort_ids):
        _fail("paired_cohort_missing_source")

    inventory_by_target = {}
    for index, target in enumerate(TARGET_COLUMNS):
        addressed = sum(probabilities[index] != 0.5 for probabilities in public.values())
        inventory_by_target[target] = {"non_gold_study_count": len(public),
                                       "addressed_count": addressed,
                                       "not_addressed_count": len(public) - addressed}

    paired_targets: dict[str, object] = {}
    overlay_records = []
    for record in records:
        study_id = record["study_id"]
        probabilities = public[study_id]
        overlay_records.append({
            "study_id": study_id,
            "fold": assignments[study_id],
            "soft_targets": dict(zip(TARGET_COLUMNS, probabilities)),
            "observed_mask": {target: probability != 0.5
                              for target, probability in zip(TARGET_COLUMNS, probabilities)},
        })
    for index, target in enumerate(TARGET_COLUMNS):
        c2_values = [(record["labels"][target], public[record["study_id"]][index])
                     for record in records]
        metrics = _comparison_metrics(c2_values, study_count=len(records))
        by_fold = {}
        for fold in (0, 1, 2):
            fold_records = [record for record in records if assignments[record["study_id"]] == fold]
            fold_values = [(record["labels"][target], public[record["study_id"]][index])
                           for record in fold_records]
            by_fold[str(fold)] = _comparison_metrics(fold_values, study_count=len(fold_records))
        if (metrics["addressed_count"] + metrics["not_addressed_count"] != EXPECTED_COHORT_STUDIES
                or sum(x["addressed_count"] for x in by_fold.values()) != metrics["addressed_count"]):
            _fail("comparison_count_invariant_failed")
        metrics["by_fold"] = by_fold
        paired_targets[target] = metrics

    validation_counts = {str(fold): sum(value == fold for value in assignments.values())
                         for fold in (0, 1, 2)}
    summary = {
        "schema_version": "report-label-source-comparison-summary-v1",
        "source_inventory": {
            "official_study_count": len(official_ids),
            "source_rows": len(official_ids),
            "gold_studies_excluded": len(gold_ids),
            "non_gold_studies": len(public),
            "target_coverage": inventory_by_target,
        },
        "paired": {"study_count": len(records), "validation_study_counts_by_fold": validation_counts,
                   "targets": paired_targets},
        "protocol": {
            "version": 1,
            "masked_value": 0.5,
            "public_binary_threshold": "> 0.5",
            "reference_interpretation": "agreement with C2 labels; neither source is ground truth",
            "metrics": ["addressed coverage", "paired prevalence", "confusion counts", "agreement",
                        "mean absolute probability difference"],
            "forbidden_claims": ["accuracy against ground truth", "Gold performance", "source superiority"],
        },
        "pins": {"source_csv_sha256": hashlib.sha256(source_raw).hexdigest(),
                 "cohort_sha256": hashlib.sha256(cohort_raw).hexdigest(),
                 "folds_sha256": hashlib.sha256(folds_raw).hexdigest(),
                 "gold_ids_sha256": GOLD_IDS_SHA256},
    }
    plan = {
        "schema_version": "report-label-source-future-comparison-plan-v1",
        "status": "prepared_not_trained",
        "baseline": {"source": "frozen C2 binary labels", "cohort_sha256": COHORT_SHA256,
                     "folds_sha256": FOLDS_SHA256, "training_reader_changed": False},
        "candidate": {"source_card_url": DATASET_CARD_URL, "dataset_version": DATASET_VERSION,
                      "file": SOURCE_CSV_NAME, "source_sha256": SOURCE_CSV_SHA256,
                      "soft_target_training_adapter": "not implemented; separate authorization required"},
        "comparison": {"same_500_study_cohort": True, "validation_counts_by_fold": validation_counts,
                       "same_fold_membership_from_pinned_fold_artifact": True,
                       "future_metrics": list(_FUTURE_METRICS),
                       "gold_ids_or_metrics_used": False,
                       "model_and_training_configuration": "unchanged; freeze from existing C2 run before training"},
        "training_authorized_in_this_task": False,
    }
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        _fail("output_must_be_new")
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except OSError:
        _fail("output_must_be_new")
    _write_json_new(output_dir / "source_comparison_summary.json", summary)
    _write_json_new(output_dir / "soft_label_overlay.json", {
        "schema_version": "report-label-soft-target-overlay-v1",
        "pins": summary["pins"], "record_count": len(overlay_records), "records": overlay_records,
    })
    _write_json_new(output_dir / "future_model_comparison_plan.json", plan)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("C:/Users/David/Downloads/archive.zip"))
    parser.add_argument("--official-train-csv", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--cohort", type=Path, default=Path("data/private/500-study-training-cohort.json"))
    parser.add_argument("--folds", type=Path, default=Path("data/private/500-study-training-folds.json"))
    parser.add_argument("--gold-ids", type=Path, default=Path("data/private/gold58_study_ids.csv"))
    parser.add_argument("--stage", type=Path,
                         default=Path("data/private/report-label-source-comparison/steven-v6"))
    parser.add_argument("--output", type=Path,
                         default=Path("data/private/report-label-source-comparison/results-v1"))
    args = parser.parse_args(argv)
    try:
        staged = stage_approved_source(args.archive, args.stage)
        summary = build_source_comparison(
            source_csv=staged["csv_path"], official_train_csv=args.official_train_csv,
            cohort_path=args.cohort, folds_path=args.folds, gold_ids_path=args.gold_ids,
            output_dir=args.output)
    except ValueError as error:
        print(json.dumps({"status": "failed", "reason_code": str(error)}, sort_keys=True))
        return 2
    print(json.dumps({"status": "complete", "official_study_count": summary["source_inventory"]["official_study_count"],
                      "non_gold_study_count": summary["source_inventory"]["non_gold_studies"],
                      "paired_study_count": summary["paired"]["study_count"],
                      "target_count": len(TARGET_COLUMNS)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
