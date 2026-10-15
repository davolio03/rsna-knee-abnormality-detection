"""Build a private, deterministic manifest for the ordered silver-label cohort."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Mapping

from rsna_knee.labels.siwc_responses import SIWCResponsesInfer
from rsna_knee.labels.subscription import (
    LABEL_COLUMNS,
    STUDY_COLUMN,
    LabelingError,
    _load_records,
    eligible_reports,
    validate_label_result,
)


MANIFEST_SCHEMA_VERSION = "baseline-training-cohort-v1"
REPORT_NORMALIZATION_VERSION = "python-casefold-collapse-whitespace-retain-punctuation-v1"
PROMPT_REVISIONS = (
    "c2-literal-evidence-v1",
    "c2-literal-evidence-v2",
    "c2-literal-evidence-v3",
)
_REPORT_WHITESPACE = re.compile(r"\s+")
_TRAIN_COLUMNS = {STUDY_COLUMN, "Report", *LABEL_COLUMNS}
_SERIES_COLUMNS = {
    STUDY_COLUMN, "SeriesInstanceUID", "Fluid_Sensitive", "Fat_Suppression",
    "Anatomical_Plane",
}


def _read_csv_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error, ValueError):
        raise ValueError("A cohort source CSV is invalid.") from None
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("A cohort source CSV contains malformed rows.")
    return rows


def _read_gold_ids(path: Path) -> set[str]:
    rows = _read_csv_rows(path, {STUDY_COLUMN})
    if not rows:
        raise ValueError("The Gold exclusion list is empty.")
    ids = [row[STUDY_COLUMN].strip() for row in rows]
    if any(not study_id for study_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("The Gold exclusion list is invalid.")
    if len(ids) != 58:
        raise ValueError("The Gold exclusion list must contain 58 unique studies.")
    return set(ids)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        raise ValueError("A cohort source could not be hashed.") from None
    return digest.hexdigest()


def _prompt_versions(journal_path: Path, checkpoint_count: int) -> list[str]:
    try:
        entries = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeError, ValueError):
        raise ValueError("The prompt provenance journal is invalid.") from None
    if not all(isinstance(entry, dict) for entry in entries):
        raise ValueError("The prompt provenance journal is invalid.")

    markers = [entry for entry in entries if entry.get("event") == "prompt_boundary"]
    previous_revision = -1
    previous_offset = 0
    for marker in markers:
        version = marker.get("prompt_version")
        offset = marker.get("start_offset")
        if (set(marker) != {"event", "prompt_version", "start_offset"}
                or version not in PROMPT_REVISIONS or type(offset) is not int
                or not previous_offset <= offset <= checkpoint_count):
            raise ValueError("Prompt provenance boundaries are incompatible with the checkpoint.")
        revision_index = PROMPT_REVISIONS.index(version)
        if revision_index <= previous_revision:
            raise ValueError("Prompt provenance boundaries are out of order.")
        previous_revision, previous_offset = revision_index, offset

    versions = []
    for index in range(checkpoint_count):
        assigned = "legacy"
        for marker in markers:
            if marker["start_offset"] > index:
                break
            assigned = marker["prompt_version"]
        versions.append(assigned)
    return versions


def _binary_flag(value: str) -> int:
    normalized = value.strip().casefold()
    if normalized in {"1", "true"}:
        return 1
    if normalized in {"0", "false"}:
        return 0
    raise ValueError("Series metadata contains a non-binary flag.")


def _series_by_study(path: Path, selected_ids: set[str]) -> dict[str, list[dict[str, object]]]:
    rows = _read_csv_rows(path, _SERIES_COLUMNS)
    result: dict[str, list[dict[str, object]]] = {study_id: [] for study_id in selected_ids}
    seen_series: set[str] = set()
    for row in rows:
        study_id = row[STUDY_COLUMN].strip()
        series_id = row["SeriesInstanceUID"].strip()
        plane = row["Anatomical_Plane"].strip()
        if not study_id or not series_id or not plane or series_id in seen_series:
            raise ValueError("Series metadata contains an invalid or duplicate key.")
        seen_series.add(series_id)
        if study_id not in selected_ids:
            continue
        result[study_id].append({
            "series_id": series_id,
            "fluid_sensitive": _binary_flag(row["Fluid_Sensitive"]),
            "fat_suppression": _binary_flag(row["Fat_Suppression"]),
            "anatomical_plane": plane,
        })
    if any(not entries for entries in result.values()):
        raise ValueError("Series metadata does not cover every cohort study.")
    for entries in result.values():
        entries.sort(key=lambda entry: str(entry["series_id"]))
    return result


def _normalized_report_hash(report: str) -> str:
    normalized = _REPORT_WHITESPACE.sub(" ", report.casefold()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_training_cohort(*, train_csv: Path, labels_jsonl: Path, series_csv: Path,
                          gold_exclusions: Path, prompt_journal: Path,
                          cohort_size: int = 500) -> dict[str, object]:
    """Validate and build a report-free private manifest without writing files."""
    if type(cohort_size) is not int or cohort_size <= 0:
        raise ValueError("Cohort size must be a positive integer.")
    train_rows = _read_csv_rows(Path(train_csv), _TRAIN_COLUMNS)
    gold_ids = _read_gold_ids(Path(gold_exclusions))

    # Apply the Gold ID mask before handing rows to the validator that reads reports.
    non_gold_rows = [row for row in train_rows if row[STUDY_COLUMN].strip() not in gold_ids]
    eligible = eligible_reports(non_gold_rows, gold_ids=gold_ids)
    if len(eligible) < cohort_size:
        raise ValueError("The eligible non-Gold cohort is smaller than requested.")
    selected = eligible[:cohort_size]
    selected_ids = [str(row[STUDY_COLUMN]).strip() for row in selected]

    checkpoint = _load_records(Path(labels_jsonl))
    if len(checkpoint) != cohort_size:
        raise ValueError("The label checkpoint does not contain the exact requested cohort.")
    checkpoint_ids = [record[STUDY_COLUMN].strip() for record in checkpoint]
    if checkpoint_ids != selected_ids or len(set(checkpoint_ids)) != cohort_size:
        raise ValueError("The label checkpoint does not match the ordered eligible prefix.")

    by_id = {str(row[STUDY_COLUMN]).strip(): row for row in selected}
    prompt_versions = _prompt_versions(Path(prompt_journal), cohort_size)
    series = _series_by_study(Path(series_csv), set(selected_ids))
    cohort_records = []
    for record, prompt_version in zip(checkpoint, prompt_versions):
        study_id = record[STUDY_COLUMN].strip()
        labels = validate_label_result(record["labels"])
        SIWCResponsesInfer.validate_source_evidence(labels, str(by_id[study_id]["Report"]))
        cohort_records.append({
            "study_id": study_id,
            "labels": {column: int(labels[column]["label"]) for column in LABEL_COLUMNS},
            "series": series[study_id],
            "prompt_version": prompt_version,
            "normalized_report_sha256": _normalized_report_hash(str(by_id[study_id]["Report"])),
        })

    provenance_counts = {"legacy": 0, **{version: 0 for version in PROMPT_REVISIONS}}
    for version in prompt_versions:
        provenance_counts[version] += 1
    input_paths = {
        "train_csv": Path(train_csv), "labels_jsonl": Path(labels_jsonl),
        "series_csv": Path(series_csv), "gold_exclusions": Path(gold_exclusions),
        "prompt_journal": Path(prompt_journal),
    }
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "configuration": {
            "cohort_size": cohort_size,
            "cohort_selection": "first eligible non-Gold rows in train.csv order",
            "gold_exclusion_count": len(gold_ids),
            "report_group_normalization": REPORT_NORMALIZATION_VERSION,
            "label_encoding": "0-or-1 integer per target",
        },
        "source_sha256": {name: _sha256(path) for name, path in input_paths.items()},
        "prompt_provenance_counts": provenance_counts,
        "records": cohort_records,
    }


def _private_destination(output_path: Path, project_root: Path) -> Path:
    root = Path(project_root).resolve()
    destination = Path(output_path)
    if not destination.is_absolute():
        destination = root / destination
    destination = Path(os.path.abspath(destination))
    private = root / "data" / "private"
    try:
        destination.relative_to(private)
    except ValueError:
        raise ValueError("Manifest output must stay under data/private.") from None

    current = root
    try:
        for component in (Path("data"), Path("data/private")):
            current = root / component
            if current.exists() and current.is_symlink():
                raise ValueError("Manifest output path contains a symlink.")
        relative_parent = destination.parent.relative_to(private)
        current = private
        for component in relative_parent.parts:
            current = current / component
            if current.exists() and current.is_symlink():
                raise ValueError("Manifest output path contains a symlink.")
        if destination.is_symlink() or (destination.exists() and not destination.is_file()):
            raise ValueError("Manifest output path is unsafe.")
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("Manifest output"):
            raise
        raise ValueError("Manifest output path is unsafe.") from None
    return destination


def write_private_manifest(manifest: Mapping[str, object], *, output_path: Path,
                           project_root: Path) -> None:
    """Create a JSON manifest exclusively under ignored data/private."""
    destination = _private_destination(Path(output_path), Path(project_root))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.resolve() != destination.parent:
        raise ValueError("Manifest output path contains a symlink.")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(destination, flags, stat.S_IRUSR | stat.S_IWUSR)
    except FileExistsError:
        raise
    except OSError:
        raise ValueError("Manifest output could not be safely created.") from None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except Exception:
        try:
            destination.unlink()
        except OSError:
            pass
        raise
