"""Pinned full-cohort dataset that preserves soft labels and observation masks."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from rsna_knee.data.dicom_benchmark import _safe_component
from rsna_knee.data.dicom_preprocessing import DicomPreprocessingError, preprocess_dicom_series
from rsna_knee.data.training_dataset import load_series_images
from rsna_knee.labels.report_source_comparison import (
    COHORT_SHA256, FOLDS_SHA256, GOLD_IDS_SHA256, SOURCE_CSV_SHA256, TARGET_COLUMNS,
)
from rsna_knee.labels.full_cohort_prefilter import build_grouped_folds


MANIFEST_SHA256 = "617ea1c267960b1e903b332a78388db3b6ab80d44374721df5a4c5e184e8961b"
FOLDS_ARTIFACT_SHA256 = "b816c7902a5d0caa68a9e818d3f913943bc61cd2e6db65c4de069f5cd94e15f9"
PREFILTER_POLICY_SHA256 = "7a1b41d3d96c137859d83d78927fd020afba8ce4034cb0addbed67e730280662"
TRAIN_SERIES_SHA256 = "573c1d80772bf41211c91b149c95677385a1c22d63f485c347f1b46c0177aef3"
OFFICIAL_TRAIN_SHA256 = "8ca2203c0e9d61c080c7a314c7cdb51c1b03a1d9eb4770819f7f34af53ef4e33"
EXPECTED_RECORDS = 4298
EXPECTED_AUDIT_ROWS = 4349
EXPECTED_UNSUPERVISED = 51


class SoftDatasetError(ValueError):
    """Stable categorical dataset failure without identifiers or paths."""

    def __init__(self, reason_code: str):
        if type(reason_code) is not str or not re.fullmatch(r"[a-z0-9_]{1,48}", reason_code):
            reason_code = "invalid_input"
        self.reason_code = reason_code
        super().__init__(reason_code)


def _read(path: str | Path, reason: str) -> tuple[object, bytes]:
    try:
        raw = Path(path).read_bytes()
        return json.loads(raw.decode("utf-8")), raw
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise SoftDatasetError(reason) from None


def _verify_manifest(manifest: object) -> list[dict[str, Any]]:
    from rsna_knee.labels.full_cohort_prefilter import _validate_probabilities

    if not isinstance(manifest, Mapping) or set(manifest) != {
        "audit_rows", "configuration", "input_pins", "records", "schema_version", "target_order",
    }:
        raise SoftDatasetError("manifest_schema_invalid")
    if (manifest.get("schema_version") != "full-cohort-soft-label-manifest-v1"
            or manifest.get("target_order") != list(TARGET_COLUMNS)):
        raise SoftDatasetError("manifest_schema_invalid")
    pins = manifest.get("input_pins")
    expected_pins = {
        "baseline_cohort_sha256": COHORT_SHA256,
        "baseline_folds_sha256": FOLDS_SHA256,
        "gold_ids_sha256": GOLD_IDS_SHA256,
        "official_train_csv_sha256": OFFICIAL_TRAIN_SHA256,
        "policy_module_sha256": PREFILTER_POLICY_SHA256,
        "source_csv_sha256": SOURCE_CSV_SHA256,
        "train_series_csv_sha256": TRAIN_SERIES_SHA256,
    }
    if not isinstance(pins, Mapping) or dict(pins) != expected_pins:
        raise SoftDatasetError("manifest_provenance_mismatch")
    configuration = manifest.get("configuration")
    if (not isinstance(configuration, Mapping)
            or configuration.get("policy_version") != "full-cohort-soft-label-prefilter-v1"
            or configuration.get("unknown_probability") != 0.5):
        raise SoftDatasetError("manifest_configuration_invalid")
    records = manifest.get("records")
    audit_rows = manifest.get("audit_rows")
    if (type(records) is not list or len(records) != EXPECTED_RECORDS
            or type(audit_rows) is not list or len(audit_rows) != EXPECTED_AUDIT_ROWS):
        raise SoftDatasetError("manifest_counts_mismatch")
    record_ids: set[str] = set()
    series_ids_all: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise SoftDatasetError("manifest_record_invalid")
        study_id = record.get("study_id")
        if type(study_id) is not str or not study_id or study_id in record_ids:
            raise SoftDatasetError("manifest_membership_invalid")
        record_ids.add(study_id)
        try:
            _validate_probabilities(record.get("soft_targets"), record.get("observed_mask"))
        except (TypeError, ValueError):
            raise SoftDatasetError("manifest_labels_invalid") from None
        series = record.get("series")
        if type(series) is not list or not series:
            raise SoftDatasetError("manifest_series_invalid")
        series_ids: set[str] = set()
        for item in series:
            if (not isinstance(item, Mapping) or set(item) != {
                "series_id", "anatomical_plane", "fluid_sensitive", "fat_suppression",
            }):
                raise SoftDatasetError("manifest_series_invalid")
            series_id = item.get("series_id")
            if type(series_id) is not str or not series_id or series_id in series_ids:
                raise SoftDatasetError("manifest_series_invalid")
            series_ids.add(series_id)
            if series_id in series_ids_all:
                raise SoftDatasetError("manifest_series_invalid")
            series_ids_all.add(series_id)
            if (item.get("anatomical_plane") not in {"Axial", "Coronal", "Sagittal"}
                    or type(item.get("fluid_sensitive")) is not int
                    or item.get("fluid_sensitive") not in {0, 1}
                    or type(item.get("fat_suppression")) is not int
                    or item.get("fat_suppression") not in {0, 1}):
                raise SoftDatasetError("manifest_series_invalid")
        group_hash = record.get("report_group_sha256")
        if (type(group_hash) is not str or not re.fullmatch(r"[0-9a-f]{64}", group_hash)
                or record.get("observed_target_count") != sum(record["observed_mask"].values())):
            raise SoftDatasetError("manifest_record_invalid")
    audit_ids: set[str] = set()
    for entry in audit_rows:
        if (not isinstance(entry, Mapping) or set(entry) != {"study_id", "reason_code"}
                or type(entry.get("study_id")) is not str or not entry["study_id"]
                or entry["study_id"] in audit_ids
                or type(entry.get("reason_code")) is not str):
            raise SoftDatasetError("manifest_audit_invalid")
        audit_ids.add(entry["study_id"])
    if not record_ids.issubset(audit_ids):
        raise SoftDatasetError("manifest_audit_membership_mismatch")
    excluded = audit_ids - record_ids
    if len(excluded) != EXPECTED_UNSUPERVISED or any(
        entry["study_id"] in excluded and entry["reason_code"] != "all_targets_unknown"
        for entry in audit_rows
    ):
        raise SoftDatasetError("manifest_exclusion_mismatch")
    if any(entry["study_id"] in record_ids and entry["reason_code"] != "eligible_at_least_one_observed_target"
           for entry in audit_rows):
        raise SoftDatasetError("manifest_eligibility_mismatch")
    return records


def load_verified_soft_artifacts(
    manifest_path: str | Path = "data/private/full-cohort-prefilter/run01/full-cohort-soft-label-manifest.json",
    folds_path: str | Path = "data/private/full-cohort-prefilter/run01/full-cohort-grouped-folds.json",
):
    """Validate exact pinned artifacts and recompute all fold assignments/weights."""
    manifest, raw_manifest = _read(manifest_path, "manifest_unavailable")
    folds, raw_folds = _read(folds_path, "folds_unavailable")
    if hashlib.sha256(raw_manifest).hexdigest() != MANIFEST_SHA256:
        raise SoftDatasetError("manifest_pin_mismatch")
    if hashlib.sha256(raw_folds).hexdigest() != FOLDS_ARTIFACT_SHA256:
        raise SoftDatasetError("folds_pin_mismatch")
    records = _verify_manifest(manifest)
    try:
        recomputed = build_grouped_folds(manifest)
    except (TypeError, ValueError, KeyError):
        raise SoftDatasetError("folds_recompute_failed") from None
    if not isinstance(folds, Mapping) or dict(folds) != recomputed:
        raise SoftDatasetError("folds_membership_mismatch")
    expected_digest = hashlib.sha256(json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8") + b"\n").hexdigest()
    if folds.get("manifest_sha256") != expected_digest:
        raise SoftDatasetError("fold_manifest_binding_mismatch")
    assignment = {entry["study_id"]: entry["fold"] for entry in folds["study_assignments"]}
    if len(assignment) != EXPECTED_RECORDS:
        raise SoftDatasetError("fold_membership_invalid")
    return records, assignment, hashlib.sha256(raw_manifest).hexdigest()


class FullCohortSoftDataset:
    """Fold-bound series dataset returning tensors, soft targets, and label masks."""

    target_names = tuple(TARGET_COLUMNS)

    def __init__(self, manifest_path: str | Path, folds_path: str | Path, image_root: str | Path,
                 *, partition: str, fold: int, output_size: int = 256, max_slices: int = 8,
                 preprocessor: Callable[..., Any] | None = None):
        if partition not in {"train", "validation"}:
            raise SoftDatasetError("partition_invalid")
        if type(fold) is not int or fold not in {0, 1, 2}:
            raise SoftDatasetError("fold_invalid")
        if type(output_size) is not int or not 8 <= output_size <= 768:
            raise SoftDatasetError("output_size_invalid")
        if type(max_slices) is not int or not 1 <= max_slices <= 8:
            raise SoftDatasetError("slice_limit_invalid")
        try:
            root = Path(image_root)
            if root.is_symlink() or not root.is_dir():
                raise OSError
            self.image_root = root.resolve(strict=True)
        except (OSError, TypeError, ValueError):
            raise SoftDatasetError("image_root_unavailable") from None
        records, assignments, self.manifest_sha256 = load_verified_soft_artifacts(
            manifest_path, folds_path,
        )
        self.partition, self.fold = partition, fold
        self.output_size, self.max_slices = output_size, max_slices
        self.preprocessor = preprocessor or preprocess_dicom_series
        self._records = tuple(record for record in records
                              if (assignments[record["study_id"]] == fold)
                              == (partition == "validation"))
        if not self._records:
            raise SoftDatasetError("partition_empty")

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int):
        import torch

        if type(index) is not int or not 0 <= index < len(self._records):
            raise IndexError("Soft-label sample index is out of range.")
        record = self._records[index]
        try:
            study_id = _safe_component(record["study_id"], "study")
            study_path = self.image_root / study_id
            if study_path.is_symlink() or not study_path.is_dir():
                raise ValueError
            images_by_series, pixel_masks_by_series = [], []
            for item in record["series"]:
                series_id = _safe_component(item["series_id"], "series")
                series_path = study_path / series_id
                if series_path.is_symlink() or not series_path.is_dir():
                    raise ValueError
                series_path.resolve(strict=True).relative_to(self.image_root)
                images, pixel_mask = load_series_images(
                    series_path, preprocessor=self.preprocessor,
                    output_size=self.output_size, max_slices=self.max_slices,
                )
                images_by_series.append(images)
                pixel_masks_by_series.append(pixel_mask)
            targets = torch.tensor([record["soft_targets"][name] for name in self.target_names],
                                   dtype=torch.float32)
            observed = torch.tensor([record["observed_mask"][name] for name in self.target_names],
                                    dtype=torch.bool)
            return images_by_series, pixel_masks_by_series, targets, observed, index
        except Exception as error:
            data_reason_code = None
            if (type(error) is DicomPreprocessingError
                    and type(getattr(error, "reason_code", None)) is str
                    and error.reason_code in DicomPreprocessingError.SAFE_REASON_CODES):
                data_reason_code = error.reason_code
            suffix = f"_{data_reason_code}" if data_reason_code else ""
            raise SoftDatasetError(f"study_load_failed{suffix}") from None
