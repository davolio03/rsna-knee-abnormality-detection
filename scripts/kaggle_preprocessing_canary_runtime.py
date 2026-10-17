"""Bounded, identifier-free Kaggle preprocessing canary runtime."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Callable


class CanaryFailure(RuntimeError):
    """A failure represented only by a bounded public reason code."""

    SAFE_REASONS = frozenset({
        "package_inventory_invalid", "package_member_hash_mismatch",
        "manifest_hash_mismatch", "folds_hash_mismatch", "manifest_invalid",
        "folds_invalid", "fold_partition_mismatch", "target_series_unavailable",
        "series_files_mismatch", "preprocessing_failed", "output_unavailable",
        "runtime_limit_reached", "canary_disabled", "decoded_metadata_mismatch",
        "duplicate_slice_position", "header_read_failed", "incomplete_intensity_metadata",
        "inconsistent_orientation", "inconsistent_pixel_layout", "inconsistent_spacing",
        "invalid_intensity_metadata", "missing_dependency", "missing_geometry",
        "pixel_decode_failed", "series_files_missing", "series_path_unavailable",
        "symlink_path", "unsupported_frame_count", "unsupported_modality_lut",
        "unsupported_pixel_layout", "invalid_limit",
    })

    def __init__(self, reason_code: str):
        safe_code = reason_code if reason_code in self.SAFE_REASONS else "preprocessing_failed"
        super().__init__(safe_code)
        self.reason_code = safe_code


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def verify_package(input_root: Path, inventory_bytes: bytes, config: dict) -> dict:
    """Verify embedded inventory, every package member, and manifest/fold pins."""
    try:
        root = Path(input_root).resolve(strict=True)
        if not root.is_dir():
            raise OSError
    except Exception:
        raise CanaryFailure("package_inventory_invalid") from None
    if _sha256(inventory_bytes) != config.get("expected_inventory_sha256"):
        raise CanaryFailure("package_inventory_invalid")
    try:
        inventory = json.loads(inventory_bytes.decode("utf-8"))
        members = inventory["members"]
        if (inventory.get("schema_version") != "kaggle-training-package-inventory-v1"
                or len(members) != config.get("expected_member_count")
                or inventory.get("package_sha256") != config.get("expected_archive_provenance")
                or inventory.get("manifest_sha256") != config.get("expected_manifest_sha256")
                or inventory.get("folds_sha256") != config.get("expected_folds_sha256")):
            raise CanaryFailure("package_inventory_invalid")
        for name, expected in members.items():
            relative = Path(name)
            if (relative.is_absolute() or ".." in relative.parts or "\\" in name
                    or not name.startswith(("src/", "data/"))):
                raise CanaryFailure("package_inventory_invalid")
            path = root.joinpath(*relative.parts)
            cursor = root
            if any((cursor := cursor / part).is_symlink() for part in relative.parts):
                raise CanaryFailure("package_member_hash_mismatch")
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(root) or not path.is_file() or _sha256(path.read_bytes()) != expected:
                raise CanaryFailure("package_member_hash_mismatch")
    except CanaryFailure:
        raise
    except Exception:
        raise CanaryFailure("package_inventory_invalid") from None

    manifest_path = root / config["manifest_relative_path"]
    folds_path = root / config["folds_relative_path"]
    try:
        manifest_raw = manifest_path.read_bytes()
        folds_raw = folds_path.read_bytes()
    except Exception:
        raise CanaryFailure("package_member_hash_mismatch") from None
    if _sha256(manifest_raw) != config.get("expected_manifest_sha256"):
        raise CanaryFailure("manifest_hash_mismatch")
    if _sha256(folds_raw) != config.get("expected_folds_sha256"):
        raise CanaryFailure("folds_hash_mismatch")
    return {"inventory": inventory, "manifest_raw": manifest_raw, "folds_raw": folds_raw}


def select_target_series(
    records: list[tuple[str, tuple[str, ...]]], assignments: dict[str, int], image_root: Path,
    *, fold: int, expected_training_count: int, expected_validation_count: int,
    target_ordinal: int, expected_file_count: int,
) -> tuple[Path, int]:
    """Select one 1-based training-series ordinal in frozen manifest order."""
    if type(target_ordinal) is not int or target_ordinal < 1:
        raise CanaryFailure("target_series_unavailable")
    training_studies = [study for study, _ in records if assignments.get(study) != fold]
    validation_studies = [study for study, _ in records if assignments.get(study) == fold]
    if (len(training_studies) != expected_training_count
            or len(validation_studies) != expected_validation_count
            or set(training_studies) | set(validation_studies) != {study for study, _ in records}
            or len(assignments) != expected_training_count + expected_validation_count):
        raise CanaryFailure("fold_partition_mismatch")
    ordinal = 0
    for study, series_ids in records:
        if assignments.get(study) == fold:
            continue
        for series in series_ids:
            ordinal += 1
            if ordinal != target_ordinal:
                continue
            folder = Path(image_root) / study / series
            try:
                if folder.is_symlink() or not folder.is_dir():
                    raise CanaryFailure("target_series_unavailable")
                files = [path for path in folder.iterdir() if path.suffix.lower() == ".dcm"]
                if (any(path.is_symlink() for path in files)
                        or len(files) != expected_file_count):
                    raise CanaryFailure("series_files_mismatch")
            except CanaryFailure:
                raise
            except Exception:
                raise CanaryFailure("target_series_unavailable") from None
            return folder, len(files)
    raise CanaryFailure("target_series_unavailable")


def run_preprocessing_canary(
    config: dict, inventory_bytes: bytes, manifest_parser: Callable[[bytes], list], *,
    input_root: Path, image_root: Path, preprocessor: Callable | None = None,
    folds_parser: Callable[[bytes], dict[str, int]] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict:
    """Verify first; preprocess one fixed training series; save aggregate JSON only."""
    started = monotonic()
    pins = verify_package(input_root, inventory_bytes, config)
    if folds_parser is None:
        folds_parser = _parse_fold_assignments
    try:
        records = manifest_parser(pins["manifest_raw"])
        assignments = folds_parser(pins["folds_raw"])
    except CanaryFailure:
        raise
    except Exception:
        raise CanaryFailure("manifest_invalid") from None
    folder, file_count = select_target_series(
        records, assignments, image_root, fold=config["fold"],
        expected_training_count=config["expected_training_count"],
        expected_validation_count=config["expected_validation_count"],
        target_ordinal=config["target_series_ordinal"],
        expected_file_count=config["expected_target_file_count"],
    )
    if preprocessor is None:
        import sys
        package_source = Path(input_root) / "src"
        sys.path.insert(0, str(package_source))
        try:
            from rsna_knee.data.dicom_preprocessing import preprocess_dicom_series
        except Exception:
            raise CanaryFailure("preprocessing_failed") from None
        preprocessor = preprocess_dicom_series
    try:
        result = preprocessor(
            folder, max_slices=config["max_slices"], output_size=config["output_size"], dcmread=None,
        )
        import numpy as np
        images = np.asarray(result.images)
        if images.size == 0 or not np.isfinite(images).all():
            raise CanaryFailure("preprocessing_failed")
        value_range = [float(images.min()), float(images.max())]
        if not all(math.isfinite(value) for value in value_range):
            raise CanaryFailure("preprocessing_failed")
    except CanaryFailure:
        raise
    except Exception as exc:
        from rsna_knee.data.dicom_preprocessing import DicomPreprocessingError
        if isinstance(exc, DicomPreprocessingError):
            raise CanaryFailure(exc.reason_code) from None
        raise CanaryFailure("preprocessing_failed") from None
    elapsed = monotonic() - started
    if elapsed > config["max_seconds"]:
        raise CanaryFailure("runtime_limit_reached")
    summary = {
        "schema_version": "kaggle-preprocessing-canary-v1",
        "status": "completed",
        "package_inventory_sha256": config["expected_inventory_sha256"],
        "package_sha256": config["expected_archive_provenance"],
        "manifest_sha256": config["expected_manifest_sha256"],
        "folds_sha256": config["expected_folds_sha256"],
        "fold": config["fold"],
        "training_study_count": config["expected_training_count"],
        "validation_study_count": config["expected_validation_count"],
        "series_ordinal": config["target_series_ordinal"],
        "input_file_count": file_count,
        "output_size": config["output_size"],
        "max_slices": config["max_slices"],
        "image_shape": [int(value) for value in images.shape],
        "finite_values": True,
        "normalized_value_range": value_range,
        "elapsed_seconds": round(elapsed, 3),
    }
    output_root = Path(config["output_dir"])
    try:
        output_root.mkdir(parents=True, exist_ok=False)
        (output_root / "preprocessing-summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
    except Exception:
        raise CanaryFailure("output_unavailable") from None
    return summary


def _parse_fold_assignments(raw: bytes) -> dict[str, int]:
    try:
        document = json.loads(raw.decode("utf-8"))
        entries = document["study_assignments"]
        assignments = {entry["study_id"]: int(entry["fold"]) for entry in entries}
        if len(assignments) != len(entries):
            raise ValueError
        return assignments
    except Exception:
        raise CanaryFailure("folds_invalid") from None
