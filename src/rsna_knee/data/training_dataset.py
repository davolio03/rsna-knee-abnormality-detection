"""Lazy, fold-bound study dataset for the provisional image baseline."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from rsna_knee.data.dicom_benchmark import _safe_component
from rsna_knee.data.dicom_preprocessing import DicomPreprocessingError, preprocess_dicom_series
from rsna_knee.labels.subscription import LABEL_COLUMNS
from rsna_knee.validation.cohort import build_grouped_training_folds


class TrainingDatasetError(ValueError):
    """Safe categorical error that never includes identifiers or file paths."""

    SAFE_DIAGNOSTIC_REASON_CODES = DicomPreprocessingError.SAFE_REASON_CODES

    def __init__(self, reason_code: str, *, data_reason_code: str | None = None) -> None:
        if not re.fullmatch(r"[a-z0-9_]{1,48}", reason_code):
            reason_code = "invalid_input"
        self.reason_code = reason_code
        self.data_reason_code = (
            data_reason_code if type(data_reason_code) is str
            and data_reason_code in self.SAFE_DIAGNOSTIC_REASON_CODES else None
        )
        super().__init__(f"Training data unavailable ({reason_code}).")


def _read_json(path: Path, code: str) -> tuple[object, bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise TrainingDatasetError(code) from None
    return value, raw


def _load_artifacts(manifest_path: str | Path, folds_path: str | Path):
    manifest, raw = _read_json(Path(manifest_path), "manifest_unavailable")
    folds, _ = _read_json(Path(folds_path), "folds_unavailable")
    try:
        digest = hashlib.sha256(raw).hexdigest()
        expected = build_grouped_training_folds(manifest, manifest_sha256=digest)
    except (TypeError, ValueError):
        raise TrainingDatasetError("manifest_invalid") from None
    if not isinstance(folds, Mapping) or folds != expected:
        raise TrainingDatasetError("folds_mismatch")
    records = manifest["records"]
    assignments = {entry["study_id"]: entry["fold"] for entry in expected["study_assignments"]}
    return records, assignments, digest


class TrainingStudyDataset:
    """One partition of validated studies, decoded on demand one study at a time."""

    def __init__(
        self,
        manifest_path: str | Path,
        folds_path: str | Path,
        image_root: str | Path,
        *,
        partition: str,
        fold: int,
        output_size: int = 256,
        max_slices: int = 8,
        preprocessor: Callable[..., Any] | None = None,
    ) -> None:
        if partition not in {"train", "validation"}:
            raise TrainingDatasetError("partition_invalid")
        if type(fold) is not int or fold not in (0, 1, 2):
            raise TrainingDatasetError("fold_invalid")
        if type(output_size) is not int or not 1 <= output_size <= 768:
            raise TrainingDatasetError("output_size_invalid")
        if type(max_slices) is not int or not 1 <= max_slices <= 8:
            raise TrainingDatasetError("slice_limit_invalid")
        try:
            root = Path(image_root)
            if not root.is_dir() or root.is_symlink():
                raise OSError
            self.image_root = root.resolve(strict=True)
        except (OSError, TypeError, ValueError):
            raise TrainingDatasetError("image_root_unavailable") from None
        records, assignments, _ = _load_artifacts(manifest_path, folds_path)
        self.partition = partition
        self.fold = fold
        self.output_size = output_size
        self.max_slices = max_slices
        self.preprocessor = preprocessor or preprocess_dicom_series
        self._records = tuple(
            record for record in records
            if (assignments[record["study_id"]] == fold) == (partition == "validation")
        )
        if not self._records:
            raise TrainingDatasetError("partition_empty")
        self.indices = tuple(range(len(self._records)))

    def __len__(self) -> int:
        return len(self._records)

    def bounded_training_subset(self, limit: int):
        """Return at most ``limit`` deterministic training rows, never validation."""
        if self.partition != "train":
            raise TrainingDatasetError("training_subset_forbidden")
        if type(limit) is not int or limit < 1:
            raise TrainingDatasetError("subset_limit_invalid")
        from torch.utils.data import Subset

        return Subset(self, list(range(min(limit, len(self)))))

    def __getitem__(self, index: int):
        import numpy as np
        import torch

        if type(index) is not int or not 0 <= index < len(self._records):
            raise IndexError("Training sample index is out of range.")
        record = self._records[index]
        try:
            study_id = _safe_component(record["study_id"], "study")
            study_path = self.image_root / study_id
            if study_path.is_symlink() or not study_path.is_dir():
                raise ValueError
            images_by_series = []
            masks_by_series = []
            for item in record["series"]:
                series_id = _safe_component(item["series_id"], "series")
                series_path = study_path / series_id
                if series_path.is_symlink() or not series_path.is_dir():
                    raise ValueError
                series_path.resolve(strict=True).relative_to(self.image_root)
                result = self.preprocessor(
                    series_path, max_slices=self.max_slices, output_size=self.output_size,
                )
                images = np.asarray(result.images)
                masks = np.asarray(result.valid_mask)
                expected_shape = (images.shape[0], 1, self.output_size, self.output_size) if images.ndim else ()
                if (images.ndim != 4 or images.shape[0] < 1 or images.shape[0] > self.max_slices
                        or images.shape != expected_shape or masks.shape != images.shape
                        or not np.issubdtype(images.dtype, np.number) or not np.isrealobj(images)
                        or not np.isfinite(images).all() or not np.isfinite(masks).all()
                        or np.any(images < 0.0) or np.any(images > 1.0)
                        or not np.isin(masks, (0, 1, False, True)).all()):
                    raise ValueError
                if not np.isfinite(images.astype(np.float32, copy=False)).all():
                    raise ValueError
                valid_mask = masks.astype(np.bool_, copy=False)
                if np.any(~valid_mask.reshape(valid_mask.shape[0], -1).any(axis=1)):
                    raise ValueError
                if np.any(images[~valid_mask] != 0):
                    raise ValueError
                images_by_series.append(torch.as_tensor(images.copy(), dtype=torch.float32))
                masks_by_series.append(torch.as_tensor(valid_mask.copy(), dtype=torch.bool))
            labels = torch.tensor([record["labels"][name] for name in LABEL_COLUMNS], dtype=torch.float32)
        except Exception as error:
            # Decoder exceptions can contain private IDs or paths; expose only a category.
            data_reason_code = None
            if (type(error) is DicomPreprocessingError
                    and type(getattr(error, "reason_code", None)) is str
                    and error.reason_code in DicomPreprocessingError.SAFE_REASON_CODES):
                data_reason_code = error.reason_code
            raise TrainingDatasetError(
                "study_load_failed", data_reason_code=data_reason_code,
            ) from None
        return images_by_series, masks_by_series, labels, index
