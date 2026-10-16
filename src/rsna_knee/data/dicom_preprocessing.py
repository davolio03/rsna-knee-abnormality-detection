"""Explicit local DICOM series adapter feeding the PP-01 array transform."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Callable

from rsna_knee.data.dicom_benchmark import DicomBenchmarkError, _geometry
from rsna_knee.data.preprocessing import (
    DecodedSlice,
    PreprocessedSeries,
    preprocess_series,
    uniformly_spaced_indices,
)


class DicomPreprocessingError(ValueError):
    """Safe categorical failure that never embeds a path or decoder detail."""

    def __init__(self, reason_code: str, message: str):
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True)
class DicomPreprocessedSeries:
    """PP-01 image tensors plus aggregate, identifier-free PP-02 metadata."""

    images: Any
    valid_mask: Any
    metadata: dict[str, Any]


_HEADER_TAGS = (
    "ImageOrientationPatient", "ImagePositionPatient", "PixelSpacing", "Rows", "Columns",
    "SamplesPerPixel", "PhotometricInterpretation", "NumberOfFrames", "RescaleSlope",
    "RescaleIntercept", "PixelPaddingValue", "PixelPaddingRangeLimit", "ModalityLUTSequence",
)
_ABSENT = object()


def _fail(code: str, message: str) -> None:
    raise DicomPreprocessingError(code, message)


def _load_reader():
    try:
        return importlib.import_module("pydicom").dcmread
    except ImportError:
        _fail("missing_dependency", "pydicom is required to read DICOM series.")


def _positive_integer(value, label: str) -> int:
    if isinstance(value, bool):
        _fail("unsupported_pixel_layout", f"DICOM {label} metadata is invalid.")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        _fail("unsupported_pixel_layout", f"DICOM {label} metadata is invalid.")
    if number < 1 or (str(number) != str(value).strip() and not isinstance(value, int)):
        _fail("unsupported_pixel_layout", f"DICOM {label} metadata is invalid.")
    return number


def _finite_dicom_number(value, label: str) -> float:
    if isinstance(value, bool) or value is None:
        _fail("invalid_intensity_metadata", f"DICOM {label} metadata is invalid.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        _fail("invalid_intensity_metadata", f"DICOM {label} metadata is invalid.")
    if not math.isfinite(number):
        _fail("invalid_intensity_metadata", f"DICOM {label} metadata is invalid.")
    return number


def _optional_number(dataset, key: str):
    value = getattr(dataset, key, _ABSENT)
    if value is _ABSENT:
        return None
    if key in {"PixelPaddingValue", "PixelPaddingRangeLimit"}:
        if isinstance(value, bool):
            _fail("invalid_intensity_metadata", f"DICOM {key} metadata is invalid.")
        try:
            integer = int(value)
        except (TypeError, ValueError, OverflowError):
            _fail("invalid_intensity_metadata", f"DICOM {key} metadata is invalid.")
        if str(integer) != str(value).strip() and not isinstance(value, Integral):
            _fail("invalid_intensity_metadata", f"DICOM {key} metadata is invalid.")
        if abs(integer) > 2**53:
            _fail("invalid_intensity_metadata", f"DICOM {key} metadata exceeds the exact float64 integer range.")
        return integer
    return _finite_dicom_number(value, key)


def _metadata(dataset):
    try:
        orientation, spacing, coordinate = _geometry(dataset)
    except DicomBenchmarkError as exc:
        _fail(exc.reason_code, str(exc))
    rows = _positive_integer(getattr(dataset, "Rows", None), "Rows")
    columns = _positive_integer(getattr(dataset, "Columns", None), "Columns")
    samples = _positive_integer(getattr(dataset, "SamplesPerPixel", None), "SamplesPerPixel")
    photo = getattr(dataset, "PhotometricInterpretation", None)
    if samples != 1 or photo not in {"MONOCHROME1", "MONOCHROME2"}:
        _fail("unsupported_pixel_layout", "Only monochrome DICOM slices are supported.")
    frames = getattr(dataset, "NumberOfFrames", 1)
    if _positive_integer(frames, "NumberOfFrames") != 1:
        _fail("unsupported_frame_count", "Multiframe DICOM objects are not supported.")
    slope = _optional_number(dataset, "RescaleSlope")
    intercept = _optional_number(dataset, "RescaleIntercept")
    padding = _optional_number(dataset, "PixelPaddingValue")
    padding_limit = _optional_number(dataset, "PixelPaddingRangeLimit")
    if (slope is None) != (intercept is None) or (padding_limit is not None and padding is None):
        _fail("incomplete_intensity_metadata", "Paired DICOM intensity metadata is incomplete.")
    if hasattr(dataset, "ModalityLUTSequence"):
        _fail("unsupported_modality_lut", "DICOM Modality LUT sequences are unsupported.")
    return {
        "orientation": orientation,
        "spacing": spacing,
        "coordinate": coordinate,
        "rows": rows,
        "columns": columns,
        "samples": samples,
        "photometric": photo,
        "slope": slope,
        "intercept": intercept,
        "padding": padding,
        "padding_limit": padding_limit,
    }


def _decoded_metadata_matches_header(left, right) -> bool:
    return (
        left["rows"] == right["rows"]
        and left["columns"] == right["columns"]
        and left["samples"] == right["samples"]
        and left["photometric"] == right["photometric"]
        and all(abs(a - b) <= 1e-5 for a, b in zip(left["orientation"], right["orientation"]))
        and all(abs(a - b) <= 1e-5 for a, b in zip(left["spacing"], right["spacing"]))
        and abs(left["coordinate"] - right["coordinate"]) <= 1e-5
        and all(left[key] == right[key] for key in ("slope", "intercept", "padding", "padding_limit"))
    )


def preprocess_dicom_series(
    series_path: str | Path,
    *,
    max_slices: int = 8,
    output_size: int = 256,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    dcmread: Callable[..., Any] | None = None,
) -> DicomPreprocessedSeries:
    """Read one explicitly chosen directory, order headers, select, decode, and run PP-01.

    A limit of one selects the first slice in ascending projected patient-space
    coordinate. No filename or instance-number order is used for slice selection.
    """
    if isinstance(max_slices, bool) or not isinstance(max_slices, int) or not 1 <= max_slices <= 8:
        _fail("invalid_limit", "max_slices must be an integer from one through eight.")
    try:
        root = Path(series_path)
        if not root.is_dir() or root.is_symlink():
            _fail("series_path_unavailable", "The selected DICOM series directory is unavailable.")
        paths = sorted(path for path in root.iterdir() if path.suffix.lower() == ".dcm")
    except DicomPreprocessingError:
        raise
    except (OSError, TypeError, ValueError):
        _fail("series_path_unavailable", "The selected DICOM series directory is unavailable.")
    if not paths:
        _fail("series_files_missing", "The selected series has no direct DICOM files.")
    reader = dcmread or _load_reader()
    headers = []
    reference = None
    seen_positions = set()
    for path in paths:
        if path.is_symlink():
            _fail("symlink_path", "The selected series contains a symlinked DICOM file.")
        try:
            dataset = reader(path, stop_before_pixels=True, specific_tags=list(_HEADER_TAGS))
        except Exception:
            _fail("header_read_failed", "A DICOM header could not be read safely.")
        header = _metadata(dataset)
        if reference is None:
            reference = header
        else:
            if any(abs(a - b) > 1e-5 for a, b in zip(reference["orientation"], header["orientation"])):
                _fail("inconsistent_orientation", "The series contains inconsistent slice orientation.")
            if any(abs(a - b) > 1e-5 for a, b in zip(reference["spacing"], header["spacing"])):
                _fail("inconsistent_spacing", "The series contains inconsistent pixel spacing.")
            if (header["rows"], header["columns"], header["samples"], header["photometric"]) != (
                reference["rows"], reference["columns"], reference["samples"], reference["photometric"]
            ):
                _fail("inconsistent_pixel_layout", "The series contains inconsistent pixel layout metadata.")
        position = round(header["coordinate"], 6)
        if position in seen_positions:
            _fail("duplicate_slice_position", "The series contains duplicate projected slice positions.")
        seen_positions.add(position)
        headers.append((position, path, header))

    ordered = sorted(headers, key=lambda item: item[0])
    indices = uniformly_spaced_indices(len(ordered), max_slices)
    selected = [ordered[index] for index in indices]
    decoded_slices = []
    for _, path, header in selected:
        try:
            dataset = reader(path)
            decoded = _metadata(dataset)
        except DicomPreprocessingError:
            raise
        except Exception:
            _fail("pixel_decode_failed", "A selected DICOM slice could not be decoded safely.")
        if not _decoded_metadata_matches_header(header, decoded):
            _fail("decoded_metadata_mismatch", "Decoded DICOM metadata differs from its scanned header.")
        try:
            pixels = dataset.pixel_array
        except Exception:
            _fail("pixel_decode_failed", "A selected DICOM slice could not be decoded safely.")
        try:
            shape = tuple(int(value) for value in pixels.shape)
        except (AttributeError, TypeError, ValueError):
            _fail("unsupported_pixel_layout", "Decoded pixels do not match monochrome slice geometry.")
        if shape != (header["rows"], header["columns"]):
            _fail("decoded_metadata_mismatch", "Decoded pixels do not match scanned slice geometry.")
        decoded_slices.append(
            DecodedSlice(
                pixels=pixels,
                photometric_interpretation=decoded["photometric"],
                slope=decoded["slope"],
                intercept=decoded["intercept"],
                pixel_padding_value=decoded["padding"],
                pixel_padding_range_limit=decoded["padding_limit"],
                pixel_spacing=tuple(decoded["spacing"]),
                modality_lut_sequence_present=False,
            )
        )
    try:
        output: PreprocessedSeries = preprocess_series(
            decoded_slices,
            max_slices=max_slices,
            output_size=output_size,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
        )
    except (ImportError, ValueError):
        _fail("preprocessing_failed", "The selected DICOM series failed PP-01 validation.")
    metadata = {
        "preprocessing_version": "pp02-v1",
        "slice_count_scanned": len(paths),
        "slice_count_selected": len(selected),
        "selected_positions_mm": [float(item[0]) for item in selected],
        "pp01": output.metadata,
    }
    return DicomPreprocessedSeries(output.images, output.valid_mask, metadata)
