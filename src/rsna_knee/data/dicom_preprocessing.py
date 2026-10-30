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


_SAFE_DIAGNOSTIC_REASON_CODES = frozenset({
    "decoded_metadata_mismatch", "duplicate_slice_position", "header_read_failed",
    "incomplete_intensity_metadata", "inconsistent_orientation", "inconsistent_pixel_layout",
    "inconsistent_spacing", "invalid_intensity_metadata", "missing_dependency", "missing_geometry",
    "pixel_decode_failed", "pixel_payload_invalid", "pixel_transfer_syntax_unsupported",
    "dicom_file_parse_invalid",
    "preprocessing_failed", "series_files_missing",
    "series_path_unavailable", "symlink_path", "unsupported_frame_count",
    "unsupported_modality_lut", "unsupported_pixel_layout", "invalid_limit",
})


class DicomPreprocessingError(ValueError):
    """Safe categorical failure that never embeds a path or decoder detail."""

    SAFE_REASON_CODES = _SAFE_DIAGNOSTIC_REASON_CODES

    def __init__(self, reason_code: str, message: str, *, recovery_details=None):
        super().__init__(message)
        self.reason_code = reason_code
        self.recovery_details = recovery_details


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
_ORIENTATION_COMPONENT_TOLERANCE = 1e-5
_SPACING_COMPONENT_TOLERANCE = 1e-5
_NATIVE_TRANSFER_SYNTAXES = frozenset({
    "1.2.840.10008.1.2", "1.2.840.10008.1.2.1", "1.2.840.10008.1.2.2",
})


def _orientation_component_matches(left: float, right: float) -> bool:
    """Apply the fixed orientation tolerance with only a two-ULP roundoff margin.

    DICOM orientation values are parsed as binary floats. This margin prevents
    subtraction/representation rounding at the exact tolerance boundary from
    becoming a false rejection; it is not a relative or scientific tolerance.
    """
    operand_scale = max(abs(left), abs(right), _ORIENTATION_COMPONENT_TOLERANCE)
    roundoff = 2 * math.ulp(operand_scale)
    return abs(left - right) <= _ORIENTATION_COMPONENT_TOLERANCE + roundoff


def _orientations_match(left, right) -> bool:
    return all(_orientation_component_matches(a, b) for a, b in zip(left, right))


def _spacings_match(left, right) -> bool:
    """Keep the existing absolute geometry limit, allowing only binary roundoff."""
    return all(
        abs(a - b) <= _SPACING_COMPONENT_TOLERANCE + 2 * math.ulp(
            max(abs(a), abs(b), _SPACING_COMPONENT_TOLERANCE)
        )
        for a, b in zip(left, right)
    )


def _fail(code: str, message: str) -> None:
    if code not in _SAFE_DIAGNOSTIC_REASON_CODES:
        code = "preprocessing_failed"
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
        and _orientations_match(left["orientation"], right["orientation"])
        and _spacings_match(left["spacing"], right["spacing"])
        and abs(left["coordinate"] - right["coordinate"]) <= 1e-5
        and all(left[key] == right[key] for key in ("slope", "intercept", "padding", "padding_limit"))
    )


def _is_known_reader_failure(error: BaseException) -> bool:
    """Recognize only pydicom's documented malformed-file exceptions."""
    try:
        from pydicom.errors import BytesLengthException, InvalidDicomError
    except ImportError:
        return False
    return type(error) is InvalidDicomError or type(error) is BytesLengthException


def _is_bytes_length_exception(error: BaseException) -> bool:
    """Match pydicom's exact malformed-element exception without eager imports."""
    try:
        from pydicom.errors import BytesLengthException
    except ImportError:
        return False
    return type(error) is BytesLengthException


def _recoverable_header_reason(error: DicomPreprocessingError) -> bool:
    return error.reason_code in {
        "header_read_failed", "missing_geometry", "inconsistent_orientation",
        "inconsistent_spacing", "inconsistent_pixel_layout", "duplicate_slice_position",
        "unsupported_frame_count", "unsupported_modality_lut", "unsupported_pixel_layout",
        "invalid_intensity_metadata", "incomplete_intensity_metadata",
        "decoded_metadata_mismatch",
    }


def _recovery_failure(code: str, message: str, reasons: dict[str, int], slice_reasons: list[dict[str, Any]]) -> None:
    raise DicomPreprocessingError(code, message, recovery_details={
        "version": 1, "slice_count_skipped": sum(reasons.values()),
        "reason_counts": dict(sorted(reasons.items())),
        "slice_reasons": sorted(slice_reasons, key=lambda item: item["slice_ordinal"]),
    })


def _native_pixel_payload_invalid(dataset, header) -> bool:
    """Check only exact byte length for native single-frame monochrome pixels."""
    file_meta = getattr(dataset, "file_meta", None)
    syntax = getattr(file_meta, "TransferSyntaxUID", None)
    if syntax is None or str(syntax) not in _NATIVE_TRANSFER_SYNTAXES:
        return False
    bits = getattr(dataset, "BitsAllocated", None)
    if type(bits) is not int or bits not in (1, 8, 16, 32, 64):
        return False
    payload = getattr(dataset, "PixelData", None)
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        return False
    samples = header["rows"] * header["columns"] * header["samples"]
    expected = (samples + 7) // 8 if bits == 1 else samples * (bits // 8)
    if expected % 2:
        expected += 1
    return len(payload) != expected


def _unsupported_transfer_syntax(dataset) -> bool:
    file_meta = getattr(dataset, "file_meta", None)
    syntax = getattr(file_meta, "TransferSyntaxUID", None)
    if syntax is None:
        return False
    from pydicom.pixels import get_decoder
    try:
        get_decoder(str(syntax))
    except NotImplementedError:
        return True
    return False


def _preprocess_recovery_v1(
    series_path: str | Path,
    *, max_slices: int, output_size: int, lower_percentile: float,
    upper_percentile: float, reader: Callable[..., Any],
) -> DicomPreprocessedSeries:
    """Explicit partial-input policy; strict mode remains the default path."""
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

    reasons: dict[str, int] = {}
    slice_reasons = []
    def record_skip(reason: str, ordinal: int) -> None:
        reasons[reason] = reasons.get(reason, 0) + 1
        slice_reasons.append({"slice_ordinal": ordinal, "reason_code": reason})
    candidates = []
    for ordinal, path in enumerate(paths, 1):
        if path.is_symlink():
            _fail("symlink_path", "The selected series contains a symlinked DICOM file.")
        try:
            dataset = reader(path, stop_before_pixels=True, specific_tags=list(_HEADER_TAGS))
            header = _metadata(dataset)
        except DicomPreprocessingError as exc:
            if not _recoverable_header_reason(exc):
                raise
            record_skip(exc.reason_code, ordinal)
            continue
        except (FileNotFoundError, EOFError):
            record_skip("header_read_failed", ordinal)
            continue
        except Exception as exc:
            if not _is_known_reader_failure(exc):
                raise
            record_skip("header_read_failed", ordinal)
            continue
        candidates.append((ordinal, path, header))
    if not candidates:
        _recovery_failure("series_files_missing", "The selected series has no usable DICOM slices.", reasons, slice_reasons)

    # Partition by the unchanged PP-02 geometry rules, preserving stable file order.
    groups = []
    for candidate in candidates:
        header = candidate[2]
        for group in groups:
            reference = group[0][2]
            if (_orientations_match(reference["orientation"], header["orientation"])
                    and _spacings_match(reference["spacing"], header["spacing"])
                    and (reference["rows"], reference["columns"], reference["samples"], reference["photometric"])
                    == (header["rows"], header["columns"], header["samples"], header["photometric"])):
                group.append(candidate)
                break
        else:
            groups.append([candidate])
    largest = max(map(len, groups))
    dominant = [group for group in groups if len(group) == largest]
    if len(dominant) != 1:
        for ordinal, _, _ in candidates:
            record_skip("inconsistent_pixel_layout", ordinal)
        _recovery_failure("inconsistent_pixel_layout", "The series has ambiguous geometry groups.", reasons, slice_reasons)
    selected_geometry = dominant[0]
    discarded = len(candidates) - len(selected_geometry)
    if discarded:
        keep = {item[0] for item in selected_geometry}
        for ordinal, _, _ in candidates:
            if ordinal not in keep:
                record_skip("inconsistent_pixel_layout", ordinal)

    by_position: dict[float, tuple[int, Path, dict[str, Any]]] = {}
    for candidate in selected_geometry:
        position = round(candidate[2]["coordinate"], 6)
        previous = by_position.get(position)
        if previous is None:
            by_position[position] = candidate
            continue
        try:
            identical = previous[1].read_bytes() == candidate[1].read_bytes()
        except OSError:
            _fail("header_read_failed", "A DICOM slice could not be compared safely.")
        if not identical:
            record_skip("duplicate_slice_position", candidate[0])
            _recovery_failure("duplicate_slice_position", "Conflicting slices share one projected position.", reasons, slice_reasons)
        record_skip("duplicate_slice_position", candidate[0])

    ordered = sorted(
        ((position, path, header) for position, (_, path, header) in by_position.items()),
        key=lambda item: item[0],
    )
    ordinal_by_path = {path: ordinal for ordinal, path, _ in selected_geometry}
    if not ordered:
        _recovery_failure("series_files_missing", "The selected series has no usable DICOM slices.", reasons, slice_reasons)
    indices = uniformly_spaced_indices(len(ordered), max_slices)
    chosen = [ordered[index] for index in indices]
    used_positions = {item[0] for item in chosen}
    decoded_by_position = {}
    def decode(candidate):
        position, path, header = candidate
        try:
            dataset = reader(path)
        except Exception as exc:
            if _is_bytes_length_exception(exc):
                raise DicomPreprocessingError("dicom_file_parse_invalid", "The DICOM element payload is malformed.") from None
            raise
        decoded = _metadata(dataset)
        if not _decoded_metadata_matches_header(header, decoded):
            _fail("decoded_metadata_mismatch", "Decoded DICOM metadata differs from its scanned header.")
        if _unsupported_transfer_syntax(dataset):
            _fail("pixel_transfer_syntax_unsupported", "The DICOM transfer syntax has no registered decoder.")
        if _native_pixel_payload_invalid(dataset, header):
            _fail("pixel_payload_invalid", "The native DICOM pixel payload length is invalid.")
        pixels = dataset.pixel_array
        shape = tuple(int(value) for value in pixels.shape)
        if shape != (header["rows"], header["columns"]):
            _fail("decoded_metadata_mismatch", "Decoded pixels do not match scanned slice geometry.")
        return DecodedSlice(
            pixels=pixels, photometric_interpretation=decoded["photometric"],
            slope=decoded["slope"], intercept=decoded["intercept"],
            pixel_padding_value=decoded["padding"], pixel_padding_range_limit=decoded["padding_limit"],
            pixel_spacing=tuple(header["spacing"]), modality_lut_sequence_present=False,
        )

    recoverable_decode_reasons = {"pixel_payload_invalid", "pixel_transfer_syntax_unsupported",
                                  "dicom_file_parse_invalid", "decoded_metadata_mismatch"}
    target_candidates = list(chosen)
    while len(decoded_by_position) < min(max_slices, len(ordered)):
        if not target_candidates:
            break
        candidate = target_candidates.pop(0)
        try:
            decoded_by_position[candidate[0]] = decode(candidate)
        except DicomPreprocessingError as exc:
            if exc.reason_code not in recoverable_decode_reasons:
                raise
            record_skip(exc.reason_code, ordinal_by_path[candidate[1]])
            replacements = sorted(
                (item for item in ordered if item[0] not in used_positions),
                key=lambda item: (abs(item[0] - candidate[0]), item[0]),
            )
            if replacements:
                replacement = replacements[0]
                used_positions.add(replacement[0])
                target_candidates.append(replacement)
        if len(target_candidates) == 0 and len(decoded_by_position) < min(max_slices, len(ordered)):
            # Exhausted valid unselected cuts; retain only decoded originals.
            break
    if not decoded_by_position:
        _recovery_failure("series_files_missing", "The selected series has no decodable DICOM slices.", reasons, slice_reasons)
    chosen_positions = sorted(decoded_by_position)
    decoded_slices = [decoded_by_position[position] for position in chosen_positions]
    reference = ordered[0][2]
    try:
        output: PreprocessedSeries = preprocess_series(
            decoded_slices, max_slices=max_slices, output_size=output_size,
            lower_percentile=lower_percentile, upper_percentile=upper_percentile,
        )
    except (ImportError, ValueError):
        _fail("preprocessing_failed", "The selected DICOM series failed PP-01 validation.")
    skipped = sum(reasons.values())
    return DicomPreprocessedSeries(output.images, output.valid_mask, {
        "preprocessing_version": "pp02-v1", "slice_count_scanned": len(paths),
        "slice_count_selected": len(decoded_slices), "selected_positions_mm": [float(position) for position in chosen_positions],
        "pp01": output.metadata,
        "recovery": {"version": 1, "slice_count_skipped": skipped,
                     "reason_counts": dict(sorted(reasons.items())),
                     "slice_reasons": sorted(slice_reasons, key=lambda item: item["slice_ordinal"])},
    })


def preprocess_dicom_series(
    series_path: str | Path,
    *,
    max_slices: int = 8,
    output_size: int = 256,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    dcmread: Callable[..., Any] | None = None,
    recovery_policy_version: int | None = None,
) -> DicomPreprocessedSeries:
    """Read one explicitly chosen directory, order headers, select, decode, and run PP-01.

    A limit of one selects the first slice in ascending projected patient-space
    coordinate. No filename or instance-number order is used for slice selection.
    """
    if isinstance(max_slices, bool) or not isinstance(max_slices, int) or not 1 <= max_slices <= 8:
        _fail("invalid_limit", "max_slices must be an integer from one through eight.")
    if recovery_policy_version not in (None, 1) or isinstance(recovery_policy_version, bool):
        _fail("invalid_limit", "Unsupported partial-input recovery policy version.")
    if recovery_policy_version == 1:
        return _preprocess_recovery_v1(
            series_path, max_slices=max_slices, output_size=output_size,
            lower_percentile=lower_percentile, upper_percentile=upper_percentile,
            reader=dcmread or _load_reader(),
        )
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
            if not _orientations_match(reference["orientation"], header["orientation"]):
                _fail("inconsistent_orientation", "The series contains inconsistent slice orientation.")
            if not _spacings_match(reference["spacing"], header["spacing"]):
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
                # All scanned headers and each decoded/header pair passed PP-02
                # geometry checks. Canonicalize metadata for PP-01's exact contract;
                # pixel-space resize does not use spacing to resample the image.
                pixel_spacing=tuple(reference["spacing"]),
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
