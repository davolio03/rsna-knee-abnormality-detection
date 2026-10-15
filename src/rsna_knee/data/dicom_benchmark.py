"""Bounded, aggregate-only DICOM reader benchmark for a private cohort."""

from __future__ import annotations

import importlib
import math
import time
from pathlib import Path

from rsna_knee.validation.cohort import _validate_manifest as _validate_cohort_manifest


class DicomBenchmarkError(ValueError):
    """A safe, categorical failure from the reader benchmark."""

    def __init__(self, reason_code: str, safe_message: str):
        super().__init__(safe_message)
        self.reason_code = reason_code


_TAGS = [
    "StudyInstanceUID", "SeriesInstanceUID", "ImageOrientationPatient",
    "ImagePositionPatient", "PixelSpacing", "Rows", "Columns",
    "SamplesPerPixel", "PhotometricInterpretation", "NumberOfFrames",
]
_DEFAULTS = {
    "max_file_bytes": 256 * 1024 * 1024,
    "max_total_input_bytes": 2 * 1024 * 1024 * 1024,
    "max_pixels_per_slice": 100_000_000,
    "max_header_files": 10_000,
    "max_seconds": 300.0,
}


def _load_dependencies():
    """Load optional image dependencies only when a run is requested."""
    try:
        return importlib.import_module("pydicom"), importlib.import_module("numpy")
    except ImportError as exc:
        raise DicomBenchmarkError(
            "missing_dependency",
            "Install pydicom and numpy in the prepared runtime before running this benchmark.",
        ) from None


def _path_is_symlink(path: Path) -> bool:
    try:
        return path.is_symlink()
    except OSError:
        _fail("filesystem_check_failed", "An image-tree path could not be checked safely.")


def _fail(code: str, message: str):
    raise DicomBenchmarkError(code, message)


def _safe_component(value, name: str) -> str:
    if (not isinstance(value, str) or not value or value in {".", ".."}
            or "/" in value or "\\" in value or "\x00" in value):
        _fail("unsafe_path_component", f"Manifest contains an unsafe {name} path component.")
    return value


def _validate_manifest(manifest):
    try:
        records = _validate_cohort_manifest(manifest)
    except (TypeError, ValueError):
        _fail("invalid_manifest", "The private cohort manifest failed schema validation.")
    seen_studies = set()
    for record in records:
        if not isinstance(record, dict):
            _fail("invalid_manifest", "The private cohort manifest contains an invalid study record.")
        study = _safe_component(record.get("study_id"), "study")
        if study in seen_studies:
            _fail("invalid_manifest", "The private cohort manifest contains duplicate study IDs.")
        seen_studies.add(study)
        series = record.get("series")
        if not isinstance(series, list) or not series:
            _fail("invalid_manifest", "A selected study has no series metadata.")
        seen_series = set()
        for item in series:
            if not isinstance(item, dict):
                _fail("invalid_manifest", "The private cohort manifest contains invalid series metadata.")
            series_id = _safe_component(item.get("series_id"), "series")
            if series_id in seen_series:
                _fail("invalid_manifest", "A study contains duplicate series IDs.")
            seen_series.add(series_id)
    return records


def _safe_child(root: Path, *parts: str) -> Path:
    candidate = root.joinpath(*parts)
    current = root
    for part in parts:
        current = current / part
        if _path_is_symlink(current):
            _fail("symlink_path", "The image tree contains a symlinked path.")
    try:
        candidate.resolve(strict=True).relative_to(root)
    except (OSError, ValueError):
        _fail("unsafe_path_component", "An image path escapes the configured image root.")
    return candidate


def _finite_vector(value, count: int):
    if isinstance(value, (str, bytes)):
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if len(result) != count or not all(math.isfinite(item) for item in result):
        return None
    return result


def _dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def _geometry(dataset):
    orientation = _finite_vector(getattr(dataset, "ImageOrientationPatient", None), 6)
    position = _finite_vector(getattr(dataset, "ImagePositionPatient", None), 3)
    spacing = _finite_vector(getattr(dataset, "PixelSpacing", None), 2)
    if orientation is None or position is None or spacing is None or any(v <= 0 for v in spacing):
        _fail("missing_geometry", "A DICOM slice is missing valid patient geometry.")
    row, column = orientation[:3], orientation[3:]
    row_norm, column_norm = math.sqrt(_dot(row, row)), math.sqrt(_dot(column, column))
    if abs(row_norm - 1.0) > 1e-4 or abs(column_norm - 1.0) > 1e-4 or abs(_dot(row, column)) > 1e-4:
        _fail("missing_geometry", "A DICOM slice has invalid patient orientation vectors.")
    normal = [row[1] * column[2] - row[2] * column[1],
              row[2] * column[0] - row[0] * column[2],
              row[0] * column[1] - row[1] * column[0]]
    return orientation, spacing, _dot(normal, position)


def _header(pydicom, path, expected_study, expected_series):
    try:
        dataset = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=_TAGS)
    except Exception:
        _fail("header_read_failed", "A DICOM header could not be read safely.")
    if getattr(dataset, "StudyInstanceUID", None) != expected_study:
        _fail("study_identity_mismatch", "A DICOM study UID does not match the selected manifest study.")
    if getattr(dataset, "SeriesInstanceUID", None) != expected_series:
        _fail("series_identity_mismatch", "A DICOM series UID does not match the selected manifest series.")
    orientation, spacing, coordinate = _geometry(dataset)
    rows, columns = getattr(dataset, "Rows", None), getattr(dataset, "Columns", None)
    samples = getattr(dataset, "SamplesPerPixel", None)
    photo = getattr(dataset, "PhotometricInterpretation", None)
    frames = getattr(dataset, "NumberOfFrames", 1)
    if (not isinstance(rows, int) or not isinstance(columns, int) or rows < 1 or columns < 1
            or not isinstance(samples, int) or samples != 1
            or photo not in {"MONOCHROME1", "MONOCHROME2"}):
        _fail("unsupported_pixel_layout", "Only two-dimensional monochrome DICOM slices are supported.")
    try:
        frames = int(frames)
    except (TypeError, ValueError):
        frames = -1
    if frames != 1:
        _fail("unsupported_frame_count", "Multiframe DICOM objects are not supported.")
    return {"path": path, "orientation": orientation, "spacing": spacing,
            "coordinate": coordinate, "rows": rows, "columns": columns}


def _check_deadline(started: float, max_seconds: float):
    if time.monotonic() - started > max_seconds:
        _fail("time_limit", "The DICOM benchmark exceeded its configured time limit.")


def run_dicom_benchmark(
    manifest,
    image_root,
    *,
    max_studies: int = 5,
    max_slices_per_series: int = 8,
    max_file_bytes: int = _DEFAULTS["max_file_bytes"],
    max_total_input_bytes: int = _DEFAULTS["max_total_input_bytes"],
    max_pixels_per_slice: int = _DEFAULTS["max_pixels_per_slice"],
    max_header_files: int = _DEFAULTS["max_header_files"],
    max_seconds: float = _DEFAULTS["max_seconds"],
):
    """Decode a deterministic, bounded manifest sample; return aggregate metadata only."""
    records = _validate_manifest(manifest)
    limits = (max_studies, max_slices_per_series, max_file_bytes,
              max_total_input_bytes, max_pixels_per_slice, max_header_files)
    if (any(type(value) is not int or value < 1 for value in limits)
            or isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float))
            or not math.isfinite(max_seconds) or max_seconds <= 0):
        _fail("invalid_limit", "All benchmark limits must be positive.")
    pydicom, numpy = _load_dependencies()
    try:
        root = Path(image_root).resolve(strict=True)
    except (OSError, TypeError, ValueError):
        _fail("image_root_unavailable", "The configured image root is unavailable.")
    try:
        root_is_directory = root.is_dir()
        root_is_symlink = _path_is_symlink(Path(image_root))
    except OSError:
        _fail("image_root_unavailable", "The configured image root is unavailable.")
    if not root_is_directory or root_is_symlink:
        _fail("image_root_unavailable", "The configured image root must be a real directory.")
    started = time.monotonic()
    selected = records[:max_studies]
    result = {"benchmark": "raw-dicom-decode-v1", "studies_requested": len(selected),
              "series_requested": 0, "header_files_scanned": 0, "slices_decoded": 0,
              "decoded_input_bytes": 0, "decoded_pixel_bytes": 0,
              "max_pixel_shape": None, "decode_failures": {}}
    for record in selected:
        study_id = record["study_id"]
        _safe_child(root, study_id)
        for series in sorted(record["series"], key=lambda item: item["series_id"]):
            series_id = series["series_id"]
            series_dir = _safe_child(root, study_id, series_id)
            try:
                series_is_directory = series_dir.is_dir()
            except OSError:
                _fail("series_directory_unavailable", "A selected series directory is unavailable.")
            if not series_is_directory:
                _fail("series_directory_missing", "A selected series directory is missing.")
            files = []
            try:
                for path in series_dir.iterdir():
                    if path.suffix.lower() != ".dcm":
                        continue
                    files.append(path)
                    if len(files) > max_header_files - result["header_files_scanned"]:
                        _fail("header_file_limit", "The DICOM header scan exceeded its configured file limit.")
            except OSError:
                _fail("series_scan_failed", "A selected series could not be enumerated safely.")
            files.sort()
            if not files:
                _fail("series_files_missing", "A selected series has no direct DICOM files.")
            result["series_requested"] += 1
            headers = []
            reference_orientation = reference_spacing = None
            seen_positions = set()
            for path in files:
                _check_deadline(started, max_seconds)
                result["header_files_scanned"] += 1
                if result["header_files_scanned"] > max_header_files:
                    _fail("header_file_limit", "The DICOM header scan exceeded its configured file limit.")
                if _path_is_symlink(path):
                    _fail("symlink_path", "The image tree contains a symlinked file.")
                try:
                    size = path.stat().st_size
                except OSError:
                    _fail("file_read_failed", "A DICOM file could not be inspected safely.")
                if size > max_file_bytes:
                    _fail("file_size_limit", "A DICOM file exceeds the configured byte limit.")
                header = _header(pydicom, path, study_id, series_id)
                if header["rows"] * header["columns"] > max_pixels_per_slice:
                    _fail("pixel_shape_limit", "A DICOM slice exceeds the configured pixel limit.")
                if reference_orientation is None:
                    reference_orientation = header["orientation"]
                    reference_spacing = header["spacing"]
                elif (any(abs(a - b) > 1e-5 for a, b in zip(reference_orientation, header["orientation"]))):
                    _fail("inconsistent_orientation", "A series contains inconsistent slice orientation.")
                elif any(abs(a - b) > 1e-5 for a, b in zip(reference_spacing, header["spacing"])):
                    _fail("inconsistent_spacing", "A series contains inconsistent pixel spacing.")
                coordinate = round(header["coordinate"], 6)
                if coordinate in seen_positions:
                    _fail("duplicate_slice_position", "A series contains duplicate projected slice positions.")
                seen_positions.add(coordinate)
                headers.append(header)
            for header in sorted(headers, key=lambda item: item["coordinate"])[:max_slices_per_series]:
                _check_deadline(started, max_seconds)
                try:
                    size = header["path"].stat().st_size
                except OSError:
                    _fail("file_read_failed", "A DICOM file could not be inspected safely.")
                if size > max_file_bytes:
                    _fail("file_size_limit", "A DICOM file exceeds the configured byte limit.")
                result["decoded_input_bytes"] += size
                if result["decoded_input_bytes"] > max_total_input_bytes:
                    _fail("total_byte_limit", "The selected DICOM files exceed the configured byte budget.")
                dataset = None
                try:
                    dataset = pydicom.dcmread(header["path"])
                    if getattr(dataset, "StudyInstanceUID", None) != study_id:
                        _fail("study_identity_mismatch", "A decoded DICOM study UID does not match the manifest.")
                    if getattr(dataset, "SeriesInstanceUID", None) != series_id:
                        _fail("series_identity_mismatch", "A decoded DICOM series UID does not match the manifest.")
                    pixels = numpy.asarray(dataset.pixel_array)
                    if getattr(pixels, "ndim", None) != 2 or list(pixels.shape) != [header["rows"], header["columns"]]:
                        _fail("unsupported_pixel_layout", "Decoded pixels do not match the monochrome slice geometry.")
                    pixel_bytes = int(pixels.nbytes)
                except DicomBenchmarkError:
                    raise
                except Exception:
                    transfer = getattr(getattr(dataset, "file_meta", None), "TransferSyntaxUID", None)
                    key = "compressed_pixel_decode_failed" if getattr(transfer, "is_compressed", False) else "pixel_decode_failed"
                    result["decode_failures"][key] = result["decode_failures"].get(key, 0) + 1
                    continue
                result["slices_decoded"] += 1
                result["decoded_pixel_bytes"] += pixel_bytes
                shape = [int(value) for value in pixels.shape]
                if result["max_pixel_shape"] is None or math.prod(shape) > math.prod(result["max_pixel_shape"]):
                    result["max_pixel_shape"] = shape
                del pixels, dataset
                _check_deadline(started, max_seconds)
    result["elapsed_seconds"] = round(max(0.0, time.monotonic() - started), 3)
    return result
