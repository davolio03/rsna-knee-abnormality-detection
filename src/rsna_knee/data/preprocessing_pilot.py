"""Five-study, ID-free harness for the provisional PP-01/PP-02 pipeline.

This module is deliberately a pilot runner, not a general cohort API. It walks
only the first five manifest records and each record's listed series, caps
selected slices at eight, and emits aggregate metadata plus anonymous panels.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Mapping
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from rsna_knee.data.dicom_preprocessing import (
    DicomPreprocessedSeries,
    DicomPreprocessingError,
    preprocess_dicom_series,
)
from rsna_knee.data.dicom_benchmark import _safe_component
from rsna_knee.data.cohort import MANIFEST_SCHEMA_VERSION


class PreprocessingPilotError(ValueError):
    """Safe top-level pilot failure without identifiers or filesystem paths."""

    def __init__(self, reason_code: str, message: str):
        super().__init__(message)
        self.reason_code = reason_code


class _InvalidSeriesResult(ValueError):
    def __init__(self, reason_code: str):
        self.reason_code = reason_code


def _validate_output(result: DicomPreprocessedSeries, output_size: int):
    try:
        import numpy as np
    except ImportError:
        raise _InvalidSeriesResult("missing_dependency") from None

    if not isinstance(result, DicomPreprocessedSeries):
        raise _InvalidSeriesResult("invalid_result")
    images = np.asarray(result.images)
    mask = np.asarray(result.valid_mask)
    if (images.ndim != 4 or images.shape[1] != 1 or images.shape[0] < 1
            or images.shape[0] > 8 or images.shape[2:] != (output_size, output_size)
            or mask.shape != images.shape):
        raise _InvalidSeriesResult("invalid_tensor_shape")
    if not np.issubdtype(images.dtype, np.number) or not np.isrealobj(images):
        raise _InvalidSeriesResult("invalid_tensor_dtype")
    if not np.isfinite(images).all() or not np.isfinite(mask).all():
        raise _InvalidSeriesResult("invalid_tensor")
    if images.size and (float(images.min()) < 0.0 or float(images.max()) > 1.0):
        raise _InvalidSeriesResult("invalid_tensor_range")
    if not np.isin(mask, (0, 1, False, True)).all():
        raise _InvalidSeriesResult("invalid_mask")
    bool_mask = mask.astype(np.bool_, copy=False)
    if np.any(images[~bool_mask] != 0):
        raise _InvalidSeriesResult("masked_pixels_nonzero")
    if not isinstance(result.metadata, dict):
        raise _InvalidSeriesResult("invalid_metadata")
    scanned = result.metadata.get("slice_count_scanned")
    selected = result.metadata.get("slice_count_selected")
    if type(scanned) is not int or scanned < 1 or type(selected) is not int:
        raise _InvalidSeriesResult("invalid_metadata")
    if selected != images.shape[0] or selected > 8:
        raise _InvalidSeriesResult("invalid_metadata")
    return images, bool_mask, scanned, selected


def _first_five_records(manifest):
    """Validate only the authorized five rows; never walk the remaining cohort."""
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise PreprocessingPilotError("invalid_manifest", "The private manifest schema is unsupported.")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) < 5:
        raise PreprocessingPilotError("insufficient_studies", "The private manifest has fewer than five studies.")
    selected = records[:5]
    seen_studies = set()
    for record in selected:
        if not isinstance(record, Mapping):
            raise PreprocessingPilotError("invalid_manifest", "A selected manifest record is invalid.")
        try:
            study_id = _safe_component(record.get("study_id"), "study")
        except ValueError:
            raise PreprocessingPilotError("invalid_manifest", "A selected study path component is invalid.") from None
        if study_id in seen_studies:
            raise PreprocessingPilotError("invalid_manifest", "Selected study records are duplicated.")
        seen_studies.add(study_id)
        series = record.get("series")
        if not isinstance(series, list) or not series:
            raise PreprocessingPilotError("invalid_manifest", "A selected study has no series metadata.")
        seen_series = set()
        for item in series:
            if not isinstance(item, Mapping):
                raise PreprocessingPilotError("invalid_manifest", "Selected series metadata is invalid.")
            try:
                series_id = _safe_component(item.get("series_id"), "series")
            except ValueError:
                raise PreprocessingPilotError("invalid_manifest", "A selected series path component is invalid.") from None
            if series_id in seen_series:
                raise PreprocessingPilotError("invalid_manifest", "A selected study contains duplicate series.")
            seen_series.add(series_id)
    return selected


def _metadata_summary(metadata: dict[str, Any], scanned: int, selected: int):
    pp01 = metadata.get("pp01")
    transforms = pp01.get("transforms") if isinstance(pp01, dict) else None
    spacing = []
    if isinstance(transforms, list):
        for transform in transforms:
            item = transform.get("native_pixel_spacing") if isinstance(transform, dict) else None
            if (isinstance(item, list) and len(item) == 2
                    and all(isinstance(value, (int, float)) and math.isfinite(value) for value in item)):
                spacing.append([float(item[0]), float(item[1])])
            else:
                spacing.append(None)
    positions = metadata.get("selected_positions_mm")
    if not isinstance(positions, list) or len(positions) != selected:
        positions = None
    elif any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in positions):
        positions = None
    return {
        "slice_count_scanned": scanned,
        "slice_count_selected": selected,
        "selected_positions_mm": positions,
        "native_pixel_spacing_by_slice": spacing if len(spacing) == selected else None,
    }


def _write_outputs(report: dict[str, Any], panels, output_dir: Path, output_size: int):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        raise PreprocessingPilotError(
            "missing_dependency", "Pillow is required to save anonymous pilot panels."
        ) from None
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = 6
    rows = max(1, (report["series_requested"] + columns - 1) // columns)
    tile_width = output_size * 2
    tile_height = output_size + 26
    montage = Image.new("RGB", (columns * tile_width, rows * tile_height), "black")
    draw = ImageDraw.Draw(montage)
    for index, panel in enumerate(panels):
        row, column = divmod(index, columns)
        x, y = column * tile_width, row * tile_height
        if panel is None:
            failure = report["series"][index]["reason_code"]
            draw.rectangle((x, y, x + tile_width - 1, y + tile_height - 1), fill=(80, 0, 0))
            draw.text((x + 3, y + 7), f"Series {index + 1:02d}: FAILED {failure}", fill="white")
            continue
        image, mask = panel
        image_panel = Image.fromarray((image * 255).round().astype("uint8"), mode="L")
        mask_panel = Image.fromarray(mask.astype("uint8") * 255, mode="L")
        montage.paste(image_panel.convert("RGB"), (x, y + 26))
        montage.paste(mask_panel.convert("RGB"), (x + output_size, y + 26))
        draw.text((x + 3, y + 7), f"Series {index + 1:02d}: processed | mask", fill="white")
    montage.save(output_dir / "preprocessing-pilot-montage.png")
    (output_dir / "preprocessing-pilot-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_preprocessing_pilot(
    manifest,
    image_root,
    *,
    series_preprocessor: Callable[..., DicomPreprocessedSeries] | None = None,
    output_size: int = 256,
    output_dir: str | Path | None = None,
):
    """Preprocess the original first five manifest studies and all their series.

    The manifest IDs are used only to construct explicit input paths and are
    never copied to the result, logs, report file, or montage labels. Errors
    are reduced to categorical codes so decoder/path details cannot leak.
    """
    if isinstance(output_size, bool) or not isinstance(output_size, int) or not 1 <= output_size <= 768:
        raise PreprocessingPilotError("invalid_limit", "output_size must be from one through 768.")
    selected = _first_five_records(manifest)
    if sum(len(record["series"]) for record in selected) != 30:
        raise PreprocessingPilotError(
            "unexpected_series_count", "The first five manifest studies must contain exactly 30 series."
        )
    processor = series_preprocessor or preprocess_dicom_series
    root = Path(image_root)
    started = time.monotonic()
    series_results = []
    panels = [None] * 30
    status_counts = Counter()
    series_index = 0
    for study_index, record in enumerate(selected, start=1):
        for _series in record["series"]:
            series_index += 1
            entry = {
                "study_index": study_index,
                "series_index": series_index,
                "status": "failed",
            }
            try:
                series_path = root / record["study_id"] / _series["series_id"]
                result = processor(series_path, max_slices=8, output_size=output_size)
                images, mask, scanned, selected_count = _validate_output(result, output_size)
                entry.update({
                    "status": "ok",
                    **_metadata_summary(result.metadata, scanned, selected_count),
                    "tensor_shape": [int(value) for value in images.shape],
                    "finite_values": True,
                    "intensity_range": [float(images.min()), float(images.max())],
                    "valid_mask_pixels": int(mask.sum()),
                    "mask_pixels": int(mask.size),
                })
                middle = len(images) // 2
                panels[series_index - 1] = (images[middle, 0].copy(), mask[middle, 0].copy())
            except DicomPreprocessingError as exc:
                entry["reason_code"] = exc.reason_code
            except _InvalidSeriesResult as exc:
                entry["reason_code"] = exc.reason_code
            except Exception:
                # Never serialize exception text: it can contain paths or identifiers.
                entry["reason_code"] = "unexpected_error"
            reason_code = entry.get("reason_code")
            if reason_code is not None and (
                not isinstance(reason_code, str) or re.fullmatch(r"[a-z0-9_]{1,48}", reason_code) is None
            ):
                entry["reason_code"] = "unexpected_error"
            status_counts[entry["status"]] += 1
            series_results.append(entry)

    status = "complete" if status_counts.get("failed", 0) == 0 else "partial"
    report = {
        "schema": "kaggle-preprocessing-pilot-v1",
        "pipeline": "pp02-v1/pp01-v1",
        "status": status,
        "studies_requested": len(selected),
        "series_requested": series_index,
        "series_succeeded": status_counts.get("ok", 0),
        "series_failed": status_counts.get("failed", 0),
        "status_counts": dict(sorted(status_counts.items())),
        "max_slices_per_series": 8,
        "output_size": output_size,
        "elapsed_seconds": round(max(0.0, time.monotonic() - started), 3),
        "panel_series_count": len(panels),
        "panel_success_count": status_counts.get("ok", 0),
        "panel_failure_placeholders": status_counts.get("failed", 0),
        "panel_description": "middle selected slice; processed image and validity mask",
        "series": series_results,
        "limitations": [
            "The first five manifest studies are a bounded pilot, not a representative cohort.",
            "PP-01 normalization and pixel-space letterboxing remain provisional.",
            "No anatomical, clinical, label-quality, or model-readiness claim is made.",
        ],
    }
    if output_dir is not None:
        _write_outputs(report, panels, Path(output_dir), output_size)
    return report
