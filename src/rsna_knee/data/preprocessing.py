"""Versioned local preprocessing for already-decoded grayscale slices.

This module deliberately does not read DICOM files or infer slice geometry. Callers
must provide slices in geometric order and explicit metadata from their decoder.
NumPy and Pillow are optional until :func:`preprocess_series` is called.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite
from numbers import Integral, Real
from typing import Any, Sequence


@dataclass(frozen=True)
class DecodedSlice:
    """One decoded 2D stored-value array and its explicit intensity metadata."""

    pixels: Any
    photometric_interpretation: str
    slope: float | None = None
    intercept: float | None = None
    pixel_padding_value: float | int | None = None
    pixel_padding_range_limit: float | int | None = None
    pixel_spacing: tuple[float, float] | None = None
    modality_lut_sequence_present: bool = False


@dataclass(frozen=True)
class PreprocessedSeries:
    """Tensor-like output, validity mask, and JSON-serializable transform record."""

    images: Any
    valid_mask: Any
    metadata: dict[str, Any]


def uniformly_spaced_indices(length: int, limit: int = 8) -> tuple[int, ...]:
    """Choose unique, ordered, endpoint-inclusive indices from an ordered series.

    The input sequence must already be in geometric order. This helper has no
    geometry fields and never sorts or otherwise reorders its caller's slices.
    """

    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise ValueError("length must be a positive integer")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    count = min(length, limit)
    if count == 1:
        return (0,)
    last = length - 1
    indices = tuple((i * last) // (count - 1) for i in range(count))
    return indices


def preprocess_series(
    slices: Sequence[DecodedSlice],
    *,
    max_slices: int = 8,
    output_size: int = 256,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> PreprocessedSeries:
    """Normalize, resize, and letterbox selected slices from one series.

    Padding is excluded in raw stored-value space before the linear modality
    transform. One percentile range is fit to all selected valid pixels in this
    series; no cohort-wide or label-dependent statistic is consulted.
    """

    import numpy as np
    from PIL import Image

    _validate_parameters(slices, max_slices, output_size, lower_percentile, upper_percentile)
    chosen_indices = uniformly_spaced_indices(len(slices), max_slices)
    selected = [slices[index] for index in chosen_indices]
    raw_arrays: list[Any] = []
    for item in selected:
        raw = np.asarray(item.pixels)
        if raw.ndim != 2 or min(raw.shape) < 1 or not np.issubdtype(raw.dtype, np.number):
            raise ValueError("each decoded slice must be a non-empty 2D numeric array")
        if not np.isrealobj(raw):
            raise ValueError("decoded pixels must be real-valued; complex pixels are unsupported")
        if not np.isfinite(raw).all():
            raise ValueError("decoded pixels must all be finite")
        if np.issubdtype(raw.dtype, np.integer):
            minimum = int(np.min(raw))
            maximum = int(np.max(raw))
            if minimum < -(2**53) or maximum > 2**53:
                raise ValueError("integer decoded pixels must fit the exact float64 integer range")
        raw_arrays.append(raw)
    _validate_series_metadata(selected, raw_arrays)

    valid_arrays: list[Any] = []
    linear_arrays: list[Any] = []
    for item, raw in zip(selected, raw_arrays):
        valid = np.ones(raw.shape, dtype=np.bool_)
        padding = item.pixel_padding_value
        range_limit = item.pixel_padding_range_limit
        if range_limit is not None and padding is None:
            raise ValueError("pixel padding range limit requires a padding value")
        if padding is not None:
            padding = _finite_real_number(padding, "pixel padding value")
            if range_limit is not None:
                range_limit = _finite_real_number(range_limit, "pixel padding range limit")
            if range_limit is None:
                valid &= raw != padding
            else:
                low, high = sorted((padding, range_limit))
                valid &= (raw < low) | (raw > high)
        if not valid.any():
            raise ValueError("each selected slice must contain at least one valid pixel")

        if item.slope is None and item.intercept is None:
            slope, intercept = 1.0, 0.0
        else:
            if item.slope is None or item.intercept is None:
                raise ValueError("rescale slope and intercept must be supplied together")
            slope = _finite_real_number(item.slope, "rescale slope")
            intercept = _finite_real_number(item.intercept, "rescale intercept")
            if slope == 0:
                raise ValueError("rescale slope must be non-zero")

        linear = raw.astype(np.float64, copy=False) * slope + intercept
        if not np.isfinite(linear).all():
            raise ValueError("rescaled pixels must all be finite")
        valid_arrays.append(valid)
        linear_arrays.append(linear)

    concatenated_valid = np.concatenate(
        [linear[valid] for linear, valid in zip(linear_arrays, valid_arrays)]
    )
    with np.errstate(over="ignore", invalid="ignore"):
        low = float(np.percentile(concatenated_valid, lower_percentile, method="linear"))
        high = float(np.percentile(concatenated_valid, upper_percentile, method="linear"))
    observed_min = float(np.min(concatenated_valid))
    observed_max = float(np.max(concatenated_valid))
    if not isfinite(low) or not isfinite(high):
        raise ValueError("normalization percentile bounds must be finite")
    used_fallback = False
    is_constant = observed_min == observed_max
    if low == high and not is_constant:
        low, high = observed_min, observed_max
        used_fallback = True
    with np.errstate(over="ignore", invalid="ignore"):
        value_range = high - low
    if not isfinite(value_range):
        raise ValueError("normalization range is not finite or representable")

    output = np.zeros((len(selected), 1, output_size, output_size), dtype=np.float32)
    output_mask = np.zeros((len(selected), 1, output_size, output_size), dtype=np.bool_)
    transforms: list[dict[str, Any]] = []

    for out_index, (item, linear, valid) in enumerate(zip(selected, linear_arrays, valid_arrays)):
        if is_constant:
            normalized = np.zeros(linear.shape, dtype=np.float32)
        else:
            normalized = np.clip((linear - low) / value_range, 0.0, 1.0).astype(np.float32)
        if item.photometric_interpretation == "MONOCHROME1" and not is_constant:
            normalized[valid] = 1.0 - normalized[valid]
        normalized[~valid] = 0.0

        height, width = normalized.shape
        scale = min(output_size / height, output_size / width)
        resized_height = max(1, min(output_size, int(floor(height * scale + 0.5))))
        resized_width = max(1, min(output_size, int(floor(width * scale + 0.5))))
        top = (output_size - resized_height) // 2
        left = (output_size - resized_width) // 2
        bottom = output_size - resized_height - top
        right = output_size - resized_width - left

        weighted = normalized * valid.astype(np.float32)
        resampled = Image.fromarray(weighted, mode="F").resize(
            (resized_width, resized_height), resample=Image.Resampling.BILINEAR
        )
        resampled_weight = Image.fromarray(valid.astype(np.float32), mode="F").resize(
            (resized_width, resized_height), resample=Image.Resampling.BILINEAR
        )
        resampled_mask = Image.fromarray(valid.astype(np.uint8) * 255, mode="L").resize(
            (resized_width, resized_height), resample=Image.Resampling.NEAREST
        )
        weighted_pixels = np.asarray(resampled, dtype=np.float32)
        valid_weight = np.asarray(resampled_weight, dtype=np.float32)
        pixels = np.zeros(weighted_pixels.shape, dtype=np.float32)
        np.divide(weighted_pixels, valid_weight, out=pixels, where=valid_weight > 0)
        mask = np.asarray(resampled_mask, dtype=np.uint8) > 0
        y1, y2 = top, top + resized_height
        x1, x2 = left, left + resized_width
        output[out_index, 0, y1:y2, x1:x2] = pixels
        output_mask[out_index, 0, y1:y2, x1:x2] = mask
        output[out_index, 0, ~output_mask[out_index, 0]] = 0.0

        spacing = item.pixel_spacing
        transforms.append(
            {
                "native_shape": [int(height), int(width)],
                "native_pixel_spacing": None if spacing is None else [float(spacing[0]), float(spacing[1])],
                "resized_shape": [resized_height, resized_width],
                "scale_y": resized_height / height,
                "scale_x": resized_width / width,
                "padding_top": top,
                "padding_bottom": bottom,
                "padding_left": left,
                "padding_right": right,
            }
        )

    metadata = {
        "version": "pp01-v1",
        "config": {
            "max_slices": max_slices,
            "output_size": output_size,
            "lower_percentile": float(lower_percentile),
            "upper_percentile": float(upper_percentile),
            "resize": "bilinear",
            "letterbox": True,
            "aspect_policy": "pixel-space",
        },
        "selected_indices": list(chosen_indices),
        "photometric_interpretation": selected[0].photometric_interpretation,
        "normalization": {
            "scope": "series",
            "method": f"percentile_{_percentile_label(lower_percentile)}_{_percentile_label(upper_percentile)}_linear",
            "lower_bound": low,
            "upper_bound": high,
            "valid_min": observed_min,
            "valid_max": observed_max,
            "used_min_max_fallback": used_fallback,
            "is_constant": is_constant,
            "padding_excluded": True,
        },
        "transforms": transforms,
    }
    return PreprocessedSeries(output, output_mask, metadata)


def _validate_parameters(
    slices: Sequence[DecodedSlice],
    max_slices: int,
    output_size: int,
    lower_percentile: float,
    upper_percentile: float,
) -> None:
    if not slices:
        raise ValueError("at least one slice is required")
    if isinstance(max_slices, bool) or not isinstance(max_slices, int) or max_slices < 1:
        raise ValueError("max_slices must be a positive integer")
    if isinstance(output_size, bool) or not isinstance(output_size, int) or output_size < 1:
        raise ValueError("output_size must be a positive integer")
    lower = _finite_real_number(lower_percentile, "lower percentile")
    upper = _finite_real_number(upper_percentile, "upper percentile")
    if not (0 <= lower < upper <= 100):
        raise ValueError("percentiles must satisfy 0 <= lower < upper <= 100")
    if any(not isinstance(item, DecodedSlice) for item in slices):
        raise ValueError("slices must contain DecodedSlice values")


def _validate_series_metadata(slices: Sequence[DecodedSlice], raw_arrays: Sequence[Any]) -> None:
    first_photo = slices[0].photometric_interpretation
    if first_photo not in {"MONOCHROME1", "MONOCHROME2"}:
        raise ValueError("only MONOCHROME1 and MONOCHROME2 are supported")
    first_spacing = _validated_spacing(slices[0].pixel_spacing)
    first_shape = raw_arrays[0].shape
    for item, raw in zip(slices, raw_arrays):
        if item.modality_lut_sequence_present:
            raise ValueError("Modality LUT sequences are unsupported")
        if item.photometric_interpretation != first_photo:
            raise ValueError("photometric interpretation must be consistent within a series")
        if _validated_spacing(item.pixel_spacing) != first_spacing:
            raise ValueError("pixel spacing must be consistent within a series")
        if raw.shape != first_shape:
            raise ValueError("decoded slice shape must be consistent within a series")


def _validated_spacing(spacing: tuple[float, float] | None) -> tuple[float, float] | None:
    if spacing is None:
        return None
    if len(spacing) != 2:
        raise ValueError("pixel spacing must contain row and column spacing")
    row = _finite_real_number(spacing[0], "row pixel spacing")
    column = _finite_real_number(spacing[1], "column pixel spacing")
    if row <= 0 or column <= 0:
        raise ValueError("pixel spacing values must be positive")
    return row, column


def _finite_real_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real numeric value, not bool or string")
    if isinstance(value, Integral) and abs(int(value)) > 2**53:
        raise ValueError(f"{name} must fit the exact float64 integer range")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be a finite real number")
    return number


def _percentile_label(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value).replace(".", "p")
