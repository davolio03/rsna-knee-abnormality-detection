"""Small, bounded DICOM decode preflight for masked-soft training inputs."""

from __future__ import annotations

import math
import time
from typing import Any


def run_bounded_decode_preflight(dataset: Any, *, max_studies: int = 8) -> dict[str, object]:
    """Decode at most eight dataset samples and return aggregate-only status."""
    if type(max_studies) is not int or not 1 <= max_studies <= 8:
        raise ValueError("preflight_limit_invalid")
    try:
        total = len(dataset)
    except Exception:
        raise ValueError("preflight_dataset_invalid") from None
    if type(total) is not int or total < 1:
        raise ValueError("preflight_dataset_empty")

    from rsna_knee.data.masked_soft_dataset import SoftDatasetError

    attempted = decoded_series = decoded_slices = 0
    started = time.perf_counter()
    for index in range(min(total, max_studies)):
        attempted += 1
        try:
            sample = dataset[index]
        except SoftDatasetError as error:
            reason = getattr(error, "reason_code", "preflight_decode_failed")
            if type(reason) is not str or not reason.isascii() or not reason.replace("_", "").isalnum():
                reason = "preflight_decode_failed"
            return {
                "status": "failed", "reason_code": reason,
                "attempted_study_count": attempted,
                "decoded_study_count": attempted - 1,
                "decoded_series_count": decoded_series,
                "decoded_slice_count": decoded_slices,
                "elapsed_seconds": max(0.0, time.perf_counter() - started),
            }
        if type(sample) not in (tuple, list) or len(sample) != 5:
            raise ValueError("preflight_sample_invalid")
        images, pixel_masks, targets, observed, _row_index = sample
        if (type(images) not in (tuple, list) or type(pixel_masks) not in (tuple, list)
                or len(images) < 1 or len(images) != len(pixel_masks)):
            raise ValueError("preflight_sample_invalid")
        if tuple(getattr(targets, "shape", ())) != (12,) or tuple(getattr(observed, "shape", ())) != (12,):
            raise ValueError("preflight_sample_invalid")
        study_slices = 0
        for image, mask in zip(images, pixel_masks):
            image_shape = tuple(getattr(image, "shape", ()))
            mask_shape = tuple(getattr(mask, "shape", ()))
            if (len(image_shape) != 4 or image_shape != mask_shape
                    or type(image_shape[0]) is not int or image_shape[0] < 1):
                raise ValueError("preflight_sample_invalid")
            study_slices += image_shape[0]
        decoded_series += len(images)
        decoded_slices += study_slices
        del sample

    elapsed = time.perf_counter() - started
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("preflight_timing_invalid")
    return {
        "status": "complete", "attempted_study_count": attempted,
        "decoded_study_count": attempted, "decoded_series_count": decoded_series,
        "decoded_slice_count": decoded_slices, "elapsed_seconds": elapsed,
    }
