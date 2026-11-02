"""Read-only preparation contract for the isolated sixteen-slice attention run."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class Local16PreparationError(ValueError):
    """Safe reason-coded refusal for an invalid local sixteen-slice plan."""

    def __init__(self, reason_code: str, message: str):
        super().__init__(message)
        self.reason_code = reason_code


def build_local16_plan(
    raw_dicom_root: str | Path,
    *,
    cache_max_slices: int | None = None,
) -> dict[str, Any]:
    """Return a new-run recipe without scanning pixels or creating output files.

    This deliberately does not accept cache paths or inspect DICOM contents.
    ``cache_max_slices`` is only a caller-declared limit check; it does not
    authenticate cache bytes, prove how a cache was built, or validate the raw
    root beyond checking that it is an ordinary directory.
    """
    if cache_max_slices is not None and (
        isinstance(cache_max_slices, bool)
        or not isinstance(cache_max_slices, int)
        or cache_max_slices != 16
    ):
        raise Local16PreparationError(
            "cache_slice_limit_mismatch",
            "The caller-declared cache limit does not match the sixteen-slice arm.",
        )

    root = Path(raw_dicom_root)
    if root.is_symlink() or not root.is_dir():
        raise Local16PreparationError(
            "raw_dicom_root_unavailable",
            "The explicit raw DICOM root is unavailable; no cache or download fallback is used.",
        )

    return {
        "experiment_id": "E050",
        "output_identity": "local16-attention-run01",
        "raw_dicom_root": str(root.resolve()),
        "cache_policy": "build_new_from_raw_dicom",
        "cache_max_slices": 16,
        "cache_authentication_performed": False,
        "raw_dicom_content_validated": False,
        "requires_new_source_fingerprint": True,
        "fold": 0,
        "gold58_excluded": True,
        "study_level_folds_unchanged": True,
        "seed": 42,
        "backbone": "RadImageNet ResNet-50",
        "comparison": "target_attention_8_vs_target_attention_16",
        "initialization": "fresh_original_radiimagenet_weights_seed_42_each_arm",
        "same_fresh_initialization_protocol": True,
        "checkpoint_warm_start": False,
        "selected_epoch": 5,
        "arms": {
            "target_attention_8": {"max_slices": 8, "seed": 42, "epochs": 5},
            "target_attention_16": {"max_slices": 16, "seed": 42, "epochs": 5},
        },
        "output_size": 224,
        "max_slices": 16,
        "normalization": "radimagenet_2x_minus_1",
        "masked_spatial_pooling": True,
        "slice_microbatch": 2,
        "optimizer": "Adam",
        "head_learning_rate": 1e-3,
        "layer4_learning_rate": 1e-5,
        "epochs_per_arm": 5,
        "batch_norm_frozen": True,
        "plan_only": True,
        "training_started": False,
        "output_created": False,
        "downloads_enabled": False,
    }
