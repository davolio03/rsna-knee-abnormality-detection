"""Small scratch 2D model for study-level multilabel training."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ScratchStudyCNN(nn.Module):
    """Shared 2D encoder with mask-aware slice/series/study mean pooling.

    Inputs are lists because studies may contain different numbers of series
    and selected slices. Each series tensor is ``(slices, 1, height, width)``.
    Masks use ``True`` for valid pixels. The batch size is one study.
    """

    def __init__(self, *, feature_width: int = 64, output_count: int = 12) -> None:
        super().__init__()
        if type(feature_width) is not int or feature_width < 8:
            raise ValueError("feature_width must be an integer of at least eight")
        if type(output_count) is not int or output_count != 12:
            raise ValueError("output_count must be twelve")
        channels = (16, 32, feature_width)
        blocks = []
        in_channels = 1
        for out_channels in channels:
            blocks.extend((nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1), nn.ReLU()))
            in_channels = out_channels
        self.encoder = nn.Sequential(*blocks)
        self.classifier = nn.Linear(feature_width, output_count)

    def forward(self, series_images: Sequence[Tensor], series_masks: Sequence[Tensor]) -> Tensor:
        if not isinstance(series_images, (list, tuple)) or not series_images:
            raise ValueError("a study must contain at least one series")
        if not isinstance(series_masks, (list, tuple)) or len(series_images) != len(series_masks):
            raise ValueError("series images and masks must have matching entries")
        device = next(self.parameters()).device
        parameter_dtype = next(self.parameters()).dtype
        series_features = []
        reference_dtype = None
        for images, masks in zip(series_images, series_masks):
            if not isinstance(images, Tensor) or not isinstance(masks, Tensor):
                raise ValueError("series images and masks must be tensors")
            if images.ndim != 4 or images.shape[0] < 1 or images.shape[1] != 1 or min(images.shape[2:]) < 8:
                raise ValueError("series images must have shape (slices, 1, height, width)")
            if masks.shape != images.shape or masks.dtype != torch.bool:
                raise ValueError("series masks must be boolean and match image shapes")
            if images.device != device or masks.device != device:
                raise ValueError("series tensors must be on the model device")
            if not images.is_floating_point() or not torch.isfinite(images).all():
                raise ValueError("series images must contain finite floating-point values")
            if images.dtype != parameter_dtype:
                raise ValueError("series image dtype must match the model parameters")
            if reference_dtype is None:
                reference_dtype = images.dtype
            elif images.dtype != reference_dtype:
                raise ValueError("all series images must use the same dtype")
            valid_per_slice = masks.flatten(1).any(dim=1)
            if not bool(valid_per_slice.all()):
                raise ValueError("every selected slice must contain valid pixels")

            masked_images = images.masked_fill(~masks, 0)
            encoded = self.encoder(masked_images)
            weights = F.interpolate(masks.to(dtype=encoded.dtype), size=encoded.shape[-2:], mode="area")
            denominator = weights.sum(dim=(2, 3)).clamp_min(torch.finfo(encoded.dtype).eps)
            pooled = (encoded * weights).sum(dim=(2, 3)) / denominator
            # Equal slice weights, then equal series weights, then one study logit.
            series_features.append(pooled.mean(dim=0))
        study_features = torch.stack(series_features, dim=0).mean(dim=0).unsqueeze(0)
        return self.classifier(study_features)
