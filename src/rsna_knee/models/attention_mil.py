"""Target-specific hierarchical attention over a study's slices and series."""
from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F

from rsna_knee.models.resnet50 import ResNet50StudyModel


def masked_target_softmax(scores: torch.Tensor, valid: torch.Tensor, *, dim: int = 0) -> torch.Tensor:
    """Softmax a [items, targets] score matrix, excluding invalid items."""
    if (not isinstance(scores, torch.Tensor) or scores.ndim != 2 or not scores.is_floating_point()
            or not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool or valid.ndim != 1
            or valid.device != scores.device or dim not in (0, 1)
            or scores.shape[dim] != valid.numel() or not torch.isfinite(scores).all()):
        raise ValueError("attention_input_invalid")
    if not bool(valid.any()):
        raise ValueError("attention_mask_empty")
    mask_shape = [1, 1]
    mask_shape[dim] = valid.numel()
    masked = scores.masked_fill(~valid.reshape(mask_shape), torch.finfo(scores.dtype).min)
    return torch.softmax(masked, dim=dim).masked_fill(~valid.reshape(mask_shape), 0.0)


def target_weighted_sum(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Reduce [items, targets] values with per-target item weights."""
    if (not isinstance(values, torch.Tensor) or not values.is_floating_point() or values.ndim != 2
            or not isinstance(weights, torch.Tensor) or weights.shape != values.shape
            or weights.device != values.device or not torch.isfinite(values).all()
            or not torch.isfinite(weights).all() or torch.any(weights < 0)):
        raise ValueError("attention_input_invalid")
    return (values * weights).sum(dim=0)


class RadImageNetAttentionStudyModel(ResNet50StudyModel):
    """ResNet50StudyModel with zero-initialized per-target slice/series score heads."""

    def __init__(self, backbone, *, seed: int = 42, slice_microbatch: int = 2):
        super().__init__(backbone, arm="radimagenet", seed=seed, slice_microbatch=slice_microbatch)
        self.slice_attention = nn.Linear(2048, 12)
        self.series_attention = nn.Linear(2048, 12)
        nn.init.zeros_(self.slice_attention.weight)
        nn.init.zeros_(self.slice_attention.bias)
        nn.init.zeros_(self.series_attention.weight)
        nn.init.zeros_(self.series_attention.bias)

    @classmethod
    def from_base(cls, base: ResNet50StudyModel) -> "RadImageNetAttentionStudyModel":
        if not isinstance(base, ResNet50StudyModel) or base.arm != "radimagenet":
            raise ValueError("attention_base_invalid")
        model = cls(copy.deepcopy(base.backbone), seed=base.head_seed if base.head_seed is not None else 42,
                    slice_microbatch=base.slice_microbatch)
        model.classifier.load_state_dict(base.classifier.state_dict(), strict=True)
        return model

    def _pooled_slice_features(self, images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        if (images.ndim != 4 or masks.shape != images.shape or images.shape[1] != 1
                or not images.is_floating_point() or masks.dtype != torch.bool
                or images.device != masks.device or images.shape[0] < 1
                or images.shape[-2] < 8 or images.shape[-1] < 8):
            raise ValueError("study_input_invalid")
        valid = masks.flatten(1).any(dim=1)
        if not bool(valid.all()):
            raise ValueError("study_mask_invalid")
        if not torch.isfinite(images).all() or torch.any(images < 0) or torch.any(images > 1):
            raise ValueError("study_input_invalid")

        pooled_slices = []
        for start in range(0, images.shape[0], self.slice_microbatch):
            image_batch = images[start:start + self.slice_microbatch]
            mask_batch = masks[start:start + self.slice_microbatch]
            rgb = image_batch.repeat(1, 3, 1, 1)
            rgb = (rgb - self.imagenet_mean.to(dtype=rgb.dtype)) / self.imagenet_std.to(dtype=rgb.dtype)
            rgb = rgb.masked_fill(~mask_batch.repeat(1, 3, 1, 1), 0.0)
            if self.training and self.checkpoint_trunk and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                features = checkpoint(self.backbone_forward, rgb, use_reentrant=False)
            else:
                features = self.backbone_forward(rgb)
            spatial_mask = F.interpolate(mask_batch.to(dtype=features.dtype), size=features.shape[-2:], mode="area")
            denominator = spatial_mask.sum(dim=(2, 3)).clamp_min(torch.finfo(features.dtype).eps)
            pooled_slices.append((features * spatial_mask).sum(dim=(2, 3)) / denominator)
        return torch.cat(pooled_slices, dim=0)

    def forward(self, images_by_series: list[torch.Tensor], masks_by_series: list[torch.Tensor]) -> torch.Tensor:
        if (not isinstance(images_by_series, (list, tuple)) or not images_by_series
                or not isinstance(masks_by_series, (list, tuple))
                or len(images_by_series) != len(masks_by_series)):
            raise ValueError("study_input_invalid")

        series_logits = []
        series_features = []
        for images, masks in zip(images_by_series, masks_by_series):
            features = self._pooled_slice_features(images, masks)
            slice_logits = self.classifier(features)
            slice_weights = masked_target_softmax(self.slice_attention(features),
                                                  torch.ones(features.shape[0], dtype=torch.bool,
                                                             device=features.device))
            series_logits.append(target_weighted_sum(slice_logits, slice_weights))
            series_features.append(features.mean(dim=0))

        logits_by_series = torch.stack(series_logits, dim=0)
        series_features_tensor = torch.stack(series_features, dim=0)
        series_weights = masked_target_softmax(self.series_attention(series_features_tensor),
                                               torch.ones(len(series_logits), dtype=torch.bool,
                                                          device=logits_by_series.device))
        return target_weighted_sum(logits_by_series, series_weights).reshape(1, 12)

    def optimizer_groups(self, *, head_lr=1e-3, backbone_lr=1e-5):
        heads = [*self.classifier.parameters(), *self.slice_attention.parameters(), *self.series_attention.parameters()]
        return [{"params": heads, "lr": head_lr, "name": "head"},
                {"params": [p for p in self.backbone.layer4.parameters() if p.requires_grad],
                 "lr": backbone_lr, "name": "layer4"}]
