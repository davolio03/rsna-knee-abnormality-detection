"""Study-level ResNet-18 adapter with explicit offline initialization."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


RESNET18_KINDS = frozenset({"resnet18_random", "resnet18_imagenet1k_v1"})
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGENET1K_V1_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
IMAGENET1K_V1_SIZE = 46830571


def _verified_weights(path: Path, expected_sha256: str, expected_size: int) -> dict[str, torch.Tensor]:
    if (not isinstance(path, Path) or type(expected_size) is not int or expected_size <= 0
            or type(expected_sha256) is not str or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)):
        raise ValueError("pretrained_weights_invalid")
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size != expected_size:
            raise ValueError
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected_sha256:
            raise ValueError
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        raise ValueError("pretrained_weights_unavailable") from None
    if (not isinstance(state, dict) or not state
            or any(type(key) is not str or not isinstance(value, torch.Tensor) for key, value in state.items())):
        raise ValueError("pretrained_weights_invalid")
    return state


class ResNet18StudyModel(nn.Module):
    """Apply an ImageNet ResNet-18 to every slice and equally average study logits."""

    def __init__(self, backbone: nn.Module, *, slice_microbatch: int = 2,
                 checkpoint_trunk: bool = True, freeze_backbone: bool = False):
        super().__init__()
        if type(slice_microbatch) is not int or not 1 <= slice_microbatch <= 8:
            raise ValueError("slice_microbatch_invalid")
        if type(freeze_backbone) is not bool:
            raise ValueError("freeze_backbone_invalid")
        self.backbone = backbone
        self.slice_microbatch = slice_microbatch
        self.freeze_backbone = freeze_backbone
        self.checkpoint_trunk = bool(checkpoint_trunk) and not freeze_backbone
        self.activation_checkpointing = self.checkpoint_trunk
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
        self.classifier = nn.Linear(512, 12)
        self.head_seed = None
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).reshape(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).reshape(1, 3, 1, 1)
        self.register_buffer("imagenet_mean", mean, persistent=False)
        self.register_buffer("imagenet_std", std, persistent=False)
        self.train(self.training)

    def encoder_parameters(self):
        yield from self.backbone.parameters()

    def backbone_forward(self, rgb: torch.Tensor) -> torch.Tensor:
        value = self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(rgb))))
        value = self.backbone.layer1(value)
        value = self.backbone.layer2(value)
        value = self.backbone.layer3(value)
        return self.backbone.layer4(value)

    def train(self, mode: bool = True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def _encode_series(self, images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        if (images.ndim != 4 or masks.shape != images.shape or images.shape[1] != 1
                or not images.is_floating_point() or masks.dtype != torch.bool
                or images.device != masks.device or images.shape[0] < 1
                or images.shape[-2] < 8 or images.shape[-1] < 8):
            raise ValueError("study_input_invalid")
        if not torch.isfinite(images).all() or torch.any(images < 0) or torch.any(images > 1):
            raise ValueError("study_input_invalid")
        valid = masks.flatten(1).any(dim=1)
        if not bool(valid.all()):
            raise ValueError("study_mask_invalid")

        pooled_slices = []
        for start in range(0, images.shape[0], self.slice_microbatch):
            image_batch = images[start:start + self.slice_microbatch]
            mask_batch = masks[start:start + self.slice_microbatch]
            rgb = image_batch.repeat(1, 3, 1, 1)
            rgb = (rgb - self.imagenet_mean.to(dtype=rgb.dtype)) / self.imagenet_std.to(dtype=rgb.dtype)
            rgb = rgb.masked_fill(~mask_batch.repeat(1, 3, 1, 1), 0.0)
            if self.freeze_backbone:
                with torch.no_grad():
                    features = self.backbone_forward(rgb)
            elif self.training and self.checkpoint_trunk and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint

                features = checkpoint(self.backbone_forward, rgb, use_reentrant=False)
            else:
                features = self.backbone_forward(rgb)
            weights = F.interpolate(mask_batch.to(dtype=features.dtype), size=features.shape[-2:], mode="area")
            denominator = weights.sum(dim=(2, 3)).clamp_min(torch.finfo(features.dtype).eps)
            pooled = (features * weights).sum(dim=(2, 3)) / denominator
            pooled_slices.append(self.classifier(pooled))
        return torch.cat(pooled_slices, dim=0).mean(dim=0)

    def forward(self, images_by_series: list[torch.Tensor], masks_by_series: list[torch.Tensor]) -> torch.Tensor:
        if (not isinstance(images_by_series, (list, tuple)) or not images_by_series
                or not isinstance(masks_by_series, (list, tuple))
                or len(images_by_series) != len(masks_by_series)):
            raise ValueError("study_input_invalid")
        series_logits = [self._encode_series(images, masks) for images, masks in zip(images_by_series, masks_by_series)]
        return torch.stack(series_logits, dim=0).mean(dim=0).reshape(1, 12)


def build_resnet18_study_model(
    model_kind: str,
    *,
    head_seed: int,
    weights_path: Path | None = None,
    expected_weights_sha256: str | None = None,
    expected_weights_size: int | None = None,
    slice_microbatch: int = 2,
    checkpoint_trunk: bool = True,
    freeze_backbone: bool = False,
) -> ResNet18StudyModel:
    """Construct one matched ResNet-18 arm without implicit downloads."""
    if model_kind not in RESNET18_KINDS or type(head_seed) is not int:
        raise ValueError("model_kind_invalid")
    if not 1 <= slice_microbatch <= 8 or type(slice_microbatch) is not int:
        raise ValueError("slice_microbatch_invalid")
    if type(freeze_backbone) is not bool:
        raise ValueError("freeze_backbone_invalid")
    if freeze_backbone and model_kind != "resnet18_imagenet1k_v1":
        raise ValueError("frozen_backbone_requires_imagenet1k_v1")
    from torchvision.models import resnet18

    backbone = resnet18(weights=None)
    if model_kind == "resnet18_imagenet1k_v1":
        if weights_path is None or expected_weights_sha256 is None or expected_weights_size is None:
            raise ValueError("pretrained_weights_required")
        if (expected_weights_sha256 != IMAGENET1K_V1_SHA256
                or expected_weights_size != IMAGENET1K_V1_SIZE):
            raise ValueError("pretrained_weights_pin_invalid")
        state = _verified_weights(Path(weights_path), expected_weights_sha256, expected_weights_size)
        try:
            backbone.load_state_dict(state, strict=True)
        except (RuntimeError, ValueError, TypeError):
            raise ValueError("pretrained_weights_incompatible") from None
    elif weights_path is not None or expected_weights_sha256 is not None or expected_weights_size is not None:
        raise ValueError("random_arm_must_not_receive_weights")
    backbone.fc = nn.Identity()
    model = ResNet18StudyModel(backbone, slice_microbatch=slice_microbatch,
                               checkpoint_trunk=checkpoint_trunk, freeze_backbone=freeze_backbone)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(head_seed)
        model.classifier.reset_parameters()
    model.model_kind = model_kind
    model.weights_sha256 = expected_weights_sha256 if model_kind == "resnet18_imagenet1k_v1" else None
    model.weights_size = expected_weights_size if model_kind == "resnet18_imagenet1k_v1" else None
    model.head_seed = head_seed
    return model
