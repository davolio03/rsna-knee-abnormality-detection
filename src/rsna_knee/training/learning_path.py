"""Bounded aggregate-only diagnostics for the scratch CNN learning path."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
from torch import Tensor
from torch.nn import functional as F

from rsna_knee.models.baseline import ScratchStudyCNN


def _valid_input(images: Sequence[Tensor], masks: Sequence[Tensor]) -> bool:
    if not isinstance(images, (list, tuple)) or not images:
        return False
    if not isinstance(masks, (list, tuple)) or len(images) != len(masks):
        return False
    for image, mask in zip(images, masks):
        if not isinstance(image, Tensor) or not isinstance(mask, Tensor):
            return False
        if image.ndim != 4 or image.shape[0] < 1 or image.shape[1] != 1:
            return False
        if min(image.shape[2:]) < 8 or image.shape != mask.shape or mask.dtype != torch.bool:
            return False
        if image.dtype != torch.float32 or image.device.type != "cpu" or mask.device.type != "cpu":
            return False
        if not bool(torch.isfinite(image).all()) or not bool(mask.flatten(1).any(dim=1).all()):
            return False
        valid = image.masked_select(mask)
        if valid.numel() < 2:
            return False
    return True


def _group_stats(parameters: Sequence[Tensor]) -> dict[str, float | bool]:
    grads = [parameter.grad.detach() for parameter in parameters if parameter.grad is not None]
    if not grads:
        return {"finite": False, "norm": 0.0}
    finite = all(bool(torch.isfinite(grad).all()) for grad in grads)
    norm = math.sqrt(sum(float(torch.sum(grad.square())) for grad in grads)) if finite else 0.0
    return {"finite": finite and math.isfinite(norm), "norm": norm}


def _parameter_delta(parameters: Sequence[Tensor], before: Sequence[Tensor]) -> dict[str, float | bool]:
    deltas = [torch.max(torch.abs(parameter.detach() - prior)) for parameter, prior in zip(parameters, before)]
    finite = bool(deltas) and all(bool(torch.isfinite(delta)) for delta in deltas)
    maximum = max((float(delta) for delta in deltas), default=0.0) if finite else 0.0
    return {"finite": finite and math.isfinite(maximum), "max_abs_delta": maximum}


def diagnose_scratch_learning_path(
    series_images: Sequence[Tensor],
    series_masks: Sequence[Tensor],
    comparison_images: Sequence[Tensor],
    comparison_masks: Sequence[Tensor],
    targets: Tensor,
    *,
    steps: int = 2,
    seed: int = 42,
) -> dict[str, object]:
    """Run a tiny CPU probe and return only privacy-safe scalar diagnostics.

    A fresh model is initialized inside a forked RNG scope, so this diagnostic
    does not perturb caller RNG state or any production model/optimizer.
    """
    invalid = {"status": "invalid_input"}
    if type(steps) is not int or not 1 <= steps <= 8 or type(seed) is not int:
        return invalid
    if not _valid_input(series_images, series_masks) or not _valid_input(comparison_images, comparison_masks):
        return invalid
    input_stds = [
        float(image.masked_select(mask).std(unbiased=False))
        for images, masks in ((series_images, series_masks), (comparison_images, comparison_masks))
        for image, mask in zip(images, masks)
    ]
    if any(not math.isfinite(value) for value in input_stds):
        return invalid
    if any(value <= 1e-12 for value in input_stds):
        return {"status": "input_not_diverse", "input_mean_abs_delta": 0.0}
    if not isinstance(targets, Tensor) or tuple(targets.shape) != (12,) or targets.dtype != torch.float32:
        return invalid
    if targets.device.type != "cpu" or not bool(torch.isfinite(targets).all()):
        return invalid
    if not bool(((targets == 0) | (targets == 1)).all()):
        return invalid

    valid_a = [image.masked_select(mask) for image, mask in zip(series_images, series_masks)]
    valid_b = [image.masked_select(mask) for image, mask in zip(comparison_images, comparison_masks)]
    mean_a = sum(float(values.sum()) for values in valid_a) / sum(values.numel() for values in valid_a)
    mean_b = sum(float(values.sum()) for values in valid_b) / sum(values.numel() for values in valid_b)
    input_delta = abs(mean_a - mean_b)
    if not math.isfinite(input_delta):
        return invalid
    same_input = len(series_images) == len(comparison_images) and all(
        left.shape == right.shape and torch.equal(left, right) and torch.equal(left_mask, right_mask)
        for left, left_mask, right, right_mask in zip(
            series_images, series_masks, comparison_images, comparison_masks
        )
    )
    if same_input:
        return {"status": "input_not_diverse", "input_mean_abs_delta": input_delta if math.isfinite(input_delta) else 0.0}

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = ScratchStudyCNN().cpu()
        model.train()
        encoder = list(model.encoder.parameters())
        head = list(model.classifier.parameters())
        groups = {"encoder": encoder, "head": head}
        initial_parameters = {
            name: [parameter.detach().clone() for parameter in parameters]
            for name, parameters in groups.items()
        }
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        before_logits = model(series_images, series_masks).detach()
        comparison_logits = model(comparison_images, comparison_masks).detach()
        input_logit_delta = float(torch.mean(torch.abs(before_logits - comparison_logits)))
        if not math.isfinite(input_logit_delta):
            return invalid
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            logits = model(series_images, series_masks).squeeze(0)
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            if not bool(torch.isfinite(loss)):
                return invalid
            loss.backward()
            gradients = {name: _group_stats(parameters) for name, parameters in groups.items()}
            optimizer.step()
        post_logits = model(series_images, series_masks).detach()
        post_delta = float(torch.mean(torch.abs(before_logits - post_logits)))
        updates = {
            name: _parameter_delta(parameters, initial_parameters[name])
            for name, parameters in groups.items()
        }

    passed = (
        input_logit_delta > 0.0
        and post_delta > 0.0
        and all(gradients[name]["finite"] and gradients[name]["norm"] > 0 for name in groups)
        and all(updates[name]["finite"] and updates[name]["max_abs_delta"] > 0 for name in groups)
    )
    return {
        "status": "ok" if passed else "learning_path_failed",
        "steps": steps,
        "input_mean_abs_delta": input_delta,
        "input_logit_delta": input_logit_delta,
        "gradients": gradients,
        "updates": updates,
        "post_update_logit_delta": post_delta,
    }
