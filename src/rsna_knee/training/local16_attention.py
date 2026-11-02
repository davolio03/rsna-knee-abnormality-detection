"""Plan the isolated fresh-initialized E050 sixteen-slice attention run."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from rsna_knee.models.resnet50 import BN_POLICY, TRAINABLE_POLICY, WEIGHT_PINS
from rsna_knee.training.masked_soft import _runtime_versions


@dataclass(frozen=True)
class Attention16Config:
    repository_root: Path
    weights_path: Path
    output_dir: Path
    device: str = "cuda"

    def validate(self) -> None:
        if self.device not in ("cuda", "cpu"):
            raise ValueError("attention16_config_invalid")


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_attention16_plan(config: Attention16Config, cache_proof: Mapping[str, Any], *,
                          cache_receipt_path: Path, source_paths: tuple[Path, ...]) -> dict[str, Any]:
    """Bind a training plan to a verified local16 cache; never creates a model or output."""
    config.validate()
    root = Path(config.repository_root).resolve()
    private_root = (root / "data/private/attention-sixteen-slices").resolve()
    output = Path(config.output_dir)
    output = output if output.is_absolute() else root / output
    output = output.resolve(strict=False)
    try:
        output.relative_to(private_root)
    except ValueError:
        raise ValueError("attention16_output_must_be_private") from None
    if output.exists() or output.is_symlink():
        raise ValueError("attention16_output_not_fresh")
    expected_proof = (type(cache_proof) is dict and cache_proof.get("status") == "verified"
                      and cache_proof.get("cache_identity") == "local16-attention-run01"
                      and cache_proof.get("max_slices_per_series") == 16
                      and cache_proof.get("studies") == 4298 and cache_proof.get("series") == 23765
                      and cache_proof.get("partition_study_counts") == {"train": 2850, "validation": 1448}
                      and cache_proof.get("no_overlap") is True
                      and cache_proof.get("no_missing_or_extra_studies") is True
                      and cache_proof.get("gold_included") is False
                      and cache_proof.get("all_archive_hashes_rechecked") is True
                      and cache_proof.get("training_started") is False)
    if not expected_proof:
        raise ValueError("attention16_cache_verification_required")
    receipt = Path(cache_receipt_path)
    if receipt.is_symlink() or not receipt.is_file():
        raise ValueError("attention16_cache_receipt_unavailable")
    weights = Path(config.weights_path)
    weights = weights if weights.is_absolute() else root / weights
    pin = WEIGHT_PINS["radimagenet"]
    if weights.is_symlink() or not weights.is_file():
        raise ValueError("attention16_weights_unavailable")
    if weights.stat().st_size != pin["size"] or _digest(weights) != pin["sha256"]:
        raise ValueError("attention16_weights_pin_mismatch")
    if any(Path(path).is_symlink() or not Path(path).is_file() for path in source_paths):
        raise ValueError("attention16_source_unavailable")
    source = {Path(path).resolve().relative_to(root).as_posix(): _digest(Path(path)) for path in source_paths}
    frozen = {"schema": "disease-attention-mil16-run-v1", "cache_identity": "local16-attention-run01",
              "fold": 0, "seed": 42, "epochs": 5, "arm": "target_attention",
              "initialization": "fresh_original_radimagenet_weights", "checkpoint_warm_start": False,
              "head_lr": 1e-3, "layer4_lr": 1e-5, "optimizer": "Adam", "slice_microbatch": 2,
              "batchnorm": BN_POLICY, "trainable": TRAINABLE_POLICY, "output_size": 224,
              "max_slices": 16, "normalization": "radimagenet_2x_minus_1",
              "objective": "unweighted_observed_soft_bce", "selected_epoch": 5,
              "gold_included": False, "cache_receipt_sha256": _digest(receipt),
              "cache_summary": {key: cache_proof[key] for key in (
                  "studies", "series", "slices", "partition_study_counts", "no_overlap",
                  "no_missing_or_extra_studies", "all_archive_hashes_rechecked")},
              "weights_sha256": pin["sha256"], "weights_size": pin["size"],
              "source": source, "runtime": _runtime_versions()}
    fingerprint = hashlib.sha256(json.dumps(frozen, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return {"status": "planned", "fingerprint": fingerprint, "configuration": frozen,
            "output_dir": str(output), "model_constructed": False, "training_started": False,
            "training_requires_explicit_future_launch": True}
