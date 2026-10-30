"""Masked soft-target objectives and global validation metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import hashlib
import math
import os
import platform
import random
import shutil
import time
from typing import Any
from pathlib import Path


PRIVATE_RUN_ROOT = Path(__file__).resolve().parents[3] / "data" / "private" / "masked-soft-training"
PINNED_RESNET18_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
PINNED_RESNET18_SIZE = 46830571


def _validate_batch(logits: Any, targets: Any, observed: Any) -> None:
    import torch

    if (not isinstance(logits, torch.Tensor) or not isinstance(targets, torch.Tensor)
            or not isinstance(observed, torch.Tensor) or logits.ndim != 2
            or logits.shape != targets.shape or logits.shape != observed.shape
            or logits.shape[0] < 1 or logits.shape[1] < 1):
        raise ValueError("batch_shape_invalid")
    if not logits.is_floating_point() or not targets.is_floating_point():
        raise ValueError("batch_dtype_invalid")
    if observed.dtype is not torch.bool:
        raise ValueError("mask_invalid")
    if logits.device != targets.device or logits.device != observed.device:
        raise ValueError("batch_device_invalid")
    if not torch.isfinite(logits).all():
        raise ValueError("logits_nonfinite")
    if not torch.isfinite(targets).all():
        raise ValueError("targets_nonfinite")
    if torch.any(targets < 0) or torch.any(targets > 1):
        raise ValueError("target_out_of_range")
    if not bool(observed.any()):
        raise ValueError("no_observed_targets")


def _class_weight_tensors(class_weights: Mapping[str, object], *, count: int, device: Any, dtype: Any):
    import torch

    if not isinstance(class_weights, Mapping) or set(class_weights) != {"positive", "negative"}:
        raise ValueError("class_weights_invalid")
    result = []
    for name in ("positive", "negative"):
        try:
            tensor = torch.as_tensor(class_weights[name], dtype=dtype, device=device)
        except (TypeError, ValueError, RuntimeError):
            raise ValueError("class_weights_invalid") from None
        if (tensor.shape != (count,) or not torch.isfinite(tensor).all()
                or torch.any(tensor <= 0)):
            raise ValueError("class_weights_invalid")
        result.append(tensor.reshape(1, count))
    return result


def masked_soft_bce(logits: Any, targets: Any, observed: Any,
                    *, class_weights: Mapping[str, object] | None = None):
    """Mean soft BCE over observed cells; optional symmetric class balancing."""
    import torch
    import torch.nn.functional as F

    _validate_batch(logits, targets, observed)
    target = targets.to(dtype=logits.dtype)
    selected_logits = logits[observed]
    selected_targets = target[observed]
    if class_weights is None:
        cell_loss = F.binary_cross_entropy_with_logits(
            selected_logits, selected_targets, reduction="none",
        )
    else:
        cell_weights = _class_weight_tensors(
            class_weights, count=logits.shape[1], device=logits.device, dtype=logits.dtype,
        )
        positive, negative = (value.expand_as(logits)[observed] for value in cell_weights)
        cell_loss = (positive * selected_targets * F.softplus(-selected_logits)
                     + negative * (1.0 - selected_targets) * F.softplus(selected_logits))
    loss = cell_loss.sum() / observed.sum()
    if not torch.isfinite(loss):
        raise ValueError("loss_nonfinite")
    return loss


class MaskedSoftMetricAccumulator:
    """Accumulate unweighted global and per-target metrics by observed cells."""

    def __init__(self, target_names: Sequence[str]):
        if (not isinstance(target_names, Sequence) or not target_names
                or isinstance(target_names, (str, bytes))
                or any(type(name) is not str or not name for name in target_names)
                or len(set(target_names)) != len(target_names)):
            raise ValueError("target_names_invalid")
        self.target_names = tuple(target_names)
        self._loss_sum = [0.0] * len(self.target_names)
        self._brier_sum = [0.0] * len(self.target_names)
        self._counts = [0] * len(self.target_names)

    def update(self, logits: Any, targets: Any, observed: Any) -> None:
        import torch
        import torch.nn.functional as F

        _validate_batch(logits, targets, observed)
        if logits.shape[1] != len(self.target_names):
            raise ValueError("target_count_invalid")
        with torch.no_grad():
            for index in range(len(self.target_names)):
                mask = observed[:, index]
                count = int(mask.sum().item())
                if count:
                    target = targets[mask, index].to(logits.dtype)
                    selected_logits = logits[mask, index]
                    losses = F.binary_cross_entropy_with_logits(selected_logits, target, reduction="none")
                    brier = (torch.sigmoid(selected_logits) - target).square()
                    if not torch.isfinite(losses).all() or not torch.isfinite(brier).all():
                        raise ValueError("metrics_nonfinite")
                    self._loss_sum[index] += float(losses.double().sum().item())
                    self._brier_sum[index] += float(brier.double().sum().item())
                    self._counts[index] += count

    def compute(self) -> dict[str, object]:
        counts = self._counts
        total = sum(counts)
        if total == 0:
            raise ValueError("no_observed_targets")
        loss_sum = sum(self._loss_sum)
        brier_sum = sum(self._brier_sum)
        if not math.isfinite(loss_sum) or not math.isfinite(brier_sum):
            raise ValueError("metrics_nonfinite")
        per_target = []
        for index, name in enumerate(self.target_names):
            count = counts[index]
            per_target.append({
                "target": name,
                "observed_count": count,
                "masked_bce": self._loss_sum[index] / count if count else None,
                "brier": self._brier_sum[index] / count if count else None,
            })
        return {"observed_cell_count": total, "masked_bce": loss_sum / total,
                "brier": brier_sum / total, "per_target": per_target}


@dataclass(frozen=True)
class MaskedSoftConfig:
    manifest_path: Path
    folds_path: Path
    image_root: Path
    output_dir: Path | None = None
    fold: int = 0
    epochs: int = 1
    max_steps: int | None = None
    output_size: int = 256
    max_slices: int = 8
    device: str = "cpu"
    learning_rate: float = 0.001
    balance_mode: str = "unweighted"
    seed: int = 42
    weights_path: Path | None = None
    weights_sha256: str | None = None
    weights_size: int | None = None
    resume_checkpoint: Path | None = None
    resume_checkpoint_sha256: str | None = None
    max_epochs_per_run: int = 100
    max_runtime_seconds: int | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "MaskedSoftConfig":
        if not isinstance(value, Mapping):
            raise ValueError("config_invalid")
        allowed = {"manifest_path", "folds_path", "image_root", "output_dir", "fold", "epochs",
                   "max_steps", "output_size", "max_slices", "device", "learning_rate",
                   "balance_mode", "seed", "weights_path", "weights_sha256", "weights_size"}
        allowed |= {"resume_checkpoint", "resume_checkpoint_sha256", "max_epochs_per_run", "max_runtime_seconds"}
        if set(value) - allowed:
            raise ValueError("config_invalid")
        defaults = {
            "manifest_path": "data/private/full-cohort-prefilter/run01/full-cohort-soft-label-manifest.json",
            "folds_path": "data/private/full-cohort-prefilter/run01/full-cohort-grouped-folds.json",
            "image_root": "data/train_images",
        }
        paths = {}
        for name in ("manifest_path", "folds_path", "image_root"):
            raw = value.get(name, defaults[name])
            if not isinstance(raw, (str, Path)) or not str(raw).strip():
                raise ValueError("config_invalid")
            paths[name] = Path(raw)
        output = value.get("output_dir")
        if output is not None and (not isinstance(output, (str, Path)) or not str(output).strip()):
            raise ValueError("config_invalid")
        weights_path = value.get("weights_path")
        weights_sha256 = value.get("weights_sha256")
        weights_size = value.get("weights_size")
        resume_checkpoint = value.get("resume_checkpoint")
        resume_checkpoint_sha256 = value.get("resume_checkpoint_sha256")
        max_runtime_seconds = value.get("max_runtime_seconds")
        if resume_checkpoint is not None and (
                not isinstance(resume_checkpoint, (str, Path)) or not str(resume_checkpoint).strip()):
            raise ValueError("config_invalid")
        if resume_checkpoint is None:
            if resume_checkpoint_sha256 is not None:
                raise ValueError("config_invalid")
        elif (type(resume_checkpoint_sha256) is not str or len(resume_checkpoint_sha256) != 64
              or any(character not in "0123456789abcdefABCDEF" for character in resume_checkpoint_sha256)):
            raise ValueError("checkpoint_sha_required")
        if (weights_path is not None and
                (not isinstance(weights_path, (str, Path)) or not str(weights_path).strip())):
            raise ValueError("config_invalid")
        if weights_path is None:
            if weights_sha256 is not None or weights_size is not None:
                raise ValueError("config_invalid")
        elif (weights_sha256 != PINNED_RESNET18_SHA256 or type(weights_size) is not int
              or weights_size != PINNED_RESNET18_SIZE):
            raise ValueError("config_invalid")
        def integer(name: str, default: int, low: int, high: int) -> int:
            result = value.get(name, default)
            if type(result) is not int or not low <= result <= high:
                raise ValueError("config_invalid")
            return result
        max_steps = value.get("max_steps")
        if max_steps is not None and (type(max_steps) is not int or not 1 <= max_steps <= 100000):
            raise ValueError("config_invalid")
        max_epochs_per_run = integer("max_epochs_per_run", 100, 1, 100)
        if max_runtime_seconds is not None and (
                type(max_runtime_seconds) is not int or not 60 <= max_runtime_seconds <= 86400):
            raise ValueError("config_invalid")
        device = value.get("device", "cpu")
        balance_mode = value.get("balance_mode", "unweighted")
        lr = value.get("learning_rate", 0.001)
        if (type(device) is not str or device not in {"cpu", "cuda"}
                or type(balance_mode) is not str
                or balance_mode not in {"unweighted", "training_only_class_weights"}
                or isinstance(lr, bool) or not isinstance(lr, (int, float))
                or not math.isfinite(float(lr)) or not 0 < float(lr) <= 1):
            raise ValueError("config_invalid")
        if type(value.get("seed", 42)) is not int or not 0 <= value.get("seed", 42) <= 2**32 - 1:
            raise ValueError("config_invalid")
        return cls(
            **paths, output_dir=Path(output) if output is not None else None,
            fold=integer("fold", 0, 0, 2), epochs=integer("epochs", 1, 1, 100),
            max_steps=max_steps, output_size=integer("output_size", 256, 8, 768),
            max_slices=integer("max_slices", 8, 1, 8), device=device,
            learning_rate=float(lr), balance_mode=balance_mode, seed=int(value.get("seed", 42)),
            weights_path=Path(weights_path) if weights_path is not None else None,
            weights_sha256=weights_sha256, weights_size=weights_size,
            resume_checkpoint=Path(resume_checkpoint) if resume_checkpoint is not None else None,
            resume_checkpoint_sha256=(resume_checkpoint_sha256.lower()
                                      if resume_checkpoint_sha256 is not None else None),
            max_epochs_per_run=max_epochs_per_run, max_runtime_seconds=max_runtime_seconds,
        )


def _validate_offline_weights(config: MaskedSoftConfig) -> dict[str, object]:
    if config.weights_path is None:
        raise ValueError("offline_weights_path_required")
    path = config.weights_path
    try:
        if path.is_symlink() or not path.is_file() or any(parent.is_symlink() for parent in path.parents):
            raise ValueError("offline_weights_unavailable")
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
    except (OSError, ValueError):
        raise ValueError("offline_weights_unavailable") from None
    if size != PINNED_RESNET18_SIZE or digest.hexdigest() != PINNED_RESNET18_SHA256:
        raise ValueError("offline_weights_pin_mismatch")
    return {"model_kind": "resnet18_imagenet1k_v1", "freeze_backbone": True,
            "optimizer": "Adam", "weights_sha256": digest.hexdigest(), "weights_size": size}


def _build_default_model(config: MaskedSoftConfig):
    """Build the frozen baseline architecture from an explicitly pinned local file."""
    from rsna_knee.models.resnet18 import build_resnet18_study_model
    provenance = _validate_offline_weights(config)
    return build_resnet18_study_model(
        "resnet18_imagenet1k_v1", head_seed=config.seed, weights_path=config.weights_path,
        expected_weights_sha256=provenance["weights_sha256"],
        expected_weights_size=provenance["weights_size"], slice_microbatch=2,
        checkpoint_trunk=True, freeze_backbone=True,
    )


def _soft_plan(config: MaskedSoftConfig) -> tuple[dict[str, object], list[dict[str, object]]]:
    if not isinstance(config, MaskedSoftConfig):
        raise ValueError("config_invalid")
    try:
        config = MaskedSoftConfig.from_mapping(asdict(config))
    except (TypeError, ValueError, OverflowError):
        raise ValueError("config_invalid") from None
    from rsna_knee.data.masked_soft_dataset import (
        FOLDS_ARTIFACT_SHA256, OFFICIAL_TRAIN_SHA256, PREFILTER_POLICY_SHA256,
        TRAIN_SERIES_SHA256, load_verified_soft_artifacts,
    )
    from rsna_knee.labels.report_source_comparison import (
        COHORT_SHA256, FOLDS_SHA256, GOLD_IDS_SHA256, SOURCE_CSV_SHA256,
    )

    records, assignment, digest = load_verified_soft_artifacts(config.manifest_path, config.folds_path)
    folds = json.loads(Path(config.folds_path).read_text(encoding="utf-8"))
    fold_row = folds["folds"][config.fold]
    train_ids = set(fold_row["train_study_ids"])
    validation_ids = set(fold_row["validation_study_ids"])
    if train_ids & validation_ids or train_ids | validation_ids != set(assignment):
        raise ValueError("fold_membership_invalid")
    if any(assignment[study_id] == config.fold for study_id in train_ids) or any(
        assignment[study_id] != config.fold for study_id in validation_ids
    ):
        raise ValueError("fold_membership_invalid")
    plan = {
        "status": "planned",
        "manifest_sha256": digest,
        "folds_sha256": hashlib.sha256(Path(config.folds_path).read_bytes()).hexdigest(),
        "source_csv_sha256": SOURCE_CSV_SHA256,
        "gold_ids_sha256": GOLD_IDS_SHA256,
        "official_train_csv_sha256": OFFICIAL_TRAIN_SHA256,
        "train_series_csv_sha256": TRAIN_SERIES_SHA256,
        "prefilter_policy_sha256": PREFILTER_POLICY_SHA256,
        "baseline_cohort_sha256": COHORT_SHA256,
        "baseline_folds_sha256": FOLDS_SHA256,
        "pinned_folds_artifact_sha256": FOLDS_ARTIFACT_SHA256,
        "fold": config.fold,
        "seed": config.seed,
        "balance_mode": config.balance_mode,
        "training_study_count": len(train_ids),
        "validation_study_count": len(validation_ids),
        "training_group_count": fold_row["training_group_count"],
        "validation_group_count": fold_row["validation_group_count"],
        "training_support": fold_row["train_support"],
        "validation_support": fold_row["validation_support"],
        "validation_metric_weighting": "unweighted_observed_cells",
        "objective": "sum(mask * soft_bce_cell_loss) / observed_cell_count",
        "training_status": "no_training_performed",
    }
    try:
        plan.update(_validate_offline_weights(config))
    except ValueError as error:
        reason = str(error)
        plan.update({"status": "blocked", "reason_code": reason,
                    "model_kind": "resnet18_imagenet1k_v1", "freeze_backbone": True,
                    "optimizer": "Adam"})
    if config.balance_mode == "training_only_class_weights":
        plan["training_only_class_weights"] = fold_row["training_only_class_weights"]
    return plan, records


def plan_masked_soft_training(config: MaskedSoftConfig) -> dict[str, object]:
    """Verify pinned metadata/folds and report only aggregate counts; no images are loaded."""
    if not isinstance(config, MaskedSoftConfig):
        raise ValueError("config_invalid")
    try:
        config = MaskedSoftConfig.from_mapping(asdict(config))
    except (TypeError, ValueError, OverflowError):
        raise ValueError("config_invalid") from None
    return _soft_plan(config)[0]


def _check_soft_output(config: MaskedSoftConfig) -> Path:
    if config.output_dir is None:
        raise ValueError("output_dir_required")
    private_root = PRIVATE_RUN_ROOT.absolute()
    output = config.output_dir.absolute()
    try:
        if output.parent != private_root:
            raise ValueError("output_path_unsafe")
        if output.exists() or output.is_symlink() or any(parent.is_symlink() for parent in output.parents):
            raise ValueError("output_must_be_new")
        if private_root.is_symlink() or any(parent.is_symlink() for parent in private_root.parents):
            raise ValueError("output_path_unsafe")
    except OSError:
        raise ValueError("output_path_unsafe") from None
    return output


_CHECKPOINT_SCHEMA = "masked-soft-checkpoint-v1"


def _runtime_versions() -> dict[str, object]:
    import torch
    try:
        import numpy
        numpy_version = str(numpy.__version__)
    except ImportError:
        numpy_version = None
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": numpy_version, "cuda": str(torch.version.cuda) if torch.version.cuda else None}


def _checkpoint_fingerprint(config: MaskedSoftConfig, plan: Mapping[str, object], *,
                            source_root: Path | None = None) -> str:
    """Bind continuation to scientific inputs, not per-execution controls or paths."""
    source_root = Path(source_root or Path(__file__).resolve().parents[2])
    package_root = source_root / "rsna_knee"
    try:
        source_files = sorted(package_root.rglob("*.py"))
        if not source_files or any(path.is_symlink() for path in source_files):
            raise ValueError("checkpoint_source_invalid")
        source_hashes = {path.relative_to(source_root).as_posix():
                         hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    except (OSError, ValueError):
        raise ValueError("checkpoint_source_invalid") from None
    fields = {
        "schema": _CHECKPOINT_SCHEMA,
        "plan": {key: plan.get(key) for key in (
            "manifest_sha256", "folds_sha256", "source_csv_sha256", "gold_ids_sha256",
            "official_train_csv_sha256", "train_series_csv_sha256", "prefilter_policy_sha256",
            "baseline_cohort_sha256", "baseline_folds_sha256", "pinned_folds_artifact_sha256",
            "fold", "training_study_count", "validation_study_count", "model_kind",
            "freeze_backbone", "weights_sha256", "weights_size",
        )},
        "config": {key: getattr(config, key) for key in (
            "fold", "epochs", "max_steps", "output_size", "max_slices", "device", "learning_rate",
            "balance_mode", "seed", "weights_sha256", "weights_size",
        )},
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": source_hashes,
        "runtime": _runtime_versions(),
    }
    encoded = json.dumps(fields, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _capture_rng_state(*, device: str) -> dict[str, object]:
    import numpy as np
    import torch

    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [str(numpy_state[0]), numpy_state[1].tolist(), int(numpy_state[2]),
                  int(numpy_state[3]), float(numpy_state[4])],
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if device == "cuda" else [],
    }


def _restore_rng_state(state: Mapping[str, object], *, device: str) -> None:
    import numpy as np
    import torch

    if device not in {"cpu", "cuda"} or set(state) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("checkpoint_rng_invalid")
    numpy_state = state["numpy"]
    if (not isinstance(numpy_state, list) or len(numpy_state) != 5
            or not isinstance(numpy_state[1], list)):
        raise ValueError("checkpoint_rng_invalid")
    cuda_states = state["torch_cuda"]
    cpu_state = state["torch_cpu"]
    if (not isinstance(cuda_states, list) or not isinstance(cpu_state, torch.Tensor)
            or cpu_state.dtype != torch.uint8 or cpu_state.ndim != 1):
        raise ValueError("checkpoint_rng_invalid")
    if device == "cpu":
        if cuda_states:
            raise ValueError("checkpoint_device_mismatch")
    elif (not torch.cuda.is_available() or not cuda_states
          or len(cuda_states) != torch.cuda.device_count()
          or any(not isinstance(item, torch.Tensor) or item.dtype != torch.uint8 or item.ndim != 1
                 for item in cuda_states)):
        raise ValueError("checkpoint_device_mismatch")
    random.Random().setstate(state["python"])
    numpy_candidate = (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                       int(numpy_state[2]), int(numpy_state[3]), float(numpy_state[4]))
    np.random.RandomState().set_state(numpy_candidate)
    random.setstate(state["python"])
    np.random.set_state(numpy_candidate)
    torch.set_rng_state(cpu_state)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("checkpoint_device_mismatch")
        torch.cuda.set_rng_state_all(cuda_states)


def _read_checkpoint(path: Path, *, fingerprint: str, total_epochs: int,
                     expected_sha256: str) -> dict[str, object]:
    import torch

    try:
        if path.is_symlink() or not path.is_file() or any(parent.is_symlink() for parent in path.parents):
            raise ValueError("checkpoint_unavailable")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise ValueError("checkpoint_sha_mismatch")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except ValueError as error:
        if str(error) == "checkpoint_sha_mismatch":
            raise
        raise ValueError("checkpoint_unavailable") from None
    except Exception:
        raise ValueError("checkpoint_unavailable") from None
    expected = {"schema", "fingerprint", "completed_epochs", "history", "model_state",
                "optimizer_state", "rng_state", "last_epoch_seconds"}
    if not isinstance(checkpoint, dict) or set(checkpoint) != expected:
        raise ValueError("checkpoint_schema_invalid")
    completed = checkpoint["completed_epochs"]
    history = checkpoint["history"]
    if (checkpoint["schema"] != _CHECKPOINT_SCHEMA or checkpoint["fingerprint"] != fingerprint
            or type(completed) is not int or not 1 <= completed < total_epochs
            or not isinstance(history, list) or len(history) != completed
            or [item.get("epoch") for item in history if isinstance(item, dict)] != list(range(1, completed + 1))
            or not isinstance(checkpoint["model_state"], dict)
            or not isinstance(checkpoint["optimizer_state"], dict)
            or not isinstance(checkpoint["rng_state"], dict)
            or isinstance(checkpoint["last_epoch_seconds"], bool)
            or not isinstance(checkpoint["last_epoch_seconds"], (int, float))
            or not math.isfinite(float(checkpoint["last_epoch_seconds"]))
            or checkpoint["last_epoch_seconds"] < 0):
        raise ValueError("checkpoint_incompatible")
    return checkpoint


def _write_checkpoint(path: Path, checkpoint: Mapping[str, object]) -> str:
    import torch

    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError("checkpoint_temp_exists")
    try:
        with temporary.open("xb") as stream:
            torch.save(dict(checkpoint), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _validate_finite_tensors(value: object) -> bool:
    import torch

    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(_validate_finite_tensors(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_validate_finite_tensors(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _validate_adam_state(optimizer: Any, *, learning_rate: float) -> None:
    import torch

    allowed = {"step", "exp_avg", "exp_avg_sq"}
    if len(optimizer.param_groups) != 1:
        raise ValueError("checkpoint_optimizer_invalid")
    group = optimizer.param_groups[0]
    if (group.get("lr") != learning_rate or tuple(group.get("betas", ())) != (0.9, 0.999)
            or group.get("eps") != 1e-8 or group.get("weight_decay") != 0
            or group.get("amsgrad") is not False or group.get("maximize") is not False):
        raise ValueError("checkpoint_optimizer_invalid")
    parameters = group.get("params")
    if (not isinstance(parameters, list) or not parameters
            or len(optimizer.state) != len(parameters)
            or any(parameter not in optimizer.state for parameter in parameters)):
        raise ValueError("checkpoint_optimizer_invalid")
    for parameter, state in optimizer.state.items():
        if not isinstance(state, Mapping) or set(state) != allowed:
            raise ValueError("checkpoint_optimizer_invalid")
        for key in ("exp_avg", "exp_avg_sq"):
            tensor = state[key]
            if (not isinstance(tensor, torch.Tensor) or tensor.shape != parameter.shape
                    or not torch.isfinite(tensor).all()):
                raise ValueError("checkpoint_optimizer_invalid")
        step = state["step"]
        if isinstance(step, torch.Tensor):
            if step.numel() != 1 or not torch.isfinite(step).all() or float(step.item()) < 0:
                raise ValueError("checkpoint_optimizer_invalid")
        elif type(step) not in {int, float} or not math.isfinite(float(step)) or step < 0:
            raise ValueError("checkpoint_optimizer_invalid")


def run_masked_soft_training(config: MaskedSoftConfig, *, execute: bool = False,
                              dataset_factory: Any = None, model_factory: Any = None) -> dict[str, object]:
    """Plan by default; execute explicitly and checkpoint each completed epoch."""
    if not isinstance(config, MaskedSoftConfig) or type(execute) is not bool:
        raise ValueError("config_invalid")
    try:
        config = MaskedSoftConfig.from_mapping(asdict(config))
    except (TypeError, ValueError, OverflowError):
        raise ValueError("config_invalid") from None
    plan, _records = _soft_plan(config)
    if not execute:
        return plan
    if config.weights_path is not None:
        _validate_offline_weights(config)
    if plan["status"] != "planned" and model_factory is None:
        raise ValueError(str(plan["reason_code"]))
    output = _check_soft_output(config)
    execution_start = time.monotonic()
    import torch

    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("device_unavailable")
    from rsna_knee.training.baseline import _seed_everything
    _seed_everything(config.seed, config.device)
    checkpoint = None
    fingerprint = _checkpoint_fingerprint(config, plan)
    if config.resume_checkpoint is not None:
        checkpoint = _read_checkpoint(config.resume_checkpoint, fingerprint=fingerprint,
                                      total_epochs=config.epochs,
                                      expected_sha256=config.resume_checkpoint_sha256)
    from rsna_knee.data.masked_soft_dataset import FullCohortSoftDataset
    from rsna_knee.labels.full_cohort_prefilter import TARGET_COLUMNS
    if dataset_factory is None:
        dataset_factory = lambda cfg, part: FullCohortSoftDataset(
            cfg.manifest_path, cfg.folds_path, cfg.image_root, partition=part, fold=cfg.fold,
            output_size=cfg.output_size, max_slices=cfg.max_slices,
        )
    train_data = dataset_factory(config, "train")
    validation_data = dataset_factory(config, "validation")
    if len(train_data) < 1 or len(validation_data) < 1:
        raise ValueError("dataset_empty")
    if model_factory is None:
        model_factory = _build_default_model
    model = model_factory(config).to(torch.device(config.device))
    fold_row = json.loads(Path(config.folds_path).read_text(encoding="utf-8"))["folds"][config.fold]
    weights = None
    if config.balance_mode == "training_only_class_weights":
        source = fold_row["training_only_class_weights"]
        weights = {side: [source[target][side] for target in TARGET_COLUMNS]
                   for side in ("positive", "negative")}
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("trainable_parameters_empty")
    optimizer = torch.optim.Adam(trainable_parameters, lr=config.learning_rate)
    history = []
    completed = 0
    last_epoch_seconds = 0.0
    if checkpoint is not None:
        try:
            if not _validate_finite_tensors(checkpoint["model_state"]):
                raise ValueError("checkpoint_state_invalid")
            model.load_state_dict(checkpoint["model_state"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            _validate_adam_state(optimizer, learning_rate=config.learning_rate)
            if not _validate_finite_tensors(model.state_dict()):
                raise ValueError("checkpoint_state_invalid")
            history = checkpoint["history"]
            completed = checkpoint["completed_epochs"]
            last_epoch_seconds = float(checkpoint["last_epoch_seconds"])
            _restore_rng_state(checkpoint["rng_state"], device=config.device)
        except (RuntimeError, TypeError, ValueError, KeyError, IndexError):
            raise ValueError("checkpoint_state_invalid") from None
    created = False
    checkpoint_written = False
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=False, exist_ok=False)
        created = True
        digest = None
        if config.resume_checkpoint is not None:
            previous = output / f"checkpoint-epoch-{completed:02d}.pt"
            previous.write_bytes(config.resume_checkpoint.read_bytes())
            digest = hashlib.sha256(previous.read_bytes()).hexdigest()
            checkpoint_written = True
        run_epoch_count = 0
        training_limit = min(len(train_data), config.max_steps) if config.max_steps else len(train_data)
        print(json.dumps({"event": "training_started", "fold": config.fold,
                          "training_study_count": plan["training_study_count"],
                          "validation_study_count": plan["validation_study_count"],
                          "training_steps_per_epoch": training_limit,
                          "validation_steps_per_epoch": len(validation_data),
                          "completed_epochs": completed, "target_epochs": config.epochs,
                          "max_epochs_per_run": config.max_epochs_per_run,
                          "max_runtime_seconds": config.max_runtime_seconds}, sort_keys=True), flush=True)
        for epoch in range(completed, config.epochs):
            elapsed = time.monotonic() - execution_start
            if run_epoch_count >= config.max_epochs_per_run:
                break
            if (config.max_runtime_seconds is not None and completed > 0
                    and elapsed + last_epoch_seconds * 1.5 >= config.max_runtime_seconds):
                break
            epoch_start = time.monotonic()
            model.train()
            train_sum, train_cells, steps = 0.0, 0, 0
            limit = training_limit
            for index in range(limit):
                images, pixel_masks, targets, observed, _row = train_data[index]
                images = [value.to(config.device, dtype=torch.float32) for value in images]
                pixel_masks = [value.to(config.device, dtype=torch.bool) for value in pixel_masks]
                targets = targets.to(config.device, dtype=torch.float32).reshape(1, -1)
                observed = observed.to(config.device, dtype=torch.bool).reshape(1, -1)
                logits = model(images, pixel_masks)
                if tuple(logits.shape) != (1, len(TARGET_COLUMNS)) or logits.shape != targets.shape or not torch.isfinite(logits).all():
                    raise ValueError("model_output_invalid")
                loss = masked_soft_bce(logits, targets, observed, class_weights=weights)
                if not torch.isfinite(loss):
                    raise ValueError("loss_nonfinite")
                cells = int(observed.sum().item())
                if cells < 1:
                    raise ValueError("no_observed_targets")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                train_sum += float(loss.detach().double().item()) * cells
                if not math.isfinite(train_sum):
                    raise ValueError("metrics_nonfinite")
                train_cells += cells
                steps += 1
                if steps % 100 == 0:
                    print(json.dumps({"event": "training_progress", "partition": "train",
                                      "epoch": epoch + 1, "completed": steps, "total": limit},
                                     sort_keys=True), flush=True)
            if steps == 0:
                raise ValueError("training_empty")
            model.eval()
            metrics = MaskedSoftMetricAccumulator(TARGET_COLUMNS)
            with torch.no_grad():
                for index in range(len(validation_data)):
                    images, pixel_masks, targets, observed, _row = validation_data[index]
                    images = [value.to(config.device, dtype=torch.float32) for value in images]
                    pixel_masks = [value.to(config.device, dtype=torch.bool) for value in pixel_masks]
                    targets = targets.to(config.device, dtype=torch.float32).reshape(1, -1)
                    observed = observed.to(config.device, dtype=torch.bool).reshape(1, -1)
                    logits = model(images, pixel_masks)
                    metrics.update(logits, targets, observed)
                    if (index + 1) % 100 == 0:
                        print(json.dumps({"event": "training_progress", "partition": "validation",
                                          "epoch": epoch + 1, "completed": index + 1,
                                          "total": len(validation_data)}, sort_keys=True), flush=True)
            history.append({"epoch": epoch + 1, "training_masked_bce": train_sum / train_cells,
                            "training_observed_cells": train_cells,
                            "training_objective_weighting": config.balance_mode,
                            "validation_objective_weighting": "unweighted",
                            "validation": metrics.compute()})
            completed = epoch + 1
            last_epoch_seconds = time.monotonic() - epoch_start
            checkpoint_path = output / f"checkpoint-epoch-{completed:02d}.pt"
            digest = _write_checkpoint(checkpoint_path, {
                "schema": _CHECKPOINT_SCHEMA, "fingerprint": fingerprint,
                "completed_epochs": completed, "history": history,
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "rng_state": _capture_rng_state(device=config.device),
                "last_epoch_seconds": last_epoch_seconds,
            })
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if (saved.get("schema") != _CHECKPOINT_SCHEMA
                    or saved.get("fingerprint") != fingerprint
                    or saved.get("completed_epochs") != completed
                    or hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != digest):
                raise ValueError("checkpoint_readback_mismatch")
            checkpoint_written = True
            print(json.dumps({"event": "epoch_checkpointed", "epoch": completed,
                              "checkpoint": checkpoint_path.name, "sha256": digest,
                              "elapsed_seconds": last_epoch_seconds}, sort_keys=True), flush=True)
            run_epoch_count += 1
        model_metadata = _validate_offline_weights(config) if model_factory is _build_default_model else {
            "model_kind": "injected_synthetic", "freeze_backbone": None,
            "optimizer": "Adam", "weights_sha256": None, "weights_size": None,
        }
        if completed == config.epochs:
            torch.save({"state_dict": model.state_dict(), "metadata": model_metadata}, output / "model-state.pt")
            status = "complete"
            training_status = "training_completed_no_quality_claim"
        else:
            status = "paused_checkpoint_available"
            training_status = "training_paused_checkpoint_available"
        latest_checkpoint = output / f"checkpoint-epoch-{completed:02d}.pt"
        report = {**plan, "status": status, "training_status": training_status,
                  "completed_epochs": completed, "target_epochs": config.epochs,
                  "epochs": history,
                  "checkpoint_path": str(latest_checkpoint) if completed else None,
                  "checkpoint_sha256": digest if completed else None,
                  "resume_source": "explicit_checkpoint" if checkpoint is not None else None,
                  "max_epochs_per_run": config.max_epochs_per_run,
                  "max_runtime_seconds": config.max_runtime_seconds,
                  "runtime_versions": _runtime_versions(), **model_metadata}
        report_path = output / "training-report.json"
        with report_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, sort_keys=True, allow_nan=False, indent=2)
            stream.write("\n")
        if json.loads(report_path.read_text(encoding="utf-8")) != report:
            raise ValueError("report_readback_mismatch")
        return report
    except Exception:
        if (created and not checkpoint_written and output.is_dir() and not output.is_symlink()
                and output.resolve().parent == output.parent.resolve()):
            shutil.rmtree(output)
        raise
