"""Bounded local training for the provisional scratch study-level baseline."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


_SAFE_FAILURE_STAGES = frozenset({
    "runtime_setup", "model_initialization", "training", "validation", "metrics",
    "checkpoint", "report_write",
})
_SAFE_SOURCE_FUNCTIONS = frozenset({
    "run_training", "_sample_loss", "_mean_loss", "_runtime_metadata", "_write_json",
})
_SAFE_EXCEPTION_FAMILIES = frozenset({
    "memory", "operating_system", "runtime", "value", "type", "index_or_key", "import", "other",
})


def _validate_failure_diagnostics(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(value, Mapping) or set(value) - {
        "stage", "exception_family", "exception_chain", "source", "data_reason_code"
    }:
        return None
    stage = value.get("stage")
    family = value.get("exception_family")
    chain = value.get("exception_chain")
    if (type(stage) is not str or stage not in _SAFE_FAILURE_STAGES
            or type(family) is not str or family not in _SAFE_EXCEPTION_FAMILIES
            or type(chain) is not list or not 1 <= len(chain) <= 4
            or chain[0] != family
            or any(type(entry) is not str or entry not in _SAFE_EXCEPTION_FAMILIES for entry in chain)):
        return None
    safe: dict[str, Any] = {"stage": stage, "exception_family": family, "exception_chain": list(chain)}
    source = value.get("source")
    if source is not None:
        if (not isinstance(source, Mapping) or set(source) != {"module", "function", "line"}
                or source.get("module") != "training.baseline"
                or type(source.get("function")) is not str
                or source.get("function") not in _SAFE_SOURCE_FUNCTIONS
                or type(source.get("line")) is not int
                or not 1 <= source["line"] <= 100000):
            return None
        safe["source"] = {
            "module": "training.baseline", "function": source["function"], "line": source["line"],
        }
    data_reason_code = value.get("data_reason_code")
    if data_reason_code is not None:
        from rsna_knee.data.training_dataset import TrainingDatasetError

        if (type(data_reason_code) is not str
                or data_reason_code not in TrainingDatasetError.SAFE_DIAGNOSTIC_REASON_CODES):
            return None
        safe["data_reason_code"] = data_reason_code
    return safe


def _safe_exception_family(error: Exception) -> str:
    torch_module = sys.modules.get("torch")
    cuda_module = getattr(torch_module, "cuda", None)
    cuda_oom_type = getattr(cuda_module, "OutOfMemoryError", None)
    if isinstance(error, MemoryError) or (
        isinstance(cuda_oom_type, type) and isinstance(error, cuda_oom_type)
    ):
        return "memory"
    if isinstance(error, OSError):
        return "operating_system"
    if isinstance(error, RuntimeError):
        return "runtime"
    if isinstance(error, ValueError):
        return "value"
    if isinstance(error, TypeError):
        return "type"
    if isinstance(error, (IndexError, KeyError)):
        return "index_or_key"
    if isinstance(error, ImportError):
        return "import"
    return "other"


def _safe_exception_chain(error: Exception) -> list[str]:
    chain: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while isinstance(current, BaseException) and id(current) not in seen and len(chain) < 4:
        seen.add(id(current))
        chain.append(_safe_exception_family(current))
        current = current.__cause__ or (
            current.__context__ if not current.__suppress_context__ else None
        )
    return chain or ["other"]


def _safe_failure_diagnostics(error: Exception, stage: str) -> dict[str, Any]:
    """Keep only coarse allowlisted failure metadata; never retain exception text."""
    if stage not in _SAFE_FAILURE_STAGES:
        stage = "runtime_setup"
    family = _safe_exception_family(error)
    diagnostics: dict[str, Any] = {
        "stage": stage,
        "exception_family": family,
        "exception_chain": _safe_exception_chain(error),
    }
    from rsna_knee.data.training_dataset import TrainingDatasetError

    if type(error) is TrainingDatasetError:
        data_reason_code = getattr(error, "data_reason_code", None)
        if (type(data_reason_code) is str
                and data_reason_code in TrainingDatasetError.SAFE_DIAGNOSTIC_REASON_CODES):
            diagnostics["data_reason_code"] = data_reason_code
    traceback = error.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        code = frame.f_code
        if (Path(code.co_filename).as_posix().endswith("/rsna_knee/training/baseline.py")
                and code.co_name in _SAFE_SOURCE_FUNCTIONS):
            diagnostics["source"] = {
                "module": "training.baseline",
                "function": code.co_name,
                "line": int(traceback.tb_lineno),
            }
        traceback = traceback.tb_next
    return diagnostics


class TrainingRunError(ValueError):
    """Safe categorical training error that omits input paths and identifiers."""

    def __init__(self, reason_code: str, *, diagnostics: Mapping[str, Any] | None = None) -> None:
        if not re.fullmatch(r"[a-z0-9_]{1,48}", reason_code):
            reason_code = "invalid_input"
        self.reason_code = reason_code
        self.diagnostics = _validate_failure_diagnostics(diagnostics)
        super().__init__(f"Training run unavailable ({reason_code}).")


@dataclass(frozen=True)
class TrainingConfig:
    mode: str
    manifest_path: Path
    folds_path: Path
    image_root: Path
    output_dir: Path
    fold: int = 0
    seed: int = 42
    epochs: int = 1
    max_steps: int | None = None
    overfit_limit: int | None = None
    output_size: int = 256
    max_slices: int = 8
    device: str = "cpu"
    learning_rate: float = 0.001
    feature_width: int = 64

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TrainingConfig":
        if not isinstance(value, Mapping):
            raise ValueError("training configuration must be an object")
        required = ("mode", "manifest_path", "folds_path", "image_root", "output_dir")
        if any(not isinstance(value.get(key), str) or not value[key].strip() for key in required):
            raise ValueError("training configuration is missing a required path or mode")
        mode = value["mode"]
        if not isinstance(mode, str) or mode not in {"overfit", "fold"}:
            raise ValueError("mode must be 'overfit' or 'fold'")

        def integer(name: str, default: int, low: int, high: int) -> int:
            number = value.get(name, default)
            if type(number) is not int or not low <= number <= high:
                raise ValueError(f"{name} is outside its allowed range")
            return number

        epochs = integer("epochs", 1, 1, 100)
        raw_steps = value.get("max_steps", 20 if mode == "overfit" else None)
        if raw_steps is not None and (type(raw_steps) is not int or raw_steps < 1):
            raise ValueError("max_steps must be a positive integer or null")
        if mode == "overfit" and (raw_steps is None or raw_steps > 40):
            raise ValueError("overfit max_steps must be between 1 and 40")
        raw_limit = value.get("overfit_limit", 8 if mode == "overfit" else None)
        if raw_limit is not None and (type(raw_limit) is not int or not 1 <= raw_limit <= 16):
            raise ValueError("overfit_limit must be between 1 and 16 or null")
        if mode == "overfit" and raw_limit is None:
            raise ValueError("overfit_limit is required in overfit mode")
        if mode == "fold" and raw_limit is not None:
            raise ValueError("overfit_limit is only valid in overfit mode")
        fold = integer("fold", 0, 0, 2)
        seed = integer("seed", 42, 0, 2**32 - 1)
        output_size = integer("output_size", 256, 8, 768)
        max_slices = integer("max_slices", 8, 1, 8)
        feature_width = integer("feature_width", 64, 8, 1024)
        device = value.get("device", "cpu")
        if not isinstance(device, str) or device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        learning_rate = value.get("learning_rate", 0.001)
        if (isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float))
                or not 0 < learning_rate <= 1):
            raise ValueError("learning_rate must be finite and in (0, 1]")
        return cls(
            mode=mode,
            manifest_path=Path(value["manifest_path"]),
            folds_path=Path(value["folds_path"]),
            image_root=Path(value["image_root"]),
            output_dir=Path(value["output_dir"]),
            fold=fold,
            seed=seed,
            epochs=epochs,
            max_steps=raw_steps,
            overfit_limit=raw_limit,
            output_size=output_size,
            max_slices=max_slices,
            device=device,
            learning_rate=float(learning_rate),
            feature_width=feature_width,
        )


def _digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, ValueError):
        raise TrainingRunError("input_unavailable") from None


def _check_new_output(config: TrainingConfig) -> None:
    output = config.output_dir
    if output.exists() or output.is_symlink():
        raise TrainingRunError("output_must_be_new")
    try:
        import os

        output_absolute = Path(os.path.abspath(output))
        inputs = tuple(Path(os.path.abspath(item)) for item in
                       (config.manifest_path, config.folds_path, config.image_root))
        if any(output_absolute == item or output_absolute in item.parents or item in output_absolute.parents
               for item in inputs):
            raise TrainingRunError("output_path_conflict")
        # Reject a symlink anywhere in the destination path to keep writes local and explicit.
        current = output_absolute
        while current != current.parent:
            if current.is_symlink():
                raise TrainingRunError("output_path_conflict")
            current = current.parent
    except (OSError, RuntimeError):
        raise TrainingRunError("output_path_invalid") from None


def _validate_config(config: TrainingConfig) -> TrainingConfig:
    """Re-validate public dataclass instances; callers may bypass the parser via replace()."""
    if not isinstance(config, TrainingConfig):
        raise TrainingRunError("config_invalid")
    path_names = ("manifest_path", "folds_path", "image_root", "output_dir")
    values: dict[str, Any] = {}
    for name in path_names:
        raw_path = getattr(config, name)
        if not isinstance(raw_path, (str, Path)) or not str(raw_path).strip():
            raise TrainingRunError("config_invalid")
        values[name] = str(raw_path)
    for name in (
        "mode", "fold", "seed", "epochs", "max_steps", "overfit_limit",
        "output_size", "max_slices", "device", "learning_rate", "feature_width",
    ):
        values[name] = getattr(config, name)
    try:
        return TrainingConfig.from_mapping(values)
    except (TypeError, ValueError, OverflowError):
        raise TrainingRunError("config_invalid") from None


def _seed_everything(seed: int, device_name: str) -> None:
    import numpy as np
    import torch

    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device_name == "cuda":
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
    except (RuntimeError, AttributeError):
        code = "cuda_determinism_unavailable" if device_name == "cuda" else "determinism_unavailable"
        raise TrainingRunError(code) from None


def _check_device(name: str) -> Any:
    import torch

    if name == "cuda":
        try:
            available = torch.cuda.is_available()
        except (RuntimeError, OSError):
            raise TrainingRunError("cuda_unavailable") from None
        if not available:
            raise TrainingRunError("cuda_unavailable")
    return torch.device(name)


def _default_dataset_factory(config: TrainingConfig, partition: str):
    from rsna_knee.data.training_dataset import TrainingStudyDataset

    return TrainingStudyDataset(
        config.manifest_path, config.folds_path, config.image_root,
        partition=partition, fold=config.fold,
        output_size=config.output_size, max_slices=config.max_slices,
    )


def _sample_loss(model: Any, sample: Any, loss_function: Any, device: Any):
    import torch

    images, masks, labels, _row_index = sample
    images = [value.to(device=device, dtype=torch.float32) for value in images]
    masks = [value.to(device=device, dtype=torch.bool) for value in masks]
    target = labels.to(device=device, dtype=torch.float32).reshape(1, 12)
    logits = model(images, masks)
    if tuple(logits.shape) != (1, 12) or not torch.isfinite(logits).all():
        raise TrainingRunError("model_output_invalid")
    loss = loss_function(logits, target)
    if not torch.isfinite(loss):
        raise TrainingRunError("loss_invalid")
    return loss, logits


def _mean_loss(model: Any, rows: Any, loss_function: Any, device: Any) -> float:
    import torch

    model.eval()
    losses = []
    with torch.no_grad():
        for row in rows:
            loss, _ = _sample_loss(model, row, loss_function, device)
            losses.append(float(loss.detach().cpu()))
    if not losses:
        raise TrainingRunError("training_subset_empty")
    return float(sum(losses) / len(losses))


def _safe_config(config: TrainingConfig) -> dict[str, Any]:
    return {
        "mode": config.mode, "fold": config.fold, "seed": config.seed,
        "epochs": config.epochs, "max_steps": config.max_steps,
        "overfit_limit": config.overfit_limit, "output_size": config.output_size,
        "max_slices": config.max_slices, "device": config.device,
        "learning_rate": config.learning_rate, "feature_width": config.feature_width,
    }


def _write_json(path: Path, value: Any) -> None:
    try:
        text = json.dumps(value, indent=2, allow_nan=False) + "\n"
        path.write_text(text, encoding="utf-8")
    except (OSError, TypeError, ValueError):
        raise TrainingRunError("output_write_failed") from None


def _reset_peak_cuda_memory(device: Any) -> None:
    """Start a truthful per-run CUDA allocation peak measurement."""
    if getattr(device, "type", None) == "cuda":
        import torch

        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _runtime_metadata(config: TrainingConfig, elapsed_seconds: float, *, device: Any) -> dict[str, Any]:
    import numpy as np
    import torch

    is_cuda = getattr(device, "type", None) == "cuda"
    peak_memory = None
    if is_cuda:
        torch.cuda.synchronize(device)
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    return {
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "architecture": platform.machine(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device": config.device,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime_version": torch.version.cuda,
        "elapsed_seconds": float(elapsed_seconds),
        "peak_cuda_memory_bytes": peak_memory,
    }


def run_training(
    config: TrainingConfig,
    *,
    execute: bool = False,
    dataset_factory: Callable[[TrainingConfig, str], Any] | None = None,
    model_factory: Callable[[TrainingConfig], Any] | None = None,
) -> dict[str, Any]:
    """Validate and plan by default; fit only after explicit ``execute=True``."""
    config = _validate_config(config)
    if type(execute) is not bool:
        raise TrainingRunError("execute_flag_invalid")
    try:
        import numpy as np
        import torch
    except ModuleNotFoundError:
        raise TrainingRunError("training_dependencies_unavailable") from None
    _check_new_output(config)
    device = _check_device(config.device) if execute else None
    factory = dataset_factory or _default_dataset_factory
    try:
        train_dataset = factory(config, "train")
        if config.mode == "overfit":
            train_dataset = train_dataset.bounded_training_subset(config.overfit_limit)
        train_count = len(train_dataset)
        if train_count < 1:
            raise TrainingRunError("training_partition_empty")
        validation_dataset = None
        if config.mode == "fold":
            validation_dataset = factory(config, "validation")
            validation_count = len(validation_dataset)
            if validation_count < 1:
                raise TrainingRunError("validation_partition_empty")
        else:
            validation_count = None
    except TrainingRunError:
        raise
    except Exception:
        # Dataset and decoder errors can carry private paths or study identifiers.
        raise TrainingRunError("dataset_validation_failed") from None

    manifest_sha256 = _digest(config.manifest_path)
    folds_sha256 = _digest(config.folds_path)
    plan = {
        "status": "plan_only",
        "schema_version": "private-training-report-v1",
        "runner_version": "local-baseline-runner-v1",
        "mode": config.mode,
        "fold": config.fold,
        "training_study_count": train_count,
        "validation_study_count": validation_count,
        "execute_required": True,
        "interpretation_limit": "Engineering smoke test only; not clinical evidence or a generalization claim.",
        "input_sha256": {"manifest": manifest_sha256, "folds": folds_sha256},
    }
    if not execute:
        return plan

    # Establish the new private destination before model creation, decoding, or fitting.
    try:
        config.output_dir.mkdir(parents=True, exist_ok=False)
    except (FileExistsError, OSError):
        raise TrainingRunError("output_must_be_new") from None

    started = time.perf_counter()
    stage = "runtime_setup"
    try:
        _seed_everything(config.seed, config.device)
        stage = "model_initialization"
        if model_factory is None:
            from rsna_knee.models.baseline import ScratchStudyCNN

            model = ScratchStudyCNN(feature_width=config.feature_width)
        else:
            model = model_factory(config)
        model = model.to(device)
        _reset_peak_cuda_memory(device)
        loss_function = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        losses_by_step: list[float] = []
        stage = "training"
        initial_loss = _mean_loss(model, train_dataset, loss_function, device) if config.mode == "overfit" else None
        model.train()
        steps_completed = 0
        planned_steps = config.epochs * len(train_dataset)
        step_limit = config.max_steps if config.max_steps is not None else planned_steps
        for _epoch in range(config.epochs):
            for row_index in range(len(train_dataset)):
                if steps_completed >= step_limit:
                    break
                loss, _ = _sample_loss(model, train_dataset[row_index], loss_function, device)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses_by_step.append(float(loss.detach().cpu()))
                steps_completed += 1
            if steps_completed >= step_limit:
                break

        if config.mode == "overfit":
            # Re-evaluate the exact same bounded training rows; never instantiate or touch validation.
            if not losses_by_step:
                raise TrainingRunError("no_training_steps")
            end_loss = _mean_loss(model, train_dataset, loss_function, device)
            report: dict[str, Any] = {
                **plan,
                "status": "completed",
                "steps_completed": steps_completed,
                "train_loss_by_step": losses_by_step,
                "train_loss_start": initial_loss,
                "train_loss_end": end_loss,
                "model_kind": "scratch_study_cnn",
                "training_subset_only": True,
                "interpretation_limit": "Bounded train-subset smoke test only; not clinical evidence, validation, or a generalization claim.",
            }
        else:
            stage = "validation"
            model.eval()
            true_labels: list[list[int]] = []
            probabilities: list[list[float]] = []
            prediction_rows: list[dict[str, Any]] = []
            with torch.no_grad():
                for row_index in range(len(validation_dataset)):
                    sample = validation_dataset[row_index]
                    _, logits = _sample_loss(model, sample, loss_function, device)
                    probs = torch.sigmoid(logits[0]).detach().cpu().to(torch.float32).numpy()
                    labels = sample[2].detach().cpu().to(torch.int64).numpy()
                    if probs.shape != (12,) or not np.isfinite(probs).all() or np.any(probs < 0) or np.any(probs > 1):
                        raise TrainingRunError("prediction_invalid")
                    true_labels.append(labels.tolist())
                    probabilities.append(probs.tolist())
                    prediction_rows.append({
                        "row_index": int(sample[3]),
                        "labels": labels.tolist(),
                        "probabilities": probs.tolist(),
                    })
            from rsna_knee.validation.metrics import multilabel_roc_auc

            stage = "metrics"
            metrics = multilabel_roc_auc(np.asarray(true_labels), np.asarray(probabilities))
            _write_json(config.output_dir / "validation_predictions.json", {
                "schema_version": "private-baseline-predictions-v1",
                "runner_version": "local-baseline-runner-v1",
                "partition": "validation",
                "fold": config.fold,
                "seed": config.seed,
                "model_kind": "scratch_study_cnn",
                "target_names": list(metrics.target_names),
                "model_config": {
                    "feature_width": config.feature_width,
                    "output_count": 12,
                    "output_size": config.output_size,
                    "max_slices": config.max_slices,
                },
                "input_sha256": {"manifest": manifest_sha256, "folds": folds_sha256},
                "runtime": _runtime_metadata(config, time.perf_counter() - started, device=device),
                "row_index_semantics": "zero-based index within this validated fold partition; no study identifiers",
                "rows": prediction_rows,
            })
            report = {
                **plan,
                "status": "completed" if steps_completed == planned_steps else "partial_training",
                "epochs_requested": config.epochs,
                "training_steps_planned": planned_steps,
                "steps_completed": steps_completed,
                "per_target_auc": dict(zip(metrics.target_names, metrics.per_target)),
                "macro_auc": metrics.macro_auc,
                "missing_class_targets": list(metrics.missing_class_targets),
                "validation_study_count": len(prediction_rows),
                "model_kind": "scratch_study_cnn",
                "interpretation_limit": "Single fixed development fold; provisional silver labels, not clinical evidence or a final generalization claim.",
            }

        stage = "checkpoint"
        try:
            # Save only the model tensor mapping; never load user-supplied checkpoints.
            torch.save(model.state_dict(), config.output_dir / "checkpoint.pt")
        except (OSError, RuntimeError, ValueError):
            raise TrainingRunError("checkpoint_write_failed") from None
        stage = "report_write"
        report["runtime"] = _runtime_metadata(config, time.perf_counter() - started, device=device)
        report["config"] = _safe_config(config)
        report["input_sha256"] = {"manifest": manifest_sha256, "folds": folds_sha256}
        _write_json(config.output_dir / "report.json", report)
        return report
    except TrainingRunError:
        raise
    except Exception as error:
        # Do not leak tensor/device/path text or delete partial private evidence.
        raise TrainingRunError(
            "training_failed", diagnostics=_safe_failure_diagnostics(error, stage)
        ) from None
