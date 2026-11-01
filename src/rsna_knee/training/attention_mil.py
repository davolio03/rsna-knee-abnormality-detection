"""Fresh paired equal-mean and target-attention RadImageNet training runs."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from rsna_knee.labels.full_cohort_prefilter import TARGET_COLUMNS
from rsna_knee.models.attention_mil import RadImageNetAttentionStudyModel
from rsna_knee.models.resnet50 import (BN_POLICY, TRAINABLE_POLICY, WEIGHT_PINS,
                                       load_resnet50_recipe)
from rsna_knee.training.masked_soft import (MaskedSoftMetricAccumulator, _capture_rng_state,
    _restore_rng_state, _runtime_versions, _validate_finite_tensors, _write_checkpoint, masked_soft_bce)
from rsna_knee.training.resnet50 import _seed_everything, _validate_optimizer


@dataclass(frozen=True)
class AttentionMILConfig:
    repository_root: Path
    weights_path: Path
    output_dir: Path
    device: str = "cuda"
    head_lr: float = 1e-3
    backbone_lr: float = 1e-5
    seed: int = 42
    epochs: int = 5

    def validate(self) -> None:
        if (self.device not in ("cpu", "cuda") or self.head_lr != 1e-3 or self.backbone_lr != 1e-5
                or self.seed != 42 or self.epochs != 5):
            raise ValueError("attention_config_invalid")


def _digest(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _weights_path(config: AttentionMILConfig, root: Path) -> Path:
    path = Path(config.weights_path)
    return path.resolve(strict=False) if path.is_absolute() else (root / path).resolve(strict=False)


def _safe_output_path(config: AttentionMILConfig, root: Path) -> Path:
    private_root = (root / "data/private/disease-attention-mil").resolve(strict=False)
    try:
        private_root.relative_to(root)
    except ValueError:
        raise ValueError("output_must_be_private") from None
    candidate = Path(config.output_dir)
    output = (candidate if candidate.is_absolute() else root / candidate).resolve(strict=False)
    try:
        output.relative_to(private_root)
    except ValueError:
        raise ValueError("output_must_be_private") from None
    current = output
    while current != root and current != private_root.parent:
        if current.exists() and current.is_symlink():
            raise ValueError("output_symlink_forbidden")
        if current == private_root:
            break
        current = current.parent
    return output


def _source_fingerprint(root: Path) -> dict[str, str]:
    source_root = Path(root).resolve()
    files = sorted((source_root / "src").rglob("*.py"))
    files.extend(source_root / "scripts" / name for name in
                 ("train_local_attention_mil.py", "local_masked_full_cache.py"))
    if source_root.is_symlink() or any(path.is_symlink() or not path.is_file() for path in files):
        raise ValueError("source_invalid")
    return {path.relative_to(source_root).as_posix(): _digest(path) for path in files}


def make_pair_plan(config: AttentionMILConfig, *, cache_validator=None) -> dict[str, object]:
    """Read and validate the private data/cache evidence without creating outputs."""
    config.validate()
    root = Path(config.repository_root).resolve()
    output_dir = _safe_output_path(config, root)
    cache_module = None
    if cache_validator is None:
        import sys
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from scripts import local_masked_full_cache as cache_module
        cache_validator = cache_module.validate_full_cache
    cache_proof = cache_validator(root)
    manifest = root / "data/private/full-cohort-prefilter/run01/full-cohort-soft-label-manifest.json"
    folds = manifest.with_name("full-cohort-grouped-folds.json")
    archive_root = root / "data/private/local-radimagenet-epoch02/full-cache-archives"
    weights_path = _weights_path(config, root)
    if weights_path.is_symlink() or not weights_path.is_file():
        raise ValueError("weights_unavailable")
    pin = WEIGHT_PINS["radimagenet"]
    if weights_path.stat().st_size != pin["size"] or _digest(weights_path) != pin["sha256"]:
        raise ValueError("weights_pin_mismatch")
    source = _source_fingerprint(root)
    if cache_module is None:
        import sys
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from scripts import local_masked_full_cache as cache_module
    receipt_path = root / "data/private/local-radimagenet-epoch02/cache-all-seven-chunks-independent-proof.json"
    data = {"manifest_sha256": _digest(manifest), "folds_sha256": _digest(folds),
            "cache_status": cache_proof.get("status"), "cache_studies": cache_proof.get("studies"),
            "cache_series": cache_proof.get("series"), "cache_receipt_sha256": _digest(receipt_path),
            "cache_archives": {name: {"bytes": size, "sha256": digest}
                               for name, (size, digest) in cache_module.PARTS.items()},
            "weights_sha256": pin["sha256"],
            "weights_size": pin["size"], "source": source}
    frozen = {"schema": "disease-attention-mil-run-v1", "seed": 42, "fold": 0, "epochs": 5,
              "arms": ["equal_mean", "target_attention"], "initialization": "fresh_original_radimagenet_weights",
              "head_lr": config.head_lr, "layer4_lr": config.backbone_lr, "optimizer": "Adam",
              "batchnorm": BN_POLICY, "trainable": TRAINABLE_POLICY, "output_size": 224,
              "max_slices": 8, "normalization": "radimagenet_2x_minus_1",
              "objective": "unweighted_observed_soft_bce", "metrics": ["masked_bce", "brier", "per_target_counts"],
              "data": data, "runtime": _runtime_versions()}
    fingerprint = hashlib.sha256(json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"status": "planned", "fingerprint": fingerprint, "configuration": frozen,
            "arm_states": {"equal_mean": "not_started", "target_attention": "not_started"},
            "training_studies": cache_proof.get("training_studies"),
            "validation_studies": cache_proof.get("validation_studies"),
            "cache_archive_directory": str(archive_root), "output_dir": str(output_dir),
            "training_requires_explicit_flag": "--train"}


def _model_for_arm(config: AttentionMILConfig, arm: str):
    root = Path(config.repository_root).resolve()
    base = load_resnet50_recipe("radimagenet", _weights_path(config, root), seed=42, slice_microbatch=2)
    return base if arm == "equal_mean" else RadImageNetAttentionStudyModel.from_base(base)


def _cache_dataset(root: Path, config: AttentionMILConfig, partition: str, cache_validated: bool):
    import sys
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts.local_masked_full_cache import FullCacheDataset
    return FullCacheDataset(root, config, partition, already_validated=cache_validated)


def _valid_resume(root: Path, fingerprint: str) -> dict[str, object] | None:
    plan_path = root / "run-plan.json"
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("resume_plan_unavailable") from None
    arm_states = plan.get("arm_states") if isinstance(plan, dict) else None
    if (not isinstance(plan, dict) or plan.get("fingerprint") != fingerprint or plan.get("status") != "planned"
            or not isinstance(arm_states, dict) or set(arm_states) != {"equal_mean", "target_attention"}
            or any(state not in {"not_started", "started", "checkpointed", "complete"}
                   for state in arm_states.values())):
        raise ValueError("resume_fingerprint_mismatch")
    return plan


def _save_atomic_report(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _set_arm_state(output: Path, plan: dict[str, object], arm: str, state: str) -> None:
    states = plan["arm_states"]
    allowed = {"not_started": {"started"}, "started": {"checkpointed", "complete"},
               "checkpointed": {"checkpointed", "complete"}, "complete": {"complete"}}
    if arm not in states or state not in allowed.get(states[arm], set()):
        raise ValueError("resume_arm_state_transition_invalid")
    updated = {**plan, "arm_states": {**states, arm: state}}
    _save_atomic_report(output / "run-plan.json", updated)
    plan["arm_states"] = updated["arm_states"]


def _preflight_pair_resume(output: Path, plan: dict[str, object]) -> None:
    states = plan["arm_states"]
    for arm in ("equal_mean", "target_attention"):
        arm_output = output / arm
        state = states[arm]
        if state == "not_started":
            if arm_output.exists():
                raise ValueError("resume_arm_state_mismatch")
            continue
        if not arm_output.is_dir():
            raise ValueError("checkpoint_missing")
        _, proof = _verified_latest_checkpoint(arm_output, arm=arm, fingerprint=plan["fingerprint"])
        if state == "complete" and proof["completed_epochs"] != 5:
            raise ValueError("checkpoint_status_mismatch")


def _validate_attention_history(history: object, completed_epochs: int) -> None:
    def finite_number(value, *, minimum=0.0, maximum=float("inf")):
        return type(value) in (int, float) and math.isfinite(value) and minimum <= value <= maximum

    if (not isinstance(history, list) or len(history) != completed_epochs
            or [row.get("epoch") for row in history if isinstance(row, dict)] != list(range(1, completed_epochs + 1))):
        raise ValueError("checkpoint_history_invalid")
    for row in history:
        if (not isinstance(row, dict) or set(row) != {
                "epoch", "training_masked_bce", "training_observed_cell_count", "validation", "elapsed_seconds"}
                or type(row["epoch"]) is not int
                or not finite_number(row["training_masked_bce"])
                or type(row["training_observed_cell_count"]) is not int
                or row["training_observed_cell_count"] <= 0
                or not finite_number(row["elapsed_seconds"], minimum=0.000001)):
            raise ValueError("checkpoint_history_invalid")
        metrics = row["validation"]
        if (not isinstance(metrics, dict) or set(metrics) != {
                "observed_cell_count", "masked_bce", "brier", "per_target"}
                or type(metrics["observed_cell_count"]) is not int or metrics["observed_cell_count"] <= 0
                or not finite_number(metrics["masked_bce"])
                or not finite_number(metrics["brier"], maximum=1.0)
                or not isinstance(metrics["per_target"], list)
                or len(metrics["per_target"]) != len(TARGET_COLUMNS)):
            raise ValueError("checkpoint_history_invalid")
        counts = []
        for target, expected_name in zip(metrics["per_target"], TARGET_COLUMNS):
            if (not isinstance(target, dict) or set(target) != {
                    "target", "observed_count", "masked_bce", "brier"}
                    or target["target"] != expected_name or type(target["observed_count"]) is not int
                    or target["observed_count"] < 0):
                raise ValueError("checkpoint_history_invalid")
            count = target["observed_count"]
            counts.append(count)
            if ((count and (not finite_number(target["masked_bce"])
                            or not finite_number(target["brier"], maximum=1.0)))
                    or (not count and (target["masked_bce"] is not None or target["brier"] is not None))):
                raise ValueError("checkpoint_history_invalid")
        if sum(counts) != metrics["observed_cell_count"]:
            raise ValueError("checkpoint_history_invalid")


def _validate_checkpoint_model(saved: object, model: torch.nn.Module) -> None:
    expected = model.state_dict()
    if (not isinstance(saved, dict) or set(saved) != set(expected)
            or not _validate_finite_tensors(saved)
            or any(not isinstance(saved[key], torch.Tensor) or saved[key].shape != expected[key].shape
                   or saved[key].dtype != expected[key].dtype for key in expected)):
        raise ValueError("checkpoint_model_invalid")


def _validate_checkpoint_optimizer(saved: object, optimizer: torch.optim.Optimizer,
                                   config: AttentionMILConfig, *, expected_steps: int) -> None:
    if (not isinstance(saved, dict) or set(saved) != {"state", "param_groups"}
            or not isinstance(saved["state"], dict) or not isinstance(saved["param_groups"], list)
            or len(saved["param_groups"]) != 2):
        raise ValueError("checkpoint_optimizer_invalid")
    names = ("head", "layer4")
    rates = (config.head_lr, config.backbone_lr)
    all_ids = []
    for saved_group, expected_group, name, rate in zip(
            saved["param_groups"], optimizer.param_groups, names, rates):
        if (not isinstance(saved_group, dict) or set(saved_group) != set(expected_group)
                or saved_group.get("name") != name or saved_group.get("lr") != rate
                or any(saved_group[key] != value for key, value in expected_group.items() if key != "params")
                or not isinstance(saved_group.get("params"), list)
                or len(saved_group["params"]) != len(expected_group["params"])):
            raise ValueError("checkpoint_optimizer_invalid")
        ids = saved_group["params"]
        if any(type(identifier) is not int for identifier in ids):
            raise ValueError("checkpoint_optimizer_invalid")
        all_ids.extend(ids)
        for identifier, parameter in zip(ids, expected_group["params"]):
            state = saved["state"].get(identifier)
            if not isinstance(state, dict) or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("checkpoint_optimizer_invalid")
            for key in ("exp_avg", "exp_avg_sq"):
                tensor = state[key]
                if (not isinstance(tensor, torch.Tensor) or tensor.shape != parameter.shape
                        or tensor.dtype != parameter.dtype or not torch.isfinite(tensor).all()):
                    raise ValueError("checkpoint_optimizer_invalid")
            step = state["step"]
            if (not isinstance(step, torch.Tensor) or step.numel() != 1
                    or step.dtype not in (torch.float32, torch.float64) or not torch.isfinite(step).all()
                    or step.item() != expected_steps
                    or torch.any(state["exp_avg_sq"] < 0)):
                raise ValueError("checkpoint_optimizer_invalid")
    if len(set(all_ids)) != len(all_ids) or set(saved["state"]) != set(all_ids):
        raise ValueError("checkpoint_optimizer_invalid")


def _validate_checkpoint_rng(state: object, *, device: str) -> None:
    if (not isinstance(state, dict) or set(state) != {"python", "numpy", "torch_cpu", "torch_cuda"}
            or device not in {"cpu", "cuda"}):
        raise ValueError("checkpoint_rng_invalid")
    try:
        random.Random().setstate(state["python"])
        numpy_state = state["numpy"]
        if (not isinstance(numpy_state, list) or len(numpy_state) != 5
                or type(numpy_state[0]) is not str or not isinstance(numpy_state[1], list)
                or type(numpy_state[2]) is not int or type(numpy_state[3]) is not int
                or type(numpy_state[4]) not in (int, float) or not math.isfinite(numpy_state[4])):
            raise ValueError
        np.random.RandomState().set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                                           numpy_state[2], numpy_state[3], float(numpy_state[4])))
    except (TypeError, ValueError, OverflowError):
        raise ValueError("checkpoint_rng_invalid") from None
    cpu_state, cuda_states = state["torch_cpu"], state["torch_cuda"]
    if (not isinstance(cpu_state, torch.Tensor) or cpu_state.dtype != torch.uint8 or cpu_state.ndim != 1
            or cpu_state.shape != torch.get_rng_state().shape or not isinstance(cuda_states, list)):
        raise ValueError("checkpoint_rng_invalid")
    if device == "cpu" and cuda_states:
        raise ValueError("checkpoint_rng_invalid")
    if device == "cuda" and (not cuda_states or len(cuda_states) != torch.cuda.device_count()
                              or any(not isinstance(value, torch.Tensor) or value.dtype != torch.uint8
                                     or value.ndim != 1 for value in cuda_states)):
        raise ValueError("checkpoint_rng_invalid")
    if device == "cuda":
        expected_cuda = torch.cuda.get_rng_state_all()
        if any(value.shape != expected.shape for value, expected in zip(cuda_states, expected_cuda)):
            raise ValueError("checkpoint_rng_invalid")


def _verified_latest_checkpoint(output: Path, *, arm: str, fingerprint: str) -> tuple[Path, dict[str, object]]:
    paths = sorted(output.glob("checkpoint-epoch-*.pt"))
    if not paths:
        raise ValueError("checkpoint_missing")
    path = paths[-1]
    sidecar = path.with_suffix(".sha256")
    if (path.is_symlink() or sidecar.is_symlink()
            or any(parent.is_symlink() for parent in (path, *path.parents))):
        raise ValueError("checkpoint_unavailable")
    if not sidecar.is_file():
        raise ValueError("checkpoint_proof_missing")
    try:
        proof = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("checkpoint_proof_unavailable") from None
    if (not isinstance(proof, dict) or set(proof) != {
            "schema", "arm", "fingerprint", "completed_epochs", "checkpoint_sha256"}
            or proof.get("schema") != "disease-attention-mil-checkpoint-proof-v1"
            or proof.get("fingerprint") != fingerprint or proof.get("arm") != arm
            or type(proof.get("completed_epochs")) is not int
            or not 1 <= proof["completed_epochs"] <= 5
            or path.name != f"checkpoint-epoch-{proof['completed_epochs']:02d}.pt"):
        raise ValueError("checkpoint_fingerprint_mismatch")
    if not path.is_file() or _digest(path) != proof["checkpoint_sha256"]:
        raise ValueError("checkpoint_sha_mismatch")
    return path, proof


def _train_arm(config: AttentionMILConfig, *, arm: str, output: Path, fingerprint: str,
               train_data, validation_data, epochs: int, resume: bool,
               checkpoint_callback=None) -> dict[str, object]:
    checkpoint_paths = sorted(output.glob("checkpoint-epoch-*.pt"))
    checkpoint_path = None
    proof = None
    if resume:
        checkpoint_path, proof = _verified_latest_checkpoint(output, arm=arm, fingerprint=fingerprint)
    elif checkpoint_paths:
        raise ValueError("checkpoint_already_exists")

    model = _model_for_arm(config, arm).to(config.device)
    optimizer = torch.optim.Adam(model.optimizer_groups(head_lr=config.head_lr, backbone_lr=config.backbone_lr))
    history = []
    start_epoch = 0
    latest_checkpoint = None
    if resume:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except Exception:
            raise ValueError("checkpoint_unavailable") from None
        expected_keys = {"schema", "status", "fingerprint", "arm", "completed_epochs", "history",
                         "model_state", "optimizer_state", "rng_state"}
        if (not isinstance(checkpoint, dict) or set(checkpoint) != expected_keys
                or checkpoint.get("schema") != "disease-attention-mil-checkpoint-v2"
                or checkpoint.get("status") != "epoch_checkpoint_complete"
                or checkpoint.get("fingerprint") != fingerprint or checkpoint.get("arm") != arm
                or type(checkpoint.get("completed_epochs")) is not int
                or checkpoint.get("completed_epochs") != proof["completed_epochs"]
                or not 1 <= checkpoint["completed_epochs"] <= 5):
            raise ValueError("checkpoint_incompatible")
        history = checkpoint["history"]
        _validate_attention_history(history, checkpoint["completed_epochs"])
        _validate_checkpoint_model(checkpoint["model_state"], model)
        _validate_checkpoint_optimizer(checkpoint["optimizer_state"], optimizer, config,
                                       expected_steps=len(train_data) * checkpoint["completed_epochs"])
        _validate_checkpoint_rng(checkpoint["rng_state"], device=config.device)

        # Restore only after every checkpoint component has passed validation.
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        _validate_optimizer(optimizer, config)
        _restore_rng_state(checkpoint["rng_state"], device=config.device)
        start_epoch = checkpoint["completed_epochs"]
        latest_checkpoint = checkpoint_path

    for epoch in range(start_epoch, epochs):
        model.train(True)
        loss_sum = 0.0
        observed_count = 0
        started = time.perf_counter()
        for index in range(len(train_data)):
            images, masks, targets, observed, _ = train_data[index]
            images = [image.to(config.device, dtype=torch.float32) for image in images]
            masks = [mask.to(config.device, dtype=torch.bool) for mask in masks]
            targets = targets.to(config.device, dtype=torch.float32).reshape(1, -1)
            observed = observed.to(config.device, dtype=torch.bool).reshape(1, -1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images, masks)
            loss = masked_soft_bce(logits, targets, observed)
            if not torch.isfinite(loss):
                raise ValueError("loss_nonfinite")
            loss.backward()
            optimizer.step()
            count = int(observed.sum().item())
            loss_sum += float(loss.detach()) * count
            observed_count += count

        model.eval()
        metrics = MaskedSoftMetricAccumulator(TARGET_COLUMNS)
        with torch.no_grad():
            for index in range(len(validation_data)):
                images, masks, targets, observed, _ = validation_data[index]
                images = [image.to(config.device, dtype=torch.float32) for image in images]
                masks = [mask.to(config.device, dtype=torch.bool) for mask in masks]
                targets = targets.to(config.device, dtype=torch.float32).reshape(1, -1)
                observed = observed.to(config.device, dtype=torch.bool).reshape(1, -1)
                metrics.update(model(images, masks), targets, observed)
        if observed_count <= 0:
            raise ValueError("no_observed_targets")
        entry = {"epoch": epoch + 1, "training_masked_bce": loss_sum / observed_count,
                 "training_observed_cell_count": observed_count,
                 "validation": metrics.compute(), "elapsed_seconds": time.perf_counter() - started}
        history.append(entry)
        payload = {"schema": "disease-attention-mil-checkpoint-v2", "status": "epoch_checkpoint_complete",
                   "fingerprint": fingerprint,
                   "arm": arm, "completed_epochs": epoch + 1, "history": history,
                   "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                   "rng_state": _capture_rng_state(device=config.device)}
        _validate_attention_history(payload["history"], payload["completed_epochs"])
        _validate_checkpoint_model(payload["model_state"], model)
        _validate_checkpoint_optimizer(payload["optimizer_state"], optimizer, config,
                                       expected_steps=len(train_data) * (epoch + 1))
        _validate_checkpoint_rng(payload["rng_state"], device=config.device)
        checkpoint_path = output / f"checkpoint-epoch-{epoch + 1:02d}.pt"
        checkpoint_sha = _write_checkpoint(checkpoint_path, payload)
        if _digest(checkpoint_path) != checkpoint_sha:
            raise ValueError("checkpoint_readback_mismatch")
        try:
            readback = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except Exception:
            raise ValueError("checkpoint_readback_mismatch") from None
        if (not isinstance(readback, dict) or set(readback) != set(payload)
                or readback.get("schema") != payload["schema"]
                or readback.get("status") != payload["status"]
                or readback.get("fingerprint") != fingerprint or readback.get("arm") != arm
                or readback.get("completed_epochs") != payload["completed_epochs"]):
            raise ValueError("checkpoint_readback_mismatch")
        _validate_attention_history(readback.get("history"), readback.get("completed_epochs"))
        _validate_checkpoint_model(readback.get("model_state"), model)
        _validate_checkpoint_optimizer(readback.get("optimizer_state"), optimizer, config,
                                       expected_steps=len(train_data) * (epoch + 1))
        _validate_checkpoint_rng(readback.get("rng_state"), device=config.device)
        sidecar = checkpoint_path.with_suffix(".sha256")
        temporary_sidecar = sidecar.with_suffix(".sha256.tmp")
        temporary_sidecar.write_text(json.dumps({"schema": "disease-attention-mil-checkpoint-proof-v1",
            "arm": arm, "fingerprint": fingerprint,
            "completed_epochs": epoch + 1, "checkpoint_sha256": checkpoint_sha},
            sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        temporary_sidecar.replace(sidecar)
        latest_checkpoint = checkpoint_path
        if checkpoint_callback is not None:
            checkpoint_callback()
    return {"arm": arm, "completed_epochs": len(history), "epochs": history,
            "checkpoint_sha256": _digest(latest_checkpoint) if latest_checkpoint is not None else None,
            "status": "complete" if len(history) == 5 else "paused_checkpoint_available"}


def run_attention_pair(config: AttentionMILConfig, *, execute: bool = False, smoke: bool = False,
                       resume: bool = False, cache_validator=None, dataset_factory=None) -> dict[str, object]:
    if smoke and execute:
        raise ValueError("execution_mode_conflict")
    plan = make_pair_plan(config, cache_validator=cache_validator)
    if not execute and not smoke:
        return plan
    if config.device != "cuda" or not torch.cuda.is_available():
        raise ValueError("cuda_required")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    if free_bytes < 3 * 1024**3:
        raise ValueError("gpu_free_memory_below_safety_margin")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    output = Path(plan["output_dir"])
    if output.exists():
        if not resume:
            raise ValueError("output_must_be_new")
        saved_plan = _valid_resume(output, plan["fingerprint"])
        plan["arm_states"] = saved_plan["arm_states"]
        _preflight_pair_resume(output, plan)
    elif resume:
        raise ValueError("resume_output_unavailable")
    else:
        output.mkdir(parents=True, exist_ok=False)
        _save_atomic_report(output / "run-plan.json", plan)
    root = Path(config.repository_root).resolve()
    if dataset_factory is None:
        dataset_factory = lambda partition: _cache_dataset(root, config, partition, True)
    train_data = dataset_factory("train")
    validation_data = dataset_factory("validation")

    if smoke:
        _seed_everything(config.seed, config.device)
        dataset = train_data
        images, masks, targets, observed, _ = dataset[0]
        model = _model_for_arm(config, "target_attention").to(config.device)
        optimizer = torch.optim.Adam(model.optimizer_groups(head_lr=config.head_lr, backbone_lr=config.backbone_lr))
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        images = [image.to(config.device, dtype=torch.float32) for image in images]
        masks = [mask.to(config.device, dtype=torch.bool) for mask in masks]
        targets = targets.to(config.device, dtype=torch.float32).reshape(1, -1)
        observed = observed.to(config.device, dtype=torch.bool).reshape(1, -1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=False):
            loss = masked_soft_bce(model(images, masks), targets, observed)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        return {"status": "smoke_complete", "training_epochs_started": False, "studies": 1,
                "elapsed_seconds": time.perf_counter() - started, "loss": float(loss.detach()),
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "free_bytes_before": free_bytes, "total_bytes": total_bytes,
                "checkpoint_written": False, "weight_update_steps": 1}

    reports = {}
    for arm in ("equal_mean", "target_attention"):
        _seed_everything(config.seed, config.device)
        arm_output = output / arm
        state = plan["arm_states"][arm]
        if state == "not_started":
            _set_arm_state(output, plan, arm, "started")
            arm_output.mkdir(parents=True, exist_ok=False)
            arm_resume = False
        else:
            arm_resume = True
        reports[arm] = _train_arm(config, arm=arm, output=arm_output, fingerprint=plan["fingerprint"],
                                  train_data=train_data, validation_data=validation_data,
                                  epochs=config.epochs, resume=arm_resume,
                                  checkpoint_callback=lambda arm=arm: _set_arm_state(output, plan, arm, "checkpointed"))
        if reports[arm]["status"] == "complete":
            _set_arm_state(output, plan, arm, "complete")
    result = {"status": "complete" if all(row["status"] == "complete" for row in reports.values()) else "paused_checkpoint_available",
              "fingerprint": plan["fingerprint"], "arms": reports, "runtime": _runtime_versions()}
    _save_atomic_report(output / "training-report.json", result)
    return result
