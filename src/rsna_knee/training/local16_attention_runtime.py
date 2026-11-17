"""Single-arm E050 runner reusing the audited attention checkpoint/training kernel."""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

from rsna_knee.training.local16_attention import Attention16Config, make_attention16_plan


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _write_atomic(path: Path, payload: bytes, *, replace_existing: bool = False) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if (path.is_symlink() or (path.exists() and not replace_existing)
            or temporary.exists() or temporary.is_symlink()):
        raise ValueError("attention16_output_not_fresh")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def attention16_source_closure(root: Path, cli_path: Path) -> tuple[Path, ...]:
    """Pin every local source module used by cache validation, model, loss, and runner."""
    root = Path(root).resolve()
    relative = (
        "src/rsna_knee/training/local16_attention.py",
        "src/rsna_knee/training/local16_attention_runtime.py",
        "src/rsna_knee/training/attention_mil.py",
        "src/rsna_knee/training/masked_soft.py",
        "src/rsna_knee/training/resnet50.py",
        "src/rsna_knee/training/baseline.py",
        "src/rsna_knee/models/attention_mil.py",
        "src/rsna_knee/models/resnet50.py",
        "src/rsna_knee/data/masked_soft_dataset.py",
        "src/rsna_knee/data/training_dataset.py",
        "src/rsna_knee/data/dicom_preprocessing.py",
        "src/rsna_knee/data/preprocessing.py",
        "src/rsna_knee/labels/full_cohort_prefilter.py",
        "src/rsna_knee/labels/report_source_comparison.py",
        "src/rsna_knee/labels/subscription.py",
        "scripts/local16_cache_chunk_plan.py",
        "scripts/local16_tensor_cache.py",
        "scripts/local16_full_cache.py",
        str(Path(cli_path).resolve().relative_to(root)).replace("\\", "/"),
    )
    return tuple(root / item for item in relative)


def run_local16_attention(
    config: Attention16Config, *, manifest_path: Path, folds_path: Path, plan_path: Path,
    source_archive_path: Path, archive_dir: Path, cache_receipt_path: Path | None = None,
    source_paths: tuple[Path, ...], execute: bool = False, resume: bool = False,
    cache_verifier: Callable[..., dict[str, Any]] | None = None,
    dataset_factory: Callable[[str], Any] | None = None,
    train_arm: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Verify cache, plan without model work by default, and optionally train only attention16."""
    if type(execute) is not bool or type(resume) is not bool or (resume and not execute):
        raise ValueError("attention16_execution_flags_invalid")
    if cache_verifier is None:
        from scripts.local16_full_cache import verify_full_cache
        cache_verifier = verify_full_cache
    proof = cache_verifier(manifest_path=manifest_path, folds_path=folds_path, plan_path=plan_path,
                           source_archive_path=source_archive_path, archive_dir=archive_dir)
    receipt_raw = _canonical_json(proof)
    root = Path(config.repository_root).resolve()
    output_candidate = Path(config.output_dir)
    output_candidate = output_candidate if output_candidate.is_absolute() else root / output_candidate
    current = output_candidate.absolute()
    while current != current.parent:
        if current.exists() and current.is_symlink():
            raise ValueError("attention16_output_symlink_forbidden")
        if current == root:
            break
        current = current.parent
    output = output_candidate.resolve(strict=False)
    expected_receipt = Path(cache_receipt_path) if cache_receipt_path is not None else None
    if expected_receipt is not None:
        if expected_receipt.is_symlink() or not expected_receipt.is_file() or expected_receipt.read_bytes() != receipt_raw:
            raise ValueError("attention16_cache_receipt_mismatch")

    # The recipe builder rejects an existing output path. Its fingerprint does not include
    # the path, so use a private, non-created probe path for resume while retaining the exact
    # production namespace in the returned/persisted plan.
    probe_output = output if not output.exists() else output.with_name(output.name + ".plan-probe")
    if probe_output.exists() or probe_output.is_symlink():
        raise ValueError("attention16_plan_probe_not_fresh")
    with tempfile.TemporaryDirectory(prefix="local16-attention-plan-") as temporary:
        receipt_probe = Path(temporary) / "cache-verification-receipt.json"
        receipt_probe.write_bytes(receipt_raw)
        planned = make_attention16_plan(
            replace(config, output_dir=probe_output), proof,
            cache_receipt_path=receipt_probe, source_paths=source_paths)
    planned["output_dir"] = str(output)
    if not execute:
        return planned

    if resume:
        if output.is_symlink() or not output.is_dir():
            raise ValueError("attention16_resume_output_unavailable")
        try:
            saved_plan = json.loads((output / "run-plan.json").read_text(encoding="utf-8"))
            saved_receipt = (output / "cache-verification-receipt.json").read_bytes()
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise ValueError("attention16_resume_plan_unavailable") from None
        if saved_plan != planned or saved_receipt != receipt_raw:
            raise ValueError("attention16_resume_fingerprint_mismatch")
        arm_output = output / "target_attention"
        if arm_output.is_symlink() or not arm_output.is_dir():
            raise ValueError("attention16_resume_checkpoint_unavailable")
    elif output.exists() or output.is_symlink():
        raise ValueError("attention16_output_not_fresh")

    import torch
    if config.device != "cuda" or not torch.cuda.is_available():
        raise ValueError("cuda_required")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    if free_bytes < 3 * 1024**3:
        raise ValueError("gpu_free_memory_below_safety_margin")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")

    if dataset_factory is None:
        from scripts.local16_full_cache import Local16CacheDataset
        dataset_factory = lambda partition: Local16CacheDataset(
            manifest_path, plan_path, archive_dir, partition)
    train_data, validation_data = dataset_factory("train"), dataset_factory("validation")
    if len(train_data) != 2850 or len(validation_data) != 1448:
        raise ValueError("attention16_dataset_count_mismatch")
    from rsna_knee.training.attention_mil import AttentionMILConfig, _train_arm
    from rsna_knee.training.resnet50 import _seed_everything
    train_arm = _train_arm if train_arm is None else train_arm
    legacy_config = AttentionMILConfig(repository_root=root, weights_path=config.weights_path,
                                       output_dir=output, device=config.device)
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
        _write_atomic(output / "cache-verification-receipt.json", receipt_raw)
        _write_atomic(output / "run-plan.json", _canonical_json(planned))
        arm_output = output / "target_attention"
        arm_output.mkdir(parents=False, exist_ok=False)
    _seed_everything(42, config.device)
    report = train_arm(legacy_config, arm="target_attention", output=arm_output,
                       fingerprint=planned["fingerprint"], train_data=train_data,
                       validation_data=validation_data, epochs=5, resume=resume)
    result = {"status": report["status"], "fingerprint": planned["fingerprint"],
              "configuration": planned["configuration"], "output_dir": str(output),
              "arm": report, "runtime": __import__(
                  "rsna_knee.training.masked_soft", fromlist=["_runtime_versions"])._runtime_versions(),
              "model_constructed": True, "training_started": report["completed_epochs"] > 0,
              "device_memory_before": {"free_bytes": free_bytes, "total_bytes": total_bytes}}
    _write_atomic(output / "training-report.json", _canonical_json(result), replace_existing=resume)
    return result
