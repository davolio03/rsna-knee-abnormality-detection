"""Audited one-GPU migration for the private RadImageNet epoch-one checkpoint.

This adapter preserves the frozen recipe package and its resume validator. It
rebinds only the environment fingerprint and the single active CUDA RNG state;
it is not a general checkpoint conversion tool.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import importlib
import json
import os
import argparse
from pathlib import Path, PurePosixPath
import platform
import shutil
import stat
import sys
import tempfile
from typing import Any, Iterator, Mapping
import zipfile


SOURCE_SHA256 = "48a64d88c9b5cf72f4fdf6d9ce9839cc2cb3b8632b9fbf4fb7f4db1cdc4f95c8"
SOURCE_ASSETS_SHA256 = "22695a38a98fb868bfce34a8fe244d02cf3c8694abc8048672c7c9edaa9db4b6"
CHECKPOINT_SHA256 = "83518ebe59f8a3ecf39b2d8da35aa4c618afeeb04a78b315b5cda13ff56014a8"
ORIGINAL_FINGERPRINT = "1275e1e977188df71008cc4814825f170bd9dc84f76fd4143e87479ae38080be"
MANIFEST_SHA256 = "617ea1c267960b1e903b332a78388db3b6ab80d44374721df5a4c5e184e8961b"
FOLDS_SHA256 = "b816c7902a5d0caa68a9e818d3f913943bc61cd2e6db65c4de069f5cd94e15f9"
WEIGHTS_SHA256 = "08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734"
WEIGHTS_SIZE = 94_345_733

OLD_RUNTIME = {
    "python": "3.13.15",
    "torch": "2.11.0+cu128",
    "numpy": "2.1.3",
    "cuda": "12.8",
}
TARGET_RUNTIME = {
    "python": "3.11.9",
    "torch": "2.11.0+cu128",
    "numpy": "2.1.3",
    "cuda": "12.8",
}
EXPECTED_PARTITION_COUNTS = {"train": 2850, "validation": 1448}
EXPECTED_ORIGINAL_CUDA_RNG_COUNT = 2


class CheckpointMigrationError(ValueError):
    """Raised when an exact migration precondition is not satisfied."""


def _fail(code: str) -> None:
    raise CheckpointMigrationError(code)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _has_symlink_component(path: Path) -> bool:
    return any(part.is_symlink() for part in (path, *path.parents))


def _pinned_file(path: Path, expected_sha256: str, error: str) -> bytes:
    path = Path(path)
    if path.is_symlink() or _has_symlink_component(path) or not path.is_file():
        _fail(error)
    try:
        raw = path.read_bytes()
    except OSError:
        _fail(error)
    if _sha256_bytes(raw) != expected_sha256:
        _fail(error)
    return raw


def validate_target_runtime(runtime: Mapping[str, Any], *, cuda_device_count: int) -> bool:
    if (type(runtime) is not dict or runtime != TARGET_RUNTIME
            or type(cuda_device_count) is not int or cuda_device_count != 1):
        _fail("target_runtime_mismatch")
    return True


def _current_target_runtime() -> dict[str, str]:
    import numpy as np
    import torch

    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "cuda": str(torch.version.cuda) if torch.version.cuda else "",
    }


@contextmanager
def _fp32_precision_policy(torch: Any) -> Iterator[None]:
    previous = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.get_float32_matmul_precision(),
    )
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous[0]
        torch.backends.cudnn.allow_tf32 = previous[1]
        torch.set_float32_matmul_precision(previous[2])


def _validate_source_name(name: str) -> bool:
    if type(name) is not str or "\\" in name:
        return False
    path = PurePosixPath(name)
    return (not path.is_absolute() and ".." not in path.parts and "." not in path.parts
            and path.as_posix() == name and name.startswith("rsna_knee/")
            and name.endswith(".py"))


def extract_pinned_source(source_archive: Path, source_assets: Path,
                          source_root: Path) -> dict[str, str]:
    """Verify and extract exactly the 29 frozen Python members into a fresh root."""
    source_archive, source_assets, source_root = map(Path, (source_archive, source_assets, source_root))
    archive_bytes = _pinned_file(source_archive, SOURCE_SHA256, "source_archive_pin_mismatch")
    assets_bytes = _pinned_file(source_assets, SOURCE_ASSETS_SHA256, "source_assets_pin_mismatch")
    try:
        assets = json.loads(assets_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        _fail("source_assets_invalid")
    members = assets.get("source_members") if type(assets) is dict else None
    if (assets.get("schema") != "resnet50-recipe-assets-v1"
            or assets.get("source_sha256") != SOURCE_SHA256
            or type(members) is not dict or len(members) != 29
            or any(not _validate_source_name(name) or type(digest) is not str
                   or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
                   for name, digest in members.items())):
        _fail("source_assets_invalid")
    if source_root.exists() or source_root.is_symlink() or _has_symlink_component(source_root):
        _fail("source_root_not_fresh")

    try:
        with zipfile.ZipFile(source_archive, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if (len(names) != 29 or len(set(names)) != 29 or set(names) != set(members)
                    or any(not _validate_source_name(name) for name in names)):
                _fail("source_member_set_mismatch")
            payloads: dict[str, bytes] = {}
            for info in infos:
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode) or info.is_dir():
                    _fail("source_member_type_invalid")
                data = archive.read(info)
                if _sha256_bytes(data) != members[info.filename]:
                    _fail("source_member_hash_mismatch")
                payloads[info.filename] = data
    except (OSError, zipfile.BadZipFile, RuntimeError):
        _fail("source_archive_invalid")

    source_root.mkdir(parents=True, exist_ok=False)
    try:
        for name, data in payloads.items():
            destination = source_root.joinpath(*PurePosixPath(name).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if _has_symlink_component(destination):
                _fail("source_extraction_path_invalid")
            with destination.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        extracted = {
            path.relative_to(source_root).as_posix(): _sha256_file(path)
            for path in source_root.rglob("*.py")
        }
        all_entries = list(source_root.rglob("*"))
        if (extracted != members or len([item for item in all_entries if item.is_file()]) != 29
                or any(item.is_symlink() or not (item.is_file() or item.is_dir()) for item in all_entries)):
            _fail("source_extraction_inventory_mismatch")
    except BaseException:
        shutil.rmtree(source_root, ignore_errors=True)
        raise
    return dict(sorted(extracted.items()))


@contextmanager
def _frozen_import(source_root: Path) -> Iterator[None]:
    """Import the validated frozen package without bytecode or cross-tree module reuse."""
    root = str(Path(source_root))
    previous_path = list(sys.path)
    previous_bytecode = sys.dont_write_bytecode
    previous_modules = {name: module for name, module in sys.modules.items()
                        if name == "rsna_knee" or name.startswith("rsna_knee.")}
    for name in previous_modules:
        sys.modules.pop(name, None)
    sys.path.insert(0, root)
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        for name in [name for name in sys.modules
                     if name == "rsna_knee" or name.startswith("rsna_knee.")]:
            sys.modules.pop(name, None)
        sys.modules.update(previous_modules)
        sys.path[:] = previous_path
        sys.dont_write_bytecode = previous_bytecode


def _load_frozen_runner(source_root: Path):
    return importlib.import_module("rsna_knee.training.resnet50")


def _canonical_selection(runner: Any, manifest_path: Path, folds_path: Path) -> str:
    if (_has_symlink_component(manifest_path) or _has_symlink_component(folds_path)
            or _sha256_file(manifest_path) != MANIFEST_SHA256
            or _sha256_file(folds_path) != FOLDS_SHA256):
        _fail("cohort_source_pin_mismatch")
    dataset_module = importlib.import_module("rsna_knee.data.masked_soft_dataset")
    try:
        records, assignments, *_ = dataset_module.load_verified_soft_artifacts(manifest_path, folds_path)
    except Exception:
        _fail("cohort_artifacts_invalid")
    train = [record for record in records if assignments[record["study_id"]] != 0]
    validation = [record for record in records if assignments[record["study_id"]] == 0]
    if (len(records) != sum(EXPECTED_PARTITION_COUNTS.values())
            or {"train": len(train), "validation": len(validation)} != EXPECTED_PARTITION_COUNTS):
        _fail("cohort_partition_count_mismatch")
    selected = {
        "train": [record["study_id"] for record in train],
        "validation": [record["study_id"] for record in validation],
    }
    return _sha256_bytes(json.dumps(selected, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _fingerprint_with_runtime(runner: Any, config: Any, selection: str,
                              runtime: Mapping[str, str], source_hashes: Mapping[str, str]) -> str:
    fields = {
        "schema": "resnet50-recipe-checkpoint-v1",
        "arm": config.arm,
        "mode": config.mode,
        "normalization": runner.WEIGHT_PINS[config.arm]["normalization"],
        "weights": runner.WEIGHT_PINS[config.arm],
        "manifest": runner.MANIFEST_SHA256,
        "folds": runner.FOLDS_ARTIFACT_SHA256,
        "fold": 0,
        "seed": 42,
        "output_size": 224,
        "max_slices": 8,
        "pooling": "mean_slice_mean_series_logits",
        "head_lr": config.head_lr,
        "layer4_lr": config.backbone_lr,
        "bn": runner.BN_POLICY,
        "trainable": runner.TRAINABLE_POLICY,
        "epochs": 5,
        "selection": selection,
        "runtime": dict(runtime),
        "source": dict(source_hashes),
    }
    return _sha256_bytes(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _deep_equal(left: Any, right: Any) -> bool:
    import torch

    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
                and left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right))
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_deep_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_deep_equal(a, b) for a, b in zip(left, right))
    return left == right


def _verify_only_allowed_changes(original: Mapping[str, Any], migrated: Mapping[str, Any]) -> None:
    import torch

    if set(original) != set(migrated):
        _fail("migration_field_set_changed")
    for key in original:
        if key == "fingerprint":
            if (migrated[key] == original[key] or type(migrated[key]) is not str
                    or len(migrated[key]) != 64):
                _fail("migration_fingerprint_change_invalid")
        elif key == "rng_state":
            old_rng, new_rng = original[key], migrated[key]
            if (not isinstance(old_rng, dict) or set(old_rng) != {"python", "numpy", "torch_cpu", "torch_cuda"}
                    or not isinstance(new_rng, dict) or set(new_rng) != set(old_rng)):
                _fail("migration_rng_state_invalid")
            for rng_key in ("python", "numpy", "torch_cpu"):
                if not _deep_equal(old_rng[rng_key], new_rng[rng_key]):
                    _fail("migration_rng_state_changed")
            old_cuda, new_cuda = old_rng["torch_cuda"], new_rng["torch_cuda"]
            if (not isinstance(old_cuda, list) or len(old_cuda) != EXPECTED_ORIGINAL_CUDA_RNG_COUNT
                    or not isinstance(new_cuda, list) or len(new_cuda) != 1
                    or not _deep_equal(new_cuda[0], old_cuda[0])):
                _fail("migration_cuda_rng_mapping_invalid")
        elif not _deep_equal(original[key], migrated[key]):
            _fail("migration_checkpoint_state_changed")


def _write_file_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        _fail("migration_output_not_fresh")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    if path.read_bytes() != payload:
        _fail("migration_output_readback_mismatch")


def migrate_epoch_one_checkpoint(*, checkpoint_path: Path, source_archive: Path,
                                 source_assets: Path, manifest_path: Path, folds_path: Path,
                                 weights_path: Path, source_root: Path,
                                 output_dir: Path) -> dict[str, Any]:
    """Validate the original checkpoint, migrate its one active CUDA RNG, and publish atomically."""
    import numpy as np
    import torch

    checkpoint_path, source_archive, source_assets, manifest_path, folds_path, weights_path, source_root, output_dir = map(
        Path, (checkpoint_path, source_archive, source_assets, manifest_path, folds_path,
              weights_path, source_root, output_dir))
    runtime = _current_target_runtime()
    validate_target_runtime(runtime, cuda_device_count=torch.cuda.device_count())
    if not torch.cuda.is_available():
        _fail("target_runtime_mismatch")
    _pinned_file(checkpoint_path, CHECKPOINT_SHA256, "original_checkpoint_pin_mismatch")
    if output_dir.exists() or output_dir.is_symlink() or _has_symlink_component(output_dir):
        _fail("migration_output_not_fresh")
    if not output_dir.parent.is_dir() or _has_symlink_component(output_dir.parent):
        _fail("migration_output_parent_invalid")
    source_hashes = extract_pinned_source(source_archive, source_assets, source_root)

    with _frozen_import(source_root):
        runner = _load_frozen_runner(source_root)
        selection = _canonical_selection(runner, manifest_path, folds_path)
        config = runner.RecipeConfig(
            arm="radimagenet", weights_path=weights_path, output_dir=output_dir / "run",
            image_root=output_dir.parent, manifest_path=manifest_path, folds_path=folds_path,
            mode="train", device="cuda", source_root=source_root,
            max_epochs_per_run=1, max_runtime_seconds=28800,
            resume_checkpoint=checkpoint_path, resume_sha256=CHECKPOINT_SHA256,
        )
        config.validate()

        original_fingerprint = _fingerprint_with_runtime(
            runner, config, selection, OLD_RUNTIME, source_hashes)
        if original_fingerprint != ORIGINAL_FINGERPRINT:
            _fail("original_fingerprint_recompute_mismatch")
        migrated_fingerprint = runner._fingerprint(config, selection)
        if migrated_fingerprint == original_fingerprint:
            _fail("target_fingerprint_unchanged")

        # Use the frozen runner's unmodified schema/hash/history validator.
        original = runner._read_resume(config, original_fingerprint)
        import torchvision  # Ensure the target model constructor's runtime package is available.
        del torchvision
        weights = _pinned_file(weights_path, WEIGHTS_SHA256, "weights_pin_mismatch")
        if len(weights) != WEIGHTS_SIZE:
            _fail("weights_pin_mismatch")
        model = importlib.import_module("rsna_knee.models.resnet50").load_resnet50_recipe(
            "radimagenet", weights_path)
        try:
            model.load_state_dict(original["model_state"], strict=True)
        except Exception:
            _fail("checkpoint_model_invalid")
        optimizer = torch.optim.Adam(model.optimizer_groups(
            head_lr=config.head_lr, backbone_lr=config.backbone_lr))
        try:
            optimizer.load_state_dict(original["optimizer_state"])
            runner._validate_optimizer(optimizer, config)
        except Exception:
            _fail("checkpoint_optimizer_invalid")

        old_cuda_states = original["rng_state"].get("torch_cuda")
        if (not isinstance(old_cuda_states, list)
                or len(old_cuda_states) != EXPECTED_ORIGINAL_CUDA_RNG_COUNT
                or any(not isinstance(state, torch.Tensor) or state.dtype != torch.uint8 or state.ndim != 1
                       for state in old_cuda_states)):
            _fail("original_cuda_rng_state_invalid")
        mapped_state = dict(original)
        mapped_state["fingerprint"] = migrated_fingerprint
        mapped_state["rng_state"] = dict(original["rng_state"])
        mapped_state["rng_state"]["torch_cuda"] = [old_cuda_states[0]]
        _verify_only_allowed_changes(original, mapped_state)

        stage_dir = Path(tempfile.mkdtemp(prefix=output_dir.name + ".tmp-", dir=output_dir.parent))
        published = False
        try:
            checkpoint_out = stage_dir / "checkpoint-epoch-01-migrated.pt"
            checkpoint_tmp = checkpoint_out.with_name(checkpoint_out.name + ".tmp")
            with checkpoint_tmp.open("xb") as stream:
                torch.save(mapped_state, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(checkpoint_tmp, checkpoint_out)
            migrated_sha = _sha256_file(checkpoint_out)
            target_config = replace(config, resume_checkpoint=checkpoint_out,
                                    resume_sha256=migrated_sha)
            readback = runner._read_resume(target_config, migrated_fingerprint)
            _verify_only_allowed_changes(original, readback)

            rng_state = readback["rng_state"]
            with _fp32_precision_policy(torch):
                if (torch.backends.cuda.matmul.allow_tf32
                        or torch.backends.cudnn.allow_tf32
                        or torch.get_float32_matmul_precision() != "highest"):
                    _fail("target_precision_policy_mismatch")
                runner._restore_rng_state(rng_state, device="cuda")
                torch.cuda.synchronize(0)
                restored = torch.cuda.get_rng_state_all()
                if len(restored) != 1 or not torch.equal(restored[0], rng_state["torch_cuda"][0]):
                    _fail("target_cuda_rng_restore_mismatch")

            script_hash = _sha256_file(Path(__file__))
            unused_state_hash = _sha256_bytes(old_cuda_states[1].cpu().numpy().tobytes())
            active_state_hash = _sha256_bytes(old_cuda_states[0].cpu().numpy().tobytes())
            receipt: dict[str, Any] = {
                "schema_version": "radimagenet-one-gpu-checkpoint-migration-v1",
                "status": "verified",
                "source_archive_sha256": SOURCE_SHA256,
                "source_member_count": len(source_hashes),
                "source_members_sha256": _sha256_bytes(json.dumps(
                    source_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")),
                "manifest_sha256": MANIFEST_SHA256,
                "folds_sha256": FOLDS_SHA256,
                "weights_sha256": WEIGHTS_SHA256,
                "original_checkpoint_sha256": CHECKPOINT_SHA256,
                "migrated_checkpoint_sha256": migrated_sha,
                "original_fingerprint": original_fingerprint,
                "migrated_fingerprint": migrated_fingerprint,
                "completed_epochs": readback["completed_epochs"],
                "history_entry_count": len(readback["history"]),
                "original_runtime": dict(OLD_RUNTIME),
                "target_runtime": runtime,
                "cuda_device_map": {"0": "0"},
                "original_cuda_rng_state_count": EXPECTED_ORIGINAL_CUDA_RNG_COUNT,
                "unused_original_cuda_rng_state_count": 1,
                "unused_original_cuda_rng_state_sha256": unused_state_hash,
                "mapped_cuda_rng_state_sha256": active_state_hash,
                "preserved_checkpoint_fields": [
                    "schema", "completed_epochs", "history", "model_state",
                    "optimizer_state", "rng_state.python", "rng_state.numpy", "rng_state.torch_cpu",
                    "last_epoch_seconds",
                ],
                "changed_checkpoint_fields": ["fingerprint", "rng_state.torch_cuda"],
                "optimizer_validation": "passed",
                "target_rng_restore_validation": "passed",
                "precision_policy": {
                    "dtype": "float32",
                    "autocast": False,
                    "cuda_matmul_tf32": False,
                    "cudnn_tf32": False,
                },
                "migration_utility_sha256": script_hash,
            }
            receipt_raw = (json.dumps(receipt, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False) + "\n").encode("utf-8")
            _write_file_atomic(stage_dir / "migration-receipt.json", receipt_raw)
            if (set(path.name for path in stage_dir.iterdir())
                    != {"checkpoint-epoch-01-migrated.pt", "migration-receipt.json"}):
                _fail("migration_output_inventory_mismatch")
            os.replace(stage_dir, output_dir)
            published = True
            return receipt
        finally:
            if not published and stage_dir.exists() and not stage_dir.is_symlink():
                shutil.rmtree(stage_dir, ignore_errors=True)


def _repo_input_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    return {
        "checkpoint_path": root / "data/private/local-radimagenet-epoch02/checkpoint-epoch-01.pt",
        "source_archive": root / "data/private/resnet50-pretraining-preparation/run01/resnet50-recipe-source.zip",
        "source_assets": root / "data/private/resnet50-pretraining-preparation/run01/upload/resnet50-recipe-assets.json",
        "manifest_path": root / "data/private/full-cohort-prefilter/run01/full-cohort-soft-label-manifest.json",
        "folds_path": root / "data/private/full-cohort-prefilter/run01/full-cohort-grouped-folds.json",
        "weights_path": root / "data/private/resnet50-pretraining-assets/run01/resnet50-radimagenet.pt",
        "source_root": root / "data/private/local-radimagenet-epoch02/frozen-recipe-source-run01",
        "output_dir": root / "data/private/local-radimagenet-epoch02/checkpoint-migration-run01",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit and migrate the pinned epoch-one checkpoint for one local CUDA device.")
    parser.add_argument("--execute", action="store_true",
                        help="write a fresh migrated checkpoint and receipt; default is read-only planning")
    args = parser.parse_args(argv)
    import torch

    paths = _repo_input_paths()
    runtime = _current_target_runtime()
    validate_target_runtime(runtime, cuda_device_count=torch.cuda.device_count())
    if not torch.cuda.is_available():
        _fail("target_runtime_mismatch")
    for key, expected in (
        ("checkpoint_path", CHECKPOINT_SHA256), ("source_archive", SOURCE_SHA256),
        ("source_assets", SOURCE_ASSETS_SHA256), ("manifest_path", MANIFEST_SHA256),
        ("folds_path", FOLDS_SHA256), ("weights_path", WEIGHTS_SHA256),
    ):
        _pinned_file(paths[key], expected, f"{key}_pin_mismatch")
    if paths["weights_path"].stat().st_size != WEIGHTS_SIZE:
        _fail("weights_pin_mismatch")
    if paths["source_root"].exists() or paths["output_dir"].exists():
        _fail("migration_output_not_fresh")
    if args.execute:
        receipt = migrate_epoch_one_checkpoint(**paths)
        print(json.dumps({"status": "verified", "completed_epochs": receipt["completed_epochs"],
                          "migrated_checkpoint_sha256": receipt["migrated_checkpoint_sha256"],
                          "migrated_fingerprint": receipt["migrated_fingerprint"]},
                         sort_keys=True, allow_nan=False))
    else:
        print(json.dumps({"status": "planned", "completed_epochs": 1,
                          "source_archive_sha256": SOURCE_SHA256,
                          "original_checkpoint_sha256": CHECKPOINT_SHA256,
                          "target_runtime": runtime, "cuda_device_count": torch.cuda.device_count(),
                          "execution_requires_explicit_flag": "--execute"},
                         sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
