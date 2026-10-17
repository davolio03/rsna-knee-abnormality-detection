"""Prepare a disabled, output-free notebook for one private CPU preprocessing canary."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def build_canary_notebook(
    output_dir: Path,
    inventory_bytes: bytes,
    config: dict,
    *,
    runtime_helper_path: Path,
    manifest_helper_path: Path,
) -> dict[str, Path]:
    """Embed exact reviewed helper bytes and pins into a fresh notebook/config pair."""
    output_dir = Path(output_dir).resolve()
    if not output_dir.is_dir():
        raise ValueError("canary output directory must already exist")
    config_path = output_dir / "preprocessing-canary-run01.json"
    notebook_path = output_dir / "preprocessing-canary-run01.ipynb"
    if config_path.exists() or config_path.is_symlink() or notebook_path.exists() or notebook_path.is_symlink():
        raise ValueError("canary notebook and config outputs must be new")

    runtime_source = Path(runtime_helper_path).resolve(strict=True).read_bytes()
    manifest_source = Path(manifest_helper_path).resolve(strict=True).read_bytes()
    try:
        inventory = json.loads(inventory_bytes.decode("utf-8"))
        members = inventory["members"]
        if (inventory.get("schema_version") != "kaggle-training-package-inventory-v1"
                or len(members) != config.get("expected_member_count")
                or inventory.get("manifest_sha256") != config.get("expected_manifest_sha256")
                or inventory.get("folds_sha256") != config.get("expected_folds_sha256")
                or "data/private/500-study-training-cohort.json" not in members
                or "data/private/500-study-training-folds.json" not in members):
            raise ValueError
    except Exception:
        raise ValueError("canary package inventory does not match config pins") from None
    prepared_config = dict(config)
    prepared_config.update({
        "expected_inventory_sha256": _sha256(inventory_bytes),
        "expected_archive_provenance": inventory["package_sha256"],
    })
    config_raw = (json.dumps(prepared_config, indent=2, sort_keys=True) + "\n").encode("utf-8")
    config_b64 = base64.b64encode(config_raw).decode("ascii")
    inventory_b64 = base64.b64encode(inventory_bytes).decode("ascii")
    runtime_b64 = base64.b64encode(runtime_source).decode("ascii")
    manifest_b64 = base64.b64encode(manifest_source).decode("ascii")
    code = f'''# CPU preprocessing canary only. No training, labels, or model execution.
RUN_CANARY = False
if RUN_CANARY:
    import base64
    import hashlib
    import json
    import sys
    from pathlib import Path

    CANARY_CONFIG_B64 = "{config_b64}"
    CANARY_CONFIG_SHA256 = "{_sha256(config_raw)}"
    CANARY_INVENTORY_B64 = "{inventory_b64}"
    CANARY_HELPER_B64 = "{runtime_b64}"
    CANARY_HELPER_B64_SHA256 = "{_sha256(runtime_b64.encode('ascii'))}"
    MANIFEST_HELPER_B64 = "{manifest_b64}"
    MANIFEST_HELPER_B64_SHA256 = "{_sha256(manifest_b64.encode('ascii'))}"

    def _decode_pinned(name, encoded, expected_hash):
        if hashlib.sha256(encoded.encode("ascii")).hexdigest() != expected_hash:
            raise RuntimeError("canary_provenance_invalid") from None
        try:
            return base64.b64decode(encoded, validate=True)
        except Exception:
            raise RuntimeError("canary_provenance_invalid") from None

    try:
        config_raw = _decode_pinned("config", CANARY_CONFIG_B64, CANARY_CONFIG_SHA256)
        inventory_raw = base64.b64decode(CANARY_INVENTORY_B64, validate=True)
        config = json.loads(config_raw.decode("utf-8"))
        runtime_raw = _decode_pinned("runtime", CANARY_HELPER_B64, CANARY_HELPER_B64_SHA256)
        manifest_raw = _decode_pinned("manifest", MANIFEST_HELPER_B64, MANIFEST_HELPER_B64_SHA256)
        runtime_ns = {{"__name__": "kaggle_preprocessing_canary_runtime"}}
        manifest_ns = {{"__name__": "kaggle_orientation_manifest_parser"}}
        exec(compile(runtime_raw, "<pinned-canary-runtime>", "exec"), runtime_ns)
        exec(compile(manifest_raw, "<pinned-manifest-parser>", "exec"), manifest_ns)
        input_root = Path(config["private_input_root"])
        summary = runtime_ns["run_preprocessing_canary"](
            config, inventory_raw, manifest_ns["extract_manifest_series"],
            input_root=input_root,
            image_root=Path(config["image_root"]),
        )
        print(json.dumps(summary, sort_keys=True))
    except Exception as exc:
        reason = getattr(exc, "reason_code", None)
        if reason not in {{"package_inventory_invalid", "package_member_hash_mismatch",
                         "manifest_hash_mismatch", "folds_hash_mismatch", "manifest_invalid",
                         "folds_invalid", "fold_partition_mismatch", "target_series_unavailable",
                         "series_files_mismatch", "preprocessing_failed", "output_unavailable",
                         "runtime_limit_reached", "canary_disabled", "canary_provenance_invalid"}}:
            reason = "preprocessing_failed"
        print(json.dumps({{"status": "failed", "reason_code": reason}}, sort_keys=True))
        raise RuntimeError(reason) from None
'''
    notebook = {
        "cells": [{
            "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": code.splitlines(keepends=True),
        }],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    config_path.write_bytes(config_raw)
    notebook_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"config_path": config_path, "notebook_path": notebook_path}
