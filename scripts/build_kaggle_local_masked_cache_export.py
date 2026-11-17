"""Build a separate, bounded Kaggle notebook for private CPU cache export."""

from __future__ import annotations

import base64
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import zipfile


EXPECTED_SOURCE_SHA256 = "48a64d88c9b5cf72f4fdf6d9ce9839cc2cb3b8632b9fbf4fb7f4db1cdc4f95c8"
EXPECTED_CODEC_SHA256 = "2af0d5084dc70cb3daee2becf7a81aedb13093461aa64ceb532527c7701e3896"
EXPECTED_CODEC_MANIFEST_SHA256 = "50f39d3f627b3dc06a8080ba51dd6424898ef34fa4c4e21a84593dd326eb776d"
EXPECTED_PREPARED_INVENTORY_SHA256 = "0421de9f96c1b73806d07788728b74deaacdb13bd3e9231175772a9a829773b3"
SUCCESSFUL_DATASET_V4_NOTEBOOK_SHA256 = "b7291f13d8b8127cf005a47032f4d70923f52dbf01978f989b7b99c6fdc15d17"


def read_successful_prepared_inventory_pin(notebook_path: Path) -> str:
    notebook_path = Path(notebook_path)
    if notebook_path.is_symlink() or not notebook_path.is_file():
        raise ValueError("authority_notebook_unavailable")
    raw = notebook_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SUCCESSFUL_DATASET_V4_NOTEBOOK_SHA256:
        raise ValueError("authority_notebook_pin_mismatch")
    try:
        notebook = json.loads(raw.decode("utf-8"))
        code = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"]
                          if cell.get("cell_type") == "code")
        tree = ast.parse(code)
        values = [ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name)
                          and target.id == "EXPECTED_INVENTORY_SHA256" for target in node.targets)]
    except (UnicodeError, json.JSONDecodeError, KeyError, TypeError, SyntaxError, ValueError):
        raise ValueError("authority_inventory_pin_unavailable") from None
    if (len(values) != 1 or type(values[0]) is not str
            or not re.fullmatch(r"[0-9a-f]{64}", values[0])
            or values[0] != EXPECTED_PREPARED_INVENTORY_SHA256):
        raise ValueError("authority_inventory_pin_mismatch")
    return values[0]


def _notebook_source(source_b64: str, cache_source: str, prepared_inventory_sha256: str) -> str:
    return f'''from pathlib import Path
from io import BytesIO
import base64, hashlib, importlib, json, os, sys, time, zipfile

# Resolve only the already-attached pinned private DatasetV4 and competition
# mounts. This notebook decodes at most eight studies from each partition.
INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working")
STAGE = "plan"  # plan | smoke_export
RUN_BOUNDED_EXPORT = False
MAX_STUDIES_PER_PARTITION = 8
EXPECTED_SOURCE_SHA256 = "{EXPECTED_SOURCE_SHA256}"
EXPECTED_CODEC_SHA256 = "{EXPECTED_CODEC_SHA256}"
EXPECTED_CODEC_MANIFEST_SHA256 = "{EXPECTED_CODEC_MANIFEST_SHA256}"
EXPECTED_PREPARED_INVENTORY_SHA256 = "{prepared_inventory_sha256}"
SOURCE_ARCHIVE_B64 = {source_b64!r}
CACHE_MODULE_SOURCE = {cache_source!r}

def _fail(code):
    raise RuntimeError(code)

def _digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def _verify_prepared_asset_root(root):
    root = Path(root)
    marker = root / "masked-training-inventory.json"
    if root.is_symlink() or marker.is_symlink() or not root.is_dir() or not marker.is_file():
        _fail("prepared_input_missing_or_ambiguous")
    if _digest(marker) != EXPECTED_PREPARED_INVENTORY_SHA256:
        _fail("prepared_inventory_pin_mismatch")
    inventory = json.loads(marker.read_text(encoding="utf-8"))
    if inventory.get("schema_version") != "masked-soft-kaggle-package-v1":
        _fail("prepared_inventory_schema_invalid")
    expected = ("masked-training-source.zip.b64", "full-cohort-soft-label-manifest.json",
                "full-cohort-grouped-folds.json", "resnet18-f37072fd.pth", "offline-codecs.zip.b64")
    if any((root / name).is_symlink() or not (root / name).is_file() for name in expected):
        _fail("prepared_asset_missing")
    for name in ("full-cohort-soft-label-manifest.json", "full-cohort-grouped-folds.json"):
        pin = inventory.get("assets", {{}}).get(name, {{}})
        path = root / name
        if pin.get("sha256") != _digest(path) or pin.get("size") != path.stat().st_size:
            _fail("prepared_asset_pin_mismatch")
    return root, inventory

def find_prepared_asset_root(input_root):
    input_root = Path(input_root)
    if input_root.is_symlink() or not input_root.is_dir():
        _fail("prepared_input_missing_or_ambiguous")
    roots = []
    for child in input_root.iterdir():
        if child.is_dir() and not child.is_symlink() and (child / "masked-training-inventory.json").is_file():
            roots.append(child)
    datasets_root = input_root / "datasets"
    if datasets_root.is_dir() and not datasets_root.is_symlink():
        for owner in datasets_root.iterdir():
            if not owner.is_dir() or owner.is_symlink():
                continue
            for dataset in owner.iterdir():
                if (dataset.is_dir() and not dataset.is_symlink()
                        and (dataset / "masked-training-inventory.json").is_file()):
                    roots.append(dataset)
    distinct = {{item.resolve(strict=True) for item in roots}}
    if len(distinct) != 1:
        _fail("prepared_input_missing_or_ambiguous")
    return _verify_prepared_asset_root(next(iter(distinct)))[0]

def find_competition_root(input_root):
    input_root = Path(input_root)
    candidates = []
    for prefix in (input_root, input_root / "competitions"):
        for folder in ("train_series", "train_images"):
            path = prefix / "rsna-knee-abnormality-detection" / folder
            if path.is_dir() and not path.is_symlink():
                candidates.append(path.resolve(strict=True))
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        _fail("competition_series_input_missing")
    if len(candidates) != 1:
        _fail("competition_input_ambiguous")
    return candidates[0]

def _check_decoders(pydicom, UID):
    required = ("1.2.840.10008.1.2.4.57", "1.2.840.10008.1.2.4.70",
                "1.2.840.10008.1.2.4.80", "1.2.840.10008.1.2.4.81",
                "1.2.840.10008.1.2.4.90")
    for value in required:
        try:
            decoder = pydicom.pixels.get_decoder(UID(value))
        except NotImplementedError:
            _fail("required_dicom_decoder_unavailable")
        if getattr(decoder, "is_available", False) is not True:
            _fail("required_dicom_decoder_unavailable")

def run_export(*, input_root=None, working_root=None, stage=None,
               run_bounded_export=None, emit=print):
    import torch
    stage = STAGE if stage is None else stage
    run_bounded_export = RUN_BOUNDED_EXPORT if run_bounded_export is None else run_bounded_export
    if stage not in {{"plan", "smoke_export"}}:
        _fail("stage_invalid")
    if stage == "plan" and run_bounded_export is not False:
        _fail("plan_must_not_export")
    if stage == "smoke_export" and run_bounded_export is not True:
        _fail("bounded_export_not_explicitly_enabled")
    if torch.cuda.is_initialized():
        _fail("cuda_must_remain_uninitialized")
    torch.set_num_threads(2)
    input_root = Path(INPUT_ROOT if input_root is None else input_root)
    assets = find_prepared_asset_root(input_root)
    image_root = find_competition_root(input_root)
    manifest_path = assets / "full-cohort-soft-label-manifest.json"
    folds_path = assets / "full-cohort-grouped-folds.json"
    working_root = Path(WORKING_ROOT if working_root is None else working_root)
    if working_root.is_symlink() or not working_root.is_dir():
        _fail("working_root_invalid")
    source_bytes = base64.b64decode(SOURCE_ARCHIVE_B64, validate=True)
    if hashlib.sha256(source_bytes).hexdigest() != EXPECTED_SOURCE_SHA256:
        _fail("source_pin_mismatch")
    source_path = working_root / "local-cache-export-source.zip"
    if source_path.exists() or source_path.is_symlink():
        _fail("source_workfile_not_fresh")
    source_path.write_bytes(source_bytes)
    with zipfile.ZipFile(BytesIO(source_bytes)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or not names or any(
                name.startswith("/") or ".." in Path(name).parts or not name.endswith(".py")
                for name in names):
            _fail("source_archive_invalid")
    sys.path.insert(0, str(source_path))
    source_manifest = importlib.import_module("rsna_knee.data.masked_soft_dataset")
    dataset_type = source_manifest.FullCohortSoftDataset
    datasets = {{}}
    for partition in ("train", "validation"):
        dataset = dataset_type(manifest_path, folds_path, image_root,
                               partition=partition, fold=0, output_size=224, max_slices=8)
        expected_count = 2850 if partition == "train" else 1448
        if len(dataset) != expected_count:
            _fail("fold_partition_count_mismatch")
        datasets[partition] = dataset
    plan = {{"status": "planned", "training_study_count": len(datasets["train"]),
             "validation_study_count": len(datasets["validation"]),
             "stage": stage, "cuda_initialized": torch.cuda.is_initialized(),
             "model_constructed": False}}
    if stage == "plan":
        emit(json.dumps(plan, sort_keys=True))
        return {{"plan": plan}}

    codec_bundle_path = assets / "offline-codecs.zip.b64"
    codec_bytes = codec_bundle_path.read_bytes()
    if hashlib.sha256(codec_bytes).hexdigest() != EXPECTED_CODEC_SHA256:
        try:
            codec_bytes = base64.b64decode(codec_bytes, validate=True)
        except Exception:
            _fail("codec_asset_pin_mismatch")
    if hashlib.sha256(codec_bytes).hexdigest() != EXPECTED_CODEC_SHA256:
        _fail("codec_asset_pin_mismatch")
    codec_zip = working_root / "local-cache-export-codecs.zip"
    if codec_zip.exists() or codec_zip.is_symlink():
        _fail("codec_workfile_not_fresh")
    codec_zip.write_bytes(codec_bytes)
    codec_bootstrap = importlib.import_module("rsna_knee.inference.offline_codec_bootstrap")
    codec_target = working_root / "local-cache-export-codecs"
    if codec_target.exists() or codec_target.is_symlink():
        _fail("codec_target_not_fresh")
    install = codec_bootstrap.install_codec_bundle(
        codec_zip, EXPECTED_CODEC_MANIFEST_SHA256, codec_target, working_root=working_root)
    installed = Path(install["target"]).resolve(strict=True)
    if installed.is_symlink() or not installed.is_dir():
        _fail("codec_install_failed")
    sys.path.insert(0, str(installed))
    pydicom = importlib.import_module("pydicom")
    uid_module = importlib.import_module("pydicom.uid")
    _check_decoders(pydicom, uid_module.UID)

    output_root = working_root / "local-mask-cache-smoke"
    if output_root.exists() or output_root.is_symlink():
        _fail("cache_output_not_fresh")
    output_root.mkdir()
    started = time.perf_counter()
    summaries = []
    cache_ns = {{"__name__": "local_masked_tensor_cache"}}
    exec(compile(CACHE_MODULE_SOURCE, "local_masked_tensor_cache.py", "exec"), cache_ns)
    for partition in ("train", "validation"):
        dataset = datasets[partition]
        count = min(MAX_STUDIES_PER_PARTITION, len(dataset))
        if count != MAX_STUDIES_PER_PARTITION:
            _fail("bounded_partition_too_small")
        cache = cache_ns["LocalMaskedTensorCache"](
            output_root / partition, partition=partition, fold=0,
            manifest_sha256=dataset.manifest_sha256)
        for index in range(count):
            record = dataset._records[index]
            sample = dataset[index]
            cache.write_sample(study_id=record["study_id"],
                               series_ids=[entry["series_id"] for entry in record["series"]],
                               sample=sample)
            emit(json.dumps({{"event": "cache_export_progress", "partition": partition,
                             "completed": index + 1, "total": count}}, sort_keys=True))
        cache.finalize()
        summaries.append({{"partition": partition, "study_count": count}})

    archive_path = working_root / "local-mask-cache-smoke.zip"
    archive = cache_ns["write_cache_archive"](output_root, archive_path)
    total_bytes = sum(path.stat().st_size for path in output_root.rglob("*") if path.is_file())
    result = {{"status": "complete", "partition_counts": summaries,
              "cache_bytes": total_bytes, "archive_bytes": archive_path.stat().st_size,
              "archive_sha256": archive["sha256"],
              "elapsed_seconds": round(time.perf_counter() - started, 3)}}
    emit(json.dumps(result, sort_keys=True))
    return result

if __name__ == "__main__":
    try:
        run_export(stage=STAGE, run_bounded_export=RUN_BOUNDED_EXPORT)
    except KeyboardInterrupt:
        raise
    except Exception:
        print(json.dumps({{"status": "failed", "reason_code": "cache_export_failed"}}, sort_keys=True))
        raise
'''


def build_cache_export_notebook(source_archive: Path, cache_module: Path, output_path: Path, *,
                                prepared_inventory_sha256: str,
                                authorized_smoke_export: bool = False) -> dict[str, str]:
    if type(authorized_smoke_export) is not bool:
        raise ValueError("authorized_smoke_flag_invalid")
    root = Path(__file__).resolve().parents[1]
    authority_notebook = root / "data/private/resnet50-pretraining-preparation/run02/kaggle_resnet50_radimagenet_full_epoch01.ipynb"
    canonical_inventory_pin = read_successful_prepared_inventory_pin(authority_notebook)
    if prepared_inventory_sha256 != canonical_inventory_pin:
        raise ValueError("prepared_inventory_pin_mismatch")
    source_archive = Path(source_archive)
    cache_module = Path(cache_module)
    output_path = Path(output_path)
    try:
        source_bytes = source_archive.read_bytes()
        cache_source = cache_module.read_text(encoding="utf-8")
    except OSError:
        raise ValueError("cache_export_input_unavailable") from None
    if hashlib.sha256(source_bytes).hexdigest() != EXPECTED_SOURCE_SHA256:
        raise ValueError("source_pin_mismatch")
    try:
        with zipfile.ZipFile(source_archive) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or not names or any(
                    name.startswith("/") or ".." in Path(name).parts or not name.endswith(".py")
                    for name in names):
                raise ValueError
    except (OSError, zipfile.BadZipFile, ValueError):
        raise ValueError("source_archive_invalid") from None
    compile(cache_source, "local_masked_tensor_cache.py", "exec")
    code = _notebook_source(base64.b64encode(source_bytes).decode("ascii"), cache_source,
                            prepared_inventory_sha256)
    if authorized_smoke_export:
        plan_flag = 'STAGE = "plan"  # plan | smoke_export'
        export_flag = "RUN_BOUNDED_EXPORT = False"
        if code.count(plan_flag) != 1 or code.count(export_flag) != 1:
            raise ValueError("smoke_export_opt_in_anchor_invalid")
        code = code.replace(plan_flag, 'STAGE = "smoke_export"  # plan | smoke_export', 1)
        code = code.replace(export_flag, "RUN_BOUNDED_EXPORT = True", 1)
    compile(code, "kaggle_local_masked_cache_export.py", "exec")
    notebook = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {},
                   "outputs": [], "source": code.splitlines(keepends=True)}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.13.0"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    raw = json.dumps(notebook, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    output_path = output_path.resolve()
    if output_path.exists() or output_path.is_symlink():
        raise ValueError("notebook_output_not_fresh")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(raw)
    if output_path.read_bytes() != raw:
        raise ValueError("notebook_readback_failed")
    return {"notebook_path": str(output_path), "notebook_sha256": hashlib.sha256(raw).hexdigest()}


def _full_chunk_notebook_source(source_b64: str, cache_source: str, plan_module_source: str,
                               plan_json: str, chunk_id: str, inventory_pin: str) -> str:
    return f'''from pathlib import Path
from io import BytesIO
import base64, hashlib, importlib, json, os, sys, time, zipfile

# Metadata-only by default. This file is scoped to exactly one authorized chunk.
INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working")
STAGE = "plan"  # plan | chunk_export
RUN_CHUNK_EXPORT = False
CHUNK_ID = {chunk_id!r}
EXPECTED_SOURCE_SHA256 = "{EXPECTED_SOURCE_SHA256}"
EXPECTED_CODEC_SHA256 = "{EXPECTED_CODEC_SHA256}"
EXPECTED_CODEC_MANIFEST_SHA256 = "{EXPECTED_CODEC_MANIFEST_SHA256}"
EXPECTED_PREPARED_INVENTORY_SHA256 = "{inventory_pin}"
SOURCE_ARCHIVE_B64 = {source_b64!r}
CHUNK_PLAN_JSON = {plan_json!r}
CACHE_MODULE_SOURCE = {cache_source!r}
CHUNK_PLAN_MODULE_SOURCE = {plan_module_source!r}

def fail(code):
    raise RuntimeError(code)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def _verify_prepared_asset_root(root):
    root = Path(root)
    marker = root / "masked-training-inventory.json"
    if root.is_symlink() or marker.is_symlink() or not root.is_dir() or not marker.is_file():
        fail("prepared_input_missing_or_ambiguous")
    if digest(marker) != EXPECTED_PREPARED_INVENTORY_SHA256:
        fail("prepared_inventory_pin_mismatch")
    inventory = json.loads(marker.read_text(encoding="utf-8"))
    if inventory.get("schema_version") != "masked-soft-kaggle-package-v1":
        fail("prepared_inventory_schema_invalid")
    expected = ("masked-training-source.zip.b64", "full-cohort-soft-label-manifest.json",
                "full-cohort-grouped-folds.json", "resnet18-f37072fd.pth", "offline-codecs.zip.b64")
    if any((root / name).is_symlink() or not (root / name).is_file() for name in expected):
        fail("prepared_asset_missing")
    for name in ("full-cohort-soft-label-manifest.json", "full-cohort-grouped-folds.json"):
        pin = inventory.get("assets", {{}}).get(name, {{}})
        path = root / name
        if pin.get("sha256") != digest(path) or pin.get("size") != path.stat().st_size:
            fail("prepared_asset_pin_mismatch")
    return root, inventory

def find_prepared_asset_root(input_root):
    input_root = Path(input_root)
    if input_root.is_symlink() or not input_root.is_dir():
        fail("prepared_input_missing_or_ambiguous")
    roots = []
    for child in input_root.iterdir():
        if child.is_dir() and not child.is_symlink() and (child / "masked-training-inventory.json").is_file():
            roots.append(child)
    datasets_root = input_root / "datasets"
    if datasets_root.is_dir() and not datasets_root.is_symlink():
        for owner in datasets_root.iterdir():
            if not owner.is_dir() or owner.is_symlink():
                continue
            for dataset in owner.iterdir():
                if (dataset.is_dir() and not dataset.is_symlink()
                        and (dataset / "masked-training-inventory.json").is_file()):
                    roots.append(dataset)
    distinct = {{item.resolve(strict=True) for item in roots}}
    if len(distinct) != 1:
        fail("prepared_input_missing_or_ambiguous")
    return _verify_prepared_asset_root(next(iter(distinct)))[0]

def find_competition_root(input_root):
    input_root = Path(input_root)
    candidates = []
    for prefix in (input_root, input_root / "competitions"):
        for folder in ("train_series", "train_images"):
            path = prefix / "rsna-knee-abnormality-detection" / folder
            if path.is_dir() and not path.is_symlink():
                candidates.append(path.resolve(strict=True))
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        fail("competition_series_input_missing")
    if len(candidates) != 1:
        fail("competition_input_ambiguous")
    return candidates[0]

def _check_decoders(pydicom, UID):
    required = ("1.2.840.10008.1.2.4.57", "1.2.840.10008.1.2.4.70",
                "1.2.840.10008.1.2.4.80", "1.2.840.10008.1.2.4.81",
                "1.2.840.10008.1.2.4.90")
    for value in required:
        try:
            decoder = pydicom.pixels.get_decoder(UID(value))
        except NotImplementedError:
            fail("required_dicom_decoder_unavailable")
        if getattr(decoder, "is_available", False) is not True:
            fail("required_dicom_decoder_unavailable")

def run_chunk(*, input_root=None, working_root=None, stage=None,
              run_chunk_export=None, emit=print):
    import torch
    stage = STAGE if stage is None else stage
    run_chunk_export = RUN_CHUNK_EXPORT if run_chunk_export is None else run_chunk_export
    if stage not in {{"plan", "chunk_export"}}:
        fail("stage_invalid")
    if stage == "plan" and run_chunk_export is not False:
        fail("plan_must_not_export")
    if stage == "chunk_export" and run_chunk_export is not True:
        fail("chunk_export_not_explicitly_enabled")
    if torch.cuda.is_initialized():
        fail("cuda_must_remain_uninitialized")
    torch.set_num_threads(2)
    input_root = Path(INPUT_ROOT if input_root is None else input_root)
    working_root = Path(WORKING_ROOT if working_root is None else working_root)
    if working_root.is_symlink() or not working_root.is_dir():
        fail("working_root_invalid")
    assets = find_prepared_asset_root(input_root)
    image_root = find_competition_root(input_root)
    manifest_path = assets / "full-cohort-soft-label-manifest.json"
    folds_path = assets / "full-cohort-grouped-folds.json"
    if digest(manifest_path) != json.loads(CHUNK_PLAN_JSON)["manifest_sha256"]:
        fail("manifest_pin_mismatch")
    if digest(folds_path) != json.loads(CHUNK_PLAN_JSON)["folds_sha256"]:
        fail("folds_pin_mismatch")
    plan_namespace = {{"__name__": "local_masked_cache_chunk_plan"}}
    exec(compile(CHUNK_PLAN_MODULE_SOURCE, "local_masked_cache_chunk_plan.py", "exec"), plan_namespace)
    plan = json.loads(CHUNK_PLAN_JSON)
    if not plan_namespace["validate_chunk_plan"](plan):
        fail("chunk_plan_invalid")
    chunk = next((item for item in plan["chunks"] if item["chunk_id"] == CHUNK_ID), None)
    if chunk is None or chunk["partition"] not in {{"train", "validation"}}:
        fail("chunk_id_invalid")
    if chunk["max_saved_output_bytes"] >= plan["output_limit_bytes"]:
        fail("chunk_saved_output_limit_exceeded")

    source_bytes = base64.b64decode(SOURCE_ARCHIVE_B64, validate=True)
    if hashlib.sha256(source_bytes).hexdigest() != EXPECTED_SOURCE_SHA256:
        fail("source_pin_mismatch")
    source_path = working_root / "local-cache-export-source.zip"
    if source_path.exists() or source_path.is_symlink():
        fail("source_workfile_not_fresh")
    source_path.write_bytes(source_bytes)
    with zipfile.ZipFile(BytesIO(source_bytes)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or not names or any(
                name.startswith("/") or ".." in Path(name).parts or not name.endswith(".py")
                for name in names):
            fail("source_archive_invalid")
    sys.path.insert(0, str(source_path))
    source_manifest = importlib.import_module("rsna_knee.data.masked_soft_dataset")
    dataset = source_manifest.FullCohortSoftDataset(
        manifest_path, folds_path, image_root, partition=chunk["partition"], fold=0,
        output_size=plan["preprocessing"]["output_size"],
        max_slices=plan["preprocessing"]["max_slices_per_series"])
    if len(dataset) != plan["partition_counts"][chunk["partition"]]:
        fail("partition_count_mismatch")
    full_digest = plan_namespace["_selection_digest"](chunk["partition"], dataset._records)
    start, stop = chunk["start_ordinal"], chunk["stop_ordinal_exclusive"]
    selected_records = dataset._records[start:stop]
    chunk_digest = plan_namespace["_selection_digest"](chunk["partition"], selected_records)
    if (full_digest != chunk["full_selection_sha256"]
            or full_digest != plan["partition_selection_sha256"][chunk["partition"]]
            or chunk_digest != chunk["chunk_selection_sha256"]
            or len(selected_records) != chunk["study_count"]
            or sum(len(record["series"]) for record in selected_records) != chunk["series_count"]):
        fail("chunk_selection_mismatch")
    plan_result = {{"status": "planned", "chunk_id": CHUNK_ID,
                   "partition": chunk["partition"], "study_count": chunk["study_count"],
                   "series_count": chunk["series_count"], "start_ordinal": start,
                   "stop_ordinal_exclusive": stop, "cuda_initialized": torch.cuda.is_initialized(),
                   "model_constructed": False}}
    if stage == "plan":
        emit(json.dumps(plan_result, sort_keys=True))
        return {{"plan": plan_result}}

    codec_bytes_b64 = (assets / "offline-codecs.zip.b64").read_bytes()
    try:
        codec_bytes = base64.b64decode(codec_bytes_b64, validate=True)
    except Exception:
        fail("codec_asset_encoding_invalid")
    if hashlib.sha256(codec_bytes).hexdigest() != EXPECTED_CODEC_SHA256:
        fail("codec_asset_pin_mismatch")
    codec_zip = working_root / "local-cache-export-codecs.zip"
    if codec_zip.exists() or codec_zip.is_symlink():
        fail("codec_workfile_not_fresh")
    codec_zip.write_bytes(codec_bytes)
    codec_bootstrap = importlib.import_module("rsna_knee.inference.offline_codec_bootstrap")
    codec_target = working_root / "local-cache-export-codecs"
    if codec_target.exists() or codec_target.is_symlink():
        fail("codec_target_not_fresh")
    installed = codec_bootstrap.install_codec_bundle(
        codec_zip, EXPECTED_CODEC_MANIFEST_SHA256, codec_target, working_root=working_root)
    installed_target = Path(installed["target"]).resolve(strict=True)
    if installed_target.is_symlink() or not installed_target.is_dir():
        fail("codec_install_failed")
    sys.path.insert(0, str(installed_target))
    pydicom = importlib.import_module("pydicom")
    uid_module = importlib.import_module("pydicom.uid")
    _check_decoders(pydicom, uid_module.UID)

    output_root = working_root / ("local-cache-" + CHUNK_ID)
    if output_root.exists() or output_root.is_symlink():
        fail("cache_output_not_fresh")
    started = time.perf_counter()
    provenance = {{
        "chunk_id": chunk["chunk_id"], "partition": chunk["partition"], "fold": 0,
        "start_ordinal": start, "stop_ordinal_exclusive": stop,
        "study_count": chunk["study_count"], "series_count": chunk["series_count"],
        "full_selection_sha256": chunk["full_selection_sha256"],
        "chunk_selection_sha256": chunk["chunk_selection_sha256"],
        "manifest_sha256": plan["manifest_sha256"], "folds_sha256": plan["folds_sha256"],
        "source_sha256": plan["source_sha256"],
        "preprocessing_source_sha256": plan["preprocessing_source_sha256"],
        "preprocessing": plan["preprocessing"], "gold_included": False,
    }}
    cache_namespace = {{"__name__": "local_masked_tensor_cache"}}
    exec(compile(CACHE_MODULE_SOURCE, "local_masked_tensor_cache.py", "exec"), cache_namespace)
    cache = cache_namespace["LocalMaskedTensorCache"](
        output_root, partition=chunk["partition"], fold=0,
        manifest_sha256=plan["manifest_sha256"], first_ordinal=start,
        provenance=provenance)
    for offset, ordinal in enumerate(range(start, stop)):
        record = dataset._records[ordinal]
        sample = dataset[ordinal]
        cache.write_sample(study_id=record["study_id"],
                           series_ids=[item["series_id"] for item in record["series"]],
                           sample=sample)
        if (offset + 1) % 25 == 0 or offset + 1 == chunk["study_count"]:
            emit(json.dumps({{"event": "chunk_export_progress", "chunk_id": CHUNK_ID,
                             "completed_studies": offset + 1,
                             "total_studies": chunk["study_count"]}}, sort_keys=True))
    cache.finalize()
    cache_bytes = sum(path.stat().st_size for path in output_root.rglob("*") if path.is_file())
    other_output_bytes = 0
    for path in working_root.rglob("*"):
        if path.is_symlink():
            fail("working_output_symlink_invalid")
        if path.is_file() and not path.is_relative_to(output_root):
            other_output_bytes += path.stat().st_size
    reserve_with_other_outputs = plan["output_reserve_bytes"] + other_output_bytes
    if 2 * cache_bytes + reserve_with_other_outputs >= plan["output_limit_bytes"]:
        fail("chunk_actual_output_limit_exceeded")
    archive_path = working_root / ("local-cache-" + CHUNK_ID + ".zip")
    archive = cache_namespace["write_cache_archive"](
        output_root, archive_path, max_output_bytes=plan["output_limit_bytes"],
        reserve_bytes=reserve_with_other_outputs)
    actual_saved_bytes = cache_bytes + archive["size_bytes"]
    if actual_saved_bytes + reserve_with_other_outputs > plan["output_limit_bytes"]:
        fail("chunk_actual_output_limit_exceeded")
    result = {{"status": "complete", "chunk_id": CHUNK_ID,
              "partition": chunk["partition"], "study_count": chunk["study_count"],
              "series_count": chunk["series_count"], "cache_bytes": cache_bytes,
              "archive_bytes": archive["size_bytes"], "archive_sha256": archive["sha256"],
              "saved_bytes_with_reserve": actual_saved_bytes + reserve_with_other_outputs,
              "elapsed_seconds": round(time.perf_counter() - started, 3)}}
    emit(json.dumps(result, sort_keys=True))
    return {{"export": result}}

if __name__ == "__main__":
    try:
        run_chunk(stage=STAGE, run_chunk_export=RUN_CHUNK_EXPORT)
    except KeyboardInterrupt:
        raise
    except Exception:
        print(json.dumps({{"status": "failed", "reason_code": "cache_chunk_export_failed"}}, sort_keys=True))
        raise SystemExit(2) from None
'''


def build_full_chunk_export_notebooks(source_archive: Path, cache_module: Path,
                                      plan_module: Path, plan_path: Path,
                                      output_dir: Path, *, authorized: bool = True) -> list[dict[str, str]]:
    if type(authorized) is not bool:
        raise ValueError("authorized_flag_invalid")
    source_archive, cache_module, plan_module, plan_path, output_dir = map(
        Path, (source_archive, cache_module, plan_module, plan_path, output_dir))
    try:
        source_bytes = source_archive.read_bytes()
        cache_source = cache_module.read_text(encoding="utf-8")
        plan_source = plan_module.read_text(encoding="utf-8")
        plan_bytes = plan_path.read_bytes()
    except OSError:
        raise ValueError("chunk_export_input_unavailable") from None
    if hashlib.sha256(source_bytes).hexdigest() != EXPECTED_SOURCE_SHA256:
        raise ValueError("source_pin_mismatch")
    try:
        plan = json.loads(plan_bytes.decode("utf-8"))
        plan_ns = {"__name__": "local_masked_cache_chunk_plan"}
        exec(compile(plan_source, "local_masked_cache_chunk_plan.py", "exec"), plan_ns)
        if not plan_ns["validate_chunk_plan"](plan):
            raise ValueError
    except (UnicodeError, json.JSONDecodeError, SyntaxError, KeyError, ValueError):
        raise ValueError("chunk_plan_invalid") from None
    if (plan.get("manifest_sha256") != plan_ns["EXPECTED_MANIFEST_SHA256"]
            or plan.get("folds_sha256") != plan_ns["EXPECTED_FOLDS_SHA256"]
            or plan.get("source_sha256") != EXPECTED_SOURCE_SHA256
            or plan.get("study_count") != plan_ns["EXPECTED_STUDY_COUNT"]
            or plan.get("series_count") != plan_ns["EXPECTED_SERIES_COUNT"]):
        raise ValueError("chunk_plan_source_pin_mismatch")
    inventory_pin = read_successful_prepared_inventory_pin(
        Path(__file__).resolve().parents[1]
        / "data/private/resnet50-pretraining-preparation/run02/kaggle_resnet50_radimagenet_full_epoch01.ipynb")
    for source, name in ((cache_source, "local_masked_tensor_cache.py"),
                         (plan_source, "local_masked_cache_chunk_plan.py")):
        compile(source, name, "exec")
    plan_json = json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False)
    plan_module_json = json.dumps(plan_source)
    cache_module_json = json.dumps(cache_source)
    source_b64 = base64.b64encode(source_bytes).decode("ascii")
    notebooks = []
    for chunk in plan["chunks"]:
        chunk_id = chunk["chunk_id"]
        code = _full_chunk_notebook_source(source_b64, cache_source, plan_source,
                                           plan_json, chunk_id, inventory_pin)
        if authorized:
            code = code.replace('STAGE = "plan"  # plan | chunk_export',
                                'STAGE = "chunk_export"  # plan | chunk_export', 1)
            code = code.replace("RUN_CHUNK_EXPORT = False", "RUN_CHUNK_EXPORT = True", 1)
        compile(code, "kaggle_local_masked_cache_chunk.py", "exec")
        notebook = {
            "cells": [{"cell_type": "code", "execution_count": None, "metadata": {},
                       "outputs": [], "source": code.splitlines(keepends=True)}],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.13.0"}},
            "nbformat": 4, "nbformat_minor": 5,
        }
        raw = json.dumps(notebook, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        output = output_dir / f"cache-export-{chunk_id}-{'authorized' if authorized else 'plan'}.ipynb"
        notebooks.append({"path": str(output), "raw": raw,
                          "sha256": hashlib.sha256(raw).hexdigest(), "chunk_id": chunk_id})
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(Path(item["path"]).exists() or Path(item["path"]).is_symlink() for item in notebooks):
        raise ValueError("chunk_notebook_output_not_fresh")
    results = []
    for item in notebooks:
        output = Path(item["path"])
        temporary = output.with_name(output.name + ".tmp")
        if temporary.exists() or temporary.is_symlink():
            raise ValueError("chunk_notebook_temporary_not_fresh")
        try:
            with temporary.open("xb") as stream:
                stream.write(item["raw"])
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, output)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        if output.read_bytes() != item["raw"]:
            raise ValueError("chunk_notebook_readback_failed")
        results.append({"chunk_id": item["chunk_id"], "notebook_path": str(output),
                        "notebook_sha256": item["sha256"]})
    return results


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    result = build_cache_export_notebook(
        root / "data/private/resnet50-pretraining-preparation/run01/resnet50-recipe-source.zip",
        root / "scripts/local_masked_tensor_cache.py",
        root / "data/private/local-radimagenet-epoch02/cache-export-smoke-run02.ipynb",
        prepared_inventory_sha256=read_successful_prepared_inventory_pin(
            root / "data/private/resnet50-pretraining-preparation/run02/kaggle_resnet50_radimagenet_full_epoch01.ipynb"),
    )
    print(json.dumps(result, sort_keys=True))
