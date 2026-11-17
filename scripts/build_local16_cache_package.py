"""Build a private, pinned plan/canary/full Kaggle package for E050 local16 cache."""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import argparse
import sys
import zipfile

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.local16_cache_chunk_plan import create_chunk_plan

PREPARED_INVENTORY_SHA256 = "0421de9f96c1b73806d07788728b74deaacdb13bd3e9231175772a9a829773b3"
CODEC_SHA256 = "2af0d5084dc70cb3daee2becf7a81aedb13093461aa64ceb532527c7701e3896"
CODEC_MANIFEST_SHA256 = "50f39d3f627b3dc06a8080ba51dd6424898ef34fa4c4e21a84593dd326eb776d"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _notebook_source(source_b64: str, source_sha: str, plan_json: str, chunk_id: str,
                     stage: str, max_studies: int | None) -> str:
    return f'''# E050 local16 CPU cache package; no model construction or training.
import base64, hashlib, importlib, json, os, sys, time, zipfile
from pathlib import Path

EXPECTED_SOURCE_SHA256 = {source_sha!r}
EXPECTED_PREPARED_INVENTORY_SHA256 = {PREPARED_INVENTORY_SHA256!r}
EXPECTED_CODEC_SHA256 = {CODEC_SHA256!r}
EXPECTED_CODEC_MANIFEST_SHA256 = {CODEC_MANIFEST_SHA256!r}
SOURCE_ARCHIVE_B64 = {source_b64!r}
PLAN = json.loads({plan_json!r})
CHUNK_ID = {chunk_id!r}
STAGE = {stage!r}  # plan | canary_export | chunk_export
MAX_STUDIES = {max_studies!r}
INPUT_ROOT = Path('/kaggle/input')
WORKING_ROOT = Path('/kaggle/working')

def fail(code):
    raise RuntimeError(code)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def find_asset_root(root):
    found = []
    for current, dirs, files in os.walk(root):
        dirs[:] = [name for name in dirs if not (Path(current) / name).is_symlink()]
        if 'masked-training-inventory.json' in files:
            found.append(Path(current))
    found = list(dict.fromkeys(found))
    if len(found) != 1:
        fail('prepared_input_missing_or_ambiguous')
    asset = found[0]
    marker = asset / 'masked-training-inventory.json'
    if digest(marker) != EXPECTED_PREPARED_INVENTORY_SHA256:
        fail('prepared_inventory_pin_mismatch')
    expected = ('full-cohort-soft-label-manifest.json', 'full-cohort-grouped-folds.json', 'offline-codecs.zip.b64')
    if any((asset / name).is_symlink() or not (asset / name).is_file() for name in expected):
        fail('prepared_asset_missing')
    return asset

def find_series_root(root):
    found = []
    for prefix in (root, root / 'competitions'):
        path = prefix / 'rsna-knee-abnormality-detection' / 'train_series'
        if path.is_dir() and not path.is_symlink():
            found.append(path.resolve())
    if len(set(found)) != 1:
        fail('competition_series_input_missing_or_ambiguous')
    return found[0]

def run(dataset_factory=None, *, input_root=None, working_root=None,
        codec_installer=None, decoder_probe=None):
    import torch
    input_root = INPUT_ROOT if input_root is None else Path(input_root)
    working_root = WORKING_ROOT if working_root is None else Path(working_root)
    if STAGE not in ('plan', 'canary_export', 'chunk_export'):
        fail('stage_invalid')
    if STAGE == 'plan' and MAX_STUDIES is not None:
        fail('plan_stage_invalid')
    if STAGE == 'canary_export' and MAX_STUDIES != 8:
        fail('canary_bound_invalid')
    if STAGE == 'chunk_export' and MAX_STUDIES is not None:
        fail('full_chunk_bound_invalid')
    if torch.cuda.is_initialized():
        fail('cuda_must_remain_uninitialized')
    torch.set_num_threads(2)
    chunk = next((row for row in PLAN['chunks'] if row['chunk_id'] == CHUNK_ID), None)
    if chunk is None:
        fail('chunk_id_invalid')
    source = base64.b64decode(SOURCE_ARCHIVE_B64, validate=True)
    if hashlib.sha256(source).hexdigest() != EXPECTED_SOURCE_SHA256:
        fail('source_pin_mismatch')
    source_path = working_root / ('local16-source-' + CHUNK_ID + '-' + STAGE + '.zip')
    if source_path.exists() or source_path.is_symlink():
        fail('source_workfile_not_fresh')
    source_path.write_bytes(source)
    with zipfile.ZipFile(source_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or not names or any(name.startswith('/') or '..' in Path(name).parts for name in names):
            fail('source_archive_invalid')
    sys.path.insert(0, str(source_path))
    from scripts.local16_cache_chunk_plan import validate_chunk_plan, _selection_digest
    from scripts.local16_tensor_cache import Local16TensorCache
    if not validate_chunk_plan(PLAN):
        fail('chunk_plan_invalid')
    assets = find_asset_root(input_root)
    manifest_path = assets / 'full-cohort-soft-label-manifest.json'
    folds_path = assets / 'full-cohort-grouped-folds.json'
    if digest(manifest_path) != PLAN['manifest_sha256'] or digest(folds_path) != PLAN['folds_sha256']:
        fail('cohort_asset_pin_mismatch')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    folds = json.loads(folds_path.read_text(encoding='utf-8'))
    if hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\\n').hexdigest() != folds.get('manifest_sha256'):
        fail('fold_manifest_binding_mismatch')
    image_root = find_series_root(input_root)
    from rsna_knee.data.masked_soft_dataset import FullCohortSoftDataset
    dataset_factory = FullCohortSoftDataset if dataset_factory is None else dataset_factory
    dataset = dataset_factory(manifest_path, folds_path, image_root,
                              partition=chunk['partition'], fold=0,
                              output_size=224, max_slices=16)
    selection = _selection_digest(chunk['partition'], dataset._records)
    if selection != PLAN['partition_selection_sha256'][chunk['partition']]:
        fail('full_partition_selection_mismatch')
    start, stop = chunk['start_ordinal'], chunk['stop_ordinal_exclusive']
    selected = dataset._records[start:stop]
    if _selection_digest(chunk['partition'], selected) != chunk['chunk_selection_sha256']:
        fail('chunk_selection_mismatch')
    end = min(stop, start + MAX_STUDIES) if MAX_STUDIES is not None else stop
    summary = {{'status':'planned', 'chunk_id':CHUNK_ID, 'partition':chunk['partition'],
                'study_count':end-start, 'series_count':sum(len(r['series']) for r in selected[:end-start]),
                'max_slices_per_series':16, 'cuda_initialized':False, 'training_started':False}}
    if STAGE == 'plan':
        print(json.dumps(summary, sort_keys=True))
        return
    codec_raw = base64.b64decode((assets / 'offline-codecs.zip.b64').read_bytes(), validate=True)
    if hashlib.sha256(codec_raw).hexdigest() != EXPECTED_CODEC_SHA256:
        fail('codec_asset_pin_mismatch')
    codec_zip = working_root / ('local16-codecs-' + CHUNK_ID + '-' + STAGE + '.zip')
    if codec_zip.exists() or codec_zip.is_symlink():
        fail('codec_workfile_not_fresh')
    codec_zip.write_bytes(codec_raw)
    codec_root = working_root / ('local16-codecs-' + CHUNK_ID + '-' + STAGE)
    if codec_root.exists() or codec_root.is_symlink():
        fail('codec_target_not_fresh')
    if codec_installer is None:
        bootstrap = importlib.import_module('rsna_knee.inference.offline_codec_bootstrap')
        installed = bootstrap.install_codec_bundle(codec_zip, EXPECTED_CODEC_MANIFEST_SHA256,
                                                   codec_root, working_root=working_root)
        sys.path.insert(0, str(Path(installed['target']).resolve(strict=True)))
    else:
        installed = codec_installer(codec_zip, EXPECTED_CODEC_MANIFEST_SHA256,
                                    codec_root, working_root=working_root)
    if decoder_probe is None:
        pydicom = importlib.import_module('pydicom')
        UID = importlib.import_module('pydicom.uid').UID
        for uid in ('1.2.840.10008.1.2.4.57', '1.2.840.10008.1.2.4.70',
                    '1.2.840.10008.1.2.4.80', '1.2.840.10008.1.2.4.81', '1.2.840.10008.1.2.4.90'):
            if not getattr(pydicom.pixels.get_decoder(UID(uid)), 'is_available', False):
                fail('required_dicom_decoder_unavailable')
    else:
        decoder_probe()
    output = working_root / ('local16-cache-canary-' + chunk['partition'] if STAGE == 'canary_export'
                             else 'local16-cache-' + CHUNK_ID)
    if output.exists() or output.is_symlink():
        fail('cache_output_not_fresh')
    provenance = {{'schema_version':'local16-cache-provenance-v1',
                  'cache_identity':'local16-attention-run01', 'max_slices_per_series':16,
                  'source_sha256':PLAN['source_sha256'], 'manifest_sha256':PLAN['manifest_sha256'],
                  'folds_sha256':PLAN['folds_sha256'],
                  'preprocessing_source_sha256':PLAN['preprocessing_source_sha256'], 'gold_included':False}}
    cache = Local16TensorCache(output, partition=chunk['partition'], fold=0,
                               manifest_sha256=PLAN['manifest_sha256'], first_ordinal=start,
                               provenance=provenance)
    for index in range(start, end):
        row = dataset._records[index]
        sample, eligible_counts = dataset.export_sample(index)
        cache.write_sample(study_id=row['study_id'], series_ids=[s['series_id'] for s in row['series']],
                           sample=sample, eligible_slice_counts=eligible_counts)
        if (index - start + 1) % 2 == 0 or index + 1 == end:
            print(json.dumps({{'event':'local16_cache_progress','chunk_id':CHUNK_ID,
                              'completed_studies':index-start+1,'total_studies':end-start}}, sort_keys=True))
    cache.finalize()
    cache_bytes = sum(path.stat().st_size for path in output.rglob('*') if path.is_file())
    if cache_bytes * 2 + PLAN['output_reserve_bytes'] > PLAN['output_limit_bytes']:
        fail('chunk_actual_output_limit_exceeded')
    archive_path = working_root / (output.name + '.zip')
    if archive_path.exists() or archive_path.is_symlink():
        fail('cache_archive_not_fresh')
    with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(output.rglob('*')):
            if path.is_symlink():
                fail('cache_output_symlink_invalid')
            if path.is_file():
                archive.write(path, path.relative_to(output).as_posix())
    archive_bytes = archive_path.stat().st_size
    if cache_bytes + archive_bytes + PLAN['output_reserve_bytes'] > PLAN['output_limit_bytes']:
        fail('chunk_actual_output_limit_exceeded')
    cache_manifest = json.loads((output/'cache-manifest.json').read_text(encoding='utf-8'))
    observed_counts = [shape[0] for record in cache_manifest['records'] for shape in record['series_shapes']]
    result = {{**summary, 'status':'complete', 'cache_bytes':cache_bytes,
              'archive_bytes':archive_bytes, 'archive_sha256':digest(archive_path),
              'saved_bytes_with_reserve':cache_bytes+archive_bytes+PLAN['output_reserve_bytes'],
              'observed_slice_count_histogram':{{str(count):observed_counts.count(count) for count in range(1,17)}},
              'series_with_more_than_eight_slices':sum(count > 8 for count in observed_counts),
              'chunk_manifest_sha256':digest(output/'cache-manifest.json')}}
    print(json.dumps(result, sort_keys=True))
    return result

if __name__ == '__main__':
    run()
'''


def _write_notebook(path: Path, code: str) -> str:
    compile(code, path.name + ".py", "exec")
    notebook = {"cells": [{"cell_type": "code", "execution_count": None,
                           "metadata": {}, "outputs": [], "source": code.splitlines(keepends=True)}],
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    raw = (json.dumps(notebook, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if path.exists() or path.is_symlink():
        raise ValueError("package_output_not_fresh")
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    if path.read_bytes() != raw:
        raise ValueError("package_readback_failed")
    return _sha(raw)


def build_local16_cache_package(root: Path, output_dir: Path) -> dict[str, object]:
    root, output_dir = Path(root), Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("package_output_not_fresh")
    private = root / "data/private/full-cohort-prefilter/run01"
    manifest_path, folds_path = private / "full-cohort-soft-label-manifest.json", private / "full-cohort-grouped-folds.json"
    if any(p.is_symlink() or not p.is_file() for p in (manifest_path, folds_path)):
        raise ValueError("chunk_export_input_unavailable")
    manifest_raw, folds_raw = manifest_path.read_bytes(), folds_path.read_bytes()
    source_path = output_dir / "local16-source.zip"
    source_path.parent.mkdir(parents=True, exist_ok=False)
    modules = sorted((root / "src").rglob("*.py")) + [root / "scripts/local16_cache_chunk_plan.py",
                                                       root / "scripts/local16_tensor_cache.py"]
    if any(p.is_symlink() or not p.is_file() for p in modules):
        raise ValueError("source_archive_member_invalid")
    with zipfile.ZipFile(source_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in modules:
            arcname = (path.relative_to(root / "src") if path.is_relative_to(root / "src")
                       else path.relative_to(root)).as_posix()
            archive.write(path, arcname)
        archive.writestr("scripts/__init__.py", '"""Embedded local16 cache helpers."""\n')
    source_raw = source_path.read_bytes()
    manifest, folds = json.loads(manifest_raw), json.loads(folds_raw)
    module_hashes = [_sha(p.read_bytes()) for p in modules if p.is_relative_to(root / "src/rsna_knee/data")]
    preprocessing_sha = hashlib.sha256("".join(sorted(module_hashes)).encode()).hexdigest()
    plan = create_chunk_plan(manifest, folds, manifest_sha256=_sha(manifest_raw), folds_sha256=_sha(folds_raw),
                             source_sha256=_sha(source_raw), preprocessing_source_sha256=preprocessing_sha)
    plan_raw = (json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    plan_path = output_dir / "local16-cache-chunk-plan.json"
    plan_path.write_bytes(plan_raw)
    source_b64, plan_json = base64.b64encode(source_raw).decode("ascii"), json.dumps(plan, sort_keys=True, separators=(",", ":"))
    artifacts = []
    for chunk in plan["chunks"]:
        cid, partition = chunk["chunk_id"], chunk["partition"]
        stages = [("plan", "plan", None)]
        if chunk["start_ordinal"] == 0:
            stages.append(("canary", "canary_export", 8))
        stages.append(("full", "chunk_export", None))
        for name, stage, max_studies in stages:
            path = output_dir / f"cache-{cid}-{name}.ipynb"
            code = _notebook_source(source_b64, _sha(source_raw), plan_json, cid, stage, max_studies)
            artifacts.append({"chunk_id": cid, "stage": name, "path": str(path), "sha256": _write_notebook(path, code)})
    summary = {"schema_version": "local16-cache-package-v1", "cache_identity": plan["cache_identity"],
               "plan_sha256": _sha(plan_raw), "source_sha256": _sha(source_raw),
               "manifest_sha256": plan["manifest_sha256"], "folds_sha256": plan["folds_sha256"],
               "preprocessing_source_sha256": preprocessing_sha, "max_slices_per_series": 16,
               "chunk_count": len(plan["chunks"]), "notebooks": artifacts,
               "stage_defaults": {"plan": "no export", "canary": "8 studies/partition, explicit opt-in", "full": "one bounded chunk, explicit opt-in"},
               "training_started": False}
    summary_path = output_dir / "package-manifest.json"
    summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=repo / "data/private/attention-sixteen-slices/cache-package-run01")
    output = parser.parse_args().output_dir
    if not output.is_absolute():
        output = repo / output
    print(json.dumps(build_local16_cache_package(repo, output), sort_keys=True, indent=2))
