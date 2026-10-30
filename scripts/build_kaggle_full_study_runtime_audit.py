"""Build a disabled, private full-study fold-zero runtime audit package."""
from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from rsna_knee.data.compatibility_audit import plan_bytes
from rsna_knee.data.model_runtime_audit import build_full_study_plan
from scripts.build_kaggle_model_runtime_audit import MODULES


def build_full_study_runtime_package(repository_root, output_dir, plan, metadata_csv, baseline_report):
    repo, output = Path(repository_root), Path(output_dir)
    if output.exists() or output.is_symlink() or any(parent.is_symlink() for parent in output.parents):
        raise ValueError('preparation_output_must_be_new')
    frozen_plan = plan_bytes(plan)
    baseline_raw = Path(baseline_report).read_bytes()
    full_plan = build_full_study_plan(plan, frozen_plan, metadata_csv, baseline_raw)
    frozen_full_plan = (json.dumps(full_plan, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    members = {}
    names = {'rsna_knee/__init__.py'}
    for module in MODULES:
        names.add('rsna_knee/' + module)
        names.add('rsna_knee/' + module.split('/')[0] + '/__init__.py')
    for name in sorted(names):
        path = repo / 'src' / name
        if path.is_symlink():
            raise ValueError('source_symlink_forbidden')
        raw = path.read_bytes()
        ast.parse(raw, filename=name)
        members[name] = raw
    buffer = io.BytesIO()
    with ZipFile(buffer, 'w', compression=ZIP_DEFLATED) as archive:
        for name, raw in members.items():
            info = ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, raw)
    source = buffer.getvalue()
    source_hash = hashlib.sha256(source).hexdigest()
    plan_hash = hashlib.sha256(frozen_plan).hexdigest()
    full_plan_hash = hashlib.sha256(frozen_full_plan).hexdigest()
    baseline_hash = hashlib.sha256(baseline_raw).hexdigest()
    code = f'''# Private offline GPU-only diagnostic; no training or submission.
RUN_FULL_STUDY_RUNTIME_AUDIT = False
if RUN_FULL_STUDY_RUNTIME_AUDIT:
    import base64, hashlib, json, sys
    from pathlib import Path
    source = base64.b64decode("{base64.b64encode(source).decode()}", validate=True)
    frozen = base64.b64decode("{base64.b64encode(frozen_plan).decode()}", validate=True)
    full_frozen = base64.b64decode("{base64.b64encode(frozen_full_plan).decode()}", validate=True)
    baseline = base64.b64decode("{base64.b64encode(baseline_raw).decode()}", validate=True)
    if (hashlib.sha256(source).hexdigest() != "{source_hash}"
            or hashlib.sha256(frozen).hexdigest() != "{plan_hash}"
            or hashlib.sha256(full_frozen).hexdigest() != "{full_plan_hash}"
            or hashlib.sha256(baseline).hexdigest() != "{baseline_hash}"):
        raise RuntimeError("full_study_runtime_payload_invalid")
    archive = Path('/kaggle/working/full-study-runtime-source.zip')
    with archive.open('xb') as stream:
        stream.write(source)
    sys.path.insert(0, str(archive))
    from rsna_knee.data.model_runtime_audit import run_model_runtime_audit
    competition = Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')
    training = Path('/kaggle/input/notebooks/davidsulleiro/rsna-knee-resnet18-frozen-backbone-run01/resnet18-frozen-backbone/run01')
    try:
        result = run_model_runtime_audit(json.loads(frozen), frozen,
            competition/'train_series.csv', competition/'train_series', training/'report.json',
            training/'checkpoint.pt', '/kaggle/working/full-study-runtime-audit/run01',
            device='cuda:0', source_archive=archive, source_archive_sha256='{source_hash}',
            full_study_plan=json.loads(full_frozen), full_study_plan_raw=full_frozen,
            baseline_report_raw=baseline)
        print(json.dumps({{'event':'full_study_runtime_summary','status':result['status'],
            'planned_studies':result['planned_studies'],'attempted_studies':result['attempted_studies'],
            'completed_studies':result['completed_studies'],'successful_studies':result['successful_studies'],
            'failed_studies':result['failed_studies'],'planned_series':result['planned_series'],
            'attempted_series':result['attempted_series'],'successful_series':result['successful_series'],
            'failed_series':result['failed_series'],'finite_logits':result['finite_logits'],
            'finite_probabilities':result['finite_probabilities'],
            'termination_reason':result['termination_reason']}}), flush=True)
    except Exception as error:
        candidate = error.args[0] if len(error.args) == 1 and type(error.args[0]) is str else ''
        allowed = {{'model_runtime_initialization_failed', 'model_runtime_audit_interrupted',
                   'model_runtime_report_storage_failed'}}
        reason = candidate if candidate in allowed else 'model_runtime_audit_failed'
        print(json.dumps({{'event':'full_study_runtime_failed','reason_code':reason}}), flush=True)
        raise RuntimeError('full_study_runtime_audit_failed') from None
else:
    print('Full-study runtime audit disabled: verify the frozen 20-study sibling-series plan and every package pin before enabling.')
'''
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}},
        'cells': [
            {'cell_type': 'markdown', 'id': 'protocol', 'metadata': {}, 'source': [
                '# RSNA Knee: full-study fold-zero runtime diagnostic\n',
                'One private, offline GPU-only diagnostic for all sibling series of exactly the same 20 studies selected in CA12. '
                'The only changed variable from CA16 is input granularity: genuine TestStudyDataset supplies every series in a study '
                'to the unchanged ResNet18StudyModel and its existing equal-series aggregation. Same checkpoint, preprocessing, '
                'head, pooling, slice microbatch, and cuda:0. This is not training, hidden-test inference, submission, or a quality '
                'claim. A cooperative 1,800-second budget is checked between studies; output is aggregate-only and contains no IDs or predictions.\n']},
            {'cell_type': 'code', 'id': 'audit', 'metadata': {}, 'execution_count': None, 'outputs': [],
             'source': code.splitlines(keepends=True)}]}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'full-study-runtime-source.zip').write_bytes(source)
    (output / 'selected-plan.json').write_bytes(frozen_plan)
    (output / 'full-study-plan.json').write_bytes(frozen_full_plan)
    (output / 'ca16-baseline-report.json').write_bytes(baseline_raw)
    inventory = {'schema_version': 'full-study-runtime-source-v1',
        'archive_sha256': source_hash, 'selected_plan_sha256': plan_hash,
        'full_study_plan_sha256': full_plan_hash, 'baseline_report_sha256': baseline_hash,
        'members': {name: hashlib.sha256(raw).hexdigest() for name, raw in members.items()},
        'remote_executed': False, 'device': 'cuda:0', 'budget_seconds': 1800,
        'scope': {'study_count': 20, 'series_count': full_plan['series_count'],
                  'series_plane_counts': full_plan['series_plane_counts']}}
    (output / 'inventory.json').write_text(json.dumps(inventory, indent=2) + '\n', encoding='utf-8')
    notebook_path = output / 'kaggle_full_study_runtime_audit.ipynb'
    notebook_path.write_text(json.dumps(notebook, indent=2) + '\n', encoding='utf-8')
    return {'notebook': str(notebook_path), 'notebook_sha256': hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
        'archive_sha256': source_hash, 'selected_plan_sha256': plan_hash,
        'full_study_plan_sha256': full_plan_hash, 'baseline_report_sha256': baseline_hash,
        'member_count': len(members), 'members': tuple(sorted(members)),
        'study_count': len(full_plan['studies']), 'series_count': inventory['scope']['series_count']}
