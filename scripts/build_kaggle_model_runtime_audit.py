"""Build a fresh disabled private source-only model/runtime audit notebook."""
from __future__ import annotations
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

from rsna_knee.data.compatibility_audit import plan_bytes
from scripts.build_kaggle_submission_package import MODULES as INFERENCE_MODULES

MODULES = tuple(m for m in INFERENCE_MODULES if not m.startswith(('models/', 'inference/'))) + (
    'data/orientation_audit.py', 'data/compatibility_audit.py',
    'data/model_runtime_audit.py', 'models/resnet18.py', 'inference/fold0.py',
    'models/baseline.py',
)


def build_model_runtime_package(repository_root, output_dir, plan):
    repo, output = Path(repository_root), Path(output_dir)
    if output.exists() or output.is_symlink() or any(p.is_symlink() for p in output.parents):
        raise ValueError('preparation_output_must_be_new')
    frozen = plan_bytes(plan)
    names = {'rsna_knee/__init__.py'}
    for module in MODULES:
        names.add('rsna_knee/' + module)
        names.add('rsna_knee/' + module.split('/')[0] + '/__init__.py')
    members = {}
    for name in sorted(names):
        path = repo / 'src' / name
        if path.is_symlink():
            raise ValueError('source_symlink_forbidden')
        members[name] = path.read_bytes()
        ast.parse(members[name], filename=name)
    buffer = io.BytesIO()
    with ZipFile(buffer, 'w', compression=ZIP_DEFLATED) as archive:
        for name, raw in members.items():
            info = ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, raw)
    source = buffer.getvalue()
    source_hash = hashlib.sha256(source).hexdigest()
    plan_hash = hashlib.sha256(frozen).hexdigest()
    code = f'''# Private offline GPU diagnostic; no training and no submission.
RUN_MODEL_RUNTIME_AUDIT = False
if RUN_MODEL_RUNTIME_AUDIT:
    import base64, hashlib, json, os, sys
    from pathlib import Path
    source = base64.b64decode("{base64.b64encode(source).decode()}", validate=True)
    frozen = base64.b64decode("{base64.b64encode(frozen).decode()}", validate=True)
    if hashlib.sha256(source).hexdigest() != "{source_hash}" or hashlib.sha256(frozen).hexdigest() != "{plan_hash}":
        raise RuntimeError("model_runtime_payload_invalid")
    archive = Path('/kaggle/working/model-runtime-source.zip')
    with archive.open('xb') as stream:
        stream.write(source)
    sys.path.insert(0, str(archive))
    from rsna_knee.data.model_runtime_audit import run_model_runtime_audit
    competition = Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')
    training = Path('/kaggle/input/notebooks/davidsulleiro/rsna-knee-resnet18-frozen-backbone-run01/resnet18-frozen-backbone/run01')
    try:
        result = run_model_runtime_audit(json.loads(frozen), frozen, competition/'train_series.csv',
            competition/'train_series', training/'report.json', training/'checkpoint.pt',
            '/kaggle/working/training-model-runtime-audit/run01', device='cuda:0',
            source_archive=archive, source_archive_sha256='{source_hash}')
        print(json.dumps({{'event':'model_runtime_summary','status':result['status'],
            'planned_series':result['planned_series'],'attempted_series':result['attempted_series'],
            'completed_series':result['completed_series'],'successful_series':result['successful_series'],
            'failed_series':result['failed_series'],'termination_reason':result['termination_reason']}}), flush=True)
    except Exception as error:
        candidate = error.args[0] if len(error.args) == 1 and type(error.args[0]) is str else ''
        allowed = {{'model_runtime_initialization_failed', 'model_runtime_audit_interrupted',
                   'model_runtime_report_storage_failed'}}
        reason = candidate if candidate in allowed else 'model_runtime_audit_failed'
        print(json.dumps({{'event':'model_runtime_failed','reason_code':reason}}), flush=True)
        raise RuntimeError('model_runtime_audit_failed') from None
else:
    print('Model/runtime audit disabled: verify all frozen pins and source members before enabling.')
'''
    notebook = {'nbformat':4, 'nbformat_minor':5,
        'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},
        'cells':[{'cell_type':'markdown','id':'protocol','metadata':{},'source':[
            '# RSNA Knee: bounded fold-zero model/runtime diagnostic\n',
            'Private offline GPU-only run on exactly 20 preselected Sagittal series from 20 distinct studies. '
            'Each series is one isolated model input; this is not full-study aggregation, training, test inference, '
            'submission, or a general compatibility claim. A cooperative 1,800-second budget is checked between series. '
            'The exact pinned fold-zero checkpoint and report are verified before safe checkpoint loading. '
            'The report contains only aggregate-safe metrics and numeric ordinals; never publish identifiers or probabilities.\n']},
            {'cell_type':'code','id':'audit','metadata':{},'execution_count':None,'outputs':[],
             'source':code.splitlines(keepends=True)}]}
    inventory = {'schema_version':'training-model-runtime-source-v1','archive_sha256':source_hash,
        'plan_sha256':plan_hash,'members':{name:hashlib.sha256(raw).hexdigest() for name,raw in members.items()},
        'remote_executed':False,'device':'cuda:0','budget_seconds':1800,'scope':'20 selected Sagittal series'}
    output.mkdir(parents=True, exist_ok=False)
    (output/'model-runtime-source.zip').write_bytes(source)
    (output/'audit-plan.json').write_bytes(frozen)
    (output/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n',encoding='utf-8')
    path=output/'kaggle_model_runtime_audit.ipynb'
    path.write_text(json.dumps(notebook,indent=2)+'\n',encoding='utf-8')
    return {'notebook':str(path),'notebook_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'archive_sha256':source_hash,'plan_sha256':plan_hash,'member_count':len(members),
            'members':tuple(sorted(members))}
