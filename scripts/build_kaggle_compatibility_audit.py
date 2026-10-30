"""Freeze a private ID-only audit plan and offline CPU-only source notebook."""
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

# Reuse only the adapter's legacy data/validation/label import closure, not models.
MODULES = tuple(m for m in INFERENCE_MODULES if not m.startswith(('models/', 'inference/'))) + (
    'data/orientation_audit.py', 'data/compatibility_audit.py',
)


def build_compatibility_package(repository_root, output_dir, plan):
    repo, output = Path(repository_root), Path(output_dir)
    candidate_mode = plan.get('schema_version') == 'training-dimension-candidates-v1'
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
    with ZipFile(buffer,'w',compression=ZIP_DEFLATED) as archive:
        for name, raw in members.items():
            info=ZipInfo(name,(2026,1,1,0,0,0));info.compress_type=ZIP_DEFLATED
            archive.writestr(info,raw)
    raw=buffer.getvalue();digest=hashlib.sha256(raw).hexdigest();plan_hash=hashlib.sha256(frozen).hexdigest()
    code=f"""# Compatibility audit only; device is CPU only. Keep private and Internet OFF.
# Attach the official competition only. No model, checkpoint, labels or reports.
RUN_AUDIT = False
if RUN_AUDIT:
    import base64, hashlib, json, os, sys
    from pathlib import Path
    print(json.dumps({{'event':'compatibility_bootstrap','device':'cpu','budget_seconds':1800}}),flush=True)
    try:
        os.environ['OMP_NUM_THREADS']='1'
        os.environ['MKL_NUM_THREADS']='1'
        source=base64.b64decode('{base64.b64encode(raw).decode()}',validate=True)
        frozen=base64.b64decode('{base64.b64encode(frozen).decode()}',validate=True)
        if hashlib.sha256(source).hexdigest() != '{digest}' or hashlib.sha256(frozen).hexdigest() != '{plan_hash}':
            raise ValueError('audit_payload_invalid')
        archive=Path('/kaggle/working/compatibility-source.zip')
        with archive.open('xb') as stream:
            stream.write(source)
        sys.path.insert(0,str(archive))
        from rsna_knee.data.compatibility_audit import run_compatibility_audit
        root=Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')
        summary=run_compatibility_audit(json.loads(frozen),root/'train_series.csv',root/'train_series',
                                      '/kaggle/working/{'training-dimension-candidates-run01' if candidate_mode else 'training-compatibility-run01'}')
        print(json.dumps({{'event':'compatibility_summary','status':summary['status'],
                          'completed_studies':summary['completed_studies'],
                          'attempted_series':summary['attempted_series'],
                          'failed_series':summary['failed_series']}}),flush=True)
    except Exception:
        print(json.dumps({{'event':'compatibility_failed','reason_code':'audit_initialization_or_storage_failed'}}),flush=True)
        raise RuntimeError('compatibility_audit_failed') from None
else:
    print('Audit disabled: verify the frozen plan and source package before enabling.')
"""
    notebook={'nbformat':4,'nbformat_minor':5,
              'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},
              'cells':[{'cell_type':'markdown','id':'protocol','metadata':{},'source':[
                  '# RSNA Knee: targeted dimension-candidate preprocessing audit\n' if candidate_mode else '# RSNA Knee: fresh-study compatibility audit\n',
                  ('Private CPU-only audit of exactly 20 preselected series from the retained out-of-range dimension examples. '
                   'It does not audit all 64 flagged series or other series in the selected studies. ' if candidate_mode else
                   'Private CPU-only audit, 100 fresh non-Gold studies, all planned series. '),
                  'Thirty-minute cooperative budget checked between series; one series may overrun. '
                  'No training, classifier inference, hidden-test probe or submission. '
                  'Uses the exact pinned current adapter, 256-pixel output and max eight slices. '
                  'Per-series audit failures continue without changing production inference. '
                  'The ID-only plan is private; outputs contain aggregate metadata and numeric ordinals only.\n']},
                       {'cell_type':'code','id':'audit','metadata':{},'execution_count':None,'outputs':[],
                        'source':code.splitlines(keepends=True)}]}
    inventory={'schema_version':'training-compatibility-source-v1','archive_sha256':digest,
               'plan_sha256':plan_hash,'members':{n:hashlib.sha256(v).hexdigest() for n,v in members.items()},
               'remote_executed':False,'device':'cpu','budget_seconds':1800,
               'scope':'20 retained dimension candidates only' if candidate_mode else '100 fresh studies'}
    output.mkdir(parents=True,exist_ok=False)
    (output/'compatibility-source.zip').write_bytes(raw)
    (output/'audit-plan.json').write_bytes(frozen)
    (output/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n',encoding='utf-8')
    path=output/'kaggle_training_compatibility_audit.ipynb'
    path.write_text(json.dumps(notebook,indent=2)+'\n',encoding='utf-8')
    return {'notebook':str(path),'notebook_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'archive_sha256':digest,'plan_sha256':plan_hash,'member_count':len(members)}
