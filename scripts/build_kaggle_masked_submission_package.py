"""Prepare new-model canary/full notebooks around the proven offline bootstrap."""
from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

from scripts.build_kaggle_submission_package import build_submission_package
from rsna_knee.inference.masked import validate_masked_report

TRAINING_SOURCE_SHA256 = "af5aadb3be99ae92e497b0ea85f66ee9ed25bdaa1b510b8861f3303700924303"
CODEC_MANIFEST_SHA256 = "50f39d3f627b3dc06a8080ba51dd6424898ef34fa4c4e21a84593dd326eb776d"


def checkpoint_fingerprint(report, training_source, expected_sha256=TRAINING_SOURCE_SHA256):
    """Reproduce the training fingerprint from the exact original source archive.

    New inference modules must not be mistaken for original training source.
    This mirrors the frozen checkpoint-v1 algorithm without loading the checkpoint.
    """
    validate_masked_report(report)
    raw=Path(training_source).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256: raise ValueError('training_source_pin_mismatch')
    with ZipFile(io.BytesIO(raw)) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)) or any(not name.startswith('rsna_knee/') or not name.endswith('.py')
                or '..' in PurePosixPath(name).parts for name in names): raise ValueError('training_source_invalid')
        source={name:hashlib.sha256(archive.read(name)).hexdigest() for name in sorted(names)}
    keys=('manifest_sha256','folds_sha256','source_csv_sha256','gold_ids_sha256',
          'official_train_csv_sha256','train_series_csv_sha256','prefilter_policy_sha256',
          'baseline_cohort_sha256','baseline_folds_sha256','pinned_folds_artifact_sha256',
          'fold','training_study_count','validation_study_count','model_kind','freeze_backbone','weights_sha256','weights_size')
    config={'fold':0,'epochs':5,'max_steps':None,'output_size':256,'max_slices':8,'device':'cuda',
            'learning_rate':.001,'balance_mode':'unweighted','seed':42,
            'weights_sha256':report['weights_sha256'],'weights_size':report['weights_size']}
    value={'schema':'masked-soft-checkpoint-v1','plan':{key:report.get(key) for key in keys},
           'config':config,'runner_sha256':source['rsna_knee/training/masked_soft.py'],
           'source_sha256':source,'runtime':report['runtime_versions']}
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def build_masked_submission_package(repository_root, output_dir, report_path, training_source, *, source_root,
                                    training_source_sha256=TRAINING_SOURCE_SHA256):
    repository=Path(repository_root).resolve(strict=True);output=Path(output_dir)
    if output.exists() or any(p.is_symlink() for p in (output,*output.parents)):
        raise ValueError('preparation_output_must_be_new')
    if (type(source_root) is not str or not source_root.startswith('/kaggle/input/')
            or '..' in PurePosixPath(source_root).parts): raise ValueError('source_mount_invalid')
    report_raw=Path(report_path).read_bytes();report=json.loads(report_raw)
    validate_masked_report(report)
    report_sha=hashlib.sha256(report_raw).hexdigest()
    fingerprint=checkpoint_fingerprint(report,training_source,training_source_sha256)
    # Reuse the actual tested bootstrap/publication generator, never the old model contract.
    with TemporaryDirectory() as td:
        base=Path(td)/'base'
        result=build_submission_package(repository,base,source_root=source_root,
            codec_manifest_sha256=CODEC_MANIFEST_SHA256,run_inference=True,recovery_policy_version=1)
        oldraw=(base/'fold0-inference-source.zip').read_bytes()
        notebook=json.loads(Path(result['notebook']).read_bytes())
        with ZipFile(io.BytesIO(oldraw)) as archive: members={name:archive.read(name) for name in archive.namelist()}
    extra=repository/'src/rsna_knee/inference/masked.py'
    if extra.is_symlink(): raise ValueError('source_symlink_forbidden')
    members['rsna_knee/inference/masked.py']=extra.read_bytes()
    buffer=io.BytesIO()
    with ZipFile(buffer,'w',compression=ZIP_DEFLATED) as archive:
        for name,raw in sorted(members.items()):
            ast.parse(raw,filename=name)
            info=ZipInfo(name,(2026,1,1,0,0,0));info.compress_type=ZIP_DEFLATED;archive.writestr(info,raw)
    raw=buffer.getvalue();digest=hashlib.sha256(raw).hexdigest()
    code=''.join(notebook['cells'][1]['source'])
    old_payload=base64.b64encode(oldraw).decode()
    if code.count(old_payload)!=1: raise ValueError('source_payload_ambiguous')
    code=code.replace(old_payload,base64.b64encode(raw).decode()).replace(hashlib.sha256(oldraw).hexdigest(),digest)
    code=code.replace('CHECKPOINT_SHA256 = None',f'CHECKPOINT_SHA256 = {report["checkpoint_sha256"]!r}')
    code=code.replace('REPORT_SHA256 = None',f'REPORT_SHA256 = {report_sha!r}')
    code=code.replace('CHECKPOINT_SIZE = None\n','').replace(' or not CHECKPOINT_SIZE','')
    code=code.replace('from rsna_knee.inference.fold0 import run_inference',
                      'from rsna_knee.inference.masked import run_masked_inference')
    code=code.replace('fold0-inference-run01','masked-inference-run01')
    start=code.index('    summary = run_inference(');end=code.index('    destination_created = False',start)
    code=code[:start]+f'''    summary = run_masked_inference(COMPETITION_ROOT, source, REPORT_SHA256,
                                   CHECKPOINT_SHA256, {fingerprint!r},
                                   '/kaggle/working/masked-inference-run01', device='cuda',
                                   max_studies=MAX_STUDIES, recovery_policy_version=RECOVERY_POLICY_VERSION)
'''+code[end:]
    start=code.index('    destination_created = False');end=code.index('    print(json.dumps(summary, sort_keys=True))',start)
    publication=code[start:end]
    code=code[:start]+"    if MAX_STUDIES is None or summary['study_count'] == summary['test_study_count']:\n"+''.join('    '+line for line in publication.splitlines(keepends=True))+code[end:]
    code=code.replace('# Inference only: attach the private output of execution 361175523.',
                      '# Inference only: completed masked-soft model, execution 363315909.')
    output.mkdir(parents=True,exist_ok=False)
    archive_path=output/'masked-inference-source.zip';archive_path.write_bytes(raw)
    result={'archive':str(archive_path),'archive_sha256':digest,'checkpoint_fingerprint':fingerprint,
            'checkpoint_sha256':report['checkpoint_sha256'],'report_sha256':report_sha,
            'training_source_sha256':training_source_sha256,'source_execution':363315909,
            'source_root':source_root,'member_count':len(members),'remote_executed':False,'submitted':False}
    for label,limit in [('canary',3),('submission',None)]:
        variant=json.loads(json.dumps(notebook))
        variant['cells'][0]['source']=['# RSNA Knee: pinned masked-soft model inference\n',
            'Internet OFF, T4x2, original environment. Attach official competition, the exact private trained output '
            'and existing offline DICOM codec input. No labels/reports are used as test features. '
            'Canary mode publishes the competition-root CSV only when it covers every visible test study.\n']
        variant_code=f'MAX_STUDIES = {limit!r}\n'+code
        compile(variant_code,f'masked-{label}','exec')
        variant['cells'][1]['source']=variant_code.splitlines(keepends=True)
        path=output/f'kaggle_masked_{label}.ipynb'
        path.write_text(json.dumps(variant,indent=2)+'\n',encoding='utf-8')
        result[label]=str(path);result[f'{label}_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    (output/'inventory.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result
