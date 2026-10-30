"""Build separate offline recipe assets and embedded-source benchmark notebooks."""
from __future__ import annotations
import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import zipfile

LEGACY_SOURCE_SHA='af5aadb3be99ae92e497b0ea85f66ee9ed25bdaa1b510b8861f3303700924303'

def sha(payload):return hashlib.sha256(payload).hexdigest()


def build_package(root,output,*,private_root=None):
    root=Path(root).resolve();output=Path(output).absolute()
    private=Path(private_root or root/'data/private').resolve()
    if (not output.is_relative_to(private) or output==private or output.exists()
            or any(p.is_symlink() for p in (output,*output.parents))):raise ValueError('output_must_be_fresh_private')
    spec=importlib.util.spec_from_file_location('legacy_masked_builder',root/'scripts/build_kaggle_masked_training_package.py')
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    from rsna_knee.models.resnet50 import WEIGHT_PINS
    old=root/'data/private/masked-soft-training-preparation/run09'
    original=(old/'masked-training-source.zip').read_bytes()
    if sha(original)!=LEGACY_SOURCE_SHA:raise ValueError('legacy_source_mismatch')
    members={}
    with zipfile.ZipFile(io.BytesIO(original)) as archive:
        for name in archive.namelist():
            if Path(name).is_absolute() or '..' in Path(name).parts or not name.endswith('.py'):raise ValueError('source_path_invalid')
            members[name]=archive.read(name)
    for name in ('rsna_knee/models/resnet50.py','rsna_knee/training/resnet50.py'):
        if name in members:raise ValueError('new_module_collision')
        members[name]=(root/'src'/name).read_bytes()
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,payload in sorted(members.items()):
            info=zipfile.ZipInfo(name,date_time=(2020,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,payload)
    source=buffer.getvalue();pins={}
    for arm,pin in WEIGHT_PINS.items():
        path=root/'data/private/resnet50-pretraining-assets/run01'/pin['filename']
        if path.is_symlink() or not path.is_file():raise ValueError('weights_missing')
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=pin['sha256'] or path.stat().st_size!=pin['size']:raise ValueError('weights_mismatch')
        pins[arm]=dict(pin)
    inventory={'schema':'resnet50-recipe-assets-v1','weights':pins,'source_sha256':sha(source),
               'source_members':{name:sha(payload) for name,payload in sorted(members.items())},
               'protocol':{'fold':0,'seed':42,'output_size':224,'max_slices':8,'benchmark_train':8,'benchmark_validation':8}}
    raw=(json.dumps(inventory,sort_keys=True,separators=(',',':'))+'\n').encode()
    oldinv=(old/'upload/masked-training-inventory.json').read_bytes()
    # Reuse the exact checked input discovery/codec validation helpers, never the old runner.
    helpers=legacy._notebook_source().split('def run_notebook(',1)[0]
    oldinventory=json.loads(oldinv)
    helpers=helpers.replace('__INVENTORY_SHA256__',sha(oldinv)).replace('__SOURCE_SHA256__',oldinventory['source_archive_sha256']).replace('__CODEC_SHA256__',legacy.CODEC_SHA256).replace('__CODEC_MANIFEST_SHA256__',legacy.CODEC_MANIFEST_SHA256)
    runtime='''
ARM = __ARM__
EXECUTE_BENCHMARK = __EXECUTE__
NEW_INVENTORY_SHA256 = __INVENTORY__
NEW_SOURCE_SHA256 = __SOURCE_SHA__
NEW_SOURCE_B64 = __SOURCE_B64__

def run_recipe_notebook():
    inputs=Path('/kaggle/input');working=Path('/kaggle/working')
    assets,old_inventory=_verify_asset_root(find_prepared_asset_root(inputs))
    competition=find_competition_root(inputs)
    # Existing Dataset V4 stays pinned, including its exact metadata and offline codecs.
    codec_bytes=verify_inputs(assets,old_inventory)
    candidates=[]
    for pattern in ('*/resnet50-recipe-assets.json','*/*/resnet50-recipe-assets.json','*/*/*/resnet50-recipe-assets.json'):
        candidates.extend(inputs.glob(pattern))
    if len(candidates)!=1:fail('recipe_assets_not_unique')
    marker=candidates[0]
    if marker.is_symlink() or digest(marker)!=NEW_INVENTORY_SHA256:fail('recipe_inventory_mismatch')
    inventory=json.loads(marker.read_text())
    if inventory.get('schema')!='resnet50-recipe-assets-v1':fail('recipe_inventory_invalid')
    for pin in inventory['weights'].values():
        path=marker.parent/pin['filename']
        if path.is_symlink() or path.stat().st_size!=pin['size'] or digest(path)!=pin['sha256']:fail('recipe_weight_mismatch')
    payload=base64.b64decode(NEW_SOURCE_B64,validate=True)
    if hashlib.sha256(payload).hexdigest()!=NEW_SOURCE_SHA256:fail('recipe_source_mismatch')
    source=working/'resnet50-recipe-source'
    if source.exists() or source.is_symlink():fail('recipe_source_not_fresh')
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)) or set(names)!=set(inventory['source_members']):fail('recipe_source_inventory_mismatch')
        for name in names:
            path=Path(name)
            if path.is_absolute() or '..' in path.parts or not name.endswith('.py') or '\\\\' in name:fail('recipe_source_path_invalid')
            if hashlib.sha256(archive.read(name)).hexdigest()!=inventory['source_members'][name]:fail('recipe_source_member_mismatch')
        source.mkdir()
        for name in names:
            target=source/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.read(name))
    sys.path.insert(0,str(source))
    runner=importlib.import_module('rsna_knee.training.resnet50')
    config=runner.RecipeConfig(arm=ARM,weights_path=marker.parent/inventory['weights'][ARM]['filename'],
        output_dir=working/'data/private/resnet50-pretraining'/ARM/'benchmark-run01',image_root=competition,
        manifest_path=assets/'full-cohort-soft-label-manifest.json',folds_path=assets/'full-cohort-grouped-folds.json',
        source_root=source,mode='benchmark',device='cuda',max_epochs_per_run=1,max_runtime_seconds=28800)
    print(json.dumps(runner.run_recipe_training(config),sort_keys=True),flush=True)
    if not EXECUTE_BENCHMARK:return
    bundle=working/'resnet50-offline-codecs.zip'
    if bundle.exists() or bundle.is_symlink():fail('codec_workfile_not_fresh')
    bundle.write_bytes(codec_bytes)
    bootstrap=importlib.import_module('rsna_knee.inference.offline_codec_bootstrap')
    installed=bootstrap.install_codec_bundle(bundle,EXPECTED_CODEC_MANIFEST_SHA256,working/'resnet50-codec-install')
    target=Path(installed['target'])
    if target.is_symlink() or not target.is_dir():fail('codec_install_failed')
    sys.path.insert(0,str(target))
    _check_required_decoders(importlib.import_module('pydicom'),importlib.import_module('pydicom.uid').UID)
    report=runner.run_recipe_training(config,execute=True)
    print(json.dumps(report,sort_keys=True),flush=True)

if __name__=='__main__':
    try:run_recipe_notebook()
    except KeyboardInterrupt:raise
    except Exception:
        print(json.dumps({'status':'failed','reason_code':'recipe_notebook_failed'}),flush=True)
        raise SystemExit(2) from None
'''
    output.mkdir(parents=True);upload=output/'upload';upload.mkdir()
    (upload/'resnet50-recipe-assets.json').write_bytes(raw)
    for pin in pins.values():shutil.copyfile(root/'data/private/resnet50-pretraining-assets/run01'/pin['filename'],upload/pin['filename'])
    archivepath=output/'resnet50-recipe-source.zip';archivepath.write_bytes(source)
    result={'output_dir':str(output),'upload_dir':str(upload),'archive_path':str(archivepath),'archive_sha256':sha(source),'inventory_sha256':sha(raw),'source_member_count':len(members)}
    for arm in pins:
        for stage in ('plan','benchmark'):
            code=helpers+runtime.replace('__ARM__',json.dumps(arm)).replace('__EXECUTE__',str(stage=='benchmark')).replace('__INVENTORY__',repr(sha(raw))).replace('__SOURCE_SHA__',repr(sha(source))).replace('__SOURCE_B64__',repr(base64.b64encode(source).decode()))
            compile(code,'recipe_notebook.py','exec')
            notebook={'cells':[{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':code.splitlines(keepends=True)}],
                      'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5}
            path=output/f'kaggle_resnet50_{arm}_{stage}.ipynb'
            encoded=(json.dumps(notebook,sort_keys=True,separators=(',',':'))+'\n').encode();path.write_bytes(encoded)
            if path.read_bytes()!=encoded:raise ValueError('notebook_readback_failed')
            result[f'{arm}_{stage}']=str(path);result[f'{arm}_{stage}_sha256']=sha(encoded)
    for pin in pins.values():
        with (upload/pin['filename']).open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=pin['sha256']:raise ValueError('asset_readback_failed')
    if (upload/'resnet50-recipe-assets.json').read_bytes()!=raw or archivepath.read_bytes()!=source:raise ValueError('package_readback_failed')
    return result


def build_full_radimagenet_notebook(root, output, *, private_root=None):
    """Export one authorized full-data epoch without mutating benchmark artifacts.

    A live quota check remains the parent's launch prerequisite. The epoch-boundary
    runtime guard cannot interrupt an initial epoch before its first checkpoint.
    """
    root = Path(root).resolve()
    output = Path(output).absolute()
    private = Path(private_root or root / 'data/private').resolve()
    if (not output.is_relative_to(private) or output == private or output.exists()
            or any(p.is_symlink() for p in (output, *output.parents))):
        raise ValueError('output_must_be_fresh_private')
    benchmark = root / 'data/private/resnet50-pretraining-preparation/run01/kaggle_resnet50_radimagenet_benchmark.ipynb'
    if benchmark.is_symlink():
        raise ValueError('benchmark_notebook_invalid')
    original = benchmark.read_bytes()
    if sha(original) != 'ac7b1924415ee7c44c26ab569efd45e63af1a0d50209419622a82280072550aa':
        raise ValueError('benchmark_notebook_pin_mismatch')
    notebook = json.loads(original)
    code = ''.join(notebook['cells'][0]['source'])
    replacements = {
        'EXECUTE_BENCHMARK': 'EXECUTE_FULL_TRAINING',
        "mode='benchmark'": "mode='train'",
        "/'benchmark-run01'": "/'full-run01'",
        'max_epochs_per_run=1,max_runtime_seconds=28800)':
            'max_epochs_per_run=1,max_runtime_seconds=28800,resume_checkpoint=None,resume_sha256=None)',
    }
    for before, after in replacements.items():
        expected_count = 2 if before == 'EXECUTE_BENCHMARK' else 1
        if code.count(before) != expected_count:
            raise ValueError('benchmark_notebook_contract_mismatch')
        code = code.replace(before, after)
    compile(code, 'radimagenet_full_epoch01.py', 'exec')
    notebook['cells'][0]['source'] = code.splitlines(keepends=True)
    payload = (json.dumps(notebook, sort_keys=True, separators=(',', ':')) + '\n').encode()
    output.mkdir(parents=True)
    path = output / 'kaggle_resnet50_radimagenet_full_epoch01.ipynb'
    path.write_bytes(payload)
    if path.read_bytes() != payload or benchmark.read_bytes() != original:
        raise ValueError('full_notebook_readback_failed')
    return {'output_dir': str(output), 'notebook_path': str(path),
            'notebook_sha256': sha(payload),
            'source_sha256': '48a64d88c9b5cf72f4fdf6d9ce9839cc2cb3b8632b9fbf4fb7f4db1cdc4f95c8',
            'inventory_sha256': '22695a38a98fb868bfce34a8fe244d02cf3c8694abc8048672c7c9edaa9db4b6'}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--full-radimagenet-first-epoch',action='store_true')
    args=parser.parse_args();builder=build_full_radimagenet_notebook if args.full_radimagenet_first_epoch else build_package
    print(json.dumps(builder(Path(__file__).resolve().parents[1],args.output_dir),sort_keys=True))
