"""Build the private offline header-only inventory notebook, disabled by default."""
import ast
import base64
import gzip
import io
import json
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED
from rsna_knee.data.header_inventory import digest, plan_bytes


def build_header_package(repository_root, output_dir, plan):
    repo, output = Path(repository_root), Path(output_dir)
    if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('preparation_output_must_be_new')
    path = repo / 'src/rsna_knee/data/header_inventory.py'
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('source_symlink_forbidden')
    source = path.read_bytes()
    ast.parse(source)
    frozen = plan_bytes(plan)
    buffer = io.BytesIO()
    with ZipFile(buffer, 'w', compression=ZIP_DEFLATED) as archive:
        info = ZipInfo('header_inventory.py', (2026, 1, 1, 0, 0, 0))
        info.compress_type = ZIP_DEFLATED
        archive.writestr(info, source)
    raw = buffer.getvalue()
    packed = gzip.compress(frozen, mtime=0)
    code = f'''# Private CPU / Internet OFF / official competition input only.
# Exactly one header per series; no pixel decoding, model or submission.
RUN_INVENTORY = False
if RUN_INVENTORY:
    import base64, gzip, hashlib, json, sys
    from pathlib import Path
    try:
        source = base64.b64decode('{base64.b64encode(raw).decode()}', validate=True)
        frozen = gzip.decompress(base64.b64decode('{base64.b64encode(packed).decode()}', validate=True))
        if hashlib.sha256(source).hexdigest() != '{digest(raw)}' or hashlib.sha256(frozen).hexdigest() != '{digest(frozen)}':
            raise ValueError('inventory_payload_invalid')
        target = Path('/kaggle/working/header-source.zip')
        with target.open('xb') as stream:
            stream.write(source)
        sys.path.insert(0, str(target))
        from header_inventory import run_header_inventory
        root = Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')
        print(json.dumps({{'event': 'header_inventory_started', 'device': 'cpu', 'budget_seconds': 1800}}), flush=True)
        summary = run_header_inventory(json.loads(frozen), root/'train_series.csv', root/'train_series',
                                       '/kaggle/working/training-header-inventory-run01')
        print(json.dumps({{'event': 'header_inventory_summary', 'status': summary['status'],
                           'completed_studies': summary['completed_studies'],
                           'completed_series': summary['completed_series'],
                           'failed_series': summary['failed_series']}}), flush=True)
    except Exception:
        print(json.dumps({{'event': 'header_inventory_failed', 'reason_code': 'initialization_or_storage_failed'}}), flush=True)
        raise RuntimeError('header_inventory_failed') from None
else:
    print('Header inventory disabled; verify source and frozen plan before enabling.')
'''
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
                'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}},
                'cells': [{'cell_type': 'markdown', 'id': 'protocol', 'metadata': {}, 'source': [
                    '# RSNA Knee: uncovered training header strata\n',
                    'Private CPU-only / Internet off / official competition input only. '
                    'All remaining fresh metadata studies, excluding Gold-58, prior 500 and audited 100. '
                    'Read one lexicographically chosen header per series, never pixels. '
                    'Thirty-minute cooperative budget may yield partial coverage. '
                    'This cannot test within-series consistency, codec capability or hidden inference. '
                    'The embedded ID-only plan must remain private; outputs use bounded categories and ordinals.\n']},
                          {'cell_type': 'code', 'id': 'inventory', 'metadata': {}, 'execution_count': None,
                           'outputs': [], 'source': code.splitlines(keepends=True)}]}
    output.mkdir(parents=True, exist_ok=False)
    (output/'header-source.zip').write_bytes(raw)
    (output/'header-plan.json').write_bytes(frozen)
    inventory = {'archive_sha256': digest(raw), 'plan_sha256': digest(frozen),
                 'members': {'header_inventory.py': digest(source)}, 'remote_executed': False,
                 'device': 'cpu', 'budget_seconds': 1800}
    (output/'inventory.json').write_text(json.dumps(inventory, indent=2)+'\n')
    path = output/'kaggle_training_header_inventory.ipynb'
    path.write_text(json.dumps(notebook, indent=2)+'\n')
    return {**inventory, 'notebook': str(path), 'notebook_sha256': digest(path.read_bytes()),
            'code_sha256': digest(code.encode()), 'code_characters': len(code)}
