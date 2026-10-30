"""Prepare a self-contained source-only, offline fold-zero inference notebook."""
from __future__ import annotations

import argparse
import ast
import base64
import email
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

# Eager legacy imports from dicom_benchmark require the cohort/label source
# closure. These files contain code only; no authentication or labeling runs.
MODULES = (
    'inference/fold0.py', 'inference/offline_codec_bootstrap.py', 'models/resnet18.py', 'models/baseline.py', 'data/dicom_preprocessing.py',
    'data/preprocessing.py', 'data/dicom_benchmark.py', 'validation/cohort.py',
    'validation/multilabel.py', 'data/cohort.py', 'labels/subscription.py',
    'labels/siwc_responses.py', 'labels/siwc_auth.py', 'labels/prompt_comparison.py',
)

CODEC_WHEELS = (
    ('pylibjpeg', '2.1.0', 'pylibjpeg-2.1.0-py3-none-any.whl', 25472,
     '25df9496a69e64e98c887fddee12a1271e275b5f74ba804f9bf98a08bb80993e',
     'https://files.pythonhosted.org/packages/3c/31/86c111e81c113e74ac1b0afde27755088b2db394579370c893d252d9615a/pylibjpeg-2.1.0-py3-none-any.whl', '>=3.10'),
    ('pylibjpeg-libjpeg', '2.4.0', 'pylibjpeg_libjpeg-2.4.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl', 3403812,
     'd2c8da12c79d1f46596b89df94091be01f932d739613e8af93d1d557ac3bab74',
     'https://files.pythonhosted.org/packages/77/00/0a07d56c596793135bc61bcace6267ad68e5a36b9dc53e9fd7f48dac865d/pylibjpeg_libjpeg-2.4.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl', '<4.0,>=3.10'),
    ('pylibjpeg-openjpeg', '2.5.0', 'pylibjpeg_openjpeg-2.5.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl', 1965708,
     'd7e8dd2e1d289d6d1cdc0e9b2df2021277052f72cdabe63771b0ca3ab3b6cc4b',
     'https://files.pythonhosted.org/packages/31/8c/2005b2f3953153a0bcde3339cb3f65546d5715418b92cc6ccd6c0613b0cd/pylibjpeg_openjpeg-2.5.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl', '<4.0,>=3.9'),
    ('pyjpegls', '1.5.1', 'pyjpegls-1.5.1-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl', 2677947,
     'fb56f056868561a1e49995bcbbe7ba8c38d56838d5c975ef5a0ff887d9189230',
     'https://files.pythonhosted.org/packages/48/db/08ce5743ca250750776bf39729d495442966a2ddb23017e61ac43567c5bf/pyjpegls-1.5.1-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl', '>=3.9'),
)
SAFE_BOOTSTRAP_FAILURE_STAGES = {
    'codec_bundle_invalid': 'codec_bundle',
    'codec_bundle_members_invalid': 'codec_bundle',
    'codec_bundle_size_exceeded': 'codec_bundle',
    'codec_manifest_pin_invalid': 'codec_bundle',
    'codec_manifest_missing': 'codec_bundle',
    'codec_manifest_pin_mismatch': 'codec_bundle',
    'codec_manifest_invalid': 'codec_bundle',
    'codec_runtime_incompatible': 'codec_runtime',
    'codec_numpy_requirement_unmet': 'codec_runtime',
    'codec_wheel_integrity_failed': 'codec_bundle',
    'codec_wheel_metadata_invalid': 'codec_bundle',
    'codec_wheel_metadata_mismatch': 'codec_bundle',
    'codec_target_must_be_new': 'codec_install',
    'codec_target_outside_working': 'codec_install',
    'codec_install_failed': 'codec_install',
    'pydicom_imported_before_codec_install': 'codec_order',
    'required_dicom_decoder_unavailable': 'decoder_preflight',
    'source_archive_pin_mismatch': 'source_archive',
    'runtime_archive_collision': 'source_archive',
}


def safe_bootstrap_diagnostic(error):
    """Return only fixed allowlisted reason and stage values for bootstrap failures."""
    candidate = error.args[0] if type(error) in (ValueError, RuntimeError) and len(error.args) == 1 else None
    if not isinstance(candidate, str) or candidate not in SAFE_BOOTSTRAP_FAILURE_STAGES:
        return 'bootstrap', 'bootstrap_failed'
    return SAFE_BOOTSTRAP_FAILURE_STAGES[candidate], candidate


def build_offline_codec_bundle(wheelhouse, output_zip):
    """Create a deterministic wheel ZIP after verifying official PyPI release pins."""
    source = Path(wheelhouse).resolve(strict=True)
    if not source.is_dir():
        raise ValueError('codec_wheelhouse_invalid')
    output = Path(output_zip)
    if output.exists() or output.is_symlink():
        raise ValueError('codec_bundle_output_must_be_new')
    expected_names = {entry[2] for entry in CODEC_WHEELS}
    if {path.name for path in source.iterdir()} != expected_names:
        raise ValueError('codec_wheel_inventory_mismatch')
    entries = []
    wheel_bytes = {}
    total_size = 0
    for name, version, filename, expected_size, expected_sha256, source_url, requires_python in CODEC_WHEELS:
        path = source / filename
        if path.is_symlink():
            raise ValueError('codec_wheel_symlink_forbidden')
        raw = path.read_bytes()
        actual_hash = hashlib.sha256(raw).hexdigest()
        if len(raw) != expected_size or actual_hash != expected_sha256:
            raise ValueError('codec_wheel_official_pin_mismatch')
        with ZipFile(io.BytesIO(raw)) as wheel:
            metadata_names = [item for item in wheel.namelist() if item.endswith('.dist-info/METADATA')]
            if len(metadata_names) != 1:
                raise ValueError('codec_wheel_metadata_invalid')
            metadata = email.message_from_bytes(wheel.read(metadata_names[0]))
        if metadata.get('Name', '').lower().replace('_', '-') != name or metadata.get('Version') != version:
            raise ValueError('codec_wheel_metadata_mismatch')
        entries.append({'name': name, 'version': version, 'filename': filename,
                        'size': len(raw), 'sha256': actual_hash,
                        'requires_dist': metadata.get_all('Requires-Dist', []),
                        'license': metadata.get('License'),
                        'license_expression': metadata.get('License-Expression'),
                        'license_files': metadata.get_all('License-File', []),
                        'license_classifiers': [value for value in metadata.get_all('Classifier', [])
                                               if value.startswith('License ::')],
                        'source_url': source_url,
                        'pypi_release_url': f'https://pypi.org/project/{name}/{version}/',
                        'requires_python': requires_python})
        wheel_bytes[filename] = raw
        total_size += len(raw)
    if total_size > 20 * 1024 * 1024:
        raise ValueError('codec_bundle_size_exceeded')
    manifest = {'schema_version': 'kaggle-offline-codec-bundle-v1',
                'index_url': 'https://pypi.org/simple',
                'target': {'implementation': 'CPython', 'python': '3.13', 'system': 'Linux',
                           'machine': 'x86_64', 'minimum_glibc': '2.17'},
                'packages': entries}
    manifest_bytes = (json.dumps(manifest, sort_keys=True, separators=(',', ':')) + '\n').encode('utf-8')
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    buffer = io.BytesIO()
    with ZipFile(buffer, 'w', compression=ZIP_DEFLATED) as archive:
        for filename, raw in [('manifest.json', manifest_bytes), *wheel_bytes.items()]:
            info = ZipInfo(filename, (2026, 1, 1, 0, 0, 0)); info.compress_type = ZIP_DEFLATED
            archive.writestr(info, raw)
    bundle_bytes = buffer.getvalue()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        stream.write(bundle_bytes)
    return {'manifest_sha256': manifest_hash,
            'archive_sha256': hashlib.sha256(bundle_bytes).hexdigest(),
            'archive_size': len(bundle_bytes), 'uncompressed_wheels_size': total_size,
            'member_count': len(entries) + 1, 'archive': str(output)}


def build_submission_package(repository_root, output_dir, *, source_root=None,
                             checkpoint_sha256=None, checkpoint_size=None, report_sha256=None,
                             codec_manifest_sha256=None, run_inference=False,
                             recovery_policy_version=None):
    repository = Path(repository_root).resolve(strict=True)
    output = Path(output_dir)
    if output.exists() or output.is_symlink(): raise ValueError('preparation_output_must_be_new')
    pins = (checkpoint_sha256, checkpoint_size, report_sha256)
    if any(value is not None for value in pins):
        import re
        if (any(type(value) is not str or not re.fullmatch('[0-9a-f]{64}', value)
                for value in (checkpoint_sha256, report_sha256))
                or type(checkpoint_size) is not int or checkpoint_size <= 0):
            raise ValueError('complete_valid_artifact_pins_required')
    if source_root is not None and (type(source_root) is not str or not source_root.startswith('/kaggle/input/')):
        raise ValueError('source_mount_invalid')
    if (codec_manifest_sha256 is not None
            and (type(codec_manifest_sha256) is not str
                 or len(codec_manifest_sha256) != 64
                 or any(char not in '0123456789abcdef' for char in codec_manifest_sha256))):
        raise ValueError('codec_manifest_pin_invalid')
    if type(run_inference) is not bool:
        raise ValueError('run_inference_flag_invalid')
    if (recovery_policy_version is not None
            and (type(recovery_policy_version) is not int or recovery_policy_version != 1)):
        raise ValueError('recovery_policy_version_invalid')
    paths = {'rsna_knee/__init__.py'}
    for module in MODULES:
        paths.add('rsna_knee/' + module)
        paths.add('rsna_knee/' + module.split('/')[0] + '/__init__.py')
    members = {}
    for name in sorted(paths):
        path = repository / 'src' / name
        if path.is_symlink(): raise ValueError('source_symlink_forbidden')
        members[name] = path.read_bytes()
        ast.parse(members[name], filename=name)
    buffer = io.BytesIO()
    with ZipFile(buffer, 'w', compression=ZIP_DEFLATED) as archive:
        for name, raw in members.items():
            info = ZipInfo(name, (2026, 1, 1, 0, 0, 0)); info.compress_type = ZIP_DEFLATED
            archive.writestr(info, raw)
    raw = buffer.getvalue(); digest = hashlib.sha256(raw).hexdigest()
    inventory = {'schema_version': 'fold0-inference-source-v1', 'source_execution': 361175523,
                 'archive_sha256': digest,
                 'members': {name: hashlib.sha256(value).hexdigest() for name, value in members.items()},
                 'blocking_inputs': ['UI-verified exact version-361175523 notebook-output mount',
                                     'checkpoint SHA-256 and byte size', 'report.json SHA-256'],
                 'remote_executed': False, 'submitted': False}
    code = f'''# Inference only: attach the private output of execution 361175523.
# Keep Internet OFF. Do not attach raw training labels or run training again.
RUN_INFERENCE = {run_inference!r}
SOURCE_ROOT = {source_root!r}  # Exact mounted resnet18-frozen-backbone/run01 directory.
CHECKPOINT_SHA256 = {checkpoint_sha256!r}  # Independently verified own-run artifact pin.
CHECKPOINT_SIZE = {checkpoint_size!r}
REPORT_SHA256 = {report_sha256!r}
CODEC_MANIFEST_SHA256 = {codec_manifest_sha256!r}
RECOVERY_POLICY_VERSION = {recovery_policy_version!r}  # None keeps strict behavior; 1 opts into bounded recovery.
SAFE_BOOTSTRAP_FAILURE_STAGES = {SAFE_BOOTSTRAP_FAILURE_STAGES!r}
COMPETITION_ROOT = '/kaggle/input/competitions/rsna-knee-abnormality-detection'

if RUN_INFERENCE:
    import base64, hashlib, json, os, sys
    from pathlib import Path
    if not SOURCE_ROOT or not CHECKPOINT_SHA256 or not CHECKPOINT_SIZE or not REPORT_SHA256:
        raise ValueError('verified_source_mount_and_artifact_pins_required')
    source = Path(SOURCE_ROOT)
    if not source.is_absolute() or not source.is_relative_to('/kaggle/input'):
        raise ValueError('source_mount_invalid')
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    payload = base64.b64decode('{base64.b64encode(raw).decode()}', validate=True)
    if hashlib.sha256(payload).hexdigest() != '{digest}':
        raise ValueError('source_archive_pin_mismatch')
    runtime_archive = Path('/kaggle/working/fold0-inference-source.zip')
    if runtime_archive.exists():
        if hashlib.sha256(runtime_archive.read_bytes()).hexdigest() != '{digest}':
            raise ValueError('runtime_archive_collision')
    else:
        with runtime_archive.open('xb') as stream:
            stream.write(payload)
    sys.path.insert(0, str(runtime_archive))
    if CODEC_MANIFEST_SHA256:
        if 'pydicom' in sys.modules:
            raise RuntimeError('pydicom_imported_before_codec_install')
        from rsna_knee.inference.offline_codec_bootstrap import install_codec_bundle
        codec_root = Path('/kaggle/input/datasets/davidsulleiro/rsna-knee-offline-dicom-codecs')
        codec_archive = codec_root / 'rsna-knee-offline-dicom-codecs.zip'
        codec_location = codec_root if (codec_root / 'manifest.json').is_file() else codec_archive
        codec_result = install_codec_bundle(codec_location, CODEC_MANIFEST_SHA256,
                                            Path('/kaggle/working/fold0-codec-target'),
                                            working_root=Path('/kaggle/working'))
        sys.path.insert(0, codec_result['target'])
        import pydicom
        from pydicom.pixels import get_decoder
        for codec, uid in (('JPEG Lossless P14', '1.2.840.10008.1.2.4.57'),
                           ('JPEG Lossless SV1', '1.2.840.10008.1.2.4.70'),
                           ('JPEG-LS Lossless', '1.2.840.10008.1.2.4.80'),
                           ('JPEG-LS Near-Lossless', '1.2.840.10008.1.2.4.81'),
                           ('JPEG 2000 Lossless', '1.2.840.10008.1.2.4.90')):
            if get_decoder(uid).is_available is not True:
                raise RuntimeError('required_dicom_decoder_unavailable')
        print(json.dumps({{'event': 'offline_codec_bootstrap', 'stage': 'ready',
                          'packages': codec_result['packages']}}, sort_keys=True), flush=True)
    from rsna_knee.inference.fold0 import run_inference
    import platform, sysconfig
    from itertools import islice
    runtime_info = {{'python_version': list(sys.version_info[:3]),
                    'python_implementation': platform.python_implementation(),
                    'system': platform.system(), 'machine': platform.machine(),
                    'sysconfig_platform': sysconfig.get_platform(),
                    'python_soabi': sysconfig.get_config_var('SOABI')}}
    try:
        from packaging.tags import sys_tags
        runtime_info['wheel_tag_candidates'] = [str(tag) for tag in islice(sys_tags(), 8)]
    except Exception:
        runtime_info['wheel_tag_candidates'] = []
    decoder_status = []
    try:
        import pydicom
        from pydicom.pixels import get_decoder
        runtime_info['pydicom_version'] = pydicom.__version__
        for label, uid in (('JPEG Lossless P14', '1.2.840.10008.1.2.4.57'),
                           ('JPEG Lossless SV1', '1.2.840.10008.1.2.4.70'),
                           ('JPEG-LS Lossless', '1.2.840.10008.1.2.4.80'),
                           ('JPEG-LS Near-Lossless', '1.2.840.10008.1.2.4.81'),
                           ('JPEG 2000 Lossless', '1.2.840.10008.1.2.4.90')):
            try:
                decoder = get_decoder(uid)
                available = decoder.is_available
                plugins = sorted(decoder.available_plugins)
                state = 'available' if available is True else 'unavailable' if available is False else 'unknown'
            except Exception:
                state = 'unknown'
                plugins = []
            decoder_status.append({{'codec': label, 'availability': state, 'plugins': plugins}})
    except Exception:
        runtime_info['pydicom_version'] = 'unknown'
        decoder_status = [{{'codec': label, 'availability': 'unknown', 'plugins': []}}
                          for label in ('JPEG Lossless P14', 'JPEG Lossless SV1',
                                        'JPEG-LS Lossless', 'JPEG-LS Near-Lossless',
                                        'JPEG 2000 Lossless')]
    print(json.dumps({{'event': 'dicom_decoder_availability', 'runtime': runtime_info,
                      'codecs': decoder_status}}, sort_keys=True), flush=True)
    summary = run_inference(COMPETITION_ROOT, source, CHECKPOINT_SHA256,
                            CHECKPOINT_SIZE, REPORT_SHA256,
                            '/kaggle/working/fold0-inference-run01', device='cuda',
                            recovery_policy_version=RECOVERY_POLICY_VERSION)
    result = Path('/kaggle/working/fold0-inference-run01/submission.csv')
    destination = Path('/kaggle/working/submission.csv')
    with destination.open('xb') as stream:
        destination_created = True
        stream.write(result.read_bytes())
    if destination.read_bytes() != result.read_bytes():
        raise ValueError('submission_publication_readback_failed')
    print(json.dumps(summary, sort_keys=True))
else:
    print('Inference disabled: verify the exact fold-zero mount and artifact pins before enabling.')
'''
    # Imports and package setup happen before the inference module can emit diagnostics.
    bootstrap_start = code.index('    if not SOURCE_ROOT')
    bootstrap_end = code.index('    summary = run_inference')
    bootstrap = code[bootstrap_start:bootstrap_end]
    guarded = (
        "    print(json.dumps({'event': 'inference_bootstrap', 'stage': 'startup'}), flush=True)\n"
        "    try:\n" + ''.join('    ' + line for line in bootstrap.splitlines(keepends=True)) +
        "    except Exception as error:\n"
        "        candidate = error.args[0] if type(error) in (ValueError, RuntimeError) and len(error.args) == 1 else None\n"
        "        if not isinstance(candidate, str) or candidate not in SAFE_BOOTSTRAP_FAILURE_STAGES:\n"
        "            candidate = None\n"
        "        stage = SAFE_BOOTSTRAP_FAILURE_STAGES.get(candidate, 'bootstrap')\n"
        "        reason_code = candidate or 'bootstrap_failed'\n"
        "        family = 'dependency' if isinstance(error, ImportError) else 'other'\n"
        "        print(json.dumps({'event': 'inference_failed', 'stage': stage, "
        "'reason_code': reason_code, 'exception_family': family}), flush=True)\n"
        "        raise RuntimeError('inference_bootstrap_failed') from None\n"
    )
    code = code[:bootstrap_start] + guarded + code[bootstrap_end:]
    publication_start = code.index("    result = Path('/kaggle/working/fold0-inference-run01/submission.csv')")
    publication_end = code.index('    print(json.dumps(summary, sort_keys=True))')
    publication = code[publication_start:publication_end]
    guarded_publication = (
        "    destination_created = False\n"
        "    try:\n" + ''.join('    ' + line for line in publication.splitlines(keepends=True)) +
        "    except Exception:\n"
        "        if destination_created:\n"
        "            try:\n"
        "                destination.unlink(missing_ok=True)\n"
        "            except OSError:\n"
        "                pass\n"
        "        print(json.dumps({'event': 'inference_failed', 'stage': 'publication', "
        "'reason_code': 'submission_publication_failed'}), flush=True)\n"
        "        raise RuntimeError('submission_publication_failed') from None\n"
    )
    code = code[:publication_start] + guarded_publication + code[publication_end:]
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                'cells': [
                    {'cell_type': 'markdown', 'id': 'protocol', 'metadata': {}, 'source': [
                        '# RSNA Knee: frozen fold-zero inference\n',
                        'Existing checkpoint only, execution 361175523. No retraining, ensemble, or thresholding. '
                        'Attach the official competition and the exact private notebook-output version. '
                        'Internet off; GPU on. Verify source mount, checkpoint hash/size and report hash before enabling. '
                        'The embedded archive contains source code only and performs no network requests.\n']},
                    {'cell_type': 'code', 'id': 'inference', 'metadata': {}, 'execution_count': None,
                     'outputs': [], 'source': code.splitlines(keepends=True)}]}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'fold0-inference-source.zip').write_bytes(raw)
    (output / 'inventory.json').write_text(json.dumps(inventory, indent=2) + '\n', encoding='utf-8')
    notebook_path = output / 'kaggle_fold0_submission.ipynb'
    notebook_path.write_text(json.dumps(notebook, indent=2) + '\n', encoding='utf-8')
    return {'notebook': str(notebook_path), 'archive_sha256': digest,
            'notebook_sha256': hashlib.sha256(notebook_path.read_bytes()).hexdigest(), 'member_count': len(members)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(build_submission_package(Path(__file__).resolve().parents[1], arguments.output_dir), indent=2))
