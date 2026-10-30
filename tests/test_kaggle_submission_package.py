"""Source-only offline submission notebook preparation tests."""
import base64
import contextlib
import textwrap
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from scripts.build_kaggle_submission_package import build_submission_package, safe_bootstrap_diagnostic


class SubmissionPackageTests(unittest.TestCase):
    def test_bootstrap_failure_diagnostics_allow_only_fixed_codes_and_stages(self):
        self.assertEqual(safe_bootstrap_diagnostic(ValueError('codec_wheel_metadata_mismatch')),
                         ('codec_bundle', 'codec_wheel_metadata_mismatch'))
        diagnostic = safe_bootstrap_diagnostic(ValueError(r'C:\private\secret\pip-log'))
        self.assertEqual(diagnostic, ('bootstrap', 'bootstrap_failed'))
        self.assertNotIn('secret', repr(diagnostic))

    def test_submission_notebook_reports_decoder_availability_before_inference(self):
        with TemporaryDirectory() as directory:
            result = build_submission_package(Path(__file__).resolve().parents[1], Path(directory) / 'preparation',
                source_root='/kaggle/input/notebooks/owner/fold0/resnet18-frozen-backbone/run01',
                checkpoint_sha256='a' * 64, checkpoint_size=123, report_sha256='b' * 64)
            code = ''.join(json.loads(Path(result['notebook']).read_text())['cells'][1]['source'])
            self.assertLess(code.index('dicom_decoder_availability'), code.index('summary = run_inference'))
            self.assertIn('JPEG Lossless SV1', code)
            self.assertIn('JPEG Lossless P14', code)
            self.assertIn('JPEG-LS Lossless', code)
            self.assertIn('JPEG-LS Near-Lossless', code)
            self.assertIn('JPEG 2000 Lossless', code)
            self.assertIn('get_decoder', code)
            self.assertIn('sysconfig.get_platform()', code)
            self.assertIn("sysconfig.get_config_var('SOABI')", code)
            self.assertIn('pydicom.__version__', code)
            self.assertIn("'/kaggle/working/submission.csv'", code)
            self.assertIn('RUN_INFERENCE = False', code)
            self.assertNotIn('pip install', code.lower())

    def test_explicit_verified_pins_remain_disabled(self):
        with TemporaryDirectory() as directory:
            result = build_submission_package(Path(__file__).resolve().parents[1], Path(directory) / 'pins',
                source_root='/kaggle/input/notebooks/owner/fold0/resnet18-frozen-backbone/run01',
                checkpoint_sha256='a' * 64, checkpoint_size=123, report_sha256='b' * 64)
            code = ''.join(json.loads(Path(result['notebook']).read_text())['cells'][1]['source'])
            self.assertIn("CHECKPOINT_SHA256 = '" + 'a' * 64 + "'", code)
            self.assertIn('RUN_INFERENCE = False', code)
            self.assertIn('inference_bootstrap', code)
            with self.assertRaises(ValueError):
                build_submission_package(Path(__file__).resolve().parents[1], Path(directory) / 'bad',
                                         checkpoint_sha256='not_a_pin')

    def test_recovery_candidate_is_explicitly_versioned_and_pins_source(self):
        with TemporaryDirectory() as directory:
            result = build_submission_package(
                Path(__file__).resolve().parents[1], Path(directory) / 'recovery-v1',
                source_root='/kaggle/input/notebooks/owner/fold0/resnet18-frozen-backbone/run01',
                checkpoint_sha256='a' * 64, checkpoint_size=123, report_sha256='b' * 64,
                codec_manifest_sha256='c' * 64, run_inference=True,
                recovery_policy_version=1,
            )
            notebook = json.loads(Path(result['notebook']).read_text())
            code = ''.join(notebook['cells'][1]['source'])
            compile(code, 'recovery-v1-notebook', 'exec')
            self.assertIn('RECOVERY_POLICY_VERSION = 1', code)
            self.assertIn('recovery_policy_version=RECOVERY_POLICY_VERSION', code)
            self.assertEqual(notebook['cells'][1]['outputs'], [])
            with ZipFile(Path(result['notebook']).parent / 'fold0-inference-source.zip') as archive:
                fold0 = archive.read('rsna_knee/inference/fold0.py').decode()
                self.assertIn('recovery-journal.json', fold0)
                self.assertEqual(hashlib.sha256(archive.read('rsna_knee/inference/fold0.py')).hexdigest(),
                                 json.loads((Path(result['notebook']).parent / 'inventory.json').read_text())['members']['rsna_knee/inference/fold0.py'])

    def test_codec_bootstrap_is_pinned_before_pydicom_and_inference_imports(self):
        with TemporaryDirectory() as directory:
            result = build_submission_package(Path(__file__).resolve().parents[1], Path(directory) / 'run07',
                source_root='/kaggle/input/notebooks/owner/fold0/resnet18-frozen-backbone/run01',
                checkpoint_sha256='a' * 64, checkpoint_size=123, report_sha256='b' * 64,
                codec_manifest_sha256='c' * 64, run_inference=True)
            code = ''.join(json.loads(Path(result['notebook']).read_text())['cells'][1]['source'])
            compile(code, 'run07-notebook', 'exec')
            self.assertIn('RUN_INFERENCE = True', code)
            self.assertIn("CODEC_MANIFEST_SHA256 = '" + 'c' * 64 + "'", code)
            self.assertLess(code.index("if 'pydicom' in sys.modules"), code.index('install_codec_bundle'))
            self.assertLess(code.index('install_codec_bundle'), code.index('import pydicom'))
            self.assertLess(code.index('required_dicom_decoder_unavailable'),
                            code.index('from rsna_knee.inference.fold0 import run_inference'))
            self.assertLess(code.index('from rsna_knee.inference.fold0 import run_inference'),
                            code.index('summary = run_inference'))
            self.assertNotIn('files.pythonhosted.org/packages/', code)
            self.assertIn("'codec_wheel_metadata_mismatch': 'codec_bundle'", code)
            self.assertNotIn('str(error)', code)
            archive = Path(result['notebook']).parent / 'fold0-inference-source.zip'
            with ZipFile(archive) as source:
                self.assertIn('rsna_knee/inference/offline_codec_bootstrap.py', source.namelist())
                helper = source.read('rsna_knee/inference/offline_codec_bootstrap.py').decode()
                self.assertIn('"--no-index"', helper)
                self.assertIn('"--no-deps"', helper)
                self.assertTrue(all(name.endswith('.py') for name in source.namelist()))
                self.assertFalse(any(name.endswith('.whl') for name in source.namelist()))

    def test_invalid_codec_manifest_pin_is_rejected(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'codec_manifest_pin_invalid'):
                build_submission_package(Path(__file__).resolve().parents[1], Path(directory) / 'bad',
                                         codec_manifest_sha256='not-a-sha')

    def test_disabled_notebook_and_exact_source_inventory(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / 'preparation'
            result = build_submission_package(Path(__file__).resolve().parents[1], output)
            notebook = json.loads(Path(result['notebook']).read_text())
            code = ''.join(notebook['cells'][1]['source'])
            compile(code, 'submission-notebook', 'exec')
            self.assertIn('RUN_INFERENCE = False', code)
            self.assertIn('CHECKPOINT_SHA256 = None', code)
            self.assertIn('REPORT_SHA256 = None', code)
            self.assertIn('SOURCE_ROOT = None', code)
            self.assertNotIn('extractall', code)
            self.assertTrue(all(not cell.get('outputs') for cell in notebook['cells']))
            inventory = json.loads((output / 'inventory.json').read_text())
            with ZipFile(output / 'fold0-inference-source.zip') as archive:
                self.assertEqual(set(archive.namelist()), set(inventory['members']))
                for name in archive.namelist():
                    self.assertTrue(name.endswith('.py'))
                    self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), inventory['members'][name])
            probe = subprocess.run([sys.executable, '-I', '-c',
                                    'import sys; sys.path.insert(0, sys.argv[1]); '
                                    'import rsna_knee.inference.fold0 as m; assert len(m.TARGETS) == 12',
                                    str(output / 'fold0-inference-source.zip')], capture_output=True, text=True)
            self.assertEqual(probe.returncode, 0, probe.stderr)
            exec(compile(code, 'disabled', 'exec'), {})
            with self.assertRaises(ValueError): build_submission_package(Path(__file__).resolve().parents[1], output)


    def test_failed_publication_removes_only_its_new_csv(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            result = build_submission_package(Path(__file__).resolve().parents[1], root / 'prep')
            code = ''.join(json.loads(Path(result['notebook']).read_text())['cells'][1]['source'])
            start = code.index('    destination_created = False\n    try:\n')
            end = code.index('    print(json.dumps(summary, sort_keys=True))')
            publication = textwrap.dedent(code[start:end]).replace('/kaggle/working/fold0-inference-run01/submission.csv', (root / 'source.csv').as_posix()).replace('/kaggle/working/submission.csv', (root / 'submission.csv').as_posix())
            output = io.StringIO()
            with patch.object(Path, 'read_bytes', side_effect=OSError('SECRET_PATH')), contextlib.redirect_stdout(output):
                with self.assertRaisesRegex(RuntimeError, '^submission_publication_failed$'):
                    exec(publication, {'Path': Path, 'json': json})
            self.assertFalse((root / 'submission.csv').exists())
            self.assertNotIn('SECRET', output.getvalue())
            (root / 'submission.csv').write_text('KEEP')
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                exec(publication, {'Path': Path, 'json': json})
            self.assertEqual((root / 'submission.csv').read_text(), 'KEEP')


if __name__ == '__main__': unittest.main()
