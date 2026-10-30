"""New-model packaging retains the proven offline inference boundaries."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import unittest
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from scripts.build_kaggle_masked_submission_package import build_masked_submission_package, checkpoint_fingerprint
from tests.test_masked_model_inference import report_fixture


class MaskedPackageTests(unittest.TestCase):
    def test_fingerprint_matches_training_algorithm_and_ignores_extraction_root(self):
        from rsna_knee.training.masked_soft import _checkpoint_fingerprint, MaskedSoftConfig
        from unittest.mock import patch
        report=report_fixture()
        with TemporaryDirectory() as td:
            root=Path(td);source=root/'rsna_knee'/'training';source.mkdir(parents=True)
            runner=Path('src/rsna_knee/training/masked_soft.py').read_bytes();(source/'masked_soft.py').write_bytes(runner)
            archive=io.BytesIO()
            with ZipFile(archive,'w') as z:z.writestr('rsna_knee/training/masked_soft.py',runner)
            raw=archive.getvalue();path=root/'source.zip';path.write_bytes(raw)
            config=MaskedSoftConfig.from_mapping({'epochs':5,'device':'cuda',
                'weights_path':'unused.pth','weights_sha256':report['weights_sha256'],
                'weights_size':report['weights_size']})
            plan={**report}
            with patch('rsna_knee.training.masked_soft._runtime_versions',return_value=report['runtime_versions']):
                expected=_checkpoint_fingerprint(config,plan,source_root=root)
            self.assertEqual(checkpoint_fingerprint(report,path,hashlib.sha256(raw).hexdigest()),expected)

    def test_build_preserves_codec_recovery_csv_and_exact_embedded_bytes(self):
        with TemporaryDirectory() as td:
            root=Path(td);report=report_fixture();rp=root/'report.json';rp.write_text(json.dumps(report))
            source=root/'training.zip'
            with ZipFile(source,'w') as z:z.writestr('rsna_knee/training/masked_soft.py',Path('src/rsna_knee/training/masked_soft.py').read_bytes())
            result=build_masked_submission_package(Path('.'),root/'out',rp,source,
                source_root='/kaggle/input/notebooks/owner/run/data/private/masked-soft-training/run04',
                training_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
            raw=Path(result['archive']).read_bytes()
            for name,limit in [('canary',3),('submission',None)]:
                notebook=json.loads(Path(result[name]).read_bytes());code=''.join(notebook['cells'][1]['source'])
                compile(code,name,'exec')
                self.assertIn('recovery_policy_version',code)
                self.assertIn('pydicom_imported_before_codec_install',code)
                self.assertIn('submission_publication_failed',code)
                self.assertIn('run_masked_inference',code)
                self.assertIn(f'MAX_STUDIES = {limit!r}',code)
                self.assertIn("summary['study_count'] == summary['test_study_count']",code)
                payload=re.search(r"payload = base64.b64decode\('([^']+)'",code).group(1)
                self.assertEqual(base64.b64decode(payload),raw)
                self.assertIn(hashlib.sha256(raw).hexdigest(),code)
                self.assertEqual(notebook['cells'][1]['outputs'],[])
            with ZipFile(io.BytesIO(raw)) as z:
                self.assertIn('rsna_knee/inference/masked.py',z.namelist())
                self.assertEqual(z.read('rsna_knee/inference/fold0.py'),Path('src/rsna_knee/inference/fold0.py').read_bytes())
