"""Source-only private disabled header inventory package proofs."""
import base64
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile
from tests.test_header_inventory import fixture, REPO
from scripts.build_kaggle_header_inventory import build_header_package


class HeaderPackageTests(unittest.TestCase):
    def test_disabled_pins_compressed_private_plan_and_isolated_model_free_import(self):
        with TemporaryDirectory(dir=REPO) as d:
            root=Path(d);args,plan=fixture(root)
            result=build_header_package(REPO,root/'prep',plan)
            nb=json.loads(Path(result['notebook']).read_text());code=''.join(nb['cells'][1]['source'])
            compile(code,'header-notebook','exec')
            self.assertIn('RUN_INVENTORY = False',code)
            self.assertTrue(all(not c.get('outputs') and c.get('execution_count') is None for c in nb['cells']))
            self.assertNotIn('torch.load',code);self.assertNotIn('extractall',code)
            with ZipFile(root/'prep/header-source.zip') as archive:
                self.assertEqual(archive.namelist(),['header_inventory.py'])
                self.assertEqual(hashlib.sha256(archive.read('header_inventory.py')).hexdigest(),result['members']['header_inventory.py'])
            raw=(root/'prep/header-source.zip').read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(),result['archive_sha256'])
            frozen=(root/'prep/header-plan.json').read_bytes()
            self.assertEqual(json.loads(frozen),plan)
            self.assertIn(base64.b64encode(gzip.compress(frozen,mtime=0)).decode(),code)
            probe=subprocess.run([sys.executable,'-I','-c',"import sys;sys.path.insert(0,sys.argv[1]);import header_inventory;assert 'torch' not in sys.modules;assert 'pydicom' not in sys.modules;assert 'rsna_knee' not in sys.modules",str(root/'prep/header-source.zip')],capture_output=True,text=True)
            self.assertEqual(probe.returncode,0,probe.stderr)
            self.assertNotIn('MUST_NOT_DECODE',frozen.decode())
            with self.assertRaises(ValueError):build_header_package(REPO,root/'prep',plan)
