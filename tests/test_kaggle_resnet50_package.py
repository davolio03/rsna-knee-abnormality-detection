"""Offline package contract: preserved legacy source and explicit benchmark-only execution."""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zipfile

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/build_kaggle_resnet50_package.py'

class PackageTests(unittest.TestCase):
    def test_package_is_offline_pinned_and_plan_default(self):
        spec=importlib.util.spec_from_file_location('resnet50_package',SCRIPT)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        root=SCRIPT.parents[1]
        with TemporaryDirectory() as td:
            result=module.build_package(root,Path(td)/'new',private_root=Path(td))
            upload=Path(result['upload_dir'])
            self.assertEqual({p.name for p in upload.iterdir()},{'resnet50-recipe-assets.json','resnet50-imagenet1k-v1.pth','resnet50-radimagenet.pt'})
            old=root/'data/private/masked-soft-training-preparation/run09/masked-training-source.zip'
            with zipfile.ZipFile(old) as a,zipfile.ZipFile(result['archive_path']) as b:
                for name in a.namelist():self.assertEqual(a.read(name),b.read(name))
                self.assertEqual(len(b.namelist()),len(a.namelist())+2)
            for arm in ('imagenet','radimagenet'):
                for stage in ('plan','benchmark'):
                    path=Path(result[f'{arm}_{stage}'])
                    code=''.join(json.loads(path.read_text())['cells'][0]['source'])
                    compile(code,str(path),'exec')
                    self.assertIn(f'ARM = "{arm}"',code)
                    self.assertIn('EXECUTE_BENCHMARK = '+str(stage=='benchmark'),code)
                    self.assertIn("mode='benchmark'",code)
                    self.assertNotIn('run_masked_soft_training(config',code)
            with self.assertRaises(ValueError):module.build_package(root,Path(result['output_dir']),private_root=Path(td))

    def test_full_radimagenet_variant_preserves_payload_and_is_one_complete_epoch(self):
        spec=importlib.util.spec_from_file_location('resnet50_package_full',SCRIPT)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        root=SCRIPT.parents[1]
        original=root/'data/private/resnet50-pretraining-preparation/run01/kaggle_resnet50_radimagenet_benchmark.ipynb'
        original_bytes=original.read_bytes()
        with TemporaryDirectory() as td:
            result=module.build_full_radimagenet_notebook(root,Path(td)/'full',private_root=Path(td))
            code=''.join(json.loads(Path(result['notebook_path']).read_text())['cells'][0]['source'])
            namespace={'__name__':'not_main'};exec(compile(code,'full.py','exec'),namespace)
            old={};exec(compile(''.join(json.loads(original_bytes)['cells'][0]['source']),'benchmark.py','exec'),{'__name__':'not_main'},old)
            self.assertEqual(namespace['NEW_SOURCE_B64'],old['NEW_SOURCE_B64'])
            self.assertEqual(namespace['NEW_SOURCE_SHA256'],old['NEW_SOURCE_SHA256'])
            self.assertEqual(namespace['NEW_INVENTORY_SHA256'],old['NEW_INVENTORY_SHA256'])
            self.assertEqual(namespace['ARM'],'radimagenet');self.assertTrue(namespace['EXECUTE_FULL_TRAINING'])
            self.assertIn("mode='train'",code);self.assertNotIn("mode='benchmark'",code)
            self.assertIn('max_epochs_per_run=1',code)
            self.assertIn('resume_checkpoint=None,resume_sha256=None',code)
            self.assertIn("/'full-run01'",code)
            self.assertNotIn('max_studies=',code)
            self.assertEqual(original.read_bytes(),original_bytes)
            with self.assertRaises(ValueError):module.build_full_radimagenet_notebook(root,Path(result['output_dir']),private_root=Path(td))
