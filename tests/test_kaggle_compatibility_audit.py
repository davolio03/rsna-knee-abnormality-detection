"""Private source-only CPU audit notebook packaging tests."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZipFile
from tests.test_compatibility_audit import fixture
from rsna_knee.data.compatibility_audit import build_audit_plan, build_dimension_candidate_plan
from scripts.build_kaggle_compatibility_audit import build_compatibility_package


class CompatibilityPackageTests(unittest.TestCase):
    def test_frozen_source_and_id_only_plan_disabled_no_models(self):
        repo=Path(__file__).resolve().parents[1]
        with TemporaryDirectory(dir=repo) as directory:
            root=Path(directory);series,gold,prior,pins=fixture(root)
            plan=build_audit_plan(series,gold,prior,pins,sample_size=3)
            result=build_compatibility_package(repo,root/'prep',plan)
            nb=json.loads(Path(result['notebook']).read_text())
            code=''.join(nb['cells'][1]['source']);compile(code,'compatibility','exec')
            self.assertIn('RUN_AUDIT = False',code)
            self.assertIn("device is CPU only",code)
            self.assertNotIn('torch.load',code);self.assertNotIn('extractall',code)
            self.assertTrue(all(not c.get('outputs') for c in nb['cells']))
            with ZipFile(root/'prep'/'compatibility-source.zip') as archive:
                self.assertFalse(any('/models/' in n or '/inference/' in n for n in archive.namelist()))
                inv=json.loads((root/'prep'/'inventory.json').read_text())
                self.assertEqual(set(inv['members']),set(archive.namelist()))
                for name in archive.namelist():
                    self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(),inv['members'][name])
            probe=subprocess.run([sys.executable,'-I','-c',"import sys;sys.path.insert(0,sys.argv[1]);import rsna_knee.data.compatibility_audit;assert 'torch' not in sys.modules",str(root/'prep'/'compatibility-source.zip')],capture_output=True,text=True)
            self.assertEqual(probe.returncode,0,probe.stderr)
            self.assertNotIn('MUST_NOT_DECODE',(root/'prep'/'audit-plan.json').read_text())
            with self.assertRaises(ValueError): build_compatibility_package(repo,root/'prep',plan)

    def test_dimension_candidate_package_is_disabled_and_scoped_to_twenty(self):
        repo=Path(__file__).resolve().parents[1]
        with TemporaryDirectory(dir=repo) as directory:
            root=Path(directory);series,_,_,pins=fixture(root)
            pins.update(audit_plan='a'*64,audit_report='b'*64)
            rows=['StudyInstanceUID,SeriesInstanceUID,Anatomical_Plane','gold,g0,Sagittal','prior,p0,Coronal']
            studies=[];ordinals=[]
            for i in range(20):
                study=f's{i:02d}';series_id=f'x{i:02d}'
                rows.append(f'{study},{series_id},Axial')
                studies.append({'study_id':study,'series':[{'series_id':series_id,'plane':'Axial'}]})
                ordinals.append({'study_ordinal':i+1,'series_ordinal':1})
            series.write_text('\n'.join(rows)+'\n')
            pins['series']=hashlib.sha256(series.read_bytes()).hexdigest()
            source_plan={'schema_version':'training-header-inventory-v1','source_sha256':pins,
                'excluded_ids':['gold','prior'],'studies':studies}
            source_path=root/'header-plan.json';source_path.write_text(json.dumps(source_plan))
            report={'schema_version':'training-header-inventory-v1','status':'complete',
                'termination_reason':'all_planned_series_attempted','plan_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),
                'planned_studies':20,'planned_series':20,'attempted_series':20,'completed_series':20,'failed_series':0,
                'flag_counts':{'uncovered_dimensions':64},'candidate_ordinals':{'uncovered_dimensions':ordinals}}
            report_path=root/'header-report.json';report_path.write_text(json.dumps(report))
            with patch('rsna_knee.data.compatibility_audit.TRUSTED_DIMENSION_REPORT_SHA256',hashlib.sha256(report_path.read_bytes()).hexdigest()), \
                    patch('rsna_knee.data.compatibility_audit.TRUSTED_HEADER_PLAN_SHA256',hashlib.sha256(source_path.read_bytes()).hexdigest()):
                plan=build_dimension_candidate_plan(report_path,source_path)
            result=build_compatibility_package(repo,root/'prep',plan)
            nb=json.loads(Path(result['notebook']).read_text())
            code=''.join(nb['cells'][1]['source'])
            self.assertIn('RUN_AUDIT = False',code)
            self.assertIn('exactly 20 preselected series', ''.join(nb['cells'][0]['source']))
            self.assertIn('does not audit all 64 flagged series', ''.join(nb['cells'][0]['source']))
            self.assertIn('/kaggle/working/training-dimension-candidates-run01',code)
            self.assertTrue(all(not c.get('outputs') for c in nb['cells']))
            with ZipFile(root/'prep'/'compatibility-source.zip') as archive:
                self.assertIn('rsna_knee/data/compatibility_audit.py',archive.namelist())
            self.assertEqual(json.loads((root/'prep'/'audit-plan.json').read_text()),plan)
