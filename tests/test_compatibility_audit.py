"""Synthetic-only fresh-study sampling and bounded preprocessing audit tests."""
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
import rsna_knee.data.compatibility_audit as compatibility_audit
from rsna_knee.data.compatibility_audit import (build_audit_plan, build_dimension_candidate_plan,
    validate_plan, run_compatibility_audit)
from rsna_knee.data.dicom_preprocessing import DicomPreprocessingError


def fixture(root):
    series = root / 'train_series.csv'
    series.write_text('StudyInstanceUID,SeriesInstanceUID,Anatomical_Plane\ngold,g0,Sagittal\nprior,p0,Coronal\na,a0,Sagittal\na,a1,Axial\nb,b0,Coronal\nc,c0,Axial\n')
    gold = root / 'gold.csv'; gold.write_text('StudyInstanceUID\ngold\n')
    manifest = root / 'manifest.json'
    manifest.write_text(json.dumps({'records': [{'study_id': 'prior', 'series': [{'series_id': 'p0'}], 'labels': 'MUST_NOT_DECODE'}]}))
    pins = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in [('series',series),('gold',gold),('prior',manifest)]}
    return series,gold,manifest,pins


class CompatibilityAuditTests(unittest.TestCase):
    def test_dimension_candidate_plan_pins_exact_twenty_and_rejects_tampering(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as d:
            root=Path(d)
            source,_,_,pins=fixture(root)
            pins.update(audit_plan='a'*64,audit_report='b'*64)
            studies=[]
            ordinals=[]
            metadata=['StudyInstanceUID,SeriesInstanceUID,Anatomical_Plane','gold,g0,Sagittal','prior,p0,Coronal']
            for i in range(20):
                study=f's{i:02d}';series_id=f'x{i:02d}'
                studies.append({'study_id':study,'series':[{'series_id':series_id,'plane':'Axial'},
                    {'series_id':f'extra{i:02d}','plane':'Coronal'}]})
                metadata.extend([f'{study},{series_id},Axial',f'{study},extra{i:02d},Coronal'])
                (root/'images'/study/series_id).mkdir(parents=True)
                (root/'images'/study/f'extra{i:02d}').mkdir(parents=True)
                ordinals.append({'study_ordinal':i+1,'series_ordinal':1})
            source.write_text('\n'.join(metadata)+'\n')
            pins['series']=hashlib.sha256(source.read_bytes()).hexdigest()
            header_plan={'schema_version':'training-header-inventory-v1','source_sha256':pins,
                'excluded_ids':['gold','prior'],'studies':studies}
            report={'schema_version':'training-header-inventory-v1','status':'complete','termination_reason':'all_planned_series_attempted',
                'plan_sha256':'a'*64,'planned_studies':20,'planned_series':40,'attempted_series':40,'completed_series':40,'failed_series':0,
                'flag_counts':{'uncovered_dimensions':64},'candidate_ordinals':{'uncovered_dimensions':ordinals}}
            plan_path=root/'header-plan.json';plan_path.write_text(json.dumps(header_plan))
            report['plan_sha256']=hashlib.sha256(plan_path.read_bytes()).hexdigest()
            report_path=root/'header-report.json';report_path.write_text(json.dumps(report))
            with patch.object(compatibility_audit,'TRUSTED_DIMENSION_REPORT_SHA256',hashlib.sha256(report_path.read_bytes()).hexdigest()), \
                    patch.object(compatibility_audit,'TRUSTED_HEADER_PLAN_SHA256',hashlib.sha256(plan_path.read_bytes()).hexdigest()):
                plan=build_dimension_candidate_plan(report_path,plan_path)
            self.assertEqual(plan['candidate_count'],20)
            self.assertEqual(len(plan['candidates']),20)
            self.assertEqual(plan['source_sha256']['series'],pins['series'])
            calls=[]
            def adapter(path,**kwargs):
                calls.append(path.name)
                arr=np.ones((1,1,256,256),np.float32)
                return SimpleNamespace(images=arr,valid_mask=arr.astype(bool))
            with contextlib.redirect_stdout(io.StringIO()):
                result=run_compatibility_audit(plan,source,root/'images',root/'out',
                    preprocessor=adapter,dcmread=lambda *a,**k:None)
            self.assertEqual(result['planned_series'],20)
            self.assertEqual(result['attempted_series'],20)
            self.assertEqual(result['completed_studies'],20)
            self.assertEqual(len(calls),20)
            self.assertNotIn('extra00',calls)
            for tamper in ('report_hash','duplicate','bounds','exclusion'):
                if tamper=='report_hash':
                    with patch.object(compatibility_audit,'TRUSTED_DIMENSION_REPORT_SHA256','0'*64), \
                            patch.object(compatibility_audit,'TRUSTED_HEADER_PLAN_SHA256',hashlib.sha256(plan_path.read_bytes()).hexdigest()), \
                            self.assertRaises(ValueError):
                        build_dimension_candidate_plan(report_path,plan_path)
                else:
                    altered=json.loads(json.dumps(plan))
                    if tamper=='duplicate': altered['candidates'][1]=altered['candidates'][0]
                    elif tamper=='bounds': altered['candidates'][0]['study_ordinal']=0
                    else: altered['candidates'][0]['study_id']='gold'
                    with self.assertRaises(ValueError): validate_plan(altered)

    def test_selection_is_deterministic_fresh_and_all_series(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as d:
            series,gold,prior,pins=fixture(Path(d))
            plan=build_audit_plan(series,gold,prior,pins,sample_size=3)
            self.assertEqual(plan,build_audit_plan(series,gold,prior,pins,sample_size=3))
            self.assertEqual({s['study_id'] for s in plan['studies']},{'a','b','c'})
            self.assertEqual(sum(len(s['series']) for s in plan['studies']),4)
            self.assertEqual(plan['fresh_study_count'],3)
            validate_plan(plan)
            for change in ('overlap','duplicate','traversal'):
                altered=json.loads(json.dumps(plan))
                if change=='overlap': altered['gold_ids'].append(altered['studies'][0]['study_id'])
                elif change=='duplicate': altered['studies'].append(altered['studies'][0])
                else: altered['studies'][0]['series'][0]['series_id']='../secret'
                with self.assertRaises(ValueError): validate_plan(altered)
            with self.assertRaises(ValueError): build_audit_plan(series,gold,prior,{**pins,'gold':'0'*64},sample_size=3)

    def test_duplicate_or_missing_exclusions_fail(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as d:
            root=Path(d); series,gold,prior,pins=fixture(root)
            for text in ['StudyInstanceUID\ngold\ngold\n','StudyInstanceUID\nabsent\n']:
                gold.write_text(text); pins['gold']=hashlib.sha256(gold.read_bytes()).hexdigest()
                with self.assertRaises(ValueError): build_audit_plan(series,gold,prior,pins,sample_size=1)

    def test_audit_continues_safe_failures_and_checkpoints(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as d:
            root=Path(d); series,gold,prior,pins=fixture(root)
            plan=build_audit_plan(series,gold,prior,pins,sample_size=3)
            for study in plan['studies']:
                for row in study['series']: (root/'train_series'/study['study_id']/row['series_id']).mkdir(parents=True)
            calls=[]
            def adapter(path,**kwargs):
                calls.append(kwargs)
                self.assertEqual(kwargs['max_slices'],8); self.assertEqual(kwargs['output_size'],256)
                if len(calls)==2: raise DicomPreprocessingError('pixel_decode_failed','SECRET/path/UID')
                if len(calls)==3: raise RuntimeError('SECRET/path/UID')
                arr=np.ones((1,1,256,256),np.float32)
                return SimpleNamespace(images=arr,valid_mask=arr.astype(bool))
            captured=io.StringIO()
            with contextlib.redirect_stdout(captured):
                report=run_compatibility_audit(plan,series,root/'train_series',root/'out',preprocessor=adapter,dcmread=lambda *a,**k:None)
            self.assertEqual((report['status'],report['attempted_series'],report['successful_series'],report['failed_series']),('complete',4,2,2))
            self.assertEqual(report['reason_counts'],{'pixel_decode_failed':1,'unexpected_error':1})
            disk=(root/'out'/'report.json').read_text(); self.assertNotIn('SECRET',disk+captured.getvalue())
            self.assertNotIn(str(root),disk+captured.getvalue())
            self.assertEqual(json.loads(disk),report)
            self.assertFalse(any('submission' in p.name for p in (root/'out').iterdir()))
            self.assertFalse((root/'out'/'report.tmp').exists())
            with self.assertRaises(ValueError): run_compatibility_audit(plan,series,root/'train_series',root/'out',preprocessor=adapter,dcmread=lambda *a,**k:None)

    def test_budget_yields_atomic_partial_without_false_coverage(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as d:
            root=Path(d); series,gold,prior,pins=fixture(root)
            plan=build_audit_plan(series,gold,prior,pins,sample_size=3)
            for study in plan['studies']:
                for row in study['series']: (root/'images'/study['study_id']/row['series_id']).mkdir(parents=True)
            ticks=[0.0]
            def clock(): return ticks[0]
            def adapter(*a,**kw):
                ticks[0]+=1900
                arr=np.ones((1,1,256,256)); return SimpleNamespace(images=arr,valid_mask=arr.astype(bool))
            with contextlib.redirect_stdout(io.StringIO()):
                result=run_compatibility_audit(plan,series,root/'images',root/'out',preprocessor=adapter,dcmread=lambda *a,**k:None,clock=clock)
            self.assertEqual(result['status'],'partial')
            self.assertEqual(result['attempted_series'],1)
            self.assertLess(result['completed_studies'],3)
            self.assertEqual(json.loads((root/'out'/'report.json').read_text()),result)

    def test_reader_observations_are_allowlisted(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as d:
            root=Path(d); series,gold,prior,pins=fixture(root)
            plan=build_audit_plan(series,gold,prior,pins,sample_size=1)
            for study in plan['studies']:
                for row in study['series']: (root/'images'/study['study_id']/row['series_id']).mkdir(parents=True)
            def reader(*a,**kw): return SimpleNamespace(Rows=2,Columns=3,file_meta=SimpleNamespace(TransferSyntaxUID='SECRET_TRANSFER_UID'))
            def adapter(path,**kw):
                kw['dcmread'](path/'fake',stop_before_pixels=True);kw['dcmread'](path/'fake')
                arr=np.ones((1,1,256,256));return SimpleNamespace(images=arr,valid_mask=arr.astype(bool))
            with contextlib.redirect_stdout(io.StringIO()):
                result=run_compatibility_audit(plan,series,root/'images',root/'out',preprocessor=adapter,dcmread=reader,decoder_probe=lambda uid: {'status':'available','plugins':['pylibjpeg','SECRET_PLUGIN']})
            raw=json.dumps(result)
            self.assertNotIn('SECRET',raw)
            self.assertEqual(result['dimensions']['rows'],[2,2])
            self.assertGreater(result['header_reads'],0);self.assertGreater(result['decoded_reads'],0)

    def test_interrupt_preserves_started_but_incomplete_counts(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as d:
            root=Path(d);series,gold,prior,pins=fixture(root)
            plan=build_audit_plan(series,gold,prior,pins,sample_size=1)
            for study in plan['studies']:
                for row in study['series']: (root/'images'/study['study_id']/row['series_id']).mkdir(parents=True)
            def interrupted(*a,**kw): raise KeyboardInterrupt
            with contextlib.redirect_stdout(io.StringIO()):
                result=run_compatibility_audit(plan,series,root/'images',root/'out',preprocessor=interrupted,dcmread=lambda *a,**k:None)
            self.assertEqual(result['status'],'partial')
            self.assertEqual(result['termination_reason'],'interrupted')
            self.assertEqual((result['attempted_series'],result['completed_series']),(1,0))
            self.assertEqual(result['completed_studies'],0)

    def test_atomic_checkpoint_failure_preserves_last_valid_json(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as d:
            root=Path(d);series,gold,prior,pins=fixture(root)
            plan=build_audit_plan(series,gold,prior,pins,sample_size=1)
            (root/'images').mkdir()
            original_replace=__import__('os').replace
            count=[0]
            def interrupted_replace(source,target):
                count[0]+=1
                if count[0]>1: raise OSError('SECRET_STORAGE_PATH')
                return original_replace(source,target)
            with patch('rsna_knee.data.compatibility_audit.os.replace',side_effect=interrupted_replace), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError,'^audit_checkpoint_failed$'):
                    run_compatibility_audit(plan,series,root/'images',root/'out',preprocessor=lambda *a,**k:None,dcmread=lambda *a,**k:None)
            record=json.loads((root/'out'/'report.json').read_text())
            self.assertEqual(record['status'],'partial')
            self.assertNotIn('SECRET',json.dumps(record))

    def test_real_adapter_selected_decode_counts_and_finite_output(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as d:
            root=Path(d);series,gold,prior,pins=fixture(root)
            plan=build_audit_plan(series,gold,prior,pins,sample_size=1)
            for study in plan['studies']:
                for row in study['series']:
                    folder=root/'images'/study['study_id']/row['series_id'];folder.mkdir(parents=True)
                    for i in range(10): (folder/f'{i:02}.dcm').write_bytes(b'synthetic')
            def reader(path,**kwargs):
                z=int(Path(path).stem)
                return SimpleNamespace(ImageOrientationPatient=[1,0,0,0,1,0],ImagePositionPatient=[0,0,z],PixelSpacing=[.5,.5],Rows=2,Columns=3,SamplesPerPixel=1,PhotometricInterpretation='MONOCHROME2',pixel_array=np.arange(6).reshape(2,3),file_meta=SimpleNamespace(TransferSyntaxUID='1.2.840.10008.1.2.1'))
            with contextlib.redirect_stdout(io.StringIO()):
                report=run_compatibility_audit(plan,series,root/'images',root/'out',dcmread=reader,decoder_probe=lambda u:{'status':'available','plugins':[]})
            count=report['planned_series']
            self.assertEqual(report['status'],'complete');self.assertEqual(report['successful_series'],count)
            self.assertEqual(report['header_reads'],10*count);self.assertEqual(report['decoded_reads'],8*count)
            self.assertEqual(report['dimensions'],{'rows':[2,2],'columns':[3,3]})
            self.assertNotIn('study_id',json.dumps(report));self.assertNotIn('series_id',json.dumps(report))

    def test_private_plan_rejects_unexpected_content(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as d:
            root=Path(d);series,gold,prior,pins=fixture(root)
            for where in ('root','study','series'):
                plan=build_audit_plan(series,gold,prior,pins,sample_size=1)
                target=plan if where=='root' else plan['studies'][0] if where=='study' else plan['studies'][0]['series'][0]
                target['report_text']='SECRET_REPORT'
                with self.assertRaises(ValueError): validate_plan(plan)

    def test_codec_probe_failure_is_unknown_not_unavailable(self):
        from rsna_knee.data.compatibility_audit import _codec_availability
        for error in (ImportError('SECRET'),NotImplementedError('SECRET'),TypeError('SECRET'),ValueError('SECRET'),RuntimeError('SECRET')):
            with self.subTest(kind=type(error).__name__), patch('pydicom.pixels.get_decoder',side_effect=error):
                self.assertEqual(_codec_availability('1.2.840.10008.1.2.1'),{'status':'unknown','plugins':[]})
        for available,status in [(False,'unavailable'),(True,'available'),(None,'unknown')]:
            decoder=SimpleNamespace(is_available=available,available_plugins=())
            with patch('pydicom.pixels.get_decoder',return_value=decoder):
                self.assertEqual(_codec_availability('1.2.840.10008.1.2.1'),{'status':status,'plugins':[]})
