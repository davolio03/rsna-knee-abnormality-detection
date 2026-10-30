"""Synthetic-only frozen inventory and genuinely header-only reader contracts."""
import contextlib
import copy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.dataelem import RawDataElement
from rsna_knee.data.compatibility_audit import build_audit_plan, plan_bytes as audit_bytes
from rsna_knee.data.header_inventory import (build_header_plan, run_header_inventory, validate_plan,
                                            plan_bytes, digest, TAGS, _BoundedHeader, _HeaderLimit)

REPO = Path(__file__).resolve().parents[1]


def fixture(root):
    series = root/'train_series.csv'
    series.write_text('StudyInstanceUID,SeriesInstanceUID,Fluid_Sensitive,Fat_Suppression,Anatomical_Plane\n'
                      '1,11,x,x,Axial\n2,21,x,x,Coronal\n3,31,x,x,Sagittal\n'
                      '4,42,x,x,Axial\n4,41,x,x,Coronal\n5,51,x,x,Sagittal\n6,61,x,x,Axial\n')
    gold = root/'gold.csv'; gold.write_text('StudyInstanceUID\n1\n')
    prior = root/'prior.json'; prior.write_text(json.dumps({'records': [{'study_id': '2', 'series': [{'series_id': '21'}], 'labels': 'MUST_NOT_DECODE'}]}))
    pins = {k: digest(p.read_bytes()) for k,p in [('series',series),('gold',gold),('prior',prior)]}
    old = build_audit_plan(series,gold,prior,pins,sample_size=1)
    oldfile = root/'audit-plan.json'; oldfile.write_bytes(audit_bytes(old))
    n = len(old['studies'][0]['series'])
    report = {'plan_sha256':digest(oldfile.read_bytes()),'status':'complete', 'termination_reason':'all_planned_series_attempted',
              'completed_studies':1,'failed_series':0, **{k:n for k in ('planned_series','attempted_series','completed_series','successful_series')},
              'series_results':[{'study_ordinal':1,'series_ordinal':j,'stage':'preprocess','status':'ok','reason_code':None} for j in range(1,n+1)]}
    reportfile=root/'report.json';reportfile.write_text(json.dumps(report))
    pins.update(audit_plan=digest(oldfile.read_bytes()),audit_report=digest(reportfile.read_bytes()))
    args=(series,gold,prior,oldfile,reportfile,pins)
    return args, build_header_plan(*args)


class NoPixels(Dataset):
    @property
    def pixel_array(self):
        raise AssertionError('PIXELS_MUST_NEVER_BE_ACCESSED')


def header():
    ds=NoPixels(); ds.file_meta=FileMetaDataset();ds.file_meta.TransferSyntaxUID='1.2.840.10008.1.2.1'
    ds.Rows=512;ds.Columns=512;ds.SamplesPerPixel=1;ds.PhotometricInterpretation='MONOCHROME2'
    ds.PixelSpacing=[0.5,0.5];ds.ImageOrientationPatient=[1,0,0,0,1,0];ds.ImagePositionPatient=[0,0,0]
    return ds


def images(root, plan):
    for study in plan['studies']:
        for row in study['series']:
            path=root/'images'/study['study_id']/row['series_id'];path.mkdir(parents=True)
            (path/'2.dcm').write_bytes(b'fake');(path/'1.dcm').write_bytes(b'fake')


class HeaderInventoryTests(unittest.TestCase):
    def test_freeze_verified_exclusions_order_schema_and_provenance(self):
        with TemporaryDirectory(dir=REPO) as d:
            args,plan=fixture(Path(d))
            self.assertEqual(plan,build_header_plan(*args))
            self.assertEqual(len(plan['studies']),3)
            self.assertEqual(len(plan['excluded_ids']),3)
            for s in plan['studies']:
                self.assertEqual(s['series'],sorted(s['series'],key=lambda r:r['series_id']))
            self.assertNotIn('MUST_NOT_DECODE',plan_bytes(plan).decode())
            for mutator in ('duplicate','traversal','labels','excluded'):
                bad=copy.deepcopy(plan)
                if mutator=='duplicate':bad['studies'].append(bad['studies'][0])
                elif mutator=='traversal':bad['studies'][0]['series'][0]['series_id']='../secret'
                elif mutator=='labels':bad['studies'][0]['labels']=[1]
                else:bad['excluded_ids'].append(bad['excluded_ids'][0])
                with self.assertRaisesRegex(ValueError,'inventory_plan_invalid'):validate_plan(bad)
            pins={**args[-1],'audit_report':'0'*64}
            with self.assertRaisesRegex(ValueError,'inventory_provenance_invalid'):build_header_plan(*args[:-1],pins)
            report=json.loads(args[4].read_text());report['series_results'][0]['status']='failed';args[4].write_text(json.dumps(report))
            pins={**args[-1],'audit_report':digest(args[4].read_bytes())}
            with self.assertRaises(ValueError):build_header_plan(*args[:-1],pins)

    def test_real_header_arguments_lexicographic_choice_and_no_pixels(self):
        with TemporaryDirectory(dir=REPO) as d:
            root=Path(d);args,plan=fixture(root);images(root,plan);calls=[]
            def reader(stream,**kw):
                self.assertTrue(kw['stop_before_pixels']);self.assertEqual(set(kw['specific_tags']),set(TAGS.values()))
                self.assertEqual(kw['defer_size'],1024);self.assertEqual(Path(stream.stream.name).name,'1.dcm')
                calls.append(1);return header()
            with contextlib.redirect_stdout(io.StringIO()):
                report=run_header_inventory(plan,args[0],root/'images',root/'out',dcmread=reader)
            n=sum(len(s['series']) for s in plan['studies'])
            self.assertEqual((report['status'],report['completed_studies'],report['successful_series']),('complete',3,n))
            self.assertEqual(len(calls),n);self.assertEqual(report['enumerated_files'],2*n)
            self.assertEqual(report['strata']['transfer_syntax'],{'explicit_vr_little_endian':n})
            self.assertEqual(report['flag_counts'],{})
            self.assertEqual(json.loads((root/'out/report.json').read_text()),report)
            with self.assertRaises(ValueError):run_header_inventory(plan,args[0],root/'images',root/'out',dcmread=reader)

    def test_safe_unknown_metadata_failures_candidates_and_continue(self):
        with TemporaryDirectory(dir=REPO) as d:
            root=Path(d);args,plan=fixture(root);images(root,plan);calls=[]
            def reader(*a,**kw):
                calls.append(1)
                if len(calls)==2:raise RuntimeError('SECRET raw/ID path decoder')
                ds=header()
                with warnings_suppressed():ds.file_meta.TransferSyntaxUID='9.999999';ds.PhotometricInterpretation='SECRET'
                ds.Rows=128;ds.NumberOfFrames=3;ds.PixelSpacing=[float('nan'),0.5]
                return ds
            captured=io.StringIO()
            with contextlib.redirect_stdout(captured):
                report=run_header_inventory(plan,args[0],root/'images',root/'out',dcmread=reader)
            self.assertEqual(report['failed_series'],1);self.assertEqual(report['status'],'complete')
            self.assertIn('uncovered_transfer_syntax',report['candidate_ordinals'])
            text=json.dumps(report)+captured.getvalue()
            for secret in ('SECRET','9.999999',str(root),'NaN'):self.assertNotIn(secret,text)
            self.assertEqual(report['reason_counts'],{'header_read_or_metadata_failed':1})

    def test_nonstring_exception_argument_is_safe_and_continues(self):
        with TemporaryDirectory(dir=REPO) as d:
            root=Path(d);args,plan=fixture(root);images(root,plan)
            def reader(*a,**kw):raise ValueError(['SECRET'])
            with contextlib.redirect_stdout(io.StringIO()):
                r=run_header_inventory(plan,args[0],root/'images',root/'out',dcmread=reader)
            self.assertEqual(r['status'],'complete')
            self.assertEqual(r['failed_series'],r['planned_series'])
            self.assertNotIn('SECRET',json.dumps(r))

    def test_deferred_huge_tags_bounded_reads_and_invalid_geometry(self):
        with TemporaryDirectory(dir=REPO) as d:
            root=Path(d);args,plan=fixture(root);images(root,plan)
            def reader(*a,**kw):
                ds=header()
                ds._dict[TAGS['PixelSpacing']]=RawDataElement(TAGS['PixelSpacing'],'DS',999999,None,0,False,True)
                ds.ImageOrientationPatient=[1,0,0,1,0,0]
                return ds
            with contextlib.redirect_stdout(io.StringIO()):
                r=run_header_inventory(plan,args[0],root/'images',root/'out',dcmread=reader)
            self.assertEqual(r['strata']['PixelSpacing'],{'invalid':r['successful_series']})
            self.assertEqual(r['strata']['ImageOrientationPatient'],{'invalid':r['successful_series']})
            stream=_BoundedHeader(io.BytesIO(b'x'*10))
            with self.assertRaises(_HeaderLimit):stream.read(2*1024*1024)
            with self.assertRaises(_HeaderLimit):stream.read()

    def test_installed_reader_genuine_synthetic_dicom_header_only(self):
        from pydicom import dcmwrite, dcmread
        from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage
        with TemporaryDirectory(dir=REPO) as d:
            root=Path(d);args,plan=fixture(root);images(root,plan)
            ds=header();ds.file_meta.TransferSyntaxUID=ExplicitVRLittleEndian
            ds.file_meta.MediaStorageSOPClassUID=SecondaryCaptureImageStorage
            ds.file_meta.MediaStorageSOPInstanceUID='1.2.3'
            ds.SOPClassUID=SecondaryCaptureImageStorage;ds.SOPInstanceUID='1.2.3'
            ds.BitsAllocated=8;ds.PixelData=b'SECRET_PIXEL_BYTES'
            for path in (root/'images').rglob('1.dcm'):dcmwrite(path,ds,enforce_file_format=True)
            seen=[]
            def reader(stream,**kwargs):
                value=dcmread(stream,**kwargs)
                self.assertNotIn(0x7fe00010,value);seen.append(1);return value
            with contextlib.redirect_stdout(io.StringIO()):
                report=run_header_inventory(plan,args[0],root/'images',root/'out',dcmread=reader)
            self.assertEqual(report['successful_series'],report['planned_series'])
            self.assertEqual(len(seen),report['planned_series'])

    def test_missing_extra_files_symlink_and_plan_mismatch(self):
        with TemporaryDirectory(dir=REPO) as d:
            root=Path(d);args,plan=fixture(root);images(root,plan)
            first=plan['studies'][0];directory=root/'images'/first['study_id']/first['series'][0]['series_id']
            (directory/'secret.txt').write_text('SECRET')
            with contextlib.redirect_stdout(io.StringIO()):r=run_header_inventory(plan,args[0],root/'images',root/'out',dcmread=lambda *a,**k:header())
            self.assertEqual(r['reason_counts'],{'inventory_path_invalid':1})
            args[0].write_text(args[0].read_text()+'7,71,x,x,Axial\n')
            with self.assertRaisesRegex(ValueError,'inventory_initialization_invalid'):run_header_inventory(plan,args[0],root/'images',root/'other')

    def test_budget_interrupt_and_atomic_last_snapshot(self):
        with TemporaryDirectory(dir=REPO) as d:
            root=Path(d);args,plan=fixture(root);images(root,plan);now=[0]
            def reader(*a,**kw):now[0]=1801;return header()
            with contextlib.redirect_stdout(io.StringIO()):
                r=run_header_inventory(plan,args[0],root/'images',root/'budget',dcmread=reader,clock=lambda:now[0])
            self.assertEqual((r['status'],r['termination_reason'],r['completed_series']),('partial','time_budget',1))
            with contextlib.redirect_stdout(io.StringIO()):
                r=run_header_inventory(plan,args[0],root/'images',root/'interrupt',dcmread=lambda *a,**k:(_ for _ in ()).throw(KeyboardInterrupt()))
            self.assertEqual((r['attempted_series'],r['completed_series'],r['termination_reason']),(1,0,'interrupted'))
            import os
            original=os.replace;calls=[]
            def replace(*a):
                calls.append(1)
                if len(calls)>1:raise OSError('SECRET')
                return original(*a)
            with patch('rsna_knee.data.header_inventory.os.replace',replace):
                with self.assertRaisesRegex(ValueError,'inventory_checkpoint_failed'):
                    run_header_inventory(plan,args[0],root/'images',root/'atomic',dcmread=lambda *a,**k:header())
            saved=json.loads((root/'atomic/report.json').read_text());self.assertEqual(saved['attempted_series'],0)


@contextlib.contextmanager
def warnings_suppressed():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');yield
