"""Full-study runtime audit plan and model integration regressions."""
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import subprocess
import sys
from zipfile import ZipFile

import torch
from torch import nn

from rsna_knee.data.model_runtime_audit import build_full_study_plan, validate_full_study_plan
from rsna_knee.data.model_runtime_audit import run_model_runtime_audit
from rsna_knee.inference.fold0 import TestStudyDataset
from rsna_knee.models.resnet18 import ResNet18StudyModel
from scripts.build_kaggle_full_study_runtime_audit import build_full_study_runtime_package


class FullStudyRuntimeAuditPlanTests(unittest.TestCase):
    def _fixture(self, root):
        metadata = root / 'train_series.csv'
        rows = [('StudyInstanceUID', 'SeriesInstanceUID', 'Anatomical_Plane')]
        for study in range(20):
            rows.extend([
                (f's{study:02}', f'r{study:02}-sag', 'Sagittal'),
                (f's{study:02}', f'r{study:02}-ax', 'Axial'),
            ])
        rows.append(('extra-study', 'extra-series', 'Coronal'))
        with metadata.open('w', newline='', encoding='utf-8') as stream:
            csv.writer(stream).writerows(rows)
        metadata_hash = hashlib.sha256(metadata.read_bytes()).hexdigest()
        plan = {
            'schema_version': 'training-dimension-candidates-v1',
            'source_sha256': {'series': metadata_hash},
            'header_report_sha256': 'b' * 64, 'header_plan_sha256': 'c' * 64,
            'header_source_sha256': {'series': metadata_hash, 'gold': 'e' * 64,
                'prior': 'f' * 64, 'audit_plan': '1' * 64, 'audit_report': '2' * 64},
            'excluded_ids': [f'excluded-{i}' for i in range(658)], 'candidate_count': 20,
            'candidates': [{'study_id': f's{i:02}', 'series_id': f'r{i:02}-sag',
                'plane': 'Sagittal', 'study_ordinal': i + 1, 'series_ordinal': 1}
                for i in range(20)],
        }
        raw = json.dumps(plan, sort_keys=True, separators=(',', ':')).encode() + b'\n'
        return plan, raw, metadata

    def test_freezes_all_siblings_for_exact_selected_studies_in_stable_order(self):
        with TemporaryDirectory() as temp:
            plan, raw, metadata = self._fixture(Path(temp))
            baseline = json.dumps({'schema_version': 'training-model-runtime-report-v1',
                'status': 'complete', 'planned_series': 20, 'completed_series': 20,
                'successful_series': 20, 'failed_series': 0, 'finite_logits': 240,
                'finite_probabilities': 240}, sort_keys=True, separators=(',', ':')).encode()
            with patch('rsna_knee.data.model_runtime_audit.SELECTED_PLAN_SHA256', hashlib.sha256(raw).hexdigest()), \
                 patch('rsna_knee.data.model_runtime_audit.HEADER_REPORT_SHA256', 'b' * 64), \
                 patch('rsna_knee.data.model_runtime_audit.HEADER_PLAN_SHA256', 'c' * 64), \
                 patch('rsna_knee.data.model_runtime_audit.TRAIN_SERIES_SHA256', hashlib.sha256(metadata.read_bytes()).hexdigest()), \
                 patch('rsna_knee.data.model_runtime_audit.CA16_REPORT_SHA256', hashlib.sha256(baseline).hexdigest()):
                full = build_full_study_plan(plan, raw, metadata, baseline)
            frozen = json.dumps(full, sort_keys=True, separators=(',', ':')).encode() + b'\n'
            self.assertEqual(len(full['studies']), 20)
            self.assertEqual(full['series_count'], 40)
            self.assertEqual(full['series_plane_counts'], {'Axial': 20, 'Coronal': 0, 'Sagittal': 20})
            self.assertEqual([row['study_ordinal'] for row in full['studies']], list(range(1, 21)))
            self.assertTrue(all(len(row['series']) == 2 for row in full['studies']))
            self.assertTrue(all([item['series_id'] for item in row['series']] ==
                [f"r{row['study_id'][1:]}-sag", f"r{row['study_id'][1:]}-ax"] for row in full['studies']))
            self.assertNotIn('extra-study', repr(full))
            with patch('rsna_knee.data.model_runtime_audit.SELECTED_PLAN_SHA256', hashlib.sha256(raw).hexdigest()), \
                 patch('rsna_knee.data.model_runtime_audit.HEADER_REPORT_SHA256', 'b' * 64), \
                 patch('rsna_knee.data.model_runtime_audit.HEADER_PLAN_SHA256', 'c' * 64), \
                 patch('rsna_knee.data.model_runtime_audit.TRAIN_SERIES_SHA256', hashlib.sha256(metadata.read_bytes()).hexdigest()), \
                 patch('rsna_knee.data.model_runtime_audit.CA16_REPORT_SHA256', hashlib.sha256(baseline).hexdigest()):
                validate_full_study_plan(full, frozen, plan, raw, metadata, baseline)
            with self.assertRaisesRegex(ValueError, 'full_study_plan_invalid'):
                forged = json.loads(frozen)
                forged['studies'].append(forged['studies'][0])
                validate_full_study_plan(forged, json.dumps(forged).encode(), plan, raw, metadata, baseline)

    def test_metadata_pin_and_selected_candidate_membership_are_required(self):
        with TemporaryDirectory() as temp:
            plan, raw, metadata = self._fixture(Path(temp))
            baseline = json.dumps({'schema_version': 'training-model-runtime-report-v1',
                'status': 'complete', 'planned_series': 20, 'completed_series': 20,
                'successful_series': 20, 'failed_series': 0, 'finite_logits': 240,
                'finite_probabilities': 240}, sort_keys=True, separators=(',', ':')).encode()
            with patch('rsna_knee.data.model_runtime_audit.SELECTED_PLAN_SHA256', hashlib.sha256(raw).hexdigest()), \
                 patch('rsna_knee.data.model_runtime_audit.HEADER_REPORT_SHA256', 'b' * 64), \
                 patch('rsna_knee.data.model_runtime_audit.HEADER_PLAN_SHA256', 'c' * 64), \
                 patch('rsna_knee.data.model_runtime_audit.TRAIN_SERIES_SHA256', hashlib.sha256(metadata.read_bytes()).hexdigest()), \
                 patch('rsna_knee.data.model_runtime_audit.CA16_REPORT_SHA256', hashlib.sha256(baseline).hexdigest()):
                full = build_full_study_plan(plan, raw, metadata, baseline)
            altered = metadata.read_text(encoding='utf-8').replace('r00-sag', 'forged-series')
            metadata.write_text(altered, encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'full_study_plan_invalid'):
                frozen = json.dumps(full, sort_keys=True, separators=(',', ':')).encode() + b'\n'
                validate_full_study_plan(full, frozen, plan, raw, metadata, baseline)

    def test_package_is_fresh_disabled_output_free_and_imports_isolated(self):
        repo = Path(__file__).resolve().parents[1]
        with TemporaryDirectory(dir=repo) as temp:
            root = Path(temp)
            plan, raw, metadata = self._fixture(root)
            baseline = json.dumps({'schema_version': 'training-model-runtime-report-v1',
                'status': 'complete', 'planned_series': 20, 'completed_series': 20,
                'successful_series': 20, 'failed_series': 0, 'finite_logits': 240,
                'finite_probabilities': 240}, sort_keys=True, separators=(',', ':')).encode()
            baseline_path = root / 'baseline.json'; baseline_path.write_bytes(baseline)
            with patch('rsna_knee.data.model_runtime_audit.SELECTED_PLAN_SHA256', hashlib.sha256(raw).hexdigest()), \
                 patch('rsna_knee.data.model_runtime_audit.HEADER_REPORT_SHA256', 'b' * 64), \
                 patch('rsna_knee.data.model_runtime_audit.HEADER_PLAN_SHA256', 'c' * 64), \
                 patch('rsna_knee.data.model_runtime_audit.TRAIN_SERIES_SHA256', hashlib.sha256(metadata.read_bytes()).hexdigest()), \
                 patch('rsna_knee.data.model_runtime_audit.CA16_REPORT_SHA256', hashlib.sha256(baseline).hexdigest()):
                result = build_full_study_runtime_package(repo, root / 'package', plan, metadata, baseline_path)
            notebook = json.loads(Path(result['notebook']).read_text(encoding='utf-8'))
            code = ''.join(notebook['cells'][1]['source'])
            self.assertIn('RUN_FULL_STUDY_RUNTIME_AUDIT = False', code)
            self.assertNotIn('run_inference(', code)
            self.assertNotIn('submission.csv', code)
            self.assertTrue(all(not cell.get('outputs') for cell in notebook['cells']))
            self.assertEqual(result['member_count'], 22)
            with ZipFile(root / 'package' / 'full-study-runtime-source.zip') as archive:
                names = set(archive.namelist())
                module = 'rsna_knee/data/model_runtime_audit.py'
                self.assertIn(module, names)
                self.assertEqual(archive.read(module), (repo / 'src' / module).read_bytes())
            probe = ('import sys;sys.path.insert(0,sys.argv[1]);'
                'from rsna_knee.data.model_runtime_audit import build_full_study_plan,run_model_runtime_audit;'
                'from rsna_knee.inference.fold0 import TestStudyDataset,load_fold0_model;'
                'from rsna_knee.models.resnet18 import ResNet18StudyModel')
            imported = subprocess.run([sys.executable, '-I', '-c', probe,
                str(root / 'package' / 'full-study-runtime-source.zip')], capture_output=True, text=True)
            self.assertEqual(imported.returncode, 0, imported.stderr)

    def test_genuine_dataset_and_resnet_receive_all_series_as_one_study(self):
        class TinyBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Identity()
                self.conv1 = nn.Conv2d(3, 512, kernel_size=1)
                self.bn1 = nn.Identity()
                self.relu = nn.ReLU()
                self.maxpool = nn.Identity()
                self.layer1 = nn.Identity()
                self.layer2 = nn.Identity()
                self.layer3 = nn.Identity()
                self.layer4 = nn.Identity()

        with TemporaryDirectory() as temp:
            root = Path(temp)
            test_csv = root / 'study.csv'
            series_csv = root / 'series.csv'
            test_csv.write_text('StudyInstanceUID\nstudy-a\n', encoding='utf-8')
            series_csv.write_text('StudyInstanceUID,SeriesInstanceUID\nstudy-a,sag\nstudy-a,ax\n', encoding='utf-8')
            image_root = root / 'train_series'
            for name in ('sag', 'ax'):
                (image_root / 'study-a' / name).mkdir(parents=True)
            class Result:
                def __init__(self, fill):
                    import numpy as np
                    self.images = np.full((2, 1, 256, 256), fill, dtype='float32')
                    self.valid_mask = np.ones_like(self.images, dtype=bool)
            dataset = TestStudyDataset(test_csv, series_csv, image_root,
                preprocessor=lambda path, **_kw: Result(0.25 if path.name == 'sag' else 0.75))
            images, masks = dataset[0]
            model = ResNet18StudyModel(TinyBackbone(), slice_microbatch=2, checkpoint_trunk=False, freeze_backbone=True)
            logits = model(images, masks)
            probabilities = torch.sigmoid(logits)
            self.assertEqual((len(images), len(masks), tuple(logits.shape)), (2, 2, (1, 12)))
            self.assertTrue(torch.isfinite(logits).all())
            self.assertTrue(torch.isfinite(probabilities).all())

    def test_full_study_runner_counts_every_sibling_and_passes_one_study_input(self):
        import numpy as np
        class Tensor:
            def __init__(self, value): self.value = value
            @property
            def shape(self): return self.value.shape
            def to(self, _device): return self
            def detach(self): return self
            def float(self): return self
            def cpu(self): return self
            def numpy(self): return self.value
            def all(self): return bool(self.value.all())
        class Cuda:
            def is_available(self): return True
            def set_device(self, _index): pass
            def synchronize(self, _device): pass
            def reset_peak_memory_stats(self, _device): pass
            def max_memory_allocated(self, _device): return 4096
            def device_count(self): return 2
        class FakeTorch:
            __version__ = 'test-torch'
            version = type('Version', (), {'cuda': 'test-cuda'})()
            cuda = Cuda()
            float32 = torch.float32
            bool = torch.bool
            @staticmethod
            def device(name): return name
            @staticmethod
            def inference_mode(): return torch.inference_mode()
            @staticmethod
            def isfinite(value): return Tensor(np.isfinite(value.value))
            @staticmethod
            def sigmoid(value): return Tensor(np.full(value.shape, .5, dtype='float32'))
        class Result:
            def __init__(self, fill):
                self.images = np.full((2, 1, 256, 256), fill, dtype='float32')
                self.valid_mask = np.ones_like(self.images, dtype=bool)
        class Model:
            def to(self, _device): return self
            def eval(self): return self
            def __call__(self, images, masks):
                assert len(images) == len(masks) == 2
                assert all(value.shape == (2, 1, 256, 256) for value in images)
                return Tensor(np.zeros((1, 12), dtype='float32'))

        repo = Path(__file__).resolve().parents[1]
        with TemporaryDirectory(dir=repo) as temp:
            root = Path(temp)
            plan, raw, metadata = self._fixture(root)
            baseline_report = {'schema_version': 'training-model-runtime-report-v1', 'status': 'complete',
                'planned_series': 20, 'completed_series': 20, 'successful_series': 20,
                'failed_series': 0, 'finite_logits': 240, 'finite_probabilities': 240}
            baseline = json.dumps(baseline_report, sort_keys=True, separators=(',', ':')).encode()
            plan_hash = hashlib.sha256(raw).hexdigest()
            metadata_hash = hashlib.sha256(metadata.read_bytes()).hexdigest()
            full = None
            with patch('rsna_knee.data.model_runtime_audit.SELECTED_PLAN_SHA256', plan_hash), \
                 patch('rsna_knee.data.model_runtime_audit.HEADER_REPORT_SHA256', 'b' * 64), \
                 patch('rsna_knee.data.model_runtime_audit.HEADER_PLAN_SHA256', 'c' * 64), \
                 patch('rsna_knee.data.model_runtime_audit.TRAIN_SERIES_SHA256', metadata_hash), \
                 patch('rsna_knee.data.model_runtime_audit.CA16_REPORT_SHA256', hashlib.sha256(baseline).hexdigest()):
                full = build_full_study_plan(plan, raw, metadata, baseline)
            full_raw = json.dumps(full, sort_keys=True, separators=(',', ':')).encode() + b'\n'
            image_root = root / 'images'
            for study in full['studies']:
                for row in study['series']:
                    (image_root / study['study_id'] / row['series_id']).mkdir(parents=True)
            training_report = root / 'training-report.json'; training_report.write_text('{}', encoding='utf-8')
            archive = root / 'source.zip'; archive.write_bytes(b'pinned-source-package')
            checkpoint = root / 'checkpoint.pt'; checkpoint.write_bytes(b'synthetic-checkpoint-bytes')
            preprocess_calls = []
            def synthetic_preprocessor(path, **_kw):
                preprocess_calls.append(path.is_dir())
                return Result(.25 if path.name.endswith('sag') else .75)
            with patch('rsna_knee.data.model_runtime_audit.SELECTED_PLAN_SHA256', plan_hash), \
                 patch('rsna_knee.data.model_runtime_audit.HEADER_REPORT_SHA256', 'b' * 64), \
                 patch('rsna_knee.data.model_runtime_audit.HEADER_PLAN_SHA256', 'c' * 64), \
                 patch('rsna_knee.data.model_runtime_audit.TRAIN_SERIES_SHA256', metadata_hash), \
                 patch('rsna_knee.data.model_runtime_audit.TRAINING_REPORT_SHA256', hashlib.sha256(b'{}').hexdigest()), \
                 patch('rsna_knee.data.model_runtime_audit.validate_training_report'), \
                 patch('rsna_knee.data.model_runtime_audit.CHECKPOINT_SHA256', hashlib.sha256(checkpoint.read_bytes()).hexdigest()), \
                 patch('rsna_knee.data.model_runtime_audit.CHECKPOINT_SIZE', checkpoint.stat().st_size), \
                 patch('rsna_knee.data.model_runtime_audit.CA16_REPORT_SHA256', hashlib.sha256(baseline).hexdigest()), \
                 patch.object(torch.Tensor, 'to', lambda self, *_args, **_kwargs: self):
                result = run_model_runtime_audit(plan, raw, metadata, image_root, training_report,
                    checkpoint, root / 'output', source_archive=archive,
                    source_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                    full_study_plan=full, full_study_plan_raw=full_raw, baseline_report_raw=baseline,
                    preprocessor=synthetic_preprocessor,
                    model_loader=lambda *_args: Model(), torch_api=FakeTorch(), budget_seconds=1800)
            self.assertEqual((result['planned_studies'], result['completed_studies'],
                result['successful_studies'], result['failed_studies']), (20, 20, 20, 0),
                result['failures'][:2])
            self.assertEqual((result['planned_series'], result['attempted_series'],
                result['completed_series'], result['successful_series'], result['failed_series']), (40, 40, 40, 40, 0))
            self.assertEqual((result['finite_logits'], result['finite_probabilities']), (240, 240))
            self.assertEqual(result['termination_reason'], 'all_planned_studies_attempted')
            self.assertEqual(result['series_plane_counts'], {'Axial': 20, 'Coronal': 0, 'Sagittal': 20})
            plan_hash = hashlib.sha256(full_raw).hexdigest()
            self.assertEqual(result['expected_provenance']['full_study_plan_sha256'], plan_hash)
            self.assertEqual(result['verified_provenance']['full_study_plan_sha256'], plan_hash)
            persisted = json.loads((root / 'output' / 'model-runtime-report.json').read_text(encoding='utf-8'))
            self.assertEqual(persisted['expected_provenance'], persisted['verified_provenance'])
            serialized = json.dumps(result, allow_nan=False)
            self.assertNotIn('s00', serialized)
            self.assertNotIn('logits_by_study', serialized)


if __name__ == '__main__':
    unittest.main()
