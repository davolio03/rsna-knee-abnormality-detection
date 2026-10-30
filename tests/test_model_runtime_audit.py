"""Bounded model/runtime audit preparation and runner tests."""
import json
import hashlib
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import subprocess
import sys
from unittest.mock import patch
from rsna_knee.models.resnet18 import ResNet18StudyModel
import torch
from torch import nn

from rsna_knee.data.model_runtime_audit import run_model_runtime_audit, validate_model_runtime_plan
from scripts.build_kaggle_model_runtime_audit import build_model_runtime_package


class ModelRuntimeAuditTests(unittest.TestCase):
    def _plan(self, series_hash='a' * 64):
        pins = {'series': series_hash, 'gold': 'e' * 64, 'prior': 'f' * 64,
                'audit_plan': '1' * 64, 'audit_report': '2' * 64}
        return {'schema_version': 'training-dimension-candidates-v1',
            'source_sha256': {'series': pins['series']}, 'header_report_sha256': 'b' * 64,
            'header_plan_sha256': 'c' * 64, 'header_source_sha256': pins,
            'excluded_ids': [f'excluded-{i}' for i in range(658)], 'candidate_count': 20,
            'candidates': [{'study_id': f's{i}', 'series_id': f'r{i}', 'plane': 'Sagittal',
                'study_ordinal': i + 1, 'series_ordinal': 1} for i in range(20)]}

    def test_plan_must_match_pins_and_exact_twenty_distinct_sagittal_studies(self):
        plan = self._plan()
        raw = json.dumps(plan, sort_keys=True, separators=(',', ':')).encode()
        with patch('rsna_knee.data.model_runtime_audit.SELECTED_PLAN_SHA256', __import__('hashlib').sha256(raw).hexdigest()), \
             patch('rsna_knee.data.model_runtime_audit.HEADER_REPORT_SHA256', 'b' * 64), \
             patch('rsna_knee.data.model_runtime_audit.HEADER_PLAN_SHA256', 'c' * 64), \
             patch('rsna_knee.data.model_runtime_audit.TRAIN_SERIES_SHA256', 'a' * 64):
            validate_model_runtime_plan(plan, raw)
            extra = dict(plan)
            extra['candidates'] = plan['candidates'] + [dict(plan['candidates'][0])]
            with self.assertRaisesRegex(ValueError, 'model_runtime_plan_invalid'):
                validate_model_runtime_plan(extra, json.dumps(extra, sort_keys=True, separators=(',', ':')).encode())
            changed = dict(plan)
            changed['candidates'] = [dict(row) for row in plan['candidates']]
            changed['candidates'][0]['plane'] = 'Axial'
            with self.assertRaisesRegex(ValueError, 'model_runtime_plan_invalid'):
                validate_model_runtime_plan(changed, json.dumps(changed, sort_keys=True, separators=(',', ':')).encode())

    def test_preparation_is_disabled_and_contains_model_runtime_sources(self):
        repo = Path(__file__).resolve().parents[1]
        with TemporaryDirectory(dir=repo) as temp:
            root = Path(temp)
            plan = self._plan()
            result = build_model_runtime_package(repo, root / 'prep', plan)
            notebook = json.loads(Path(result['notebook']).read_text(encoding='utf-8'))
            code = ''.join(notebook['cells'][1]['source'])
            self.assertIn('RUN_MODEL_RUNTIME_AUDIT = False', code)
            self.assertNotIn('run_inference(', code)
            self.assertNotIn('TestStudyDataset', code)
            self.assertTrue(all(not cell.get('outputs') for cell in notebook['cells']))
            self.assertTrue({'rsna_knee/data/model_runtime_audit.py',
                             'rsna_knee/inference/fold0.py',
                             'rsna_knee/models/resnet18.py',
                             'rsna_knee/models/baseline.py'}.issubset(set(result['members'])))
            probe = ("import sys;sys.path.insert(0,sys.argv[1]);"
                "from rsna_knee.data.model_runtime_audit import validate_model_runtime_plan;"
                "from rsna_knee.inference.fold0 import load_fold0_model;"
                "from rsna_knee.models.resnet18 import ResNet18StudyModel")
            imported = subprocess.run([sys.executable, '-I', '-c', probe,
                str(root/'prep'/'model-runtime-source.zip')], capture_output=True, text=True)
            self.assertEqual(imported.returncode, 0, imported.stderr)

    def test_synthetic_cuda_execution_runs_single_series_inputs_and_reports_no_ids_or_values(self):
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
            calls = 0
            fail_on_call = None
            def is_available(self): return True
            def set_device(self, index): self.assert_index = index
            def synchronize(self, _device):
                self.calls += 1
                if self.calls == self.fail_on_call:
                    raise RuntimeError('must-not-leak-cuda-runtime-path')
            def reset_peak_memory_stats(self, _device): pass
            def max_memory_allocated(self, _device): return 4096
            def device_count(self): return 2

        class FakeTorch:
            __version__ = 'test-torch'
            version = type('Version', (), {'cuda':'test-cuda'})()
            cuda = Cuda()
            float32 = object()
            bool = object()
            @staticmethod
            def device(name): return name
            @staticmethod
            def inference_mode(): return nullcontext()
            @staticmethod
            def as_tensor(value, dtype=None): return Tensor(value)
            @staticmethod
            def isfinite(value): return Tensor(__import__('numpy').isfinite(value.value))
            @staticmethod
            def sigmoid(value): return Tensor(__import__('numpy').full(value.shape, .5))

        class Result:
            images = __import__('numpy').ones((2, 1, 256, 256), dtype='float32')
            valid_mask = __import__('numpy').ones((2, 1, 256, 256), dtype='bool')

        class Model:
            def __init__(self): self.calls = 0
            def to(self, _device): return self
            def eval(self): return self
            def __call__(self, images, masks):
                self.calls += 1
                assert len(images) == len(masks) == 1
                assert images[0].shape == masks[0].shape == (2, 1, 256, 256)
                return Tensor(__import__('numpy').zeros((1, 12), dtype='float32'))

        model = Model()
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            metadata = root / 'train_series.csv'
            metadata.write_text('StudyInstanceUID,SeriesInstanceUID,Anatomical_Plane\n' +
                ''.join(f"s{i},r{i},Sagittal\n" for i in range(20)), encoding='utf-8')
            metadata_hash = hashlib.sha256(metadata.read_bytes()).hexdigest()
            plan = self._plan(metadata_hash)
            raw_plan = json.dumps(plan, sort_keys=True, separators=(',', ':')).encode()
            image_root = root / 'images'
            for i in range(20):
                (image_root / f's{i}' / f'r{i}').mkdir(parents=True)
            training_report = root / 'report.json'; training_report.write_text('{}', encoding='utf-8')
            source_archive = root / 'source.zip'; source_archive.write_bytes(b'test-source-package')
            source_archive_hash = hashlib.sha256(source_archive.read_bytes()).hexdigest()
            checkpoint = root/'checkpoint.pt'; checkpoint.write_bytes(b'x')
            checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            with patch('rsna_knee.data.model_runtime_audit.SELECTED_PLAN_SHA256', hashlib.sha256(raw_plan).hexdigest()), \
                 patch('rsna_knee.data.model_runtime_audit.HEADER_REPORT_SHA256', 'b' * 64), \
                 patch('rsna_knee.data.model_runtime_audit.HEADER_PLAN_SHA256', 'c' * 64), \
                 patch('rsna_knee.data.model_runtime_audit.TRAIN_SERIES_SHA256', metadata_hash), \
                 patch('rsna_knee.data.model_runtime_audit.TRAINING_REPORT_SHA256', hashlib.sha256(b'{}').hexdigest()), \
                 patch('rsna_knee.data.model_runtime_audit.validate_training_report'), \
                 patch('rsna_knee.data.model_runtime_audit.CHECKPOINT_SHA256', checkpoint_hash), \
                 patch('rsna_knee.data.model_runtime_audit.CHECKPOINT_SIZE', 1):
                result = run_model_runtime_audit(plan, raw_plan, metadata, image_root,
                    training_report, checkpoint, root/'output',
                    preprocessor=lambda *_args, **_kwargs: Result(),
                    model_loader=lambda *_args: model, torch_api=FakeTorch(), budget_seconds=1800,
                    source_archive=source_archive, source_archive_sha256=source_archive_hash)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual((result['planned_series'], result['attempted_series'], result['successful_series']), (20, 20, 20))
            self.assertEqual((result['finite_logits'], result['finite_probabilities']), (240, 240))
            self.assertEqual(result['peak_gpu_bytes'], 4096)
            self.assertEqual(model.calls, 20)
            emitted = (root/'output'/'model-runtime-report.json').read_text(encoding='utf-8')
            self.assertNotIn('s0', emitted)
            self.assertNotIn('r0', emitted)
            self.assertNotIn('probability_min', emitted)
            self.assertNotIn('probability_max', emitted)
            verified = result['verified_provenance']
            self.assertEqual(verified, {
                'plan_sha256': hashlib.sha256(raw_plan).hexdigest(),
                'training_series_sha256': metadata_hash,
                'training_report_sha256': hashlib.sha256(b'{}').hexdigest(),
                'checkpoint_sha256': checkpoint_hash, 'checkpoint_size': 1,
                'source_archive_sha256': source_archive_hash})
            self.assertEqual(result['expected_provenance'], verified)
            def flaky_preprocess(path, **_kwargs):
                if path.name == 'r0':
                    raise RuntimeError('must-not-leak-s0-r0')
                return Result()
            with patch.multiple('rsna_knee.data.model_runtime_audit',
                    SELECTED_PLAN_SHA256=hashlib.sha256(raw_plan).hexdigest(),
                    HEADER_REPORT_SHA256='b' * 64, HEADER_PLAN_SHA256='c' * 64,
                    TRAIN_SERIES_SHA256=metadata_hash,
                    TRAINING_REPORT_SHA256=hashlib.sha256(b'{}').hexdigest(),
                    CHECKPOINT_SHA256=checkpoint_hash, CHECKPOINT_SIZE=1), \
                    patch('rsna_knee.data.model_runtime_audit.validate_training_report'):
                partial = run_model_runtime_audit(plan, raw_plan, metadata, image_root,
                    training_report, checkpoint, root/'partial', preprocessor=flaky_preprocess,
                    model_loader=lambda *_args: Model(), torch_api=FakeTorch(), budget_seconds=1800,
                    source_archive=source_archive, source_archive_sha256=source_archive_hash)
                self.assertEqual((partial['status'], partial['completed_series'], partial['successful_series'],
                                  partial['failed_series']), ('partial', 20, 19, 1))
                partial_text = (root/'partial'/'model-runtime-report.json').read_text(encoding='utf-8')
                self.assertNotIn('must-not-leak', partial_text)
                self.assertIn('"series_ordinal":1', partial_text)
                self.assertIn('"reason_code":"unexpected_error"', partial_text)

                FakeTorch.cuda.is_available = lambda: False
                with self.assertRaisesRegex(ValueError, 'model_runtime_initialization_failed'):
                    run_model_runtime_audit(plan, raw_plan, metadata, image_root, training_report,
                        checkpoint, root/'no-cuda', model_loader=lambda *_args: self.fail('must not load'),
                        torch_api=FakeTorch(), budget_seconds=1800,
                        source_archive=source_archive, source_archive_sha256=source_archive_hash)
                no_cuda = json.loads((root/'no-cuda'/'model-runtime-report.json').read_text(encoding='utf-8'))
                self.assertEqual(no_cuda['failures'][-1]['stage'], 'cuda_initialization')
                FakeTorch.cuda.is_available = lambda: True

                ticks = iter(range(0, 10000, 100))
                budget = run_model_runtime_audit(plan, raw_plan, metadata, image_root, training_report,
                    checkpoint, root/'budget', preprocessor=lambda *_a, **_k: Result(),
                    model_loader=lambda *_args: Model(), torch_api=FakeTorch(),
                    clock=lambda: next(ticks), budget_seconds=1800,
                    source_archive=source_archive, source_archive_sha256=source_archive_hash)
                self.assertEqual(budget['status'], 'partial')
                self.assertEqual(budget['termination_reason'], 'cooperative_budget_reached')
                self.assertLess(budget['attempted_series'], budget['planned_series'])

                fake_cuda = FakeTorch()
                fake_cuda.cuda.calls = 0
                fake_cuda.cuda.fail_on_call = 2
                with self.assertRaisesRegex(ValueError, 'model_runtime_audit_interrupted'):
                    interrupted = run_model_runtime_audit(plan, raw_plan, metadata, image_root,
                        training_report, checkpoint, root/'interrupted',
                        model_loader=lambda *_args: Model(), torch_api=fake_cuda,
                        source_archive=source_archive, source_archive_sha256=source_archive_hash)
                interrupted_report = json.loads((root/'interrupted'/'model-runtime-report.json').read_text())
                self.assertEqual((interrupted_report['status'], interrupted_report['termination_reason']),
                                 ('partial', 'runtime_interrupted'))
                self.assertEqual((interrupted_report['attempted_series'], interrupted_report['completed_series'],
                                  interrupted_report['successful_series'], interrupted_report['failed_series']),
                                 (0, 0, 0, 0))
                self.assertEqual(interrupted_report['runtime_failure']['stage'], 'series_boundary_synchronize')
                self.assertEqual(interrupted_report['runtime_failure']['series_ordinal'], 1)

                original_replace = __import__('rsna_knee.data.model_runtime_audit', fromlist=['os']).os.replace
                replace_calls = 0
                def fail_terminal_replace(source, target):
                    nonlocal replace_calls
                    replace_calls += 1
                    if replace_calls == 2:
                        raise OSError('must-not-leak-private-storage-path')
                    original_replace(source, target)
                with patch('rsna_knee.data.model_runtime_audit.os.replace', side_effect=fail_terminal_replace):
                    with self.assertRaisesRegex(ValueError, 'model_runtime_report_storage_failed'):
                        run_model_runtime_audit(plan, raw_plan, metadata, image_root, training_report,
                            checkpoint, root/'storage-failure', preprocessor=lambda *_a, **_k: Result(),
                            model_loader=lambda *_args: Model(), torch_api=FakeTorch(),
                            source_archive=source_archive, source_archive_sha256=source_archive_hash)
                stored = json.loads((root/'storage-failure'/'model-runtime-report.json').read_text())
                self.assertEqual(replace_calls, 2)
                self.assertEqual((stored['attempted_series'], stored['completed_series'],
                                  stored['successful_series'], stored['failed_series']), (0, 0, 0, 0))

    def test_real_resnet18_adapter_forward_with_synthetic_tiny_backbone(self):
        class TinyBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = nn.Conv2d(3, 512, kernel_size=1, bias=False)
                self.conv1 = self.projection
                self.bn1 = nn.Identity()
                self.relu = nn.ReLU()
                self.maxpool = nn.AvgPool2d(16)
                self.layer1 = nn.Identity()
                self.layer2 = nn.Identity()
                self.layer3 = nn.Identity()
                self.layer4 = nn.Identity()
            def forward(self, images):
                return images

        model = ResNet18StudyModel(TinyBackbone(), slice_microbatch=2,
            checkpoint_trunk=False, freeze_backbone=True).eval()
        images = torch.ones((2, 1, 256, 256), dtype=torch.float32)
        masks = torch.ones_like(images, dtype=torch.bool)
        with torch.inference_mode():
            logits = model([images], [masks])
        self.assertEqual(tuple(logits.shape), (1, 12))
        self.assertTrue(torch.isfinite(logits).all())

    def test_runtime_sync_interruption_and_report_storage_failure_preserve_series_counts(self):
        class Model:
            def to(self, _device): return self
            def eval(self): return self

        class Result:
            images = __import__('numpy').ones((1, 1, 256, 256), dtype='float32')
            valid_mask = __import__('numpy').ones((1, 1, 256, 256), dtype='bool')

        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            metadata = root/'train_series.csv'
            metadata.write_text('StudyInstanceUID,SeriesInstanceUID,Anatomical_Plane\n' +
                ''.join(f's{i},r{i},Sagittal\n' for i in range(20)), encoding='utf-8')
            metadata_hash = hashlib.sha256(metadata.read_bytes()).hexdigest()
            plan = self._plan(metadata_hash)
            plan_raw = json.dumps(plan, sort_keys=True, separators=(',', ':')).encode()
            images = root/'images'
            for i in range(20): (images/f's{i}'/f'r{i}').mkdir(parents=True)
            training_report = root/'report.json'; training_report.write_text('{}', encoding='utf-8')
            checkpoint = root/'checkpoint.pt'; checkpoint.write_bytes(b'x')
            source_archive = root/'source.zip'; source_archive.write_bytes(b'test-source-package')
            source_archive_hash = hashlib.sha256(source_archive.read_bytes()).hexdigest()

            class Cuda:
                calls = 0
                def is_available(self): return True
                def set_device(self, _index): pass
                def synchronize(self, _device):
                    self.calls += 1
                    if self.calls == 2: raise RuntimeError('must-not-leak-runtime-failure')
                def reset_peak_memory_stats(self, _device): pass
                def max_memory_allocated(self, _device): return 0
                def device_count(self): return 2
            fake_torch = torch
            cuda = Cuda()
            with patch('rsna_knee.data.model_runtime_audit.SELECTED_PLAN_SHA256', hashlib.sha256(plan_raw).hexdigest()), \
                 patch('rsna_knee.data.model_runtime_audit.HEADER_REPORT_SHA256', 'b'*64), \
                 patch('rsna_knee.data.model_runtime_audit.HEADER_PLAN_SHA256', 'c'*64), \
                 patch('rsna_knee.data.model_runtime_audit.TRAIN_SERIES_SHA256', metadata_hash), \
                 patch('rsna_knee.data.model_runtime_audit.TRAINING_REPORT_SHA256', hashlib.sha256(b'{}').hexdigest()), \
                 patch('rsna_knee.data.model_runtime_audit.validate_training_report'), \
                 patch('rsna_knee.data.model_runtime_audit.CHECKPOINT_SHA256', hashlib.sha256(b'x').hexdigest()), \
                 patch('rsna_knee.data.model_runtime_audit.CHECKPOINT_SIZE', 1), \
                 patch.object(fake_torch.cuda, 'is_available', return_value=True), \
                 patch.object(fake_torch.cuda, 'set_device'), \
                 patch.object(fake_torch.cuda, 'synchronize', side_effect=cuda.synchronize), \
                 patch.object(fake_torch.cuda, 'reset_peak_memory_stats'), \
                 patch.object(fake_torch.cuda, 'device_count', return_value=2):
                with self.assertRaisesRegex(ValueError, 'model_runtime_audit_interrupted'):
                    run_model_runtime_audit(plan, plan_raw, metadata, images, training_report,
                        checkpoint, root/'interrupted', model_loader=lambda *_args: Model(),
                        torch_api=fake_torch, source_archive=source_archive,
                        source_archive_sha256=source_archive_hash)
                interrupted = json.loads((root/'interrupted'/'model-runtime-report.json').read_text())
                self.assertEqual((interrupted['status'], interrupted['termination_reason']), ('partial', 'runtime_interrupted'))
                self.assertEqual((interrupted['attempted_series'], interrupted['completed_series'],
                                  interrupted['successful_series'], interrupted['failed_series']), (0, 0, 0, 0))
                self.assertEqual(interrupted['runtime_failure']['stage'], 'series_boundary_synchronize')
                self.assertEqual(interrupted['runtime_failure']['series_ordinal'], 1)

                calls = 0
                original_replace = __import__('rsna_knee.data.model_runtime_audit', fromlist=['os']).os.replace
                def fail_terminal_replace(source, target):
                    nonlocal calls
                    calls += 1
                    if calls == 2: raise OSError('must-not-leak-storage-path')
                    original_replace(source, target)
                with patch('rsna_knee.data.model_runtime_audit.os.replace', side_effect=fail_terminal_replace):
                    with self.assertRaisesRegex(ValueError, 'model_runtime_report_storage_failed'):
                        run_model_runtime_audit(plan, plan_raw, metadata, images, training_report,
                            checkpoint, root/'storage-failure', model_loader=lambda *_args: Model(),
                            torch_api=fake_torch, source_archive=source_archive,
                            source_archive_sha256=source_archive_hash)
                self.assertEqual(calls, 2)
                stored = json.loads((root/'storage-failure'/'model-runtime-report.json').read_text())
                self.assertEqual((stored['attempted_series'], stored['completed_series'],
                                  stored['successful_series'], stored['failed_series']), (0, 0, 0, 0))


if __name__ == '__main__':
    unittest.main()
