"""Synthetic-only inference contract and safe checkpoint round-trip tests."""
import csv
import contextlib
import io
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18

from rsna_knee.models.resnet18 import ResNet18StudyModel
from rsna_knee.inference.fold0 import (
    TARGETS, TestStudyDataset, load_fold0_model, validate_training_report,
    write_submission, run_inference,
)


def report_fixture():
    return {
        'status': 'completed', 'mode': 'fold', 'fold': 0,
        'training_study_count': 333, 'validation_study_count': 167,
        'steps_completed': 1665, 'training_steps_planned': 1665, 'epochs_requested': 5,
        'model_kind': 'resnet18_imagenet1k_v1',
        'input_sha256': {
            'manifest': 'e88c61c22a463737f16951b7c76db03834034aecd1b31dc61b4ce495080b2be2',
            'folds': 'b82b376b0086201e3364b11a0203b62890e628442075a52ea4a9e09ee0e778ac'},
        'config': {'fold': 0, 'seed': 42, 'model_seed': 42, 'output_size': 256,
                   'max_slices': 8, 'slice_microbatch': 2, 'freeze_backbone': True,
                   'epochs': 5, 'learning_rate': .001, 'model_kind': 'resnet18_imagenet1k_v1'},
        'model_initialization': {
            'architecture': 'torchvision_resnet18', 'initialization': 'imagenet1k_v1',
            'model_seed': 42, 'freeze_backbone': True, 'trainable_parameter_count': 6156,
            'head_trainable_parameter_count': 6156, 'slice_microbatch': 2,
            'activation_checkpointing': False, 'batchnorm_running_statistics': 'frozen_eval_affine_frozen',
            'weights_sha256': 'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec',
            'weights_size': 46830571},
    }


class FoldZeroTests(unittest.TestCase):
    def test_target_contract_matches_training(self):
        from rsna_knee.labels.subscription import LABEL_COLUMNS
        self.assertEqual(TARGETS, LABEL_COLUMNS)

    def test_report_rejects_other_folds_and_preprocessing(self):
        validate_training_report(report_fixture())
        for field, value in [('fold', 1), ('status', 'partial_training')]:
            report = report_fixture(); report[field] = value
            with self.assertRaises(ValueError): validate_training_report(report)
        report = report_fixture(); report['config']['output_size'] = 64
        with self.assertRaises(ValueError): validate_training_report(report)

    def test_safe_roundtrip_logits_and_bad_states(self):
        torch.set_num_threads(1)
        backbone = resnet18(weights=None); backbone.fc = nn.Identity()
        original = ResNet18StudyModel(backbone, freeze_backbone=True).eval()
        image = torch.rand(1, 1, 32, 32); mask = torch.ones_like(image, dtype=torch.bool)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            torch.save(original.state_dict(), path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            loaded = load_fold0_model(path, digest, path.stat().st_size)
            with torch.no_grad():
                torch.testing.assert_close(original([image], [mask]), loaded([image], [mask]), rtol=0, atol=0)
            with self.assertRaises(ValueError): load_fold0_model(path, '0' * 64, path.stat().st_size)
            for state in [{'classifier.bias': torch.zeros(12)}, {'x': torch.tensor(float('nan'))}, {'x': 'not_tensor'}]:
                torch.save(state, path); digest = hashlib.sha256(path.read_bytes()).hexdigest()
                with self.assertRaises(ValueError): load_fold0_model(path, digest, path.stat().st_size)

    def test_csv_order_header_and_invalid_predictions(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'submission.csv'
            write_submission(path, ['studyB', 'studyA'], np.full((2, 12), .4))
            with path.open(newline='') as stream: rows = list(csv.reader(stream))
            self.assertEqual(rows[0], ['StudyInstanceUID', *TARGETS])
            self.assertEqual([row[0] for row in rows[1:]], ['studyB', 'studyA'])
            for ids, values in [(['a', 'a'], np.zeros((2, 12))), (['a'], np.zeros((2, 12))),
                                (['a'], np.full((1, 12), np.nan)), (['a'], np.full((1, 12), 2))]:
                with self.assertRaises(ValueError): write_submission(Path(directory) / 'bad.csv', ids, values)

    def test_test_dataset_all_series_order_and_safe_paths(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); images = root / 'test_series'; images.mkdir()
            (root / 'test.csv').write_text('StudyInstanceUID\nstudyB\nstudyA\n')
            series = root / 'test_series.csv'
            series.write_text('StudyInstanceUID,SeriesInstanceUID\nstudyA,s3\nstudyB,s1\nstudyB,s2\n')
            for study, names in [('studyA', ['s3']), ('studyB', ['s1', 's2'])]:
                for name in names: (images / study / name).mkdir(parents=True)
            decoder = lambda *a, **k: SimpleNamespace(images=np.ones((2, 1, 256, 256)), valid_mask=np.ones((2, 1, 256, 256), dtype=bool))
            data = TestStudyDataset(root / 'test.csv', series, images, preprocessor=decoder)
            self.assertEqual(data.study_ids, ('studyB', 'studyA'))
            self.assertEqual(len(data[0][0]), 2)
            (images / 'studyB' / 's2').rmdir()
            with self.assertRaises(ValueError): data[0]
            for text in ['StudyInstanceUID,SeriesInstanceUID\nstudyB,../escape\n',
                         'StudyInstanceUID,SeriesInstanceUID\nstudyB,s1\nstudyB,s1\n',
                         'StudyInstanceUID,SeriesInstanceUID\nstudyB,s1\n']:
                series.write_text(text)
                with self.assertRaises(ValueError): TestStudyDataset(root / 'test.csv', series, images)

    def test_runner_synthetic_csv_and_aggregate_report(self):
        class FakeModel(nn.Module):
            def forward(self, images, masks): return torch.arange(12, dtype=torch.float32).reshape(1, 12)
        class FakeData:
            study_ids = ('studyB', 'studyA')
            series_count = 3
            def __len__(self): return 2
            def __getitem__(self, index):
                if index >= 2: raise IndexError
                image = torch.ones(1, 1, 8, 8)
                return [image], [image.bool()]
        with TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'source'; source.mkdir()
            (source / 'report.json').write_text(json.dumps(report_fixture()))
            digest = hashlib.sha256((source / 'report.json').read_bytes()).hexdigest()
            with patch('rsna_knee.inference.fold0.load_fold0_model', return_value=FakeModel()), patch('rsna_knee.inference.fold0.TestStudyDataset', return_value=FakeData()):
                captured = io.StringIO()
                with contextlib.redirect_stdout(captured):
                    report = run_inference(root, source, '0' * 64, 1, digest, root / 'out', device='cpu')
            self.assertEqual(report['study_count'], 2)
            self.assertNotIn('studyB', json.dumps(report))
            self.assertTrue((root / 'out' / 'submission.csv').is_file())
            with (root / 'out' / 'submission.csv').open(newline='') as stream:
                csv_rows = list(csv.reader(stream))
            self.assertEqual([row[0] for row in csv_rows[1:]], list(FakeData.study_ids))
            expected = torch.sigmoid(torch.arange(12, dtype=torch.float32)).numpy()
            np.testing.assert_array_equal(np.asarray([row[1:] for row in csv_rows[1:]], dtype=float), np.tile(expected, (2, 1)))
            events = [json.loads(row) for row in captured.getvalue().splitlines()]
            self.assertEqual({event['stage'] for event in events}, {'startup', 'input_metadata', 'checkpoint', 'preprocess', 'model', 'write'})
            self.assertEqual(events[-1]['event'], 'inference_completed')

    def test_opt_in_runner_writes_identifier_free_recovery_journal_and_all_rows(self):
        class FakeModel(nn.Module):
            def forward(self, images, masks): return torch.zeros((1, 12))
        class FakeData:
            study_ids = ('studyA', 'studyB')
            series_count = 3
            extra_series_metadata_count = 1
            duplicate_series_metadata_count = 0
            def __init__(self, *args, **kwargs):
                self.last_recovery = None
                assert kwargs['recovery_policy_version'] == 1
            def __len__(self): return 2
            def __getitem__(self, index):
                image = torch.ones(1, 1, 8, 8)
                self.last_recovery = {'version': 1, 'study_ordinal': index + 1,
                                      'series_skipped': 1 if index == 0 else 0,
                                      'slice_skipped': 0,
                                      'reason_counts': {'pixel_payload_invalid': 1} if index == 0 else {}}
                return [image], [image.bool()]
        with TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'source'; source.mkdir()
            (source / 'report.json').write_text(json.dumps(report_fixture()))
            digest = hashlib.sha256((source / 'report.json').read_bytes()).hexdigest()
            with patch('rsna_knee.inference.fold0.load_fold0_model', return_value=FakeModel()), \
                 patch('rsna_knee.inference.fold0.TestStudyDataset', FakeData), contextlib.redirect_stdout(io.StringIO()):
                report = run_inference(root, source, '0' * 64, 1, digest, root / 'out',
                                       device='cpu', recovery_policy_version=1)
            journal = json.loads((root / 'out' / 'recovery-journal.json').read_text())
            self.assertEqual(report['study_count'], 2)
            self.assertEqual([entry['study_ordinal'] for entry in journal['studies']], [1, 2])
            self.assertNotIn('studyA', json.dumps(journal))
            self.assertNotIn('pixel payload', json.dumps(journal).lower())
            with (root / 'out' / 'submission.csv').open(newline='') as stream:
                self.assertEqual([row[0] for row in list(csv.reader(stream))[1:]], list(FakeData.study_ids))

    def test_opt_in_runner_persists_journal_on_fatal_study(self):
        from rsna_knee.inference.fold0 import InferenceError
        class FakeModel(nn.Module):
            def forward(self, images, masks): return torch.zeros((1, 12))
        class FakeData:
            study_ids = ('studyA',)
            series_count = 1
            extra_series_metadata_count = 0
            duplicate_series_metadata_count = 0
            def __init__(self, *args, **kwargs): self.last_recovery = None
            def __len__(self): return 1
            def __getitem__(self, index):
                self.last_recovery = {'version': 1, 'study_ordinal': 1,
                                      'series_skipped': 1, 'reason_counts': {'pixel_payload_invalid': 1}}
                raise InferenceError('study_series_missing', study_ordinal=1)
        with TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'source'; source.mkdir()
            (source / 'report.json').write_text(json.dumps(report_fixture()))
            digest = hashlib.sha256((source / 'report.json').read_bytes()).hexdigest()
            with patch('rsna_knee.inference.fold0.load_fold0_model', return_value=FakeModel()), \
                 patch('rsna_knee.inference.fold0.TestStudyDataset', FakeData), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(InferenceError):
                    run_inference(root, source, '0' * 64, 1, digest, root / 'out',
                                  device='cpu', recovery_policy_version=1)
            journal = json.loads((root / 'out' / 'recovery-journal.json').read_text())
            self.assertEqual(journal['failure']['reason_code'], 'study_series_missing')
            self.assertEqual(journal['studies'][0]['study_ordinal'], 1)
            self.assertNotIn('studyA', json.dumps(journal))
            self.assertFalse((root / 'out' / 'submission.csv').exists())


    def test_dataset_preserves_only_allowlisted_reasons_and_ordinals(self):
        from rsna_knee.data.dicom_preprocessing import DicomPreprocessingError
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'test.csv').write_text('StudyInstanceUID\nsecretStudyA\nsecretStudyB\n')
            (root / 'series.csv').write_text('StudyInstanceUID,SeriesInstanceUID\nsecretStudyA,s1\nsecretStudyB,s2\nsecretStudyB,s3\n')
            for study, name in [('secretStudyA', 's1'), ('secretStudyB', 's2'), ('secretStudyB', 's3')]:
                (root / 'images' / study / name).mkdir(parents=True)
            for reason in ['inconsistent_orientation', 'duplicate_slice_position', 'pixel_decode_failed', 'secret_reason']:
                def decoder(path, **kwargs):
                    if path.name == 's3':
                        raise DicomPreprocessingError(reason, 'SECRET medical /private/path')
                    return SimpleNamespace(images=np.ones((1, 1, 256, 256)), valid_mask=np.ones((1, 1, 256, 256), dtype=bool))
                data = TestStudyDataset(root / 'test.csv', root / 'series.csv', root / 'images', preprocessor=decoder)
                self.assertEqual(len(data[0][0]), 1)
                with self.assertRaises(ValueError) as caught:
                    data[1]
                error = caught.exception
                self.assertEqual(error.reason_code, reason if reason != 'secret_reason' else 'preprocessing_failed')
                self.assertEqual((error.study_ordinal, error.series_ordinal), (2, 2))
                self.assertNotIn('SECRET', str(error))
                self.assertNotIn('secret', str(error))
                self.assertTrue(error.__suppress_context__)
            def unknown_decoder(*args, **kwargs): raise RuntimeError('SECRET')
            data = TestStudyDataset(root / 'test.csv', root / 'series.csv', root / 'images', preprocessor=unknown_decoder)
            with self.assertRaises(ValueError) as caught:
                data[0]
            self.assertEqual(caught.exception.exception_family, 'runtime')
            self.assertEqual(caught.exception.reason_code, 'unexpected_error')

    def test_opt_in_recovery_keeps_all_test_rows_and_skips_only_missing_series(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); images = root / 'images'; images.mkdir()
            (root / 'test.csv').write_text('StudyInstanceUID\nstudyA\nstudyB\n')
            (root / 'series.csv').write_text(
                'StudyInstanceUID,SeriesInstanceUID\nstudyA,missing\nstudyA,good\n'
                'studyA,good\nstudyB,goodB\nextra-study,extra-series\n'
            )
            (images / 'studyA' / 'good').mkdir(parents=True)
            (images / 'studyB' / 'goodB').mkdir(parents=True)
            def decoder(path, **kwargs):
                image = np.ones((1, 1, 256, 256), dtype=np.float32)
                return SimpleNamespace(images=image, valid_mask=image.astype(bool))
            data = TestStudyDataset(
                root / 'test.csv', root / 'series.csv', images,
                preprocessor=decoder, recovery_policy_version=1,
            )
            self.assertEqual(data.study_ids, ('studyA', 'studyB'))
            self.assertEqual(data.series_count, 3)
            self.assertEqual(data.extra_series_metadata_count, 1)
            self.assertEqual(data.duplicate_series_metadata_count, 1)
            self.assertEqual(len(data[0][0]), 1)
            self.assertEqual(data.last_recovery, {
                'version': 1, 'study_ordinal': 1,
                'series_skipped': 1, 'slice_skipped': 0,
                'reason_counts': {'series_path_unavailable': 1},
                'series_reasons': [{'series_ordinal': 1, 'reason_code': 'series_path_unavailable'}],
                'slice_reasons': [],
            })
            self.assertEqual(len(data[1][0]), 1)

    def test_recovery_metadata_conflicts_and_all_bad_study_fail_closed(self):
        from rsna_knee.data.dicom_preprocessing import DicomPreprocessingError
        from rsna_knee.inference.fold0 import InferenceError
        with TemporaryDirectory() as directory:
            root = Path(directory); images = root / 'images'; images.mkdir()
            (root / 'test.csv').write_text('StudyInstanceUID\nstudyA\n')
            series = root / 'series.csv'
            series.write_text('StudyInstanceUID,SeriesInstanceUID\nstudyA,s1\nother,s1\n')
            with self.assertRaises(ValueError):
                TestStudyDataset(root / 'test.csv', series, images, recovery_policy_version=1)
            series.write_text('StudyInstanceUID,SeriesInstanceUID\nstudyA,../escape\n')
            with self.assertRaises(ValueError):
                TestStudyDataset(root / 'test.csv', series, images, recovery_policy_version=1)
            series.write_text('StudyInstanceUID,SeriesInstanceUID\nstudyA,s1\n')
            (images / 'studyA' / 's1').mkdir(parents=True)
            def bad(*args, **kwargs):
                raise DicomPreprocessingError('series_files_missing', 'SECRET')
            data = TestStudyDataset(root / 'test.csv', series, images, preprocessor=bad, recovery_policy_version=1)
            with self.assertRaisesRegex(InferenceError, 'study_series_missing'):
                data[0]

    def test_recovery_rejects_same_series_duplicate_with_conflicting_full_row(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); images = root / 'images'; images.mkdir()
            (root / 'test.csv').write_text('StudyInstanceUID\nstudyA\n')
            series = root / 'series.csv'
            series.write_text(
                'StudyInstanceUID,SeriesInstanceUID,Anatomical_Plane\n'
                'studyA,s1,Axial\nstudyA,s1,Sagittal\n'
            )
            with self.assertRaisesRegex(ValueError, 'series_metadata_invalid'):
                TestStudyDataset(root / 'test.csv', series, images, recovery_policy_version=1)

    def test_recovery_does_not_keep_prior_study_journal_after_failure(self):
        from rsna_knee.inference.fold0 import InferenceError
        with TemporaryDirectory() as directory:
            root = Path(directory); images = root / 'images'; images.mkdir()
            (root / 'test.csv').write_text('StudyInstanceUID\nstudyA\nstudyB\n')
            (root / 'series.csv').write_text(
                'StudyInstanceUID,SeriesInstanceUID\nstudyA,s1\nstudyB,s2\n'
            )
            (images / 'studyA' / 's1').mkdir(parents=True)
            (images / 'studyB' / 's2').mkdir(parents=True)
            def decoder(path, **kwargs):
                if path.name == 's1':
                    return SimpleNamespace(
                        images=np.ones((1, 1, 256, 256), dtype=np.float32),
                        valid_mask=np.ones((1, 1, 256, 256), dtype=bool),
                        metadata={'recovery': {'version': 1, 'reason_counts': {'missing_geometry': 1},
                                               'slice_reasons': [{'slice_ordinal': 1, 'reason_code': 'missing_geometry'}]}},
                    )
                raise MemoryError('SECRET')
            data = TestStudyDataset(root / 'test.csv', root / 'series.csv', images,
                                    preprocessor=decoder, recovery_policy_version=1)
            data[0]
            self.assertIsNotNone(data.last_recovery)
            with self.assertRaises(InferenceError):
                data[1]
            self.assertIsNone(data.last_recovery)

    def test_recovery_does_not_downgrade_dangling_symlinked_study_parent(self):
        from rsna_knee.inference.fold0 import InferenceError
        with TemporaryDirectory() as directory:
            root = Path(directory); images = root / 'images'; images.mkdir()
            (root / 'test.csv').write_text('StudyInstanceUID\nstudyA\n')
            series = root / 'series.csv'
            series.write_text('StudyInstanceUID,SeriesInstanceUID\nstudyA,s1\n')
            data = TestStudyDataset(root / 'test.csv', series, images, recovery_policy_version=1)
            study_path = images / 'studyA'
            with patch.object(Path, 'is_symlink', autospec=True,
                              side_effect=lambda path: path == study_path):
                with self.assertRaises(InferenceError) as caught:
                    data[0]
            self.assertEqual(caught.exception.reason_code, 'series_path_invalid')

    def test_runner_failure_has_safe_diagnostics_and_no_csv(self):
        from rsna_knee.inference.fold0 import InferenceError
        class FakeModel(nn.Module):
            def forward(self, images, masks): return torch.zeros((1, 12))
        class FakeData:
            study_ids = ('SECRET_UID_A', 'SECRET_UID_B')
            series_count = 3
            def __len__(self): return 2
            def __getitem__(self, index):
                if index == 1:
                    raise InferenceError('pixel_decode_failed', study_ordinal=2, series_ordinal=2)
                value = torch.ones(1, 1, 8, 8)
                return [value], [value.bool()]
        with TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'source'; source.mkdir()
            (source / 'report.json').write_text(json.dumps(report_fixture()))
            digest = hashlib.sha256((source / 'report.json').read_bytes()).hexdigest()
            output = io.StringIO()
            with patch('rsna_knee.inference.fold0.load_fold0_model', return_value=FakeModel()), patch('rsna_knee.inference.fold0.TestStudyDataset', return_value=FakeData()), contextlib.redirect_stdout(output):
                with self.assertRaises(InferenceError):
                    run_inference(root, source, '0' * 64, 1, digest, root / 'out', device='cpu')
            rows = [json.loads(row) for row in output.getvalue().splitlines()]
            failure = rows[-1]
            self.assertEqual(failure['event'], 'inference_failed')
            self.assertEqual(failure['stage'], 'preprocess')
            self.assertEqual(failure['reason_code'], 'pixel_decode_failed')
            self.assertEqual((failure['study_ordinal'], failure['series_ordinal'], failure['completed']), (2, 2, 1))
            self.assertEqual(json.loads((root / 'out' / 'inference-diagnostics.json').read_text()), failure)
            self.assertFalse((root / 'out' / 'submission.csv').exists())
            self.assertNotIn('SECRET', output.getvalue())
            self.assertNotIn(str(root), output.getvalue())
            self.assertIn('inference_progress', [row['event'] for row in rows])

    def test_unknown_errors_and_invalid_checkpoint_are_bounded(self):
        from rsna_knee.inference.fold0 import InferenceError
        for failure, family in [(RuntimeError('SECRET path/UID'), 'runtime'), (type('SECRETException', (Exception,), {})('SECRET'), 'other')]:
            with TemporaryDirectory() as directory:
                root = Path(directory); source = root / 'source'; source.mkdir()
                (source / 'report.json').write_text(json.dumps(report_fixture()))
                digest = hashlib.sha256((source / 'report.json').read_bytes()).hexdigest()
                output = io.StringIO()
                class EmptyData:
                    series_count = 0
                    def __len__(self): return 0
                with patch('rsna_knee.inference.fold0.load_fold0_model', side_effect=failure), patch('rsna_knee.inference.fold0.TestStudyDataset', return_value=EmptyData()), contextlib.redirect_stdout(output):
                    with self.assertRaises(InferenceError):
                        run_inference(root, source, '0' * 64, 1, digest, root / 'out', device='cpu')
                record = json.loads(output.getvalue().splitlines()[-1])
                self.assertEqual(record['stage'], 'checkpoint')
                self.assertEqual(record['exception_family'], family)
                self.assertEqual(record['reason_code'], 'unexpected_error')
                self.assertNotIn('SECRET', output.getvalue())
                self.assertFalse((root / 'out' / 'submission.csv').exists())


    def test_failure_stages_invalid_pins_and_output_no_overwrite(self):
        from rsna_knee.inference.fold0 import InferenceError
        class FakeData:
            series_count = 1
            study_ids = ('PRIVATE_UID',)
            def __len__(self): return 1
            def __getitem__(self, index):
                image = torch.ones(1, 1, 8, 8)
                return [image], [image.bool()]
        class FakeModel(nn.Module):
            def forward(self, images, masks): return torch.zeros((1, 12))
        with TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'source'; source.mkdir()
            (source / 'report.json').write_text(json.dumps(report_fixture()))
            digest = hashlib.sha256((source / 'report.json').read_bytes()).hexdigest()
            scenarios = [
                ('metadata', patch('rsna_knee.inference.fold0.TestStudyDataset', side_effect=ValueError('test_metadata_invalid')), 'input_metadata', 'test_metadata_invalid'),
                ('checkpoint', patch('rsna_knee.inference.fold0.load_fold0_model', side_effect=ValueError('checkpoint_invalid')), 'checkpoint', 'checkpoint_invalid'),
                ('model', patch.object(FakeModel, 'forward', side_effect=MemoryError('SECRET')), 'model', 'unexpected_error'),
                ('write', patch('rsna_knee.inference.fold0.write_submission', side_effect=OSError('SECRET')), 'write', 'unexpected_error'),
            ]
            for name, failure_patch, expected_stage, expected_reason in scenarios:
                output = io.StringIO()
                with patch('rsna_knee.inference.fold0.TestStudyDataset', return_value=FakeData()), patch('rsna_knee.inference.fold0.load_fold0_model', return_value=FakeModel()), failure_patch, contextlib.redirect_stdout(output):
                    with self.assertRaises(InferenceError):
                        run_inference(root, source, '0' * 64, 1, digest, root / name, device='cpu')
                record = json.loads(output.getvalue().splitlines()[-1])
                self.assertEqual((record['stage'], record['reason_code']), (expected_stage, expected_reason))
                self.assertFalse((root / name / 'submission.csv').exists())
                self.assertNotIn('SECRET', output.getvalue())
                self.assertNotIn('PRIVATE_UID', output.getvalue())
            prior = root / 'prior'; prior.mkdir()
            (prior / 'submission.csv').write_text('KEEP')
            (prior / 'inference-diagnostics.json').write_text('KEEP DIAGNOSTICS')
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(InferenceError):
                run_inference(root, source, '0' * 64, 1, digest, prior, device='cpu')
            self.assertEqual((prior / 'submission.csv').read_text(), 'KEEP')
            self.assertEqual((prior / 'inference-diagnostics.json').read_text(), 'KEEP DIAGNOSTICS')

    def test_forged_reason_never_reads_error_text(self):
        from rsna_knee.data.dicom_preprocessing import DicomPreprocessingError
        class HostileError(DicomPreprocessingError):
            def __str__(self): raise AssertionError('raw text must not be read')
            def __repr__(self): raise AssertionError('raw repr must not be read')
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'test.csv').write_text('StudyInstanceUID\nprivateUID\n')
            (root / 'series.csv').write_text('StudyInstanceUID,SeriesInstanceUID\nprivateUID,s1\n')
            (root / 'images' / 'privateUID' / 's1').mkdir(parents=True)
            for forged in ['SECRET_UID/path', ['inconsistent_orientation'], None]:
                def decoder(*args, **kwargs): raise HostileError(forged, 'SECRET')
                data = TestStudyDataset(root / 'test.csv', root / 'series.csv', root / 'images', preprocessor=decoder)
                with self.assertRaises(ValueError) as caught:
                    data[0]
                self.assertEqual(caught.exception.reason_code, 'preprocessing_failed')
                self.assertNotIn('SECRET', str(caught.exception))


if __name__ == '__main__': unittest.main()
