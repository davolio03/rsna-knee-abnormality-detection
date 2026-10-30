"""Offline inference for the explicitly selected, trusted frozen fold-zero model."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from rsna_knee.data.dicom_preprocessing import DicomPreprocessingError, preprocess_dicom_series
from rsna_knee.models.resnet18 import ResNet18StudyModel, IMAGENET1K_V1_SHA256, IMAGENET1K_V1_SIZE

TARGETS = ('ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture')
SOURCE_EXECUTION = 361175523
SOURCE_INPUT_PINS = {
    'manifest': 'e88c61c22a463737f16951b7c76db03834034aecd1b31dc61b4ce495080b2be2',
    'folds': 'b82b376b0086201e3364b11a0203b62890e628442075a52ea4a9e09ee0e778ac',
}


# Never derive diagnostics from an exception's free-text message or class name.
_SAFE_REASONS = DicomPreprocessingError.SAFE_REASON_CODES | frozenset({
    'artifact_pin_required', 'identifier_invalid', 'image_root_invalid', 'series_path_invalid',
    'test_metadata_invalid', 'duplicate_study', 'series_metadata_invalid', 'study_series_missing',
    'preprocessed_tensor_invalid', 'source_report_provenance_mismatch', 'checkpoint_invalid',
    'submission_invalid', 'submission_readback_failed', 'device_invalid', 'output_must_be_new',
    'output_path_invalid', 'source_mount_invalid', 'report_pin_mismatch', 'cuda_required',
    'prediction_invalid', 'unexpected_error',
})


class InferenceError(ValueError):
    """Identifier-free failure; ordinal positions are one-based, never source IDs."""
    def __init__(self, reason_code, *, study_ordinal=None, series_ordinal=None, exception_family='validation'):
        self.reason_code = reason_code if type(reason_code) is str and reason_code in _SAFE_REASONS else 'unexpected_error'
        self.study_ordinal = study_ordinal if type(study_ordinal) is int and study_ordinal > 0 else None
        self.series_ordinal = series_ordinal if type(series_ordinal) is int and series_ordinal > 0 else None
        self.exception_family = exception_family if type(exception_family) is str and exception_family in {'memory', 'io', 'validation', 'runtime', 'type', 'dependency', 'other'} else 'other'
        super().__init__(self.reason_code)


def _exception_family(error):
    if isinstance(error, InferenceError):
        return error.exception_family
    for kind, family in ((MemoryError, 'memory'), (torch.OutOfMemoryError, 'memory'), (OSError, 'io'), (ValueError, 'validation'),
                         (RuntimeError, 'runtime'), (TypeError, 'type'), (ImportError, 'dependency')):
        if isinstance(error, kind):
            return family
    return 'other'


def _safe_reason(error):
    if isinstance(error, InferenceError):
        return error.reason_code
    if isinstance(error, DicomPreprocessingError):
        code = error.reason_code
        return code if type(code) is str and code in DicomPreprocessingError.SAFE_REASON_CODES else 'preprocessing_failed'
    # Exact built-in categorical errors from this module only; no substring matching.
    if type(error) is ValueError and len(error.args) == 1:
        code = error.args[0]
        if type(code) is str and code in _SAFE_REASONS:
            return code
    return 'unexpected_error'


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()


def _pin(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('artifact_pin_required')


def _component(value):
    if type(value) is not str or re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]{0,127}', value) is None or value in {'.', '..'}:
        raise ValueError('identifier_invalid')
    return value


def _contained(root, *parts):
    root = Path(root)
    if root.is_symlink() or not root.is_dir(): raise ValueError('image_root_invalid')
    current = root
    for part in parts:
        current = current / _component(part)
        if current.is_symlink() or not current.is_dir(): raise ValueError('series_path_invalid')
    current.resolve(strict=True).relative_to(root.resolve(strict=True))
    return current


def _csv(path, required):
    try:
        with Path(path).open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            if (reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames)
                    or not set(required).issubset(reader.fieldnames)):
                raise ValueError
            rows = list(reader)
            if not rows or any(None in row or any(row.get(name) is None for name in required) for row in rows):
                raise ValueError
            return rows
    except (OSError, UnicodeError, csv.Error, ValueError):
        raise ValueError('test_metadata_invalid') from None


class TestStudyDataset:
    """Test-only metadata reader; no reports, labels, or cohort artifacts are read."""
    def __init__(self, test_csv, series_csv, image_root, *, preprocessor=None,
                 recovery_policy_version=None):
        if recovery_policy_version not in (None, 1) or isinstance(recovery_policy_version, bool):
            raise ValueError('test_metadata_invalid')
        self.recovery_policy_version = recovery_policy_version
        self.extra_series_metadata_count = 0
        self.duplicate_series_metadata_count = 0
        self.last_recovery = None
        rows = _csv(test_csv, ('StudyInstanceUID',))
        self.study_ids = tuple(_component(row['StudyInstanceUID']) for row in rows)
        if len(set(self.study_ids)) != len(self.study_ids): raise ValueError('duplicate_study')
        grouped = {key: [] for key in self.study_ids}
        seen = {}
        for row in _csv(series_csv, ('StudyInstanceUID', 'SeriesInstanceUID')):
            study = _component(row['StudyInstanceUID']); series = _component(row['SeriesInstanceUID'])
            if series in seen:
                previous_study, previous_row = seen[series]
                if previous_study != study or previous_row != row:
                    raise ValueError('series_metadata_invalid')
                if recovery_policy_version == 1:
                    self.duplicate_series_metadata_count += 1
                    continue
                raise ValueError('series_metadata_invalid')
            seen[series] = (study, dict(row))
            if study not in grouped:
                if recovery_policy_version == 1:
                    self.extra_series_metadata_count += 1
                    continue
                raise ValueError('series_metadata_invalid')
            grouped[study].append(series)
        if any(not names for names in grouped.values()): raise ValueError('study_series_missing')
        self.image_root = Path(image_root)
        _contained(self.image_root)
        self._series = tuple(tuple(grouped[key]) for key in self.study_ids)
        self.series_count = sum(map(len, self._series))
        self.preprocessor = preprocessor or preprocess_dicom_series

    def __len__(self): return len(self.study_ids)

    def __getitem__(self, index):
        if type(index) is not int or not 0 <= index < len(self): raise IndexError('test_index_invalid')
        self.last_recovery = None
        images, masks = [], []
        reason_counts = {}
        skipped = 0
        skipped_series = []
        skipped_slices = []
        for series_index, series in enumerate(self._series[index]):
            try:
                try:
                    path = _contained(self.image_root, self.study_ids[index], series)
                except ValueError as error:
                    target = self.image_root / self.study_ids[index] / series
                    symlink_component = any(
                        component.is_symlink()
                        for component in (self.image_root / self.study_ids[index], target)
                    )
                    if (self.recovery_policy_version == 1 and type(error) is ValueError
                            and error.args == ('series_path_invalid',)
                            and not symlink_component
                            and not target.exists() and not target.is_symlink()):
                        reason = 'series_path_unavailable'
                        reason_counts[reason] = reason_counts.get(reason, 0) + 1
                        skipped += 1
                        skipped_series.append({'series_ordinal': series_index + 1,
                                               'reason_code': reason})
                        self.last_recovery = {'version': 1, 'study_ordinal': index + 1,
                                              'series_skipped': skipped,
                                              'slice_skipped': len(skipped_slices),
                                              'reason_counts': dict(sorted(reason_counts.items())),
                                              'series_reasons': list(skipped_series),
                                              'slice_reasons': list(skipped_slices)}
                        continue
                    raise
                if self.recovery_policy_version == 1:
                    result = self.preprocessor(path, max_slices=8, output_size=256,
                                               recovery_policy_version=1)
                else:
                    result = self.preprocessor(path, max_slices=8, output_size=256)
                image = np.asarray(result.images); mask = np.asarray(result.valid_mask)
                if (image.ndim != 4 or not 1 <= image.shape[0] <= 8
                        or image.shape[1:] != (1, 256, 256) or mask.shape != image.shape
                        or not np.isrealobj(image) or not np.isfinite(image).all()
                        or not np.isin(mask, (0, 1)).all() or (image < 0).any() or (image > 1).any()
                        or not mask.reshape(len(mask), -1).any(axis=1).all()
                        or np.any(image[~mask.astype(bool)] != 0)):
                    raise ValueError('preprocessed_tensor_invalid')
                if self.recovery_policy_version == 1:
                    series_recovery = getattr(result, 'metadata', {}).get('recovery', {})
                    if series_recovery:
                        if type(series_recovery) is not dict or series_recovery.get('version') != 1:
                            raise ValueError('preprocessed_tensor_invalid')
                        for reason, count in series_recovery.get('reason_counts', {}).items():
                            if reason not in _RECOVERABLE_SERIES_REASONS or type(count) is not int or count < 1:
                                raise ValueError('preprocessed_tensor_invalid')
                            reason_counts[reason] = reason_counts.get(reason, 0) + count
                        for detail in series_recovery.get('slice_reasons', []):
                            ordinal = detail.get('slice_ordinal')
                            reason = detail.get('reason_code')
                            if type(ordinal) is not int or ordinal < 1 or reason not in _RECOVERABLE_SERIES_REASONS:
                                raise ValueError('preprocessed_tensor_invalid')
                            skipped_slices.append({'series_ordinal': series_index + 1,
                                                   'slice_ordinal': ordinal, 'reason_code': reason})
                images.append(torch.as_tensor(image.copy(), dtype=torch.float32))
                masks.append(torch.as_tensor(mask.copy(), dtype=torch.bool))
            except DicomPreprocessingError as error:
                if self.recovery_policy_version == 1 and error.reason_code in _RECOVERABLE_SERIES_REASONS:
                    reason_counts[error.reason_code] = reason_counts.get(error.reason_code, 0) + 1
                    skipped += 1
                    details = error.recovery_details
                    if details is not None:
                        if type(details) is not dict or details.get('version') != 1:
                            raise InferenceError('preprocessed_tensor_invalid', study_ordinal=index + 1,
                                                 series_ordinal=series_index + 1) from None
                        for reason, count in details.get('reason_counts', {}).items():
                            if reason not in _RECOVERABLE_SERIES_REASONS or type(count) is not int or count < 1:
                                raise InferenceError('preprocessed_tensor_invalid', study_ordinal=index + 1,
                                                     series_ordinal=series_index + 1) from None
                            reason_counts[reason] = reason_counts.get(reason, 0) + count
                        for detail in details.get('slice_reasons', []):
                            ordinal = detail.get('slice_ordinal')
                            reason = detail.get('reason_code')
                            if type(ordinal) is not int or ordinal < 1 or reason not in _RECOVERABLE_SERIES_REASONS:
                                raise InferenceError('preprocessed_tensor_invalid', study_ordinal=index + 1,
                                                     series_ordinal=series_index + 1) from None
                            skipped_slices.append({'series_ordinal': series_index + 1,
                                                   'slice_ordinal': ordinal, 'reason_code': reason})
                    skipped_series.append({'series_ordinal': series_index + 1,
                                           'reason_code': error.reason_code})
                    self.last_recovery = {'version': 1, 'study_ordinal': index + 1,
                                          'series_skipped': skipped,
                                          'slice_skipped': len(skipped_slices),
                                          'reason_counts': dict(sorted(reason_counts.items())),
                                          'series_reasons': list(skipped_series),
                                          'slice_reasons': list(skipped_slices)}
                    continue
                raise InferenceError(_safe_reason(error), study_ordinal=index + 1,
                                     series_ordinal=series_index + 1, exception_family=_exception_family(error)) from None
            except Exception as error:
                raise InferenceError(_safe_reason(error), study_ordinal=index + 1,
                                     series_ordinal=series_index + 1, exception_family=_exception_family(error)) from None
        if not images:
            if self.recovery_policy_version == 1:
                self.last_recovery = {'version': 1, 'study_ordinal': index + 1,
                                      'series_skipped': skipped, 'slice_skipped': len(skipped_slices),
                                      'reason_counts': dict(sorted(reason_counts.items())),
                                      'series_reasons': list(skipped_series),
                                      'slice_reasons': list(skipped_slices)}
            raise InferenceError('study_series_missing', study_ordinal=index + 1,
                                 exception_family='validation')
        self.last_recovery = ({'version': 1, 'study_ordinal': index + 1,
                               'series_skipped': skipped, 'slice_skipped': len(skipped_slices),
                               'reason_counts': dict(sorted(reason_counts.items())),
                               'series_reasons': list(skipped_series),
                               'slice_reasons': list(skipped_slices)}
                              if self.recovery_policy_version == 1 else None)
        return images, masks


_RECOVERABLE_SERIES_REASONS = frozenset({
    'series_files_missing', 'series_path_unavailable', 'header_read_failed', 'missing_geometry',
    'inconsistent_orientation', 'inconsistent_spacing', 'inconsistent_pixel_layout',
    'duplicate_slice_position', 'unsupported_frame_count', 'unsupported_modality_lut',
    'unsupported_pixel_layout', 'invalid_intensity_metadata', 'incomplete_intensity_metadata',
    'decoded_metadata_mismatch', 'pixel_payload_invalid', 'pixel_transfer_syntax_unsupported',
    'dicom_file_parse_invalid',
    'preprocessing_failed',
})


def validate_training_report(report):
    expected = {'status': 'completed', 'mode': 'fold', 'fold': 0,
                'training_study_count': 333, 'validation_study_count': 167,
                'steps_completed': 1665, 'training_steps_planned': 1665, 'epochs_requested': 5,
                'model_kind': 'resnet18_imagenet1k_v1', 'input_sha256': SOURCE_INPUT_PINS}
    config = {'fold': 0, 'seed': 42, 'model_seed': 42, 'output_size': 256, 'max_slices': 8,
              'slice_microbatch': 2, 'freeze_backbone': True, 'epochs': 5, 'learning_rate': .001,
              'model_kind': 'resnet18_imagenet1k_v1'}
    initialization = {'architecture': 'torchvision_resnet18', 'initialization': 'imagenet1k_v1',
                      'model_seed': 42, 'freeze_backbone': True, 'trainable_parameter_count': 6156,
                      'head_trainable_parameter_count': 6156, 'slice_microbatch': 2,
                      'activation_checkpointing': False, 'batchnorm_running_statistics': 'frozen_eval_affine_frozen',
                      'weights_sha256': IMAGENET1K_V1_SHA256, 'weights_size': IMAGENET1K_V1_SIZE}
    def matches(actual, wanted):
        return isinstance(actual, dict) and all(type(actual.get(k)) is type(v) and actual[k] == v for k, v in wanted.items())
    if not matches(report, expected) or not matches(report.get('config'), config) or not matches(report.get('model_initialization'), initialization):
        raise ValueError('source_report_provenance_mismatch')


def load_fold0_model(checkpoint, expected_sha256, expected_size):
    """Verify before safe loading; never tolerate partial keys or a random head."""
    _pin(expected_sha256)
    path = Path(checkpoint)
    try:
        if (type(expected_size) is not int or expected_size <= 0 or path.is_symlink()
                or path.stat().st_size != expected_size or _hash(path) != expected_sha256): raise ValueError
        state = torch.load(path, map_location='cpu', weights_only=True)
        if (not isinstance(state, dict) or not state or any(type(k) is not str or not isinstance(v, torch.Tensor)
                or not torch.isfinite(v).all() for k, v in state.items())): raise ValueError
        from torchvision.models import resnet18
        backbone = resnet18(weights=None); backbone.fc = nn.Identity()
        model = ResNet18StudyModel(backbone, slice_microbatch=2, checkpoint_trunk=False, freeze_backbone=True)
        model.load_state_dict(state, strict=True)
        return model.eval()
    except Exception:
        raise ValueError('checkpoint_invalid') from None


def write_submission(path, study_ids, probabilities):
    ids = tuple(_component(key) for key in study_ids)
    values = np.asarray(probabilities)
    if (not ids or len(set(ids)) != len(ids) or values.shape != (len(ids), 12)
            or not np.isrealobj(values) or not np.isfinite(values).all()
            or (values < 0).any() or (values > 1).any()): raise ValueError('submission_invalid')
    path = Path(path)
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['StudyInstanceUID', *TARGETS])
        writer.writerows([key, *row.tolist()] for key, row in zip(ids, values))
    with path.open(encoding='utf-8', newline='') as stream: rows = list(csv.reader(stream))
    if rows[0] != ['StudyInstanceUID', *TARGETS] or tuple(row[0] for row in rows[1:]) != ids:
        raise ValueError('submission_readback_failed')
    if not np.array_equal(np.asarray([row[1:] for row in rows[1:]], dtype=float), values):
        raise ValueError('submission_readback_failed')


def run_inference(competition_root, source_root, checkpoint_sha256, checkpoint_size, report_sha256, output_dir, *, device='cuda', recovery_policy_version=None):
    """Run once against the mounted test corpus; expose only aggregate progress."""
    started = time.perf_counter()
    stage = 'startup'
    completed = total = series_count = 0
    study_ordinal = None
    output = None
    output_created = False
    recovery_journal = []
    data = None

    def emit(event, **extra):
        record = {'event': event, 'stage': stage, 'completed': completed, 'total': total,
                  'series_count': series_count, 'study_ordinal': study_ordinal,
                  'elapsed_seconds': time.perf_counter() - started, **extra}
        print(json.dumps(record, allow_nan=False, sort_keys=True), flush=True)
        return record

    try:
        emit('inference_stage')
        source = Path(source_root)
        output = Path(output_dir)
        if output.exists() or output.is_symlink():
            raise ValueError('output_must_be_new')
        for path in (output, *output.parents):
            if path.is_symlink():
                raise ValueError('output_path_invalid')
        output.mkdir(parents=True, exist_ok=False)
        output_created = True
        if recovery_policy_version not in (None, 1) or isinstance(recovery_policy_version, bool):
            raise ValueError('test_metadata_invalid')
        stage = 'input_metadata'
        emit('inference_stage')
        competition = Path(competition_root)
        if recovery_policy_version == 1:
            data = TestStudyDataset(competition / 'test.csv', competition / 'test_series.csv',
                                    competition / 'test_series', recovery_policy_version=1)
        else:
            data = TestStudyDataset(competition / 'test.csv', competition / 'test_series.csv', competition / 'test_series')
        total = len(data)
        series_count = data.series_count
        stage = 'checkpoint'
        emit('inference_stage')
        _pin(report_sha256)
        if device not in {'cpu', 'cuda'}:
            raise ValueError('device_invalid')
        if source.is_symlink() or not source.is_dir():
            raise ValueError('source_mount_invalid')
        report_path = source / 'report.json'
        if report_path.is_symlink() or _hash(report_path) != report_sha256:
            raise ValueError('report_pin_mismatch')
        validate_training_report(json.loads(report_path.read_text(encoding='utf-8')))
        model = load_fold0_model(source / 'checkpoint.pt', checkpoint_sha256, checkpoint_size)
        torch.set_num_threads(1)
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('cuda_required')
        model = model.to(device).eval()
        probabilities = []
        with torch.inference_mode():
            for index in range(total):
                study_ordinal = index + 1
                stage = 'preprocess'
                emit('inference_stage')
                try:
                    images, masks = data[index]
                except Exception:
                    if (recovery_policy_version == 1 and getattr(data, 'last_recovery', None)
                            and (not recovery_journal or recovery_journal[-1] != data.last_recovery)):
                        recovery_journal.append(data.last_recovery)
                    raise
                if recovery_policy_version == 1:
                    recovery_journal.append(data.last_recovery)
                stage = 'model'
                emit('inference_stage')
                logits = model([v.to(device) for v in images], [v.to(device) for v in masks])
                if logits.shape != (1, 12) or not torch.isfinite(logits).all():
                    raise ValueError('prediction_invalid')
                probabilities.append(torch.sigmoid(logits[0]).cpu().numpy())
                completed += 1
                emit('inference_progress')
        stage = 'write'
        emit('inference_stage')
        write_submission(output / 'submission.csv', data.study_ids, np.asarray(probabilities))
        if recovery_policy_version == 1:
            with (output / 'recovery-journal.json').open('x', encoding='utf-8') as stream:
                stream.write(json.dumps({'version': 1, 'studies': recovery_journal,
                                         'extra_series_metadata_rows': data.extra_series_metadata_count,
                                         'duplicate_series_metadata_rows': data.duplicate_series_metadata_count},
                                        sort_keys=True, allow_nan=False) + '\n')
        summary = {'status': 'completed', 'source_execution': SOURCE_EXECUTION, 'fold': 0,
                   'checkpoint_sha256': checkpoint_sha256, 'checkpoint_size': checkpoint_size,
                   'report_sha256': report_sha256, 'study_count': total, 'series_count': series_count,
                   'targets': list(TARGETS), 'output_size': 256, 'max_slices': 8, 'slice_microbatch': 2,
                   'pooling': 'equal_mean_slice_then_series_logits_then_sigmoid',
                   'device': device, 'elapsed_seconds': time.perf_counter() - started}
        if recovery_policy_version == 1:
            summary['recovery_policy_version'] = 1
            summary['recovery_journal'] = 'recovery-journal.json'
            summary['series_skipped'] = sum(item['series_skipped'] for item in recovery_journal)
            summary['slice_skipped'] = sum(item['slice_skipped'] for item in recovery_journal)
        with (output / 'inference-report.json').open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        emit('inference_completed')
        return summary
    except Exception as error:
        reason = _safe_reason(error)
        series_ordinal = error.series_ordinal if isinstance(error, InferenceError) else None
        failure = emit('inference_failed', reason_code=reason, series_ordinal=series_ordinal,
                       exception_family=_exception_family(error))
        if output_created:
            # A failed run must never publish a partially written submission.
            try:
                (output / 'submission.csv').unlink(missing_ok=True)
                if recovery_policy_version == 1:
                    if (getattr(data, 'last_recovery', None)
                            and (not recovery_journal or recovery_journal[-1] != data.last_recovery)):
                        recovery_journal.append(data.last_recovery)
                    with (output / 'recovery-journal.json').open('x', encoding='utf-8') as stream:
                        stream.write(json.dumps({'version': 1, 'studies': recovery_journal,
                                                 'failure': failure}, sort_keys=True, allow_nan=False) + '\n')
                with (output / 'inference-diagnostics.json').open('x', encoding='utf-8') as stream:
                    stream.write(json.dumps(failure, indent=2, allow_nan=False) + '\n')
            except OSError:
                emit('diagnostics_storage_failed', reason_code='unexpected_error', exception_family='io')
        raise InferenceError(reason, study_ordinal=study_ordinal, series_ordinal=series_ordinal,
                             exception_family=_exception_family(error)) from None
