"""Bounded fold-zero model/runtime diagnostic on a pinned selected-series sample."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import csv
import tempfile
import time
import warnings

import numpy as np
import torch

from rsna_knee.data.compatibility_audit import CANDIDATE_PLAN_VERSION, _metadata, validate_plan
from rsna_knee.data.dicom_preprocessing import DicomPreprocessingError, preprocess_dicom_series
from rsna_knee.inference.fold0 import TestStudyDataset, load_fold0_model, validate_training_report

SELECTED_PLAN_SHA256 = '4eba5107a3854b7bfde4f8915314c4955cf914fd29f01a986ecb333de5019a57'
HEADER_REPORT_SHA256 = '7297a162743e3399411dc6e6a53eb24527e145bb8c5dc2f1264f2d25a796639e'
HEADER_PLAN_SHA256 = 'e2970e19bbfbe96a962e4414be5c7adfcf95daaa5cf7612de8d8b747918527fb'
TRAIN_SERIES_SHA256 = '573c1d80772bf41211c91b149c95677385a1c22d63f485c347f1b46c0177aef3'
TRAINING_REPORT_SHA256 = '5daef84f7ed8ad9aa7173989fe80fc168a900fc196025feeb98ee9c8c1792dc1'
CHECKPOINT_SHA256 = 'd85f6531aeb8aa3035f05997ef3dfbb2f88a97c288c07b8786a13795fdf33447'
CHECKPOINT_SIZE = 44_811_979
CA16_REPORT_SHA256 = '0e0bb3a5548ebffb28d26e1d40f92b19c2fd3cb0e94b63771b7e7a546a6cc1e2'
BUDGET_SECONDS = 1800


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _safe_reason(error) -> str:
    if isinstance(error, DicomPreprocessingError):
        allowed = DicomPreprocessingError.SAFE_REASON_CODES
        reason = error.reason_code
        if type(reason) is str and reason in allowed:
            return reason
    safe_codes = DicomPreprocessingError.SAFE_REASON_CODES | {
        'selected_path_invalid', 'preprocessed_tensor_invalid', 'training_metadata_pin_mismatch',
        'selected_series_membership_invalid', 'training_report_pin_mismatch', 'cuda_required',
        'budget_invalid', 'prediction_invalid', 'checkpoint_invalid', 'source_archive_pin_mismatch',
        'checkpoint_pin_mismatch', 'report_storage_failed', 'full_study_plan_invalid',
        'full_study_baseline_invalid',
    }
    if type(error) is ValueError and len(error.args) == 1 and type(error.args[0]) is str:
        reason = error.args[0]
        if reason in safe_codes:
            return reason
    if isinstance(error, (MemoryError, torch.OutOfMemoryError)):
        return 'resource_exhausted'
    return 'unexpected_error'


def _contained(root: Path, study: str, series: str) -> Path:
    if (type(study) is not str or type(series) is not str
            or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}', study) is None
            or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}', series) is None):
        raise ValueError('selected_path_invalid')
    base = Path(root)
    current = base
    for part in (study, series):
        current = current / part
        if current.is_symlink() or not current.is_dir():
            raise ValueError('selected_path_invalid')
    current.resolve(strict=True).relative_to(base.resolve(strict=True))
    return current


def validate_model_runtime_plan(plan, plan_raw: bytes) -> None:
    """Verify CA12 pins and exact 20 distinct Sagittal candidates before reading data."""
    try:
        validate_plan(plan)
        if (_digest(plan_raw) != SELECTED_PLAN_SHA256
                or type(plan) is not dict or plan.get('schema_version') != CANDIDATE_PLAN_VERSION
                or plan.get('header_report_sha256') != HEADER_REPORT_SHA256
                or plan.get('header_plan_sha256') != HEADER_PLAN_SHA256
                or plan.get('source_sha256') != {'series': TRAIN_SERIES_SHA256}
                or type(plan.get('excluded_ids')) is not list or len(plan['excluded_ids']) != 658
                or len(plan.get('candidates', [])) != 20):
            raise ValueError
        seen_studies, seen_pairs = set(), set()
        for row in plan['candidates']:
            if row.get('plane') != 'Sagittal' or row['study_id'] in seen_studies:
                raise ValueError
            pair = (row['study_id'], row['series_id'])
            if pair in seen_pairs:
                raise ValueError
            seen_studies.add(row['study_id'])
            seen_pairs.add(pair)
        if len(seen_studies) != 20:
            raise ValueError
    except Exception:
        raise ValueError('model_runtime_plan_invalid') from None


def _validate_membership(plan, series_csv: Path) -> None:
    raw = series_csv.read_bytes()
    actual_hash = _digest(raw)
    if actual_hash != TRAIN_SERIES_SHA256:
        raise ValueError('training_metadata_pin_mismatch')
    grouped = _metadata(raw)
    for row in plan['candidates']:
        actual = grouped.get(row['study_id'], [])
        matches = [item for item in actual if item['series_id'] == row['series_id']]
        if len(matches) != 1 or matches[0]['plane'] != row['plane']:
            raise ValueError('selected_series_membership_invalid')
    return actual_hash


def _validate_ca16_baseline(raw: bytes) -> None:
    try:
        report = json.loads(raw)
        if (_digest(raw) != CA16_REPORT_SHA256 or type(report) is not dict
                or report.get('schema_version') != 'training-model-runtime-report-v1'
                or report.get('status') != 'complete'
                or report.get('planned_series') != 20 or report.get('completed_series') != 20
                or report.get('successful_series') != 20 or report.get('failed_series') != 0
                or report.get('finite_logits') != 240 or report.get('finite_probabilities') != 240):
            raise ValueError
    except Exception:
        raise ValueError('full_study_baseline_invalid') from None


def build_full_study_plan(plan, plan_raw: bytes, series_csv, baseline_report_raw: bytes) -> dict:
    """Freeze every metadata-listed sibling series for the same pinned 20 studies."""
    validate_model_runtime_plan(plan, plan_raw)
    _validate_ca16_baseline(baseline_report_raw)
    raw = Path(series_csv).read_bytes()
    if _digest(raw) != TRAIN_SERIES_SHA256:
        raise ValueError('full_study_plan_invalid')
    grouped = _metadata(raw)
    studies = []
    for ordinal, candidate in enumerate(plan['candidates'], start=1):
        series = grouped.get(candidate['study_id'], [])
        if not series or not any(row['series_id'] == candidate['series_id']
                                 and row['plane'] == candidate['plane'] for row in series):
            raise ValueError('full_study_plan_invalid')
        studies.append({'study_ordinal': ordinal, 'study_id': candidate['study_id'],
                        'series': [{'series_id': row['series_id'], 'plane': row['plane']}
                                   for row in series]})
    plane_counts = {plane: sum(row['plane'] == plane for study in studies for row in study['series'])
                    for plane in ('Axial', 'Coronal', 'Sagittal')}
    return {'schema_version': 'full-study-runtime-plan-v1',
            'selected_plan_sha256': _digest(plan_raw),
            'training_series_sha256': _digest(raw),
            'baseline_report_sha256': _digest(baseline_report_raw),
            'study_count': len(studies),
            'series_count': sum(len(row['series']) for row in studies),
            'series_plane_counts': plane_counts,
            'studies': studies}


def validate_full_study_plan(full_plan, full_plan_raw: bytes, plan, plan_raw: bytes,
                             series_csv, baseline_report_raw: bytes) -> None:
    """Re-derive ordered sibling membership from both pinned inputs; never trust supplied IDs."""
    try:
        expected = build_full_study_plan(plan, plan_raw, series_csv, baseline_report_raw)
        frozen = (json.dumps(expected, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
        if (type(full_plan) is not dict or full_plan != expected or full_plan_raw != frozen
                or len(full_plan['studies']) != 20
                or len({row['study_id'] for row in full_plan['studies']}) != 20
                or sum(len(row['series']) for row in full_plan['studies']) < 20):
            raise ValueError
    except Exception:
        raise ValueError('full_study_plan_invalid') from None


def _full_study_dataset(full_plan, image_root, preprocessor, output):
    """Use the genuine test dataset contract with a frozen, private 20-study metadata view."""
    with tempfile.TemporaryDirectory(prefix='.study-metadata-', dir=output) as temporary:
        root = Path(temporary)
        test_csv, series_csv = root / 'studies.csv', root / 'series.csv'
        with test_csv.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['StudyInstanceUID'])
            writer.writerows([[row['study_id']] for row in full_plan['studies']])
        with series_csv.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['StudyInstanceUID', 'SeriesInstanceUID'])
            for study in full_plan['studies']:
                writer.writerows([[study['study_id'], row['series_id']] for row in study['series']])
        return TestStudyDataset(test_csv, series_csv, image_root, preprocessor=preprocessor)


def _atomic_report(output: Path, report: dict) -> None:
    target = output / 'model-runtime-report.json'
    temporary = output / '.model-runtime-report.tmp'
    raw = (json.dumps(report, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
    try:
        with temporary.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except OSError:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise ValueError('report_storage_failed') from None


def _environment(torch_api) -> dict:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            import torchvision
        torchvision_version = str(torchvision.__version__).split('+')[0]
    except Exception:
        torchvision_version = 'unavailable'
    return {
        'python': platform.python_version(),
        'torch': str(torch_api.__version__).split('+')[0],
        'torchvision': torchvision_version,
        'cuda_runtime': str(torch_api.version.cuda or 'unavailable'),
        'device': 'cuda:0',
        'device_count': int(torch_api.cuda.device_count()),
    }


def run_model_runtime_audit(plan, plan_raw, series_csv, image_root, training_report,
                            checkpoint, output_dir, *, device='cuda:0',
                            source_archive, source_archive_sha256,
                            full_study_plan=None, full_study_plan_raw=None,
                            baseline_report_raw=None,
                            preprocessor=None, model_loader=None, torch_api=torch,
                            clock=time.perf_counter, budget_seconds=BUDGET_SECONDS):
    """Run isolated-series baseline or all-sibling study inference, with aggregate-only evidence."""
    full_study_mode = full_study_plan is not None or full_study_plan_raw is not None
    if full_study_mode and (full_study_plan is None or full_study_plan_raw is None
                            or baseline_report_raw is None):
        raise ValueError('full_study_plan_invalid')
    output = Path(output_dir)
    if output.exists() or output.is_symlink() or any(parent.is_symlink() for parent in output.parents):
        raise ValueError('output_must_be_new')
    output.mkdir(parents=True, exist_ok=False)
    started = clock()
    report = {
        'schema_version': ('full-study-model-runtime-report-v1' if full_study_mode
                           else 'training-model-runtime-report-v1'), 'status': 'initializing',
        'termination_reason': 'initialization_pending', 'planned_series': 20,
        'attempted_series': 0, 'completed_series': 0, 'successful_series': 0,
        'failed_series': 0, 'finite_logits': 0, 'finite_probabilities': 0,
        'failures': [], 'logit_min': None, 'logit_max': None,
        'elapsed_seconds': 0.0, 'budget_seconds': budget_seconds,
        'checkpoint_status': 'not_loaded',
        'expected_provenance': {
            'plan_sha256': SELECTED_PLAN_SHA256,
            'training_series_sha256': TRAIN_SERIES_SHA256,
            'training_report_sha256': TRAINING_REPORT_SHA256,
            'checkpoint_sha256': CHECKPOINT_SHA256,
            'checkpoint_size': CHECKPOINT_SIZE,
            'source_archive_sha256': source_archive_sha256,
        },
        'verified_provenance': {}, 'runtime_failure': None,
        'peak_gpu_bytes': 0, 'environment': {},
    }
    if full_study_mode:
        report.update({'planned_studies': 20, 'attempted_studies': 0, 'completed_studies': 0,
                       'successful_studies': 0, 'failed_studies': 0,
                       'preprocessed_series': 0, 'full_study_plan_sha256': None})
        report['series_success_stage'] = 'preprocessing'
    runtime_started = False
    current_stage = 'initialization'
    current_ordinal = None

    def save(status=None, termination=None):
        report['elapsed_seconds'] = max(0.0, float(clock() - started))
        if status is not None:
            report['status'] = status
        if termination is not None:
            report['termination_reason'] = termination
        _atomic_report(output, report)

    try:
        current_stage = 'plan_validation'
        validate_model_runtime_plan(plan, plan_raw)
        report['verified_provenance']['plan_sha256'] = _digest(plan_raw)
        current_stage = 'training_metadata_validation'
        report['verified_provenance']['training_series_sha256'] = _validate_membership(plan, Path(series_csv))
        if full_study_mode:
            current_stage = 'full_study_plan_validation'
            validate_full_study_plan(full_study_plan, full_study_plan_raw, plan, plan_raw,
                                     series_csv, baseline_report_raw)
            report['planned_series'] = sum(len(row['series']) for row in full_study_plan['studies'])
            report['series_plane_counts'] = full_study_plan['series_plane_counts'].copy()
            report['full_study_plan_sha256'] = _digest(full_study_plan_raw)
            report['expected_provenance']['full_study_plan_sha256'] = _digest(full_study_plan_raw)
            report['verified_provenance']['full_study_plan_sha256'] = _digest(full_study_plan_raw)
            report['expected_provenance']['baseline_report_sha256'] = CA16_REPORT_SHA256
            report['verified_provenance']['baseline_report_sha256'] = _digest(baseline_report_raw)
        current_stage = 'training_report_validation'
        report_raw = Path(training_report).read_bytes()
        if _digest(report_raw) != TRAINING_REPORT_SHA256:
            raise ValueError('training_report_pin_mismatch')
        validate_training_report(json.loads(report_raw))
        report['verified_provenance']['training_report_sha256'] = _digest(report_raw)
        current_stage = 'source_archive_validation'
        archive_path = Path(source_archive)
        if (type(source_archive_sha256) is not str
                or not re.fullmatch(r'[0-9a-f]{64}', source_archive_sha256)
                or archive_path.is_symlink() or not archive_path.is_file()
                or _digest_file(archive_path) != source_archive_sha256):
            raise ValueError('source_archive_pin_mismatch')
        report['verified_provenance']['source_archive_sha256'] = _digest_file(archive_path)
        current_stage = 'cuda_initialization'
        if device != 'cuda:0' or not torch_api.cuda.is_available():
            raise ValueError('cuda_required')
        torch_api.cuda.set_device(0)
        gpu = torch_api.device('cuda:0')
        torch_api.cuda.synchronize(gpu)
        torch_api.cuda.reset_peak_memory_stats(gpu)
        if budget_seconds <= 0:
            raise ValueError('budget_invalid')
        report['environment'] = _environment(torch_api)
        loader = model_loader or load_fold0_model
        current_stage = 'checkpoint_safe_load'
        checkpoint_path = Path(checkpoint)
        if (checkpoint_path.is_symlink() or not checkpoint_path.is_file()
                or checkpoint_path.stat().st_size != CHECKPOINT_SIZE
                or _digest_file(checkpoint_path) != CHECKPOINT_SHA256):
            raise ValueError('checkpoint_pin_mismatch')
        report['verified_provenance']['checkpoint_sha256'] = _digest_file(checkpoint_path)
        report['verified_provenance']['checkpoint_size'] = checkpoint_path.stat().st_size
        report['checkpoint_status'] = 'pin_verified'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = loader(checkpoint, CHECKPOINT_SHA256, CHECKPOINT_SIZE)
        model = model.to(gpu).eval()
        report['checkpoint_status'] = 'verified_and_loaded'
        preprocess = preprocessor or preprocess_dicom_series
        study_dataset = (_full_study_dataset(full_study_plan, image_root, preprocess, output)
                         if full_study_mode else None)
        report['status'] = 'running'
        report['termination_reason'] = 'in_progress'
        current_stage = 'initial_report_checkpoint'
        save()
        runtime_started = True
        with torch_api.inference_mode():
            work_items = (enumerate(full_study_plan['studies'], start=1) if full_study_mode
                          else enumerate(plan['candidates'], start=1))
            for ordinal, row in work_items:
                current_ordinal = ordinal
                study_preprocessed_count = 0
                current_stage = 'series_boundary_synchronize'
                torch_api.cuda.synchronize(gpu)
                current_stage = 'budget_check'
                if clock() - started >= budget_seconds:
                    report['termination_reason'] = 'cooperative_budget_reached'
                    break
                if full_study_mode:
                    report['attempted_studies'] += 1
                else:
                    report['attempted_series'] += 1
                try:
                    current_stage = 'preprocess'
                    if full_study_mode:
                        images, masks = study_dataset[ordinal - 1]
                        series_count = len(images)
                        study_preprocessed_count = series_count
                        report['attempted_series'] += series_count
                        report['preprocessed_series'] += series_count
                        report['successful_series'] += series_count
                        image_tensors = [image.to(gpu) for image in images]
                        mask_tensors = [mask.to(gpu) for mask in masks]
                    else:
                        directory = _contained(Path(image_root), row['study_id'], row['series_id'])
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            result = preprocess(directory, max_slices=8, output_size=256)
                        images = np.asarray(result.images)
                        masks = np.asarray(result.valid_mask)
                        if (images.ndim != 4 or not 1 <= images.shape[0] <= 8
                                or images.shape[1:] != (1, 256, 256) or masks.shape != images.shape
                                or not np.isrealobj(images) or not np.isfinite(images).all()
                                or (images < 0).any() or (images > 1).any()
                                or not np.isin(masks, (0, 1)).all()
                                or not masks.reshape(len(masks), -1).any(axis=1).all()
                                or np.any(images[~masks.astype(bool)] != 0)):
                            raise ValueError('preprocessed_tensor_invalid')
                        image_tensors = [torch_api.as_tensor(images.copy(), dtype=torch_api.float32).to(gpu)]
                        mask_tensors = [torch_api.as_tensor(masks.copy(), dtype=torch_api.bool).to(gpu)]
                    torch_api.cuda.synchronize(gpu)
                    current_stage = 'model_inference'
                    logits = model(image_tensors, mask_tensors)
                    if tuple(logits.shape) != (1, 12) or not bool(torch_api.isfinite(logits).all()):
                        raise ValueError('prediction_invalid')
                    probabilities = torch_api.sigmoid(logits)
                    if tuple(probabilities.shape) != (1, 12) or not bool(torch_api.isfinite(probabilities).all()):
                        raise ValueError('prediction_invalid')
                    logit_values = logits.detach().float().cpu().numpy()
                    probability_values = probabilities.detach().float().cpu().numpy()
                    torch_api.cuda.synchronize(gpu)
                    if not np.isfinite(logit_values).all() or not np.isfinite(probability_values).all():
                        raise ValueError('prediction_invalid')
                    report['finite_logits'] += 12
                    report['finite_probabilities'] += 12
                    report['logit_min'] = float(np.min(logit_values)) if report['logit_min'] is None else min(report['logit_min'], float(np.min(logit_values)))
                    report['logit_max'] = float(np.max(logit_values)) if report['logit_max'] is None else max(report['logit_max'], float(np.max(logit_values)))
                    if full_study_mode:
                        report['successful_studies'] += 1
                    else:
                        report['successful_series'] += 1
                except Exception as error:
                    if full_study_mode:
                        failed_series_ordinal = getattr(error, 'series_ordinal', None)
                        if failed_series_ordinal is not None:
                            report['attempted_series'] += failed_series_ordinal
                            report['preprocessed_series'] += max(0, failed_series_ordinal - 1)
                            report['successful_series'] += max(0, failed_series_ordinal - 1)
                            report['failed_series'] += 1
                            study_preprocessed_count = failed_series_ordinal
                        report['failed_studies'] += 1
                        failure = {'study_ordinal': ordinal, 'stage': current_stage,
                                   'reason_code': _safe_reason(error)}
                        if failed_series_ordinal is not None:
                            failure['series_ordinal'] = failed_series_ordinal
                        report['failures'].append(failure)
                    else:
                        report['failed_series'] += 1
                        report['failures'].append({'series_ordinal': ordinal, 'stage': current_stage,
                                                   'reason_code': _safe_reason(error)})
                if full_study_mode:
                    report['completed_studies'] += 1
                    report['completed_series'] += study_preprocessed_count
                    current_stage = 'gpu_peak_memory_read'
                else:
                    report['completed_series'] += 1
                current_stage = 'gpu_peak_memory_read'
                report['peak_gpu_bytes'] = max(report['peak_gpu_bytes'], int(torch_api.cuda.max_memory_allocated(gpu)))
                current_stage = 'report_checkpoint'
                save('running', 'in_progress')
        complete = (report['completed_studies'] == report['planned_studies'] if full_study_mode
                    else report['completed_series'] == report['planned_series'])
        status = ('complete' if complete and (report['failed_studies'] == 0 if full_study_mode
                                              else report['failed_series'] == 0) else 'partial')
        termination = (('all_planned_studies_attempted' if full_study_mode else 'all_planned_series_attempted')
                       if complete else report['termination_reason'])
        save(status, termination)
        return report.copy()
    except Exception as error:
        reason = _safe_reason(error)
        if reason == 'report_storage_failed':
            raise ValueError('model_runtime_report_storage_failed') from None
        if runtime_started:
            ordinal_key = 'study_ordinal' if full_study_mode else 'series_ordinal'
            report['runtime_failure'] = {ordinal_key: current_ordinal,
                                         'stage': current_stage, 'reason_code': reason}
            try:
                save('partial', 'runtime_interrupted')
            except ValueError:
                raise ValueError('model_runtime_report_storage_failed') from None
            raise ValueError('model_runtime_audit_interrupted') from None
        report['failures'].append({'series_ordinal': None,
            'stage': current_stage, 'reason_code': reason})
        try:
            save('failed', 'initialization_failed')
        except ValueError:
            raise ValueError('model_runtime_report_storage_failed') from None
        raise ValueError('model_runtime_initialization_failed') from None
