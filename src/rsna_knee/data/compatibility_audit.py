"""Fresh-study compatibility audit; never train, infer predictions, or print IDs."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import warnings

from rsna_knee.data.dicom_preprocessing import DicomPreprocessingError, preprocess_dicom_series
from rsna_knee.data.orientation_audit import extract_manifest_series

PLAN_VERSION = 'training-compatibility-v1'
CANDIDATE_PLAN_VERSION = 'training-dimension-candidates-v1'
HEADER_PLAN_VERSION = 'training-header-inventory-v1'
DIMENSION_CANDIDATE_COUNT = 20
TRUSTED_DIMENSION_REPORT_SHA256 = '7297a162743e3399411dc6e6a53eb24527e145bb8c5dc2f1264f2d25a796639e'
TRUSTED_HEADER_PLAN_SHA256 = 'e2970e19bbfbe96a962e4414be5c7adfcf95daaa5cf7612de8d8b747918527fb'
PLANES = frozenset({'Sagittal', 'Coronal', 'Axial'})
PLUGINS = frozenset({'pydicom', 'pylibjpeg', 'gdcm', 'pillow'})
SYNTAXES = {
    '1.2.840.10008.1.2': 'implicit_vr_little_endian',
    '1.2.840.10008.1.2.1': 'explicit_vr_little_endian',
    '1.2.840.10008.1.2.2': 'explicit_vr_big_endian',
    '1.2.840.10008.1.2.4.50': 'jpeg_baseline',
    '1.2.840.10008.1.2.4.70': 'jpeg_lossless',
    '1.2.840.10008.1.2.4.80': 'jpeg_ls_lossless',
    '1.2.840.10008.1.2.4.90': 'jpeg2000_lossless',
    '1.2.840.10008.1.2.4.91': 'jpeg2000',
    '1.2.840.10008.1.2.5': 'rle',
}


def _identifier(value):
    if type(value) is not str or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}', value) is None:
        raise ValueError('identifier_invalid')
    return value


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def plan_bytes(plan):
    validate_plan(plan)
    return (json.dumps(plan, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def build_dimension_candidate_plan(report_file, header_plan_file):
    """Freeze only the 20 retained out-of-range dimension examples from pinned evidence."""
    try:
        report_path, source_plan_path = Path(report_file), Path(header_plan_file)
        if report_path.is_symlink() or source_plan_path.is_symlink():
            raise ValueError
        report_raw, source_plan_raw = report_path.read_bytes(), source_plan_path.read_bytes()
        if (_digest(report_raw) != TRUSTED_DIMENSION_REPORT_SHA256
                or _digest(source_plan_raw) != TRUSTED_HEADER_PLAN_SHA256):
            raise ValueError
        report, source_plan = json.loads(report_raw), json.loads(source_plan_raw)
        if (type(report) is not dict or report.get('schema_version') != HEADER_PLAN_VERSION
                or report.get('status') != 'complete'
                or report.get('termination_reason') != 'all_planned_series_attempted'
                or report.get('failed_series') != 0
                or report.get('attempted_series') != report.get('planned_series')
                or report.get('completed_series') != report.get('planned_series')
                or type(source_plan) is not dict or source_plan.get('schema_version') != HEADER_PLAN_VERSION
                or set(source_plan) != {'schema_version', 'source_sha256', 'excluded_ids', 'studies'}
                or report.get('plan_sha256') != TRUSTED_HEADER_PLAN_SHA256):
            raise ValueError
        pins = source_plan['source_sha256']
        if (type(pins) is not dict or not {'series', 'gold', 'prior', 'audit_plan', 'audit_report'}.issubset(pins)
                or any(type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None
                       for value in pins.values())):
            raise ValueError
        excluded = source_plan['excluded_ids']
        if type(excluded) is not list or len(set(excluded)) != len(excluded):
            raise ValueError
        excluded = {_identifier(value) for value in excluded}
        raw_studies = source_plan['studies']
        if (type(raw_studies) is not list or not raw_studies
                or report.get('planned_studies') != len(raw_studies)
                or report.get('planned_series') != sum(len(row.get('series', [])) for row in raw_studies
                    if type(row) is dict)):
            raise ValueError
        study_ids, series_ids = set(), set()
        for source_study in raw_studies:
            if type(source_study) is not dict or set(source_study) != {'study_id', 'series'}:
                raise ValueError
            source_study_id = _identifier(source_study['study_id'])
            if source_study_id in study_ids or source_study_id in excluded:
                raise ValueError
            study_ids.add(source_study_id)
            source_rows = source_study['series']
            if type(source_rows) is not list or not source_rows:
                raise ValueError
            for source_row in source_rows:
                if (type(source_row) is not dict or set(source_row) != {'series_id', 'plane'}
                        or source_row['plane'] not in PLANES):
                    raise ValueError
                source_series_id = _identifier(source_row['series_id'])
                if source_series_id in series_ids:
                    raise ValueError
                series_ids.add(source_series_id)
        candidates = report.get('candidate_ordinals', {}).get('uncovered_dimensions')
        if (report.get('flag_counts', {}).get('uncovered_dimensions', 0) < DIMENSION_CANDIDATE_COUNT
                or type(candidates) is not list or len(candidates) != DIMENSION_CANDIDATE_COUNT):
            raise ValueError
        selected, seen = [], set()
        for pair in candidates:
            if (type(pair) is not dict or set(pair) != {'study_ordinal', 'series_ordinal'}
                    or type(pair['study_ordinal']) is not int or type(pair['series_ordinal']) is not int
                    or not 1 <= pair['study_ordinal'] <= len(raw_studies)):
                raise ValueError
            study_ordinal, series_ordinal = pair['study_ordinal'], pair['series_ordinal']
            source_study = raw_studies[study_ordinal - 1]
            rows = source_study.get('series') if type(source_study) is dict else None
            if type(rows) is not list or not 1 <= series_ordinal <= len(rows):
                raise ValueError
            study_id = _identifier(source_study.get('study_id'))
            row = rows[series_ordinal - 1]
            if (type(row) is not dict or set(row) != {'series_id', 'plane'}
                    or study_id in excluded or row.get('plane') not in PLANES):
                raise ValueError
            series_id = _identifier(row.get('series_id'))
            identity = (study_id, series_id)
            if identity in seen:
                raise ValueError
            seen.add(identity)
            selected.append({'study_id': study_id, 'study_ordinal': study_ordinal,
                             'series_id': series_id, 'series_ordinal': series_ordinal,
                             'plane': row['plane']})
        plan = {'schema_version': CANDIDATE_PLAN_VERSION,
                'source_sha256': {'series': pins['series']},
                'header_report_sha256': TRUSTED_DIMENSION_REPORT_SHA256,
                'header_plan_sha256': TRUSTED_HEADER_PLAN_SHA256,
                'header_source_sha256': dict(pins), 'excluded_ids': sorted(excluded),
                'candidate_count': DIMENSION_CANDIDATE_COUNT, 'candidates': selected}
        validate_plan(plan)
        return plan
    except (OSError, UnicodeError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise ValueError('dimension_candidate_plan_invalid') from None


def _metadata(raw):
    try:
        reader = csv.DictReader(raw.decode('utf-8-sig').splitlines())
        if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError
        if not {'StudyInstanceUID', 'SeriesInstanceUID', 'Anatomical_Plane'}.issubset(reader.fieldnames):
            raise ValueError
        grouped, seen = {}, set()
        for row in reader:
            study = _identifier(row['StudyInstanceUID'])
            series = _identifier(row['SeriesInstanceUID'])
            plane = row['Anatomical_Plane']
            if None in row or series in seen or plane not in PLANES:
                raise ValueError
            seen.add(series)
            grouped.setdefault(study, []).append({'series_id': series, 'plane': plane})
        if not grouped:
            raise ValueError
        return grouped
    except (UnicodeError, KeyError, TypeError, ValueError, csv.Error):
        raise ValueError('series_metadata_invalid') from None


def build_audit_plan(series_csv, gold_csv, prior_manifest, expected_sha256, *, sample_size=100, seed=42):
    """Freeze outcome-blind hash-ranked study IDs and all their metadata series."""
    try:
        raw = {}
        for name, file in [('series', series_csv), ('gold', gold_csv), ('prior', prior_manifest)]:
            path = Path(file)
            if path.is_symlink():
                raise ValueError
            raw[name] = path.read_bytes()
            if _digest(raw[name]) != expected_sha256[name]:
                raise ValueError
        groups = _metadata(raw['series'])
        reader = csv.DictReader(raw['gold'].decode('utf-8-sig').splitlines())
        if reader.fieldnames != ['StudyInstanceUID']:
            raise ValueError
        gold = [_identifier(row['StudyInstanceUID']) for row in reader]
        previous = extract_manifest_series(raw['prior'])  # Selectively skips labels.
        prior = [_identifier(study) for study, series in previous]
        if (not gold or not prior or len(set(gold)) != len(gold) or len(set(prior)) != len(prior)
                or set(gold) & set(prior) or not (set(gold) | set(prior)).issubset(groups)):
            raise ValueError
        for study, series in previous:
            if len(set(series)) != len(series):
                raise ValueError
            if set(series) != {row['series_id'] for row in groups[study]}:
                raise ValueError
        fresh = set(groups) - set(gold) - set(prior)
        if type(sample_size) is not int or not 1 <= sample_size <= len(fresh) or type(seed) is not int:
            raise ValueError
        ranked = sorted(fresh, key=lambda key: (_digest(f'compatibility-seed-{seed}:{key}'.encode()), key))
        plan = {'schema_version': PLAN_VERSION, 'seed': seed, 'sample_size': sample_size,
                'population_study_count': len(groups), 'fresh_study_count': len(fresh),
                'source_sha256': dict(expected_sha256), 'gold_ids': sorted(gold), 'prior_ids': sorted(prior),
                'studies': [{'study_id': key, 'series': groups[key]} for key in ranked[:sample_size]]}
        validate_plan(plan)
        return plan
    except (OSError, UnicodeError, KeyError, TypeError, ValueError, csv.Error):
        raise ValueError('audit_plan_invalid') from None


def validate_plan(plan):
    try:
        if type(plan) is dict and plan.get('schema_version') == CANDIDATE_PLAN_VERSION:
            _validate_dimension_candidate_plan(plan)
            return
        if (type(plan) is not dict or set(plan) != {'schema_version', 'seed', 'sample_size',
                'population_study_count', 'fresh_study_count', 'source_sha256', 'gold_ids', 'prior_ids', 'studies'}
                or plan['schema_version'] != PLAN_VERSION
                or type(plan['seed']) is not int or type(plan['sample_size']) is not int
                or not 1 <= plan['sample_size'] <= 100):
            raise ValueError
        pins = plan['source_sha256']
        if set(pins) != {'series', 'gold', 'prior'} or any(type(v) is not str or re.fullmatch('[0-9a-f]{64}', v) is None for v in pins.values()):
            raise ValueError
        exclusions = []
        for key in ('gold_ids', 'prior_ids'):
            values = plan[key]
            if type(values) is not list or not values or len(set(values)) != len(values):
                raise ValueError
            exclusions.append({_identifier(v) for v in values})
        if exclusions[0] & exclusions[1]:
            raise ValueError
        studies, series = set(), set()
        if type(plan['studies']) is not list or len(plan['studies']) != plan['sample_size']:
            raise ValueError
        for study in plan['studies']:
            if type(study) is not dict or set(study) != {'study_id', 'series'} or type(study['series']) is not list:
                raise ValueError
            key = _identifier(study['study_id'])
            if key in studies or key in exclusions[0] | exclusions[1] or not study['series']:
                raise ValueError
            studies.add(key)
            for row in study['series']:
                if type(row) is not dict or set(row) != {'series_id', 'plane'}:
                    raise ValueError
                key = _identifier(row['series_id'])
                if key in series or row['plane'] not in PLANES:
                    raise ValueError
                series.add(key)
        for key in ('population_study_count', 'fresh_study_count'):
            if type(plan[key]) is not int or plan[key] < plan['sample_size']:
                raise ValueError
        if plan['population_study_count'] - sum(map(len, exclusions)) != plan['fresh_study_count']:
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise ValueError('audit_plan_invalid') from None


def _validate_dimension_candidate_plan(plan):
    expected = {'schema_version', 'source_sha256', 'header_report_sha256', 'header_plan_sha256',
                'header_source_sha256', 'excluded_ids', 'candidate_count', 'candidates'}
    if type(plan) is not dict or set(plan) != expected or plan['candidate_count'] != DIMENSION_CANDIDATE_COUNT:
        raise ValueError
    for name, value in [('header_report_sha256', plan['header_report_sha256']),
                        ('header_plan_sha256', plan['header_plan_sha256'])]:
        if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
            raise ValueError
    pins = plan['header_source_sha256']
    if (type(pins) is not dict or not {'series', 'gold', 'prior', 'audit_plan', 'audit_report'}.issubset(pins)
            or any(type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None
                   for value in pins.values())
            or plan['source_sha256'] != {'series': pins['series']}):
        raise ValueError
    exclusions = plan['excluded_ids']
    if type(exclusions) is not list or len(exclusions) != len(set(exclusions)):
        raise ValueError
    excluded = {_identifier(value) for value in exclusions}
    candidates = plan['candidates']
    if type(candidates) is not list or len(candidates) != DIMENSION_CANDIDATE_COUNT:
        raise ValueError
    seen, ordinals = set(), set()
    for row in candidates:
        if type(row) is not dict or set(row) != {'study_id', 'study_ordinal', 'series_id', 'series_ordinal', 'plane'}:
            raise ValueError
        study, series = _identifier(row['study_id']), _identifier(row['series_id'])
        if (study in excluded or row['plane'] not in PLANES or type(row['study_ordinal']) is not int
                or type(row['series_ordinal']) is not int or row['study_ordinal'] < 1 or row['series_ordinal'] < 1
                or (study, series) in seen or (row['study_ordinal'], row['series_ordinal']) in ordinals):
            raise ValueError
        seen.add((study, series))
        ordinals.add((row['study_ordinal'], row['series_ordinal']))


def _codec_availability(uid):
    """Use pydicom's installed public decoder API, not assumed package presence."""
    try:
        from pydicom.pixels import get_decoder
        decoder = get_decoder(uid)
        available = decoder.is_available
        if type(available) is not bool:
            return {'status': 'unknown', 'plugins': []}
        return {'status': 'available' if available else 'unavailable',
                'plugins': list(decoder.available_plugins)}
    except Exception:
        # A missing/unsupported probe API or UID is not evidence of no codec.
        return {'status': 'unknown', 'plugins': []}


def _safe_reason(error):
    if isinstance(error, DicomPreprocessingError):
        code = error.reason_code
        if type(code) is str and code in DicomPreprocessingError.SAFE_REASON_CODES:
            return code
    if type(error) is ValueError and len(error.args) == 1 and error.args[0] in ('audit_path_invalid', 'tensor_invalid'):
        return error.args[0]
    return 'unexpected_error'


def run_compatibility_audit(plan, series_csv, image_root, output_dir, *, preprocessor=None,
                            dcmread=None, decoder_probe=None, clock=time.perf_counter):
    """Audit every planned series; continue audit failures, never alter inference."""
    validate_plan(plan)
    started = clock()
    try:
        raw_metadata = Path(series_csv).read_bytes()
    except OSError:
        raise ValueError('series_metadata_unavailable') from None
    groups = _metadata(raw_metadata)
    if _digest(raw_metadata) != plan['source_sha256']['series']:
        raise ValueError('series_provenance_mismatch')
    if plan['schema_version'] == CANDIDATE_PLAN_VERSION:
        if not set(plan['excluded_ids']).issubset(groups):
            raise ValueError('plan_metadata_mismatch')
        for candidate in plan['candidates']:
            rows = groups.get(candidate['study_id'])
            if rows is None or {'series_id': candidate['series_id'], 'plane': candidate['plane']} not in rows:
                raise ValueError('plan_metadata_mismatch')
    else:
        # Rebuild the deterministic selection from metadata and frozen exclusions.
        fresh = set(groups) - set(plan['gold_ids']) - set(plan['prior_ids'])
        ranked = sorted(fresh, key=lambda key: (_digest(f"compatibility-seed-{plan['seed']}:{key}".encode()), key))
        if (len(groups) != plan['population_study_count'] or len(fresh) != plan['fresh_study_count']
                or not (set(plan['gold_ids']) | set(plan['prior_ids'])).issubset(groups)
                or [s['study_id'] for s in plan['studies']] != ranked[:plan['sample_size']]
                or any(groups[s['study_id']] != s['series'] for s in plan['studies'])):
            raise ValueError('plan_metadata_mismatch')
    root, output = Path(image_root), Path(output_dir)
    if root.is_symlink() or not root.is_dir() or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('audit_path_invalid')
    if output.exists():
        raise ValueError('audit_output_must_be_new')
    output.mkdir(parents=True, exist_ok=False)
    candidate_mode = plan['schema_version'] == CANDIDATE_PLAN_VERSION
    planned_series = len(plan['candidates']) if candidate_mode else sum(len(s['series']) for s in plan['studies'])
    planned_studies = len({row['study_id'] for row in plan['candidates']}) if candidate_mode else len(plan['studies'])
    report = {'schema_version': plan['schema_version'], 'plan_sha256': _digest(plan_bytes(plan)), 'status': 'partial',
              'planned_studies': planned_studies, 'planned_series': planned_series,
              'attempted_series': 0, 'completed_series': 0, 'termination_reason': 'in_progress', 'successful_series': 0, 'failed_series': 0, 'completed_studies': 0,
              'header_reads': 0, 'decoded_reads': 0, 'header_read_attempts': 0, 'decoded_read_attempts': 0,
              'successful_selected_slices': 0, 'planned_plane_counts': {}, 'dimensions': {'rows': None, 'columns': None},
              'transfer_syntax': {}, 'decoder_availability': {}, 'reason_counts': {}, 'plane_counts': {},
              'series_results': [], 'budget_seconds': 1800, 'elapsed_seconds': 0.0,
              'codec_limit': 'availability does not prove decode support; only successful selected-slice preprocessing is observed',
              'read_count_limit': 'decoded_reads counts selected pixel-body reads; successful_selected_slices proves finite adapter outputs'}
    if candidate_mode:
        for row in plan['candidates']:
            report['planned_plane_counts'][row['plane']] = report['planned_plane_counts'].get(row['plane'], 0) + 1
    else:
        for study in plan['studies']:
            for row in study['series']:
                report['planned_plane_counts'][row['plane']] = report['planned_plane_counts'].get(row['plane'], 0) + 1

    def checkpoint():
        report['elapsed_seconds'] = clock() - started
        try:
            with (output / 'report.tmp').open('x', encoding='utf-8') as stream:
                json.dump(report, stream, sort_keys=True, allow_nan=False)
                stream.write('\n')
            os.replace(output / 'report.tmp', output / 'report.json')
        except OSError:
            raise ValueError('audit_checkpoint_failed') from None

    checkpoint()
    current_syntax = set()
    probe = decoder_probe or _codec_availability

    def reader(path, **kwargs):
        kind = 'header_reads' if kwargs.get('stop_before_pixels') else 'decoded_reads'
        attempt = 'header_read_attempts' if kwargs.get('stop_before_pixels') else 'decoded_read_attempts'
        report[attempt] += 1
        value = dcmread(path, **kwargs)
        report[kind] += 1
        uid = getattr(getattr(value, 'file_meta', None), 'TransferSyntaxUID', None)
        category = SYNTAXES.get(uid, 'other') if isinstance(uid, str) else 'other'
        report['transfer_syntax'].setdefault(category, {'header_reads': 0, 'decoded_reads': 0, 'successful_series': 0})[kind] += 1
        if kind == 'decoded_reads':
            current_syntax.add(category)
        if category not in report['decoder_availability']:
            try:
                available = probe(uid)
                status = available.get('status')
                plugins = available.get('plugins', [])
                report['decoder_availability'][category] = {
                    'status': status if status in {'available', 'unavailable'} else 'unknown',
                    'plugins': sorted(p for p in plugins if type(p) is str and p in PLUGINS)}
            except Exception:
                report['decoder_availability'][category] = {'status': 'unknown', 'plugins': []}
        for source, name in [('Rows', 'rows'), ('Columns', 'columns')]:
            size = getattr(value, source, None)
            try:
                size = int(size) if not isinstance(size, bool) else 0
                if not 1 <= size <= 1_000_000:
                    continue
                old = report['dimensions'][name]
                report['dimensions'][name] = [size, size] if old is None else [min(old[0], size), max(old[1], size)]
            except (TypeError, ValueError, OverflowError):
                pass
        return value

    try:
        if dcmread is None:
            import pydicom
            dcmread = pydicom.dcmread
        adapter = preprocessor or preprocess_dicom_series
        import numpy as np
        if candidate_mode:
            work = [(row['study_id'], row, row['study_ordinal'], row['series_ordinal'],
                     index == len(plan['candidates']) - 1 or plan['candidates'][index + 1]['study_id'] != row['study_id'])
                    for index, row in enumerate(plan['candidates'])]
        else:
            work = [(study['study_id'], row, study_index, series_index,
                     series_index == len(study['series']))
                    for study_index, study in enumerate(plan['studies'], start=1)
                    for series_index, row in enumerate(study['series'], start=1)]
        for study_id, row, study_index, series_index, last_in_study in work:
            if clock() - started >= 1800:
                report['termination_reason'] = 'time_budget'
                checkpoint()
                return report
            current_syntax.clear()
            reason = None
            report['attempted_series'] += 1
            checkpoint()
            try:
                study_path = root / study_id
                series_path = study_path / row['series_id']
                if study_path.is_symlink() or series_path.is_symlink() or not series_path.is_dir():
                    raise ValueError('audit_path_invalid')
                # Third-party warning strings may contain raw metadata. Exceptions
                # still reach the unchanged adapter and its safe categorical errors.
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    result = adapter(series_path, output_size=256, max_slices=8, dcmread=reader)
                images, mask = np.asarray(result.images), np.asarray(result.valid_mask)
                if (images.ndim != 4 or not 1 <= len(images) <= 8 or images.shape[1:] != (1,256,256)
                        or mask.shape != images.shape or not np.isfinite(images).all()
                        or not np.isrealobj(images) or (images < 0).any() or (images > 1).any()
                        or not np.isin(mask, (0,1)).all() or not mask.reshape(len(mask),-1).any(axis=1).all()
                        or (images[~mask.astype(bool)] != 0).any()):
                    raise ValueError('tensor_invalid')
            except Exception as error:
                reason = _safe_reason(error)
            report['completed_series'] += 1
            report['plane_counts'][row['plane']] = report['plane_counts'].get(row['plane'], 0) + 1
            outcome = 'successful_series' if reason is None else 'failed_series'
            report[outcome] += 1
            if reason:
                report['reason_counts'][reason] = report['reason_counts'].get(reason,0) + 1
            else:
                report['successful_selected_slices'] += len(images)
                for category in current_syntax:
                    report['transfer_syntax'][category]['successful_series'] += 1
            record = {'stage': 'preprocess', 'study_ordinal': study_index, 'series_ordinal': series_index, 'status': 'ok' if reason is None else 'failed', 'reason_code': reason}
            report['series_results'].append(record)
            checkpoint()
            print(json.dumps({'event':'compatibility_progress', **record, 'attempted_series':report['attempted_series'], 'planned_series':report['planned_series'], 'elapsed_seconds':report['elapsed_seconds']}),flush=True)
            if last_in_study:
                report['completed_studies'] += 1
        report['status'] = 'complete'
        report['termination_reason'] = 'all_planned_series_attempted'
    except KeyboardInterrupt:
        report['status'] = 'partial'
        report['termination_reason'] = 'interrupted'
    except Exception as error:
        report['status'] = 'failed'
        report['termination_reason'] = 'missing_dependency' if isinstance(error, ImportError) else 'audit_failed'
    checkpoint()
    print(json.dumps({'event':'compatibility_finished','status':report['status'],'attempted_series':report['attempted_series'],'planned_series':report['planned_series']}),flush=True)
    return report
