"""Private one-header-per-series strata inventory; never decode pixels."""
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

VERSION = 'training-header-inventory-v1'
PLANES = {'Axial', 'Coronal', 'Sagittal'}
TAGS = {'Rows': 0x00280010, 'Columns': 0x00280011, 'NumberOfFrames': 0x00280008,
        'SamplesPerPixel': 0x00280002, 'PhotometricInterpretation': 0x00280004,
        'PixelSpacing': 0x00280030, 'ImageOrientationPatient': 0x00200037,
        'ImagePositionPatient': 0x00200032}
SYNTAXES = {'1.2.840.10008.1.2': 'implicit_vr_little_endian',
            '1.2.840.10008.1.2.1': 'explicit_vr_little_endian',
            '1.2.840.10008.1.2.2': 'explicit_vr_big_endian',
            '1.2.840.10008.1.2.1.99': 'deflated_explicit_vr_little_endian',
            '1.2.840.10008.1.2.4.50': 'jpeg_baseline',
            '1.2.840.10008.1.2.4.51': 'jpeg_extended',
            '1.2.840.10008.1.2.4.57': 'jpeg_lossless',
            '1.2.840.10008.1.2.4.70': 'jpeg_lossless_sv1',
            '1.2.840.10008.1.2.4.80': 'jpeg_ls_lossless',
            '1.2.840.10008.1.2.4.81': 'jpeg_ls_near_lossless',
            '1.2.840.10008.1.2.4.90': 'jpeg2000_lossless',
            '1.2.840.10008.1.2.4.91': 'jpeg2000', '1.2.840.10008.1.2.5': 'rle'}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _id(value):
    # Strict numeric DICOM UID components, never path syntax or arbitrary names.
    if type(value) is not str or len(value) > 128 or re.fullmatch(r'[0-9]+(?:\.[0-9]+)*', value) is None:
        raise ValueError('inventory_plan_invalid')
    return value


def _metadata(raw):
    reader = csv.DictReader(raw.decode('utf-8-sig').splitlines())
    if (reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames)
            or not {'StudyInstanceUID', 'SeriesInstanceUID', 'Anatomical_Plane'}.issubset(reader.fieldnames)
            or not set(reader.fieldnames).issubset({'StudyInstanceUID', 'SeriesInstanceUID', 'Anatomical_Plane', 'Fluid_Sensitive', 'Fat_Suppression'})):
        raise ValueError('inventory_metadata_invalid')
    groups, seen = {}, set()
    for row in reader:
        study, series = _id(row['StudyInstanceUID']), _id(row['SeriesInstanceUID'])
        if None in row or series in seen or row['Anatomical_Plane'] not in PLANES:
            raise ValueError('inventory_metadata_invalid')
        seen.add(series)
        groups.setdefault(study, []).append({'series_id': series, 'plane': row['Anatomical_Plane']})
    for rows in groups.values():
        rows.sort(key=lambda r: r['series_id'])
    return groups


def _selected(groups, excluded):
    ids = sorted(set(groups) - set(excluded), key=lambda s: (digest(f'header-seed-42:{s}'.encode()), s))
    return [{'study_id': s, 'series': groups[s]} for s in ids]


def validate_plan(plan):
    try:
        if type(plan) is not dict or set(plan) != {'schema_version', 'source_sha256', 'excluded_ids', 'studies'} or plan['schema_version'] != VERSION:
            raise ValueError
        pins = plan['source_sha256']
        if set(pins) != {'series', 'gold', 'prior', 'audit_plan', 'audit_report'} or any(type(v) is not str or re.fullmatch('[0-9a-f]{64}', v) is None for v in pins.values()):
            raise ValueError
        excluded = plan['excluded_ids']
        if type(excluded) is not list or not excluded or len(set(excluded)) != len(excluded):
            raise ValueError
        for value in excluded:
            _id(value)
        studies, series = set(), set()
        if type(plan['studies']) is not list or not plan['studies']:
            raise ValueError
        for study in plan['studies']:
            if type(study) is not dict or set(study) != {'study_id', 'series'}:
                raise ValueError
            key = _id(study['study_id'])
            if key in studies or key in excluded or type(study['series']) is not list or not study['series']:
                raise ValueError
            studies.add(key)
            for row in study['series']:
                if type(row) is not dict or set(row) != {'series_id', 'plane'} or row['plane'] not in PLANES:
                    raise ValueError
                key = _id(row['series_id'])
                if key in series:
                    raise ValueError
                series.add(key)
    except (KeyError, TypeError, ValueError):
        raise ValueError('inventory_plan_invalid') from None


def plan_bytes(plan):
    validate_plan(plan)
    return (json.dumps(plan, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def build_header_plan(series_csv, gold_csv, prior_manifest, audit_plan_file, report_file, pins):
    """Local-only freeze: verify exclusions and previous completed audit first."""
    from rsna_knee.data.compatibility_audit import build_audit_plan, plan_bytes as audit_bytes
    try:
        files = {'series': series_csv, 'gold': gold_csv, 'prior': prior_manifest,
                 'audit_plan': audit_plan_file, 'audit_report': report_file}
        raw = {}
        for name, file in files.items():
            path = Path(file)
            if path.is_symlink():
                raise ValueError
            raw[name] = path.read_bytes()
            if digest(raw[name]) != pins[name]:
                raise ValueError
        old = json.loads(raw['audit_plan'])
        rebuilt = build_audit_plan(series_csv, gold_csv, prior_manifest,
                                  {k: pins[k] for k in ('series', 'gold', 'prior')},
                                  sample_size=old['sample_size'], seed=old['seed'])
        if audit_bytes(rebuilt) != raw['audit_plan']:
            raise ValueError
        report = json.loads(raw['audit_report'])
        pairs = {(i, j) for i, s in enumerate(old['studies'], 1) for j, _ in enumerate(s['series'], 1)}
        rows = report['series_results']
        observed = [(r['study_ordinal'], r['series_ordinal']) for r in rows]
        n = len(pairs)
        if (report['plan_sha256'] != pins['audit_plan'] or report['status'] != 'complete'
                or report['termination_reason'] != 'all_planned_series_attempted'
                or report['failed_series'] != 0 or report['completed_studies'] != len(old['studies'])
                or any(report[k] != n for k in ('planned_series', 'attempted_series', 'completed_series', 'successful_series'))
                or observed != sorted(pairs)
                or any(r['stage'] != 'preprocess' or r['status'] != 'ok' or r['reason_code'] is not None for r in rows)):
            raise ValueError
        groups = _metadata(raw['series'])
        excluded = sorted(set(old['gold_ids'] + old['prior_ids'] + [s['study_id'] for s in old['studies']]))
        plan = {'schema_version': VERSION, 'source_sha256': dict(pins), 'excluded_ids': excluded,
                'studies': _selected(groups, excluded)}
        validate_plan(plan)
        return plan
    except (OSError, UnicodeError, KeyError, TypeError, ValueError):
        raise ValueError('inventory_provenance_invalid') from None


class _HeaderLimit(ValueError):
    pass


class _BoundedHeader:
    """Bound actual bytes read, including file-meta and malformed huge tags."""
    def __init__(self, stream):
        self.stream, self.remaining = stream, 1024 * 1024

    def read(self, size=-1):
        if size < 0 or size > self.remaining:
            raise _HeaderLimit()
        raw = self.stream.read(size)
        self.remaining -= len(raw)
        return raw

    def seek(self, *args):
        return self.stream.seek(*args)

    def tell(self):
        return self.stream.tell()


_INVALID = object()


def _value(dataset, keyword):
    # Never trigger deferred oversized tag reads. Require installed public API.
    elem = dataset.get_item(TAGS[keyword], keep_deferred=True)
    if elem is None:
        return None
    if getattr(elem, 'length', 0) > 1024 or getattr(elem, 'value', None) is None:
        return _INVALID
    value = getattr(dataset, keyword)
    return value


def _integer(value):
    if value is None:
        return None
    if value is _INVALID or isinstance(value, bool) or not isinstance(value, (str, int)) or len(str(value)) > 12:
        return _INVALID
    try:
        number = int(value)
        return number if 1 <= number <= 1_000_000 and str(number) == str(value).strip() else _INVALID
    except (ValueError, OverflowError):
        return _INVALID


def _vector(value, size, positive=False):
    if value is None:
        return 'missing'
    try:
        if value is _INVALID or len(value) != size:
            return 'invalid'
        numbers = []
        for v in value:
            if isinstance(v, bool) or len(str(v)) > 64:
                return 'invalid'
            n = float(v)
            if not math.isfinite(n) or abs(n) > 1e9 or (positive and n <= 0):
                return 'invalid'
            numbers.append(n)
        if size == 6:
            a, b = numbers[:3], numbers[3:]
            if abs(sum(x*x for x in a)-1) > 1e-4 or abs(sum(x*x for x in b)-1) > 1e-4 or abs(sum(x*y for x,y in zip(a,b))) > 1e-4:
                return 'invalid'
        return 'valid'
    except (TypeError, ValueError, OverflowError):
        return 'invalid'


def _strata(dataset):
    uid = getattr(getattr(dataset, 'file_meta', None), 'TransferSyntaxUID', None)
    syntax = SYNTAXES.get(uid, 'other') if isinstance(uid, str) and len(uid) <= 128 else 'other'
    strata = {'transfer_syntax': syntax}
    flags, ranges = [], {}
    if syntax != 'explicit_vr_little_endian':
        flags.append('uncovered_transfer_syntax')
    for key in ('Rows', 'Columns', 'NumberOfFrames', 'SamplesPerPixel'):
        value = _integer(_value(dataset, key))
        strata[key] = 'missing' if value is None else 'invalid' if value is _INVALID else 'single' if value == 1 else 'multiple'
        if value is not None and value is not _INVALID:
            ranges[key] = value
        if key in ('Rows', 'Columns') and (value is None or value is _INVALID or not 256 <= value <= 1280):
            flags.append('uncovered_dimensions')
        elif key == 'NumberOfFrames' and value is not None and value != 1:
            flags.append('unusual_frames')
        elif key == 'SamplesPerPixel' and value != 1:
            flags.append('unusual_samples')
    photo = _value(dataset, 'PhotometricInterpretation')
    strata['photometric'] = photo if type(photo) is str and photo in {'MONOCHROME1', 'MONOCHROME2', 'RGB', 'PALETTE COLOR', 'YBR_FULL', 'YBR_FULL_422'} else 'missing' if photo is None else 'other'
    if strata['photometric'] not in {'MONOCHROME1', 'MONOCHROME2'}:
        flags.append('unusual_photometric')
    for key, size, positive in [('PixelSpacing', 2, True), ('ImageOrientationPatient', 6, False), ('ImagePositionPatient', 3, False)]:
        strata[key] = _vector(_value(dataset, key), size, positive)
        if strata[key] != 'valid':
            flags.append('unusual_geometry')
    return strata, sorted(set(flags)), ranges


def _header_path(root, study, series):
    path = root / study / series
    if any(p.is_symlink() for p in (root, root / study, path)) or not path.is_dir():
        raise ValueError('inventory_path_invalid')
    files = []
    for item in path.iterdir():
        if item.is_symlink() or not item.is_file() or item.suffix != '.dcm':
            raise ValueError('inventory_path_invalid')
        try:
            _id(item.stem)
        except ValueError:
            raise ValueError('inventory_path_invalid') from None
        files.append(item)
    if not files:
        raise ValueError('inventory_files_missing')
    return min(files, key=lambda p: p.name), len(files)


def run_header_inventory(plan, series_csv, image_root, output_dir, *, dcmread=None, clock=time.perf_counter):
    """Continue safe per-series header failures, never change production behavior."""
    validate_plan(plan)
    try:
        raw = Path(series_csv).read_bytes()
        groups = _metadata(raw)
        if digest(raw) != plan['source_sha256']['series'] or not set(plan['excluded_ids']).issubset(groups) or _selected(groups, plan['excluded_ids']) != plan['studies']:
            raise ValueError
        root, output = Path(image_root), Path(output_dir)
        if not root.is_dir() or any(p.is_symlink() for p in (root, *root.parents, output, *output.parents)) or output.exists():
            raise ValueError
    except (OSError, UnicodeError, KeyError, TypeError, ValueError, csv.Error):
        raise ValueError('inventory_initialization_invalid') from None
    output.mkdir(parents=True, exist_ok=False)
    started = clock()
    report = {'schema_version': VERSION, 'plan_sha256': digest(plan_bytes(plan)), 'status': 'partial',
              'termination_reason': 'in_progress', 'planned_studies': len(plan['studies']),
              'planned_series': sum(len(s['series']) for s in plan['studies']), 'completed_studies': 0,
              'attempted_series': 0, 'completed_series': 0, 'successful_series': 0, 'failed_series': 0,
              'header_read_attempts': 0, 'header_reads': 0, 'enumerated_files': 0,
              'plane_counts': {}, 'strata': {}, 'ranges': {}, 'flag_counts': {}, 'reason_counts': {},
              'candidate_ordinals': {}, 'budget_seconds': 1800, 'elapsed_seconds': 0.0,
              'limit': 'One selected header per series; no within-series consistency, pixel, codec or model capability proof.'}

    def checkpoint():
        report['elapsed_seconds'] = max(0.0, clock() - started)
        try:
            with (output / 'report.tmp').open('x', encoding='utf-8') as stream:
                json.dump(report, stream, sort_keys=True, allow_nan=False)
                stream.write('\n')
            os.replace(output / 'report.tmp', output / 'report.json')
        except OSError:
            raise ValueError('inventory_checkpoint_failed') from None

    def count(mapping, key):
        mapping[key] = mapping.get(key, 0) + 1

    checkpoint()
    try:
        if dcmread is None:
            from pydicom import dcmread
        for i, study in enumerate(plan['studies'], 1):
            for j, row in enumerate(study['series'], 1):
                if clock() - started >= 1800:
                    report['termination_reason'] = 'time_budget'
                    checkpoint()
                    return report
                report['attempted_series'] += 1
                checkpoint()
                flags, reason = [], None
                try:
                    path, size = _header_path(root, study['study_id'], row['series_id'])
                    report['enumerated_files'] += size
                    report['header_read_attempts'] += 1
                    with path.open('rb') as stream, warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        dataset = dcmread(_BoundedHeader(stream), stop_before_pixels=True,
                                          specific_tags=tuple(TAGS.values()), defer_size=1024)
                        report['header_reads'] += 1
                        strata, flags, ranges = _strata(dataset)
                    for key, value in strata.items():
                        count(report['strata'].setdefault(key, {}), value)
                    for key, value in ranges.items():
                        old = report['ranges'].get(key, [value, value])
                        report['ranges'][key] = [min(old[0], value), max(old[1], value)]
                except _HeaderLimit:
                    reason = 'header_byte_limit'
                except Exception as error:
                    reason = error.args[0] if type(error) is ValueError and len(error.args) == 1 and type(error.args[0]) is str and error.args[0] in {'inventory_path_invalid', 'inventory_files_missing'} else 'header_read_or_metadata_failed'
                report['completed_series'] += 1
                count(report['plane_counts'], row['plane'])
                report['failed_series' if reason else 'successful_series'] += 1
                if reason:
                    count(report['reason_counts'], reason)
                    flags.append(reason)
                for flag in flags:
                    count(report['flag_counts'], flag)
                    candidates = report['candidate_ordinals'].setdefault(flag, [])
                    if len(candidates) < 20:
                        candidates.append({'study_ordinal': i, 'series_ordinal': j})
                checkpoint()
                # Bounded aggregate progress, no identifiers or raw metadata.
                if report['completed_series'] % 100 == 0:
                    print(json.dumps({'event': 'header_progress', 'completed_series': report['completed_series'], 'failed_series': report['failed_series'], 'elapsed_seconds': report['elapsed_seconds']}), flush=True)
            report['completed_studies'] += 1
        report['status'], report['termination_reason'] = 'complete', 'all_planned_series_attempted'
    except KeyboardInterrupt:
        report['termination_reason'] = 'interrupted'
    except Exception:
        report['status'], report['termination_reason'] = 'failed', 'inventory_failed'
    checkpoint()
    print(json.dumps({'event': 'header_inventory_finished', 'status': report['status'], 'completed_series': report['completed_series'], 'failed_series': report['failed_series']}), flush=True)
    return report
