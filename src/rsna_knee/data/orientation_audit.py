"""Redacted, pixel-free DICOM orientation measurements for authorized audits."""

from __future__ import annotations

import math
import json
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any


ORIENTATION_COMPONENT_TOLERANCE = 1e-5
ORIENTATION_TAGS = ("ImageOrientationPatient",)


class _JSONCursor:
    """Small selective JSON reader that skips unneeded values without decoding them."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.index = 0
        self.depth = 0
        self.decoder = json.JSONDecoder()

    def whitespace(self) -> None:
        while self.index < len(self.text) and self.text[self.index] in " \t\r\n":
            self.index += 1

    def take(self, expected: str) -> None:
        self.whitespace()
        if self.index >= len(self.text) or self.text[self.index] != expected:
            raise ValueError("manifest_invalid")
        self.index += 1

    def string(self) -> str:
        self.whitespace()
        try:
            value, end = self.decoder.raw_decode(self.text, self.index)
        except (json.JSONDecodeError, TypeError, ValueError):
            raise ValueError("manifest_invalid") from None
        if type(value) is not str:
            raise ValueError("manifest_invalid")
        self.index = end
        return value

    def skip_string(self) -> None:
        self.whitespace()
        self.take('"')
        escaped = False
        while self.index < len(self.text):
            char = self.text[self.index]
            self.index += 1
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                return
        raise ValueError("manifest_invalid")

    def skip_value(self) -> None:
        self.whitespace()
        self.depth += 1
        if self.depth > 128:
            raise ValueError("manifest_invalid")
        try:
            char = self.text[self.index:self.index + 1]
            if char == '"':
                self.skip_string()
            elif char == "{":
                self.index += 1
                self.whitespace()
                if self.text[self.index:self.index + 1] != "}":
                    while True:
                        self.skip_string()
                        self.take(":")
                        self.skip_value()
                        self.whitespace()
                        if self.text[self.index:self.index + 1] == "}":
                            break
                        self.take(",")
                self.take("}")
            elif char == "[":
                self.index += 1
                self.whitespace()
                if self.text[self.index:self.index + 1] != "]":
                    while True:
                        self.skip_value()
                        self.whitespace()
                        if self.text[self.index:self.index + 1] == "]":
                            break
                        self.take(",")
                self.take("]")
            else:
                start = self.index
                while self.index < len(self.text) and self.text[self.index] not in ",]} \t\r\n":
                    self.index += 1
                token = self.text[start:self.index]
                try:
                    json.loads(token)
                except (json.JSONDecodeError, TypeError, ValueError):
                    raise ValueError("manifest_invalid") from None
        finally:
            self.depth -= 1


def extract_manifest_series(raw_manifest: bytes) -> list[tuple[str, tuple[str, ...]]]:
    """Extract ordered study/series keys while lexically skipping all other fields.

    In particular, the label values are not JSON-decoded or returned.
    """
    try:
        text = raw_manifest.decode("utf-8")
        cursor = _JSONCursor(text)
        records: list[tuple[str, tuple[str, ...]]] | None = None
        cursor.take("{")
        cursor.whitespace()
        if cursor.text[cursor.index:cursor.index + 1] != "}":
            while True:
                key = cursor.string()
                cursor.take(":")
                if key == "records":
                    records = _parse_records(cursor)
                else:
                    cursor.skip_value()
                cursor.whitespace()
                if cursor.text[cursor.index:cursor.index + 1] == "}":
                    break
                cursor.take(",")
        cursor.take("}")
        cursor.whitespace()
        if cursor.index != len(cursor.text) or records is None or not records:
            raise ValueError("manifest_invalid")
        return records
    except (UnicodeDecodeError, IndexError, TypeError, ValueError):
        raise ValueError("manifest_invalid") from None


def _parse_records(cursor: _JSONCursor) -> list[tuple[str, tuple[str, ...]]]:
    records: list[tuple[str, tuple[str, ...]]] = []
    cursor.take("[")
    cursor.whitespace()
    if cursor.text[cursor.index:cursor.index + 1] == "]":
        cursor.take("]")
        return records
    while True:
        cursor.take("{")
        study_id = None
        series = None
        cursor.whitespace()
        if cursor.text[cursor.index:cursor.index + 1] != "}":
            while True:
                key = cursor.string()
                cursor.take(":")
                if key == "study_id":
                    study_id = cursor.string()
                elif key == "series":
                    series = _parse_series(cursor)
                else:
                    cursor.skip_value()
                cursor.whitespace()
                if cursor.text[cursor.index:cursor.index + 1] == "}":
                    break
                cursor.take(",")
        cursor.take("}")
        if type(study_id) is not str or not study_id or not series:
            raise ValueError("manifest_invalid")
        records.append((study_id, tuple(series)))
        cursor.whitespace()
        if cursor.text[cursor.index:cursor.index + 1] == "]":
            break
        cursor.take(",")
    cursor.take("]")
    return records


def _parse_series(cursor: _JSONCursor) -> list[str]:
    series: list[str] = []
    cursor.take("[")
    cursor.whitespace()
    if cursor.text[cursor.index:cursor.index + 1] == "]":
        cursor.take("]")
        return series
    while True:
        cursor.take("{")
        series_id = None
        cursor.whitespace()
        if cursor.text[cursor.index:cursor.index + 1] != "}":
            while True:
                key = cursor.string()
                cursor.take(":")
                if key == "series_id":
                    series_id = cursor.string()
                else:
                    cursor.skip_value()
                cursor.whitespace()
                if cursor.text[cursor.index:cursor.index + 1] == "}":
                    break
                cursor.take(",")
        cursor.take("}")
        if type(series_id) is not str or not series_id:
            raise ValueError("manifest_invalid")
        series.append(series_id)
        cursor.whitespace()
        if cursor.text[cursor.index:cursor.index + 1] == "]":
            break
        cursor.take(",")
    cursor.take("]")
    return series


def _valid_orientation(value: object) -> tuple[float, ...] | None:
    if isinstance(value, (str, bytes)):
        return None
    try:
        values = tuple(float(part) for part in value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    if len(values) != 6 or not all(math.isfinite(part) for part in values):
        return None
    row, column = values[:3], values[3:]
    if _norm(row) == 0.0 or _norm(column) == 0.0:
        return None
    if _norm(_cross(_unit(row), _unit(column))) == 0.0:
        return None
    return values


def _norm(vector: Sequence[float]) -> float:
    return math.hypot(*vector)


def _cross(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _unit(vector: Sequence[float]) -> tuple[float, ...]:
    magnitude = _norm(vector)
    return tuple(value / magnitude for value in vector)


def _angle_degrees(left: Sequence[float], right: Sequence[float]) -> float:
    unit_left = _unit(left)
    unit_right = _unit(right)
    cross_magnitude = _norm(_cross(unit_left, unit_right))
    dot_product = math.fsum(a * b for a, b in zip(unit_left, unit_right))
    return math.degrees(math.atan2(cross_magnitude, dot_product))


def audit_orientation_headers(
    orientations: Iterable[object],
    *,
    threshold: float = ORIENTATION_COMPONENT_TOLERANCE,
) -> dict[str, int | float | None]:
    """Measure component differences and geometric angles without retaining IDs."""
    if type(threshold) not in (int, float) or not math.isfinite(float(threshold)) or threshold != 1e-5:
        raise ValueError("The orientation threshold is fixed by the audited preprocessor.")
    raw = list(orientations)
    valid = [item for item in (_valid_orientation(value) for value in raw) if item is not None]
    result: dict[str, int | float | None] = {
        "header_count": len(raw),
        "valid_orientation_count": len(valid),
        "invalid_orientation_count": len(raw) - len(valid),
        "component_count_above_threshold": 0,
        "header_count_above_threshold": 0,
        "max_component_delta": None,
        "max_component_spread": None,
        "max_row_angle_degrees": None,
        "max_column_angle_degrees": None,
        "max_normal_angle_degrees": None,
        "reference_header_ordinal": None,
        "reference_is_first_header": False,
    }
    if not valid:
        return result

    reference = valid[0]
    reference_ordinal = next(
        index for index, value in enumerate(raw, start=1) if _valid_orientation(value) is not None
    )
    max_delta = 0.0
    component_count = 0
    header_count = 0
    for orientation in valid[1:]:
        deltas = [abs(value - baseline) for value, baseline in zip(orientation, reference)]
        max_delta = max(max_delta, *deltas)
        component_count += sum(delta > threshold for delta in deltas)
        header_count += int(any(delta > threshold for delta in deltas))

    spreads = [max(values) - min(values) for values in zip(*valid)]
    reference_row = reference[:3]
    reference_column = reference[3:]
    reference_row = _unit(reference_row)
    reference_column = _unit(reference_column)
    reference_normal = _cross(reference_row, reference_column)
    row_angles = [_angle_degrees(item[:3], reference_row) for item in valid]
    column_angles = [_angle_degrees(item[3:], reference_column) for item in valid]
    normal_angles = [
        _angle_degrees(_cross(_unit(item[:3]), _unit(item[3:])), reference_normal)
        for item in valid
    ]
    result.update({
        "component_count_above_threshold": component_count,
        "header_count_above_threshold": header_count,
        "max_component_delta": max_delta,
        "max_component_spread": max(spreads),
        "max_row_angle_degrees": max(row_angles),
        "max_column_angle_degrees": max(column_angles),
        "max_normal_angle_degrees": max(normal_angles),
        "reference_header_ordinal": reference_ordinal,
        "reference_is_first_header": reference_ordinal == 1,
    })
    return result


def scan_first_inconsistent_series(
    series_groups: Iterable[Iterable[str | Path]],
    dcmread: Callable[..., Any],
    *,
    enabled: bool = True,
    provenance_valid: bool = True,
    max_headers: int = 10000,
    max_seconds: float = 600.0,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Scan complete ordered series until the first threshold breach or a cap.

    ``series_groups`` must already follow the exact training study/series/file
    order. No path or exception text is returned or retained in the result.
    """
    if not enabled:
        return {"status": "disabled", "headers_read": 0}
    if not provenance_valid:
        return {"status": "blocked", "blocked_reason": "provenance_invalid", "headers_read": 0}
    if type(max_headers) is not int or max_headers < 1:
        raise ValueError("max_headers must be a positive integer.")
    if type(max_seconds) not in (int, float) or not math.isfinite(float(max_seconds)) or max_seconds <= 0:
        raise ValueError("max_seconds must be finite and positive.")

    started = monotonic()
    headers_read = 0
    reader_failures = 0
    last_series_headers = 0
    series_seen = False
    for series_ordinal, paths in enumerate(series_groups, start=1):
        series_seen = True
        if not paths:
            raise ValueError("series_files_missing")
        values: list[object] = []
        series_failures = 0
        for path in paths:
            if headers_read >= max_headers:
                summary = audit_orientation_headers(values)
                return {
                    "status": "partial", "partial": True,
                    "partial_reason": "header_limit_reached",
                    "headers_read": headers_read, "series_ordinal": series_ordinal,
                    "headers_read_in_series": len(values),
                    "series_complete": False, "header_read_failure_count": reader_failures,
                    "series_header_read_failure_count": series_failures,
                    "candidate_exceeds_threshold": bool(
                        summary["reference_is_first_header"]
                        and summary["header_count_above_threshold"]
                    ),
                    "measurements": summary,
                }
            if monotonic() - started >= max_seconds:
                summary = audit_orientation_headers(values)
                return {
                    "status": "partial", "partial": True,
                    "partial_reason": "runtime_limit_reached",
                    "headers_read": headers_read, "series_ordinal": series_ordinal,
                    "headers_read_in_series": len(values),
                    "series_complete": False, "header_read_failure_count": reader_failures,
                    "series_header_read_failure_count": series_failures,
                    "candidate_exceeds_threshold": bool(
                        summary["reference_is_first_header"]
                        and summary["header_count_above_threshold"]
                    ),
                    "measurements": summary,
                }
            headers_read += 1
            try:
                dataset = dcmread(
                    path,
                    stop_before_pixels=True,
                    specific_tags=list(ORIENTATION_TAGS),
                )
                values.append(getattr(dataset, "ImageOrientationPatient", None))
            except Exception:
                reader_failures += 1
                series_failures += 1
                values.append(None)

        summary = audit_orientation_headers(values)
        last_series_headers = len(values)
        if summary["reference_is_first_header"] and summary["header_count_above_threshold"]:
            return {
                "status": "complete", "partial": False,
                "headers_read": headers_read, "series_ordinal": series_ordinal,
                "headers_read_in_series": len(values),
                "series_complete": True, "header_read_failure_count": reader_failures,
                "series_header_read_failure_count": series_failures,
                "inconsistent_orientation": True, "measurements": summary,
            }

    if not series_seen or headers_read == 0:
        raise ValueError("series_files_missing")

    return {
        "status": "complete", "partial": False, "headers_read": headers_read,
        "series_ordinal": None, "series_complete": True,
        "headers_read_in_series": last_series_headers,
        "header_read_failure_count": reader_failures,
        "series_header_read_failure_count": 0,
        "inconsistent_orientation": False,
        "measurements": None,
    }
