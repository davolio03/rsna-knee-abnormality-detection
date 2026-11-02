"""Deterministic whole-study chunk plan for the isolated 16-slice cache."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Sequence

SCHEMA = "local16-cache-chunk-plan-v1"
MAX_SLICES = 16
SERIES_BYTES_UPPER_BOUND = 16 * 224 * 224 * 5
OUTPUT_LIMIT_BYTES = 20_000_000_000
OUTPUT_RESERVE_BYTES = 1_000_000_000
MAX_SERIES_PER_CHUNK = 2300
_HASH = re.compile(r"[0-9a-f]{64}\Z")


def _selection_digest(partition: str, rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256(b"local16-cache-selection-v1\n" + partition.encode() + b"\n")
    for ordinal, row in enumerate(rows):
        digest.update(str(ordinal).encode() + b"\0" + row["study_id"].encode() + b"\0")
        for series in row["series"]:
            digest.update(series["series_id"].encode() + b"\0")
    return digest.hexdigest()


def create_chunk_plan(manifest: Mapping[str, Any], folds: Mapping[str, Any], *,
                      manifest_sha256: str, folds_sha256: str, source_sha256: str,
                      preprocessing_source_sha256: str,
                      max_series_per_chunk: int = MAX_SERIES_PER_CHUNK) -> dict[str, Any]:
    for value in (manifest_sha256, folds_sha256, source_sha256, preprocessing_source_sha256):
        if type(value) is not str or not _HASH.fullmatch(value):
            raise ValueError("chunk_plan_pin_invalid")
    if type(max_series_per_chunk) is not int or not 1 <= max_series_per_chunk <= MAX_SERIES_PER_CHUNK:
        raise ValueError("series_chunk_cap_invalid")
    if type(manifest) is not dict or manifest.get("schema_version") != "full-cohort-soft-label-manifest-v1" or type(manifest.get("records")) is not list:
        raise ValueError("chunk_plan_manifest_invalid")
    if (type(folds) is not dict or folds.get("manifest_sha256") != hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n").hexdigest()
            or type(folds.get("study_assignments")) is not list):
        raise ValueError("chunk_plan_manifest_binding_mismatch")
    assignments = {}
    for row in folds["study_assignments"]:
        if (type(row) is not dict or type(row.get("study_id")) is not str or row["study_id"] in assignments
                or type(row.get("fold")) is not int or row["fold"] not in (0, 1, 2)):
            raise ValueError("chunk_plan_assignment_invalid")
        assignments[row["study_id"]] = row["fold"]
    rows, ids = manifest["records"], set()
    for row in rows:
        if type(row) is not dict or type(row.get("study_id")) is not str or row["study_id"] in ids or type(row.get("series")) is not list or not row["series"]:
            raise ValueError("chunk_plan_record_invalid")
        ids.add(row["study_id"])
        series_ids = [s.get("series_id") for s in row["series"] if type(s) is dict]
        if len(series_ids) != len(row["series"]) or any(type(i) is not str or not i for i in series_ids) or len(set(series_ids)) != len(series_ids):
            raise ValueError("chunk_plan_series_invalid")
    if set(assignments) != ids:
        raise ValueError("chunk_plan_assignment_membership_mismatch")
    partitions = {"train": [r for r in rows if assignments[r["study_id"]] != 0],
                  "validation": [r for r in rows if assignments[r["study_id"]] == 0]}
    chunks, p_hashes = [], {}
    for partition, selected in partitions.items():
        p_hash = _selection_digest(partition, selected)
        p_hashes[partition] = p_hash
        start = 0
        while start < len(selected):
            stop, count = start, 0
            while stop < len(selected):
                next_count = len(selected[stop]["series"])
                if next_count > max_series_per_chunk:
                    raise ValueError("study_exceeds_series_chunk_cap")
                if count and count + next_count > max_series_per_chunk:
                    break
                count += next_count
                stop += 1
            raw_bound = count * SERIES_BYTES_UPPER_BOUND
            saved_bound = 2 * raw_bound + OUTPUT_RESERVE_BYTES
            if saved_bound >= OUTPUT_LIMIT_BYTES:
                raise ValueError("chunk_saved_output_limit_exceeded")
            chunks.append({"chunk_id": f"{partition}-{sum(c['partition'] == partition for c in chunks):03d}",
                           "partition": partition, "start_ordinal": start,
                           "stop_ordinal_exclusive": stop, "study_count": stop - start,
                           "series_count": count, "max_uncompressed_tensor_bytes": raw_bound,
                           "max_saved_output_bytes": saved_bound,
                           "chunk_selection_sha256": _selection_digest(partition, selected[start:stop])})
            start = stop
    result = {"schema_version": SCHEMA, "cache_identity": "local16-attention-run01", "fold": 0,
              "gold_included": False, "max_slices_per_series": MAX_SLICES,
              "output_size": 224, "study_count": len(rows), "series_count": sum(len(r["series"]) for r in rows),
              "partition_counts": {key: len(value) for key, value in partitions.items()},
              "manifest_sha256": manifest_sha256, "folds_sha256": folds_sha256,
              "source_sha256": source_sha256, "preprocessing_source_sha256": preprocessing_source_sha256,
              "max_series_per_chunk": max_series_per_chunk, "series_bytes_upper_bound": SERIES_BYTES_UPPER_BOUND,
              "output_limit_bytes": OUTPUT_LIMIT_BYTES, "output_reserve_bytes": OUTPUT_RESERVE_BYTES,
              "partition_selection_sha256": p_hashes, "chunks": chunks}
    validate_chunk_plan(result)
    return result


def validate_chunk_plan(plan: Mapping[str, Any]) -> bool:
    if (type(plan) is not dict or plan.get("schema_version") != SCHEMA
            or plan.get("cache_identity") != "local16-attention-run01" or plan.get("fold") != 0
            or plan.get("gold_included") is not False or plan.get("max_slices_per_series") != 16
            or plan.get("output_size") != 224 or plan.get("series_bytes_upper_bound") != SERIES_BYTES_UPPER_BOUND
            or plan.get("max_series_per_chunk", MAX_SERIES_PER_CHUNK) > MAX_SERIES_PER_CHUNK
            or plan.get("output_limit_bytes") != OUTPUT_LIMIT_BYTES
            or plan.get("output_reserve_bytes") != OUTPUT_RESERVE_BYTES
            or type(plan.get("chunks")) is not list):
        raise ValueError("chunk_plan_schema_invalid")
    counts = plan.get("partition_counts")
    if type(counts) is not dict or set(counts) != {"train", "validation"}:
        raise ValueError("chunk_plan_coverage_invalid")
    for key in ("manifest_sha256", "folds_sha256", "source_sha256", "preprocessing_source_sha256"):
        if type(plan.get(key)) is not str or not _HASH.fullmatch(plan[key]):
            raise ValueError("chunk_plan_pin_invalid")
    for partition in ("train", "validation"):
        part = [row for row in plan["chunks"] if type(row) is dict and row.get("partition") == partition]
        if not part or sum(row.get("study_count", 0) for row in part) != counts[partition]:
            raise ValueError("chunk_plan_coverage_invalid")
        next_start = 0
        for i, row in enumerate(part):
            series = row.get("series_count")
            raw = series * SERIES_BYTES_UPPER_BOUND if type(series) is int else -1
            if (row.get("chunk_id") != f"{partition}-{i:03d}" or row.get("start_ordinal") != next_start
                    or type(row.get("stop_ordinal_exclusive")) is not int
                    or row["stop_ordinal_exclusive"] - next_start != row.get("study_count")
                    or type(series) is not int or not 1 <= series <= plan["max_series_per_chunk"]
                    or row.get("max_uncompressed_tensor_bytes") != raw
                    or row.get("max_saved_output_bytes") != 2 * raw + OUTPUT_RESERVE_BYTES
                    or row["max_saved_output_bytes"] >= OUTPUT_LIMIT_BYTES
                    or type(row.get("chunk_selection_sha256")) is not str or not _HASH.fullmatch(row["chunk_selection_sha256"])):
                raise ValueError("chunk_plan_coverage_invalid")
            next_start = row["stop_ordinal_exclusive"]
    if sum(counts.values()) != plan.get("study_count") or len(plan["chunks"]) == 0:
        raise ValueError("chunk_plan_coverage_invalid")
    return True
