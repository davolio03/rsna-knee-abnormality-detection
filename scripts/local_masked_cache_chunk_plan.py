"""Deterministic whole-study chunk planning for private selected-slice export."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import zipfile


EXPECTED_MANIFEST_SHA256 = "617ea1c267960b1e903b332a78388db3b6ab80d44374721df5a4c5e184e8961b"
EXPECTED_FOLDS_SHA256 = "b816c7902a5d0caa68a9e818d3f913943bc61cd2e6db65c4de069f5cd94e15f9"
EXPECTED_SOURCE_SHA256 = "48a64d88c9b5cf72f4fdf6d9ce9839cc2cb3b8632b9fbf4fb7f4db1cdc4f95c8"
EXPECTED_STUDY_COUNT = 4298
EXPECTED_TRAIN_COUNT = 2850
EXPECTED_VALIDATION_COUNT = 1448
EXPECTED_SERIES_COUNT = 23765
OUTPUT_LIMIT_BYTES = 20_000_000_000
SERIES_BYTES_UPPER_BOUND = 2_007_040  # 8 * 224 * 224 * (float32 pixel + bool mask)
OUTPUT_RESERVE_BYTES = 1_000_000_000
_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
_PREPROCESSING_MODULES = (
    "rsna_knee/data/masked_soft_dataset.py",
    "rsna_knee/data/dicom_preprocessing.py",
    "rsna_knee/data/training_dataset.py",
    "rsna_knee/data/preprocessing.py",
)


def _selection_digest(partition: str, records: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    digest.update(b"local-masked-cache-selection-v1\n")
    digest.update(partition.encode("ascii") + b"\n")
    for ordinal, record in enumerate(records):
        digest.update(str(ordinal).encode("ascii") + b"\0")
        digest.update(record["study_id"].encode("utf-8") + b"\0")
        for series in record["series"]:
            digest.update(series["series_id"].encode("utf-8") + b"\0")
    return digest.hexdigest()


def _make_chunks(partition: str, records: Sequence[Mapping[str, Any]], *,
                 max_series_per_chunk: int, full_digest: str) -> list[dict[str, Any]]:
    chunk_ranges = []
    start = 0
    series_count = 0
    for ordinal, record in enumerate(records):
        count = len(record["series"])
        if count > max_series_per_chunk:
            raise ValueError("study_exceeds_series_chunk_cap")
        if series_count and series_count + count > max_series_per_chunk:
            chunk_ranges.append((start, ordinal, series_count))
            start, series_count = ordinal, 0
        series_count += count
    if series_count:
        chunk_ranges.append((start, len(records), series_count))

    chunks = []
    for chunk_index, (start, stop, series_count) in enumerate(chunk_ranges):
        chunk_records = records[start:stop]
        raw_bound = series_count * SERIES_BYTES_UPPER_BOUND
        saved_output_bound = 2 * raw_bound + OUTPUT_RESERVE_BYTES
        if saved_output_bound >= OUTPUT_LIMIT_BYTES:
            raise ValueError("chunk_saved_output_limit_exceeded")
        chunks.append({
            "chunk_id": f"{partition}-{chunk_index:03d}",
            "partition": partition,
            "chunk_index": chunk_index,
            "start_ordinal": start,
            "stop_ordinal_exclusive": stop,
            "study_count": stop - start,
            "series_count": series_count,
            "full_selection_sha256": full_digest,
            "chunk_selection_sha256": _selection_digest(partition, chunk_records),
            "max_uncompressed_tensor_bytes": raw_bound,
            "max_saved_output_bytes": saved_output_bound,
        })
    return chunks


def validate_chunk_plan(plan: Mapping[str, Any]) -> bool:
    if (type(plan) is not dict or plan.get("schema_version") != "local-masked-cache-chunk-plan-v1"
            or plan.get("fold") != 0 or plan.get("gold_included") is not False
            or type(plan.get("chunks")) is not list):
        raise ValueError("chunk_plan_schema_invalid")
    counts = plan.get("partition_counts")
    partition_digests = plan.get("partition_selection_sha256")
    cap, limit, reserve = (plan.get("max_series_per_chunk"), plan.get("output_limit_bytes"),
                           plan.get("output_reserve_bytes"))
    if (type(counts) is not dict or set(counts) != {"train", "validation"}
            or any(type(counts[name]) is not int or counts[name] < 1 for name in counts)
            or type(partition_digests) is not dict or set(partition_digests) != {"train", "validation"}
            or any(type(partition_digests[name]) is not str
                   or not _HASH_RE.fullmatch(partition_digests[name]) for name in partition_digests)
            or type(cap) is not int or not 1 <= cap <= 4000
            or limit != OUTPUT_LIMIT_BYTES or reserve != OUTPUT_RESERVE_BYTES):
        raise ValueError("chunk_plan_schema_invalid")
    if any(type(chunk) is not dict for chunk in plan["chunks"]):
        raise ValueError("chunk_plan_coverage_invalid")
    observed_partitions = [chunk.get("partition") for chunk in plan["chunks"]]
    if (any(partition not in {"train", "validation"} for partition in observed_partitions)
            or observed_partitions != sorted(observed_partitions,
                                             key=lambda value: (value != "train", value))):
        raise ValueError("chunk_plan_coverage_invalid")
    all_ids = set()
    expected_total_studies = 0
    expected_total_series = 0
    for partition in ("train", "validation"):
        chunks = [chunk for chunk in plan["chunks"] if type(chunk) is dict
                  and chunk.get("partition") == partition]
        if not chunks or len(chunks) != sum(
                1 for chunk in plan["chunks"] if type(chunk) is dict
                and chunk.get("chunk_id", "").startswith(partition + "-")):
            raise ValueError("chunk_plan_coverage_invalid")
        next_ordinal = total_series = 0
        for index, chunk in enumerate(chunks):
            chunk_id = chunk.get("chunk_id")
            start = chunk.get("start_ordinal")
            stop = chunk.get("stop_ordinal_exclusive")
            studies = chunk.get("study_count")
            series = chunk.get("series_count")
            raw_bytes = chunk.get("max_uncompressed_tensor_bytes")
            saved_bytes = chunk.get("max_saved_output_bytes")
            if (type(chunk_id) is not str or chunk_id in all_ids
                    or chunk_id != f"{partition}-{index:03d}"
                    or type(start) is not int or start != next_ordinal
                    or type(stop) is not int or stop <= start
                    or type(studies) is not int or stop - start != studies
                    or type(series) is not int or not 1 <= series <= cap
                    or chunk.get("full_selection_sha256") != partition_digests[partition]
                    or type(chunk.get("chunk_selection_sha256")) is not str
                    or not _HASH_RE.fullmatch(chunk["chunk_selection_sha256"])
                    or raw_bytes != series * SERIES_BYTES_UPPER_BOUND
                    or saved_bytes != 2 * raw_bytes + reserve
                    or saved_bytes >= limit):
                raise ValueError("chunk_plan_coverage_invalid")
            all_ids.add(chunk_id)
            next_ordinal = stop
            total_series += series
        if next_ordinal != counts[partition]:
            raise ValueError("chunk_plan_coverage_invalid")
        expected_total_studies += next_ordinal
        expected_total_series += total_series
    if (expected_total_studies != plan.get("study_count")
            or expected_total_series != plan.get("series_count")
            or len(all_ids) != len(plan["chunks"])):
        raise ValueError("chunk_plan_coverage_invalid")
    full_digest = hashlib.sha256(
        b"local-masked-cache-full-selection-v1\n"
        + b"train\0" + bytes.fromhex(partition_digests["train"])
        + b"validation\0" + bytes.fromhex(partition_digests["validation"])
    ).hexdigest()
    if plan.get("full_selection_sha256") != full_digest:
        raise ValueError("chunk_plan_coverage_invalid")
    return True


def create_chunk_plan(manifest: Mapping[str, Any], folds: Mapping[str, Any], *,
                      manifest_sha256: str, folds_sha256: str, source_sha256: str,
                      preprocessing_source_members: Mapping[str, str],
                      max_series_per_chunk: int = 4000) -> dict[str, Any]:
    """Build a deterministic plan without copying, decoding, or writing study data."""
    for value in (manifest_sha256, folds_sha256, source_sha256):
        if type(value) is not str or not _HASH_RE.fullmatch(value):
            raise ValueError("chunk_plan_pin_invalid")
    if (type(max_series_per_chunk) is not int or max_series_per_chunk < 1
            or max_series_per_chunk > 4000):
        raise ValueError("series_chunk_cap_invalid")
    if (type(manifest) is not dict or manifest.get("schema_version") != "full-cohort-soft-label-manifest-v1"
            or type(manifest.get("records")) is not list):
        raise ValueError("chunk_plan_manifest_invalid")
    if (type(folds) is not dict or folds.get("manifest_sha256") is None
            or type(folds.get("study_assignments")) is not list):
        raise ValueError("chunk_plan_folds_invalid")
    if folds.get("manifest_sha256") != hashlib.sha256(json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8") + b"\n").hexdigest():
        raise ValueError("chunk_plan_manifest_binding_mismatch")

    records = manifest["records"]
    assignments = {}
    for entry in folds["study_assignments"]:
        if (type(entry) is not dict or type(entry.get("study_id")) is not str
                or type(entry.get("fold")) is not int or entry["fold"] not in {0, 1, 2}
                or entry["study_id"] in assignments):
            raise ValueError("chunk_plan_assignment_invalid")
        assignments[entry["study_id"]] = entry["fold"]
    study_ids = set()
    for record in records:
        if (type(record) is not dict or type(record.get("study_id")) is not str
                or record["study_id"] in study_ids or type(record.get("series")) is not list
                or not record["series"]):
            raise ValueError("chunk_plan_record_invalid")
        study_ids.add(record["study_id"])
        series_ids = set()
        for series in record["series"]:
            if (type(series) is not dict or type(series.get("series_id")) is not str
                    or not series["series_id"] or series["series_id"] in series_ids):
                raise ValueError("chunk_plan_series_invalid")
            series_ids.add(series["series_id"])
    if set(assignments) != study_ids:
        raise ValueError("chunk_plan_assignment_membership_mismatch")

    train_records = [record for record in records if assignments[record["study_id"]] != 0]
    validation_records = [record for record in records if assignments[record["study_id"]] == 0]
    train_digest = _selection_digest("train", train_records)
    validation_digest = _selection_digest("validation", validation_records)
    full_digest = hashlib.sha256(
        b"local-masked-cache-full-selection-v1\n"
        + b"train\0" + bytes.fromhex(train_digest)
        + b"validation\0" + bytes.fromhex(validation_digest)
    ).hexdigest()
    chunks = [
        *_make_chunks("train", train_records, max_series_per_chunk=max_series_per_chunk,
                      full_digest=train_digest),
        *_make_chunks("validation", validation_records, max_series_per_chunk=max_series_per_chunk,
                      full_digest=validation_digest),
    ]
    preprocessing = {
        "output_size": 224,
        "max_slices_per_series": 8,
        "image_dtype": "float32",
        "image_range": [0.0, 1.0],
        "pixel_mask_dtype": "bool",
        "target_dtype": "float32",
        "observed_mask_dtype": "bool",
        "series_byte_upper_bound": SERIES_BYTES_UPPER_BOUND,
    }
    canonical_modules = {name: preprocessing_source_members[name] for name in _PREPROCESSING_MODULES}
    if any(type(digest) is not str or not _HASH_RE.fullmatch(digest) for digest in canonical_modules.values()):
        raise ValueError("chunk_plan_preprocessing_pin_invalid")
    preprocessing_digest = hashlib.sha256(json.dumps(
        canonical_modules, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    series_count = sum(len(record["series"]) for record in records)
    plan = {
        "schema_version": "local-masked-cache-chunk-plan-v1",
        "fold": 0,
        "gold_included": False,
        "study_count": len(records),
        "partition_counts": {"train": len(train_records), "validation": len(validation_records)},
        "series_count": series_count,
        "manifest_sha256": manifest_sha256,
        "folds_sha256": folds_sha256,
        "source_sha256": source_sha256,
        "preprocessing_source_members": canonical_modules,
        "preprocessing_source_sha256": preprocessing_digest,
        "preprocessing": preprocessing,
        "max_series_per_chunk": max_series_per_chunk,
        "output_limit_bytes": OUTPUT_LIMIT_BYTES,
        "output_reserve_bytes": OUTPUT_RESERVE_BYTES,
        "full_selection_sha256": full_digest,
        "partition_selection_sha256": {"train": train_digest, "validation": validation_digest},
        "chunks": chunks,
    }
    validate_chunk_plan(plan)
    return plan


def build_pinned_full_chunk_plan(manifest_path: Path, folds_path: Path,
                                 source_archive_path: Path, output_path: Path, *,
                                 max_series_per_chunk: int = 4000) -> dict[str, Any]:
    manifest_path, folds_path = Path(manifest_path), Path(folds_path)
    source_archive_path, output_path = Path(source_archive_path), Path(output_path)
    for path in (manifest_path, folds_path, source_archive_path):
        if path.is_symlink() or not path.is_file():
            raise ValueError("chunk_plan_input_unavailable")
    manifest_bytes, folds_bytes, source_bytes = (
        manifest_path.read_bytes(), folds_path.read_bytes(), source_archive_path.read_bytes())
    manifest_sha, folds_sha, source_sha = map(hashlib.sha256,
                                               (manifest_bytes, folds_bytes, source_bytes))
    manifest_sha, folds_sha, source_sha = manifest_sha.hexdigest(), folds_sha.hexdigest(), source_sha.hexdigest()
    if (manifest_sha != EXPECTED_MANIFEST_SHA256 or folds_sha != EXPECTED_FOLDS_SHA256
            or source_sha != EXPECTED_SOURCE_SHA256):
        raise ValueError("chunk_plan_source_pin_mismatch")
    manifest, folds = json.loads(manifest_bytes), json.loads(folds_bytes)
    with zipfile.ZipFile(source_archive_path) as archive:
        source_members = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()}
    plan = create_chunk_plan(manifest, folds, manifest_sha256=manifest_sha,
                             folds_sha256=folds_sha, source_sha256=source_sha,
                             preprocessing_source_members=source_members,
                             max_series_per_chunk=max_series_per_chunk)
    if (plan["study_count"] != EXPECTED_STUDY_COUNT
            or plan["partition_counts"] != {"train": EXPECTED_TRAIN_COUNT,
                                            "validation": EXPECTED_VALIDATION_COUNT}
            or plan["series_count"] != EXPECTED_SERIES_COUNT
            or sum(chunk["study_count"] for chunk in plan["chunks"]) != EXPECTED_STUDY_COUNT
            or sum(chunk["series_count"] for chunk in plan["chunks"]) != EXPECTED_SERIES_COUNT):
        raise ValueError("chunk_plan_cohort_count_mismatch")
    if any(chunk["max_saved_output_bytes"] >= OUTPUT_LIMIT_BYTES for chunk in plan["chunks"]):
        raise ValueError("chunk_plan_output_limit_exceeded")
    raw = (json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    if output_path.exists() or output_path.is_symlink():
        raise ValueError("chunk_plan_output_not_fresh")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError("chunk_plan_temporary_not_fresh")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return plan
