"""Independent stream verifier and dataset adapter for downloaded local16 cache ZIPs."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any
import zipfile

from scripts.local16_cache_chunk_plan import validate_chunk_plan
from scripts.local16_tensor_cache import IDENTITY, SCHEMA

TARGETS = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fail(code: str):
    raise ValueError(code)


def verify_full_cache(*, manifest_path: Path, folds_path: Path, plan_path: Path,
                      source_archive_path: Path, archive_dir: Path) -> dict[str, Any]:
    """Verify exact fold-zero cohort coverage, archive hashes, per-study tensors, and 16 identity."""
    import numpy as np
    for path in (manifest_path, folds_path, plan_path, source_archive_path):
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            _fail("local16_input_unavailable")
    manifest_raw, folds_raw, plan_raw = (Path(path).read_bytes() for path in (manifest_path, folds_path, plan_path))
    manifest, folds, plan = json.loads(manifest_raw), json.loads(folds_raw), json.loads(plan_raw)
    if not validate_chunk_plan(plan):
        _fail("local16_chunk_plan_invalid")
    actual_hashes = {"manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
                     "folds_sha256": hashlib.sha256(folds_raw).hexdigest(),
                     "source_sha256": sha256(source_archive_path)}
    if any(actual_hashes[key] != plan[key] for key in actual_hashes):
        _fail("local16_source_pin_mismatch")
    if (manifest.get("schema_version") != "full-cohort-soft-label-manifest-v1"
            or folds.get("manifest_sha256") != hashlib.sha256(
                json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n").hexdigest()):
        _fail("local16_manifest_binding_mismatch")
    assignments = {row["study_id"]: row["fold"] for row in folds.get("study_assignments", [])}
    records = manifest.get("records", [])
    if len(assignments) != len(records) or set(assignments) != {row["study_id"] for row in records}:
        _fail("local16_fold_membership_mismatch")
    partitions = {name: [row for row in records if (assignments[row["study_id"]] == 0) == (name == "validation")]
                  for name in ("train", "validation")}
    if {key: len(value) for key, value in partitions.items()} != plan["partition_counts"]:
        _fail("local16_partition_count_mismatch")
    archive_dir = Path(archive_dir)
    if archive_dir.is_symlink() or not archive_dir.is_dir():
        _fail("local16_archive_directory_unavailable")
    expected_archive_names = {f"local16-cache-{chunk['chunk_id']}.zip" for chunk in plan["chunks"]}
    actual_archive_names = {path.name for path in archive_dir.iterdir() if path.is_file() and not path.is_symlink()}
    if actual_archive_names != expected_archive_names or any(path.is_symlink() for path in archive_dir.iterdir()):
        _fail("local16_archive_inventory_mismatch")
    seen, counts, slice_counts = set(), {"train": 0, "validation": 0}, []
    slice_histogram = {str(value): 0 for value in range(1, 17)}
    series_over_eight = 0
    archive_rows = []
    for chunk in plan["chunks"]:
        path = archive_dir / f"local16-cache-{chunk['chunk_id']}.zip"
        try:
            with zipfile.ZipFile(path) as archive:
                infos = archive.infolist()
                names = [item.filename for item in infos]
                if len(names) != len(set(names)) or any(item.is_dir() for item in infos):
                    _fail("local16_archive_members_invalid")
                if archive.testzip() is not None:
                    _fail("local16_archive_crc_invalid")
                cache_manifest = json.loads(archive.read("cache-manifest.json"))
                members = {name for name in names}
                expected_records = partitions[chunk["partition"]][chunk["start_ordinal"]:chunk["stop_ordinal_exclusive"]]
                expected_names = {"cache-manifest.json", *(f"{chunk['partition']}/sample-{i:06d}.npz" for i in range(len(expected_records)))}
                if members != expected_names:
                    _fail("local16_archive_membership_invalid")
                provenance = {"schema_version": "local16-cache-provenance-v1", "cache_identity": IDENTITY,
                              "max_slices_per_series": 16, "source_sha256": plan["source_sha256"],
                              "manifest_sha256": plan["manifest_sha256"], "folds_sha256": plan["folds_sha256"],
                              "preprocessing_source_sha256": plan["preprocessing_source_sha256"], "gold_included": False}
                if (cache_manifest.get("schema_version") != SCHEMA or cache_manifest.get("cache_identity") != IDENTITY
                        or cache_manifest.get("partition") != chunk["partition"] or cache_manifest.get("fold") != 0
                        or cache_manifest.get("manifest_sha256") != plan["manifest_sha256"]
                        or cache_manifest.get("first_ordinal") != chunk["start_ordinal"]
                        or cache_manifest.get("provenance") != provenance
                        or cache_manifest.get("sample_count") != len(expected_records)
                        or len(cache_manifest.get("records", [])) != len(expected_records)):
                    _fail("local16_cache_manifest_contract_invalid")
                for index, (cache_row, cohort_row) in enumerate(zip(cache_manifest["records"], expected_records)):
                    study_digest = hashlib.sha256(cohort_row["study_id"].encode()).hexdigest()
                    series_ids = [series["series_id"] for series in cohort_row["series"]]
                    series_digest = hashlib.sha256(b"\0".join(value.encode() for value in series_ids)).hexdigest()
                    if (cache_row.get("ordinal") != chunk["start_ordinal"] + index
                            or cache_row.get("study_sha256") != study_digest or study_digest in seen
                            or cache_row.get("series_count") != len(series_ids)
                            or cache_row.get("series_order_sha256") != series_digest
                            or type(cache_row.get("eligible_slice_counts")) is not list
                            or len(cache_row["eligible_slice_counts"]) != len(series_ids)
                            or any(type(value) is not int or value < 1 for value in cache_row["eligible_slice_counts"])
                            or cache_row.get("max_slices_per_series") != 16
                            or cache_row.get("file") != f"{chunk['partition']}/sample-{index:06d}.npz"):
                        _fail("local16_cache_membership_invalid")
                    raw = archive.read(cache_row["file"])
                    if len(raw) != cache_row.get("size_bytes") or hashlib.sha256(raw).hexdigest() != cache_row.get("sha256"):
                        _fail("local16_cache_sample_hash_mismatch")
                    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
                        count = cache_row["series_count"]
                        expected_keys = {"targets", "observed", *(f"images_{i:03d}" for i in range(count)),
                                         *(f"masks_{i:03d}" for i in range(count))}
                        if set(data.files) != expected_keys:
                            _fail("local16_cache_payload_invalid")
                        actual_shapes, slices = [], 0
                        for i in range(count):
                            image, mask = data[f"images_{i:03d}"], data[f"masks_{i:03d}"]
                            if (image.dtype != np.float32 or mask.dtype != np.bool_ or image.ndim != 4
                                    or image.shape != mask.shape or not 1 <= image.shape[0] <= 16
                                    or image.shape[0] != min(cache_row["eligible_slice_counts"][i], 16)
                                    or image.shape[1:] != (1, 224, 224) or not np.isfinite(image).all()
                                    or np.any(image < 0) or np.any(image > 1)
                                    or np.any(~mask.reshape(mask.shape[0], -1).any(axis=1))
                                    or np.any(image[~mask] != 0)):
                                _fail("local16_cache_tensor_invalid")
                            actual_shapes.append(list(image.shape))
                            slices += image.shape[0]
                            slice_histogram[str(image.shape[0])] += 1
                            series_over_eight += image.shape[0] > 8
                        if cache_row.get("series_shapes") != actual_shapes:
                            _fail("local16_cache_shape_receipt_mismatch")
                        targets = data["targets"]
                        observed = data["observed"]
                        expected_targets = np.asarray([cohort_row["soft_targets"][name] for name in TARGETS], dtype=np.float32)
                        expected_observed = np.asarray([cohort_row["observed_mask"][name] for name in TARGETS], dtype=np.bool_)
                        if (targets.dtype != np.float32 or observed.dtype != np.bool_ or targets.shape != (12,)
                                or observed.shape != (12,) or not np.array_equal(targets, expected_targets)
                                or not np.array_equal(observed, expected_observed)):
                            _fail("local16_cache_label_mismatch")
                    seen.add(study_digest)
                    counts[chunk["partition"]] += 1
                    slice_counts.append(slices)
                archive_rows.append({"chunk_id": chunk["chunk_id"], "archive_bytes": path.stat().st_size,
                                     "archive_sha256": sha256(path), "studies": len(expected_records)})
        except (OSError, zipfile.BadZipFile, KeyError, UnicodeError, json.JSONDecodeError):
            _fail("local16_archive_invalid")
    if counts != plan["partition_counts"] or len(seen) != plan["study_count"] or series_over_eight < 1:
        _fail("local16_full_cohort_coverage_invalid")
    return {"status": "verified", "cache_identity": IDENTITY, "max_slices_per_series": 16,
            "studies": len(seen), "series": plan["series_count"], "slices": sum(slice_counts),
            "observed_slice_count_histogram": slice_histogram,
            "series_with_more_than_eight_slices": series_over_eight,
            "partition_study_counts": counts, "no_overlap": True, "no_missing_or_extra_studies": True,
            "gold_included": False, "training_started": False, "all_archive_hashes_rechecked": True,
            "archives": archive_rows}


class Local16CacheDataset:
    """dataset_factory adapter for a previously verified ZIP set."""

    target_names = TARGETS

    def __init__(self, manifest_path: Path, plan_path: Path, archive_dir: Path, partition: str):
        if partition not in ("train", "validation"):
            _fail("local16_partition_invalid")
        self.manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
        self.archive_dir, self.partition = Path(archive_dir), partition
        self.rows = []
        for row in self.plan["chunks"]:
            if row["partition"] == partition:
                archive_path = self.archive_dir / f"local16-cache-{row['chunk_id']}.zip"
                with zipfile.ZipFile(archive_path) as archive:
                    cache_manifest = json.loads(archive.read("cache-manifest.json"))
                self.rows.extend((archive_path, record) for record in cache_manifest["records"])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import numpy as np
        import torch
        if type(index) is not int or not 0 <= index < len(self.rows):
            raise IndexError("local16_sample_index_invalid")
        archive_path, record = self.rows[index]
        with zipfile.ZipFile(archive_path) as archive:
            raw = archive.read(record["file"])
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            _fail("local16_cache_sample_hash_mismatch")
        with np.load(io.BytesIO(raw), allow_pickle=False) as data:
            count = record["series_count"]
            images = [torch.from_numpy(data[f"images_{i:03d}"].copy()) for i in range(count)]
            masks = [torch.from_numpy(data[f"masks_{i:03d}"].copy()) for i in range(count)]
            return images, masks, torch.from_numpy(data["targets"].copy()), torch.from_numpy(data["observed"].copy()), index
