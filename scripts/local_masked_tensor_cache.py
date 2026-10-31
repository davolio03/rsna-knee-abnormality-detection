"""Private, lossless cache for the existing masked-soft dataset contract.

The cache deliberately lives outside the packaged training source tree. It is
an opt-in adapter for transferring preprocessed selected slices, not a new
preprocessing implementation.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import zipfile


_SCHEMA = "local-masked-tensor-cache-v1"
_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _as_numpy(value: Any, *, dtype: str, code: str):
    import numpy as np

    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
    except Exception:
        raise ValueError(code) from None
    if array.dtype != np.dtype(dtype):
        raise ValueError(code)
    return np.ascontiguousarray(array)


class LocalMaskedTensorCache:
    """Write or read one fold partition as atomic per-study NPZ files."""

    def __init__(self, root: Path, *, partition: str, fold: int,
                 manifest_sha256: str, create: bool = True,
                 first_ordinal: int = 0, provenance: Mapping[str, Any] | None = None):
        if partition not in {"train", "validation"} or type(fold) is not int or fold not in {0, 1, 2}:
            raise ValueError("cache_identity_invalid")
        if type(manifest_sha256) is not str or not _HASH_RE.fullmatch(manifest_sha256):
            raise ValueError("cache_manifest_pin_invalid")
        if type(first_ordinal) is not int or first_ordinal < 0:
            raise ValueError("cache_first_ordinal_invalid")
        if provenance is not None:
            try:
                provenance = json.loads(json.dumps(provenance, sort_keys=True, allow_nan=False))
            except (TypeError, ValueError):
                raise ValueError("cache_provenance_invalid") from None
            if type(provenance) is not dict:
                raise ValueError("cache_provenance_invalid")
        self.root = Path(root)
        if self.root.is_symlink():
            raise ValueError("cache_root_invalid")
        self.partition = partition
        self.fold = fold
        self.manifest_sha256 = manifest_sha256
        self.first_ordinal = first_ordinal
        self.provenance = provenance
        self.partition_root = self.root / partition
        self._records: list[dict[str, Any]] = []
        if create:
            if self.root.exists():
                raise ValueError("cache_root_not_fresh")
            self.partition_root.mkdir(parents=True, exist_ok=False)

    def __len__(self) -> int:
        return len(self._records)

    def write_sample(self, *, study_id: str, series_ids: Sequence[str], sample: Any) -> None:
        import numpy as np

        if not isinstance(study_id, str) or not study_id or not isinstance(series_ids, (tuple, list)):
            raise ValueError("cache_sample_identity_invalid")
        if not series_ids or any(type(item) is not str or not item for item in series_ids):
            raise ValueError("cache_series_order_invalid")
        if len(set(series_ids)) != len(series_ids):
            raise ValueError("cache_series_order_invalid")
        if type(sample) not in (tuple, list) or len(sample) != 5:
            raise ValueError("cache_sample_invalid")
        images, masks, targets, observed, row_index = sample
        if (type(images) not in (tuple, list) or type(masks) not in (tuple, list)
                or len(images) != len(series_ids) or len(masks) != len(series_ids)):
            raise ValueError("cache_series_count_mismatch")
        expected_ordinal = self.first_ordinal + len(self._records)
        if type(row_index) is not int or row_index != expected_ordinal:
            raise ValueError("cache_row_order_invalid")

        payload: dict[str, Any] = {}
        shapes = []
        for index, (image, mask) in enumerate(zip(images, masks)):
            image_array = _as_numpy(image, dtype="float32", code="cache_image_invalid")
            mask_array = _as_numpy(mask, dtype="bool", code="cache_mask_invalid")
            if (image_array.ndim != 4 or image_array.shape != mask_array.shape
                    or image_array.shape[0] < 1 or image_array.shape[1] != 1
                    or image_array.shape[2] != image_array.shape[3]
                    or not np.isfinite(image_array).all()
                    or np.any(image_array < 0.0) or np.any(image_array > 1.0)
                    or np.any(~mask_array.reshape(mask_array.shape[0], -1).any(axis=1))
                    or np.any(image_array[~mask_array] != 0.0)):
                raise ValueError("cache_image_mask_contract_invalid")
            payload[f"images_{index:03d}"] = image_array
            payload[f"masks_{index:03d}"] = mask_array
            shapes.append(list(image_array.shape))

        target_array = _as_numpy(targets, dtype="float32", code="cache_targets_invalid")
        observed_array = _as_numpy(observed, dtype="bool", code="cache_observed_invalid")
        if (target_array.shape != (12,) or observed_array.shape != (12,)
                or not np.isfinite(target_array).all()
                or np.any(target_array < 0.0) or np.any(target_array > 1.0)):
            raise ValueError("cache_labels_invalid")
        payload["targets"] = target_array
        payload["observed"] = observed_array

        filename = f"sample-{len(self._records):06d}.npz"
        destination = self.partition_root / filename
        temporary = self.partition_root / (filename + ".tmp")
        if destination.exists() or temporary.exists():
            raise ValueError("cache_sample_file_not_fresh")
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        raw = destination.read_bytes()
        self._records.append({
            "ordinal": expected_ordinal,
            "study_sha256": _sha256(study_id.encode("utf-8")),
            "series_count": len(series_ids),
            "series_shapes": shapes,
            "file": f"{self.partition}/{filename}",
            "size_bytes": len(raw),
            "sha256": _sha256(raw),
        })

    def finalize(self) -> Path:
        if not self._records:
            raise ValueError("cache_partition_empty")
        manifest = {
            "schema_version": _SCHEMA,
            "fold": self.fold,
            "partition": self.partition,
            "manifest_sha256": self.manifest_sha256,
            "first_ordinal": self.first_ordinal,
            "provenance": self.provenance,
            "sample_count": len(self._records),
            "records": self._records,
        }
        raw = (json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        destination = self.root / "cache-manifest.json"
        temporary = self.root / "cache-manifest.json.tmp"
        if destination.exists() or temporary.exists():
            raise ValueError("cache_manifest_not_fresh")
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        return destination

    @classmethod
    def open(cls, root: Path, *, partition: str, fold: int,
             manifest_sha256: str, expected_study_ids: Sequence[str],
             first_ordinal: int = 0,
             expected_provenance: Mapping[str, Any] | None = None) -> "LocalMaskedTensorCache":
        import numpy as np

        try:
            root = Path(root)
            if root.is_symlink() or not root.is_dir():
                raise ValueError
            manifest_path = root / "cache-manifest.json"
            if manifest_path.is_symlink() or not manifest_path.is_file():
                raise ValueError
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            raise ValueError("cache_manifest_unavailable") from None
        if (type(manifest) is not dict or manifest.get("schema_version") != _SCHEMA
                or manifest.get("partition") != partition or manifest.get("fold") != fold
                or manifest.get("manifest_sha256") != manifest_sha256
                or manifest.get("first_ordinal") != first_ordinal
                or type(manifest.get("records")) is not list
                or manifest.get("sample_count") != len(manifest["records"])
                or len(manifest["records"]) != len(expected_study_ids)):
            raise ValueError("cache_manifest_contract_mismatch")
        if (expected_provenance is not None
                and manifest.get("provenance") != dict(expected_provenance)):
            raise ValueError("cache_provenance_mismatch")

        instance = cls.__new__(cls)
        instance.root, instance.partition = root, partition
        instance.fold, instance.manifest_sha256 = fold, manifest_sha256
        instance.first_ordinal = first_ordinal
        instance.provenance = manifest.get("provenance")
        instance.partition_root = root / partition
        instance._records = manifest["records"]
        if instance.partition_root.is_symlink() or not instance.partition_root.is_dir():
            raise ValueError("cache_partition_unavailable")
        expected_files = {"cache-manifest.json"}
        for record in instance._records:
            if type(record) is dict and type(record.get("file")) is str:
                expected_files.add(record["file"])
        actual_files: set[str] = set()
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError("cache_unlisted_file")
            relative = path.relative_to(root).as_posix()
            if path.is_dir():
                if relative != partition:
                    raise ValueError("cache_unlisted_file")
            elif path.is_file():
                actual_files.add(relative)
            else:
                raise ValueError("cache_unlisted_file")
        if actual_files != expected_files:
            raise ValueError("cache_unlisted_file")
        for index, (record, study_id) in enumerate(zip(instance._records, expected_study_ids)):
            expected_ordinal = first_ordinal + index
            if (type(record) is not dict or record.get("ordinal") != expected_ordinal
                    or record.get("study_sha256") != _sha256(str(study_id).encode("utf-8"))
                    or type(record.get("series_count")) is not int or record["series_count"] < 1
                    or record.get("file") != f"{partition}/sample-{index:06d}.npz"
                    or type(record.get("sha256")) is not str or not _HASH_RE.fullmatch(record["sha256"])):
                raise ValueError("cache_membership_or_order_mismatch")
            path = root / record["file"]
            if path.is_symlink() or not path.is_file():
                raise ValueError("cache_file_unavailable")
            raw = path.read_bytes()
            if len(raw) != record.get("size_bytes") or _sha256(raw) != record["sha256"]:
                raise ValueError("cache_file_hash_mismatch")
            try:
                with np.load(path, allow_pickle=False) as archive:
                    expected_keys = {"targets", "observed"}
                    for series_index in range(record["series_count"]):
                        expected_keys.update((f"images_{series_index:03d}", f"masks_{series_index:03d}"))
                    if set(archive.files) != expected_keys:
                        raise ValueError
                    if (archive["targets"].dtype != np.float32 or archive["observed"].dtype != np.bool_
                            or archive["targets"].shape != (12,) or archive["observed"].shape != (12,)):
                        raise ValueError
            except (OSError, ValueError, KeyError):
                raise ValueError("cache_file_payload_invalid") from None
        return instance

    def __getitem__(self, index: int):
        import numpy as np
        import torch

        if type(index) is not int or not 0 <= index < len(self._records):
            raise IndexError("Cache sample index is out of range.")
        record = self._records[index]
        path = self.root / record["file"]
        try:
            with np.load(path, allow_pickle=False) as archive:
                images = [torch.from_numpy(archive[f"images_{i:03d}"].copy())
                          for i in range(record["series_count"])]
                masks = [torch.from_numpy(archive[f"masks_{i:03d}"].copy())
                         for i in range(record["series_count"])]
                targets = torch.from_numpy(archive["targets"].copy())
                observed = torch.from_numpy(archive["observed"].copy())
        except (OSError, ValueError, KeyError):
            raise ValueError("cache_file_payload_invalid") from None
        return images, masks, targets, observed, self.first_ordinal + index


def write_cache_archive(root: Path, destination: Path, *, zip_factory=zipfile.ZipFile,
                        max_output_bytes: int | None = None,
                        reserve_bytes: int = 0) -> dict[str, Any]:
    """Create a deterministic-path ZIP and publish it only after close succeeds."""
    root = Path(root)
    destination = Path(destination)
    if (root.is_symlink() or not root.is_dir() or destination.is_symlink()
            or destination.exists()):
        raise ValueError("cache_archive_path_invalid")
    if (max_output_bytes is not None and (type(max_output_bytes) is not int or max_output_bytes < 1)):
        raise ValueError("cache_archive_limit_invalid")
    if type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError("cache_archive_reserve_invalid")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError("cache_archive_temporary_not_fresh")
    try:
        with zip_factory(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            for path in sorted(root.rglob("*")):
                if path.is_symlink():
                    raise ValueError("cache_archive_symlink_invalid")
                if path.is_file():
                    archive.write(path, path.relative_to(root).as_posix())
        with temporary.open("r+b") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        if max_output_bytes is not None:
            cache_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
            if cache_bytes + temporary.stat().st_size + reserve_bytes > max_output_bytes:
                raise ValueError("cache_archive_output_limit_exceeded")
        os.replace(temporary, destination)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    raw = destination.read_bytes()
    return {"path": str(destination), "size_bytes": len(raw), "sha256": _sha256(raw)}
