"""Isolated, identity-bound cache format for genuine sixteen-slice E050 tensors."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


SCHEMA = "local-masked-tensor-cache16-v1"
IDENTITY = "local16-attention-run01"
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_PROVENANCE_KEYS = {"schema_version", "cache_identity", "max_slices_per_series", "source_sha256",
                    "manifest_sha256", "folds_sha256", "preprocessing_source_sha256", "gold_included"}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        result = json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError):
        raise ValueError("cache_provenance_invalid") from None
    if (type(result) is not dict or set(result) != _PROVENANCE_KEYS
            or result.get("schema_version") != "local16-cache-provenance-v1"
            or result.get("cache_identity") != IDENTITY
            or result.get("max_slices_per_series") != 16 or result.get("gold_included") is not False
            or any(type(result.get(key)) is not str or not _HASH.fullmatch(result[key])
                   for key in ("source_sha256", "manifest_sha256", "folds_sha256",
                               "preprocessing_source_sha256"))):
        raise ValueError("cache_provenance_invalid")
    return result


def _array(value: Any, dtype: str, error: str):
    import numpy as np
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
    except Exception:
        raise ValueError(error) from None
    if array.dtype != np.dtype(dtype):
        raise ValueError(error)
    return np.ascontiguousarray(array)


class Local16TensorCache:
    """Write or validate one isolated per-study 16-slice chunk."""

    def __init__(self, root: Path, *, partition: str, fold: int, manifest_sha256: str,
                 first_ordinal: int = 0, provenance: Mapping[str, Any], create: bool = True):
        if partition not in {"train", "validation"} or type(fold) is not int or fold != 0:
            raise ValueError("cache_identity_invalid")
        if type(manifest_sha256) is not str or not _HASH.fullmatch(manifest_sha256):
            raise ValueError("cache_manifest_pin_invalid")
        if type(first_ordinal) is not int or first_ordinal < 0:
            raise ValueError("cache_first_ordinal_invalid")
        self.provenance = _provenance(provenance)
        if self.provenance["manifest_sha256"] != manifest_sha256:
            raise ValueError("cache_provenance_mismatch")
        self.root = Path(root)
        if self.root.is_symlink():
            raise ValueError("cache_root_invalid")
        self.partition, self.fold = partition, fold
        self.manifest_sha256, self.first_ordinal = manifest_sha256, first_ordinal
        self.partition_root = self.root / partition
        self._records: list[dict[str, Any]] = []
        if create:
            if self.root.exists():
                raise ValueError("cache_root_not_fresh")
            self.partition_root.mkdir(parents=True, exist_ok=False)

    def __len__(self) -> int:
        return len(self._records)

    def write_sample(self, *, study_id: str, series_ids: Sequence[str], sample: Any,
                     eligible_slice_counts: Sequence[int]) -> None:
        import numpy as np
        if type(study_id) is not str or not study_id or type(series_ids) not in (list, tuple):
            raise ValueError("cache_sample_identity_invalid")
        if not series_ids or any(type(item) is not str or not item for item in series_ids) or len(set(series_ids)) != len(series_ids):
            raise ValueError("cache_series_order_invalid")
        if (type(eligible_slice_counts) not in (list, tuple) or len(eligible_slice_counts) != len(series_ids)
                or any(type(value) is not int or value < 1 for value in eligible_slice_counts)):
            raise ValueError("cache_eligible_slice_counts_invalid")
        if type(sample) not in (tuple, list) or len(sample) != 5:
            raise ValueError("cache_sample_invalid")
        images, masks, targets, observed, ordinal = sample
        if (type(images) not in (list, tuple) or type(masks) not in (list, tuple)
                or len(images) != len(series_ids) or len(masks) != len(series_ids)):
            raise ValueError("cache_series_count_mismatch")
        if type(ordinal) is not int or ordinal != self.first_ordinal + len(self._records):
            raise ValueError("cache_row_order_invalid")
        payload: dict[str, Any] = {}
        shapes = []
        for index, (image, mask, eligible_count) in enumerate(zip(images, masks, eligible_slice_counts)):
            image = _array(image, "float32", "cache_image_invalid")
            mask = _array(mask, "bool", "cache_mask_invalid")
            if image.ndim == 4 and not 1 <= image.shape[0] <= 16:
                raise ValueError("cache_slice_count_invalid")
            if image.ndim == 4 and image.shape[0] != min(eligible_count, 16):
                raise ValueError("cache_eligible_slice_count_mismatch")
            if (image.ndim != 4 or image.shape != mask.shape
                    or image.shape[1:] != (1, 224, 224) or not np.isfinite(image).all()
                    or np.any(image < 0) or np.any(image > 1)
                    or np.any(~mask.reshape(mask.shape[0], -1).any(axis=1)) or np.any(image[~mask] != 0)):
                raise ValueError("cache_image_mask_contract_invalid")
            payload[f"images_{index:03d}"] = image
            payload[f"masks_{index:03d}"] = mask
            shapes.append(list(image.shape))
        targets = _array(targets, "float32", "cache_targets_invalid")
        observed = _array(observed, "bool", "cache_observed_invalid")
        if (targets.shape != (12,) or observed.shape != (12,) or not np.isfinite(targets).all()
                or np.any(targets < 0) or np.any(targets > 1)):
            raise ValueError("cache_labels_invalid")
        payload["targets"], payload["observed"] = targets, observed
        filename = f"sample-{len(self._records):06d}.npz"
        destination = self.partition_root / filename
        temporary = destination.with_name(destination.name + ".tmp")
        if destination.exists() or temporary.exists():
            raise ValueError("cache_sample_file_not_fresh")
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        raw = destination.read_bytes()
        self._records.append({"ordinal": ordinal, "study_sha256": _sha256_bytes(study_id.encode()),
                              "series_count": len(series_ids), "series_order_sha256": _sha256_bytes(
                                  b"\0".join(item.encode() for item in series_ids)),
                              "max_slices_per_series": 16, "eligible_slice_counts": list(eligible_slice_counts),
                              "series_shapes": shapes,
                              "file": f"{self.partition}/{filename}", "size_bytes": len(raw),
                              "sha256": _sha256_bytes(raw)})

    def finalize(self) -> Path:
        if not self._records:
            raise ValueError("cache_partition_empty")
        value = {"schema_version": SCHEMA, "cache_identity": IDENTITY, "fold": self.fold,
                 "partition": self.partition, "manifest_sha256": self.manifest_sha256,
                 "first_ordinal": self.first_ordinal, "provenance": self.provenance,
                 "sample_count": len(self._records), "records": self._records}
        path = self.root / "cache-manifest.json"
        temp = path.with_name(path.name + ".tmp")
        if path.exists() or temp.exists():
            raise ValueError("cache_manifest_not_fresh")
        with temp.open("xb") as stream:
            stream.write((json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        return path

    @classmethod
    def open(cls, root: Path, *, partition: str, fold: int, manifest_sha256: str,
             expected_study_ids: Sequence[str], expected_series_counts: Sequence[int],
             expected_series_ids: Sequence[Sequence[str]],
             expected_eligible_slice_counts: Sequence[Sequence[int]],
             first_ordinal: int = 0, expected_provenance: Mapping[str, Any]):
        import numpy as np
        root = Path(root)
        try:
            if root.is_symlink() or not root.is_dir():
                raise ValueError
            path = root / "cache-manifest.json"
            if path.is_symlink() or not path.is_file():
                raise ValueError
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            raise ValueError("cache_manifest_unavailable") from None
        provenance = _provenance(expected_provenance)
        if (type(manifest) is not dict or manifest.get("schema_version") != SCHEMA
                or manifest.get("cache_identity") != IDENTITY or manifest.get("partition") != partition
                or manifest.get("fold") != fold or manifest.get("manifest_sha256") != manifest_sha256
                or manifest.get("provenance") != provenance or manifest.get("first_ordinal") != first_ordinal
                or type(manifest.get("records")) is not list
                or len(manifest["records"]) != len(expected_study_ids)
                or len(expected_series_counts) != len(expected_study_ids)
                or len(expected_series_ids) != len(expected_study_ids)
                or len(expected_eligible_slice_counts) != len(expected_study_ids)):
            raise ValueError("cache_manifest_contract_mismatch")
        part = root / partition
        if part.is_symlink() or not part.is_dir():
            raise ValueError("cache_partition_unavailable")
        expected_files = {"cache-manifest.json", *(f"{partition}/sample-{i:06d}.npz" for i in range(len(expected_study_ids)))}
        actual_files = set()
        for item in root.rglob("*"):
            if item.is_symlink():
                raise ValueError("cache_unlisted_file")
            if item.is_file():
                actual_files.add(item.relative_to(root).as_posix())
        if actual_files != expected_files:
            raise ValueError("cache_unlisted_file")
        instance = cls.__new__(cls)
        instance.root, instance.partition_root = root, part
        instance.partition, instance.fold = partition, fold
        instance.manifest_sha256, instance.first_ordinal = manifest_sha256, first_ordinal
        instance.provenance, instance._records = provenance, manifest["records"]
        for index, (record, study, series_count, series_ids, eligible_counts) in enumerate(
                zip(instance._records, expected_study_ids, expected_series_counts, expected_series_ids,
                    expected_eligible_slice_counts)):
            if (type(series_ids) not in (tuple, list) or len(series_ids) != series_count
                    or any(type(value) is not str or not value for value in series_ids)
                    or len(set(series_ids)) != len(series_ids)
                    or type(eligible_counts) not in (tuple, list) or len(eligible_counts) != series_count
                    or any(type(value) is not int or value < 1 for value in eligible_counts)):
                raise ValueError("cache_membership_or_order_mismatch")
            if (type(record) is not dict or record.get("ordinal") != first_ordinal + index
                    or record.get("study_sha256") != _sha256_bytes(study.encode())
                    or record.get("series_count") != series_count or record.get("max_slices_per_series") != 16
                    or record.get("eligible_slice_counts") != list(eligible_counts)
                    or record.get("series_order_sha256") != _sha256_bytes(b"\0".join(value.encode() for value in series_ids))
                    or record.get("file") != f"{partition}/sample-{index:06d}.npz"):
                raise ValueError("cache_membership_or_order_mismatch")
            data_path = root / record["file"]
            raw = data_path.read_bytes()
            if len(raw) != record.get("size_bytes") or _sha256_bytes(raw) != record.get("sha256"):
                raise ValueError("cache_file_hash_mismatch")
            try:
                with np.load(data_path, allow_pickle=False) as data:
                    keys = {"targets", "observed", *(f"images_{i:03d}" for i in range(series_count)),
                            *(f"masks_{i:03d}" for i in range(series_count))}
                    if set(data.files) != keys or data["targets"].shape != (12,) or data["observed"].shape != (12,):
                        raise ValueError
                    shapes = []
                    for series_index in range(series_count):
                        image, mask = data[f"images_{series_index:03d}"], data[f"masks_{series_index:03d}"]
                        if (image.dtype != np.float32 or mask.dtype != np.bool_ or image.ndim != 4
                                or image.shape != mask.shape or not 1 <= image.shape[0] <= 16
                                or image.shape[0] != min(record["eligible_slice_counts"][series_index], 16)
                                or image.shape[1:] != (1, 224, 224) or not np.isfinite(image).all()
                                or np.any(image < 0) or np.any(image > 1)
                                or np.any(~mask.reshape(mask.shape[0], -1).any(axis=1)) or np.any(image[~mask] != 0)):
                            raise ValueError
                        shapes.append(list(image.shape))
                    if record.get("series_shapes") != shapes:
                        raise ValueError
            except (OSError, ValueError, KeyError):
                raise ValueError("cache_file_payload_invalid") from None
        return instance

    def __getitem__(self, index: int):
        import numpy as np
        import torch
        if type(index) is not int or not 0 <= index < len(self._records):
            raise IndexError("cache_sample_index_invalid")
        record = self._records[index]
        with np.load(self.root / record["file"], allow_pickle=False) as data:
            images = [torch.from_numpy(data[f"images_{i:03d}"].copy()) for i in range(record["series_count"])]
            masks = [torch.from_numpy(data[f"masks_{i:03d}"].copy()) for i in range(record["series_count"])]
            return images, masks, torch.from_numpy(data["targets"].copy()), torch.from_numpy(data["observed"].copy()), self.first_ordinal + index
