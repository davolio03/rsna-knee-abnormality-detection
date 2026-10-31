import json
import tempfile
import unittest
from pathlib import Path
import zipfile

import numpy as np
import torch

from scripts.local_masked_tensor_cache import LocalMaskedTensorCache, write_cache_archive


class LocalMaskedTensorCacheTests(unittest.TestCase):
    def sample(self):
        return (
            [np.arange(12, dtype=np.float32).reshape(3, 1, 2, 2) / 12],
            [np.ones((3, 1, 2, 2), dtype=np.bool_)],
            np.linspace(0, 1, 12, dtype=np.float32),
            np.asarray([True, False] * 6, dtype=np.bool_),
            0,
        )

    def test_round_trip_preserves_float_pixels_boolean_masks_and_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            cache = LocalMaskedTensorCache(root, partition="train", fold=0,
                                           manifest_sha256="a" * 64)
            cache.write_sample(study_id="study-private-1", series_ids=["series-a"], sample=self.sample())
            cache.finalize()

            reader = LocalMaskedTensorCache.open(root, partition="train", fold=0,
                                                  manifest_sha256="a" * 64,
                                                  expected_study_ids=["study-private-1"])
            images, masks, targets, observed, row = reader[0]
            np.testing.assert_array_equal(images[0], self.sample()[0][0])
            np.testing.assert_array_equal(masks[0], self.sample()[1][0])
            np.testing.assert_array_equal(targets, self.sample()[2])
            np.testing.assert_array_equal(observed, self.sample()[3])
            self.assertEqual((images[0].dtype, masks[0].dtype, targets.dtype, observed.dtype),
                             (torch.float32, torch.bool, torch.float32, torch.bool))
            self.assertEqual(row, 0)

    def test_tampered_shard_fails_integrity_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            cache = LocalMaskedTensorCache(root, partition="validation", fold=0,
                                           manifest_sha256="b" * 64)
            cache.write_sample(study_id="study-private-2", series_ids=["series-b"], sample=self.sample())
            cache.finalize()
            shard = root / "validation" / "sample-000000.npz"
            shard.write_bytes(shard.read_bytes() + b"tampered")
            with self.assertRaisesRegex(ValueError, "cache_file_hash_mismatch"):
                LocalMaskedTensorCache.open(root, partition="validation", fold=0,
                                            manifest_sha256="b" * 64,
                                            expected_study_ids=["study-private-2"])

    def test_unlisted_shard_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            cache = LocalMaskedTensorCache(root, partition="train", fold=0,
                                           manifest_sha256="d" * 64)
            cache.write_sample(study_id="study-private-4", series_ids=["series-d"], sample=self.sample())
            cache.finalize()
            (root / "train" / "unlisted.npz").write_bytes(b"extra")
            with self.assertRaisesRegex(ValueError, "cache_unlisted_file"):
                LocalMaskedTensorCache.open(root, partition="train", fold=0,
                                            manifest_sha256="d" * 64,
                                            expected_study_ids=["study-private-4"])

    def test_partial_export_has_no_completion_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            cache = LocalMaskedTensorCache(root, partition="train", fold=0,
                                           manifest_sha256="c" * 64)
            cache.write_sample(study_id="study-private-3", series_ids=["series-c"], sample=self.sample())
            self.assertFalse((root / "cache-manifest.json").exists())

    def test_interrupted_archive_write_never_publishes_final_zip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            root.mkdir()
            (root / "payload.txt").write_text("private", encoding="utf-8")
            archive = Path(temporary) / "cache.zip"

            def fail_after_partial(path, *args, **kwargs):
                Path(path).write_bytes(b"partial-zip")
                raise OSError("injected interrupted archive")

            with self.assertRaisesRegex(OSError, "injected interrupted archive"):
                write_cache_archive(root, archive, zip_factory=fail_after_partial)
            self.assertFalse(archive.exists())
            self.assertFalse(archive.with_name(archive.name + ".tmp").exists())

    def test_successful_archive_is_published_only_as_closed_zip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            root.mkdir()
            (root / "payload.txt").write_text("private", encoding="utf-8")
            archive = Path(temporary) / "cache.zip"
            result = write_cache_archive(root, archive)
            self.assertEqual(result["size_bytes"], archive.stat().st_size)
            self.assertFalse(archive.with_name(archive.name + ".tmp").exists())
            with zipfile.ZipFile(archive) as reader:
                self.assertEqual(reader.namelist(), ["payload.txt"])
                self.assertEqual(reader.read("payload.txt"), b"private")

    def test_chunk_cache_preserves_global_ordinals_and_provenance(self):
        provenance = {
            "chunk_id": "train-000",
            "start_ordinal": 12,
            "stop_ordinal": 13,
            "full_selection_sha256": "e" * 64,
            "chunk_selection_sha256": "f" * 64,
            "source_sha256": "a" * 64,
            "folds_sha256": "b" * 64,
            "preprocessing": {"output_size": 224, "max_slices": 8},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "chunk"
            cache = LocalMaskedTensorCache(root, partition="train", fold=0,
                                           manifest_sha256="c" * 64,
                                           first_ordinal=12, provenance=provenance)
            sample = list(self.sample())
            sample[4] = 12
            cache.write_sample(study_id="study-private-12", series_ids=["series-l"], sample=sample)
            cache.finalize()
            reader = LocalMaskedTensorCache.open(root, partition="train", fold=0,
                                                  manifest_sha256="c" * 64,
                                                  expected_study_ids=["study-private-12"],
                                                  first_ordinal=12,
                                                  expected_provenance=provenance)
            self.assertEqual(reader._records[0]["ordinal"], 12)
            self.assertEqual(reader[0][4], 12)
            wrong = dict(provenance, chunk_id="train-001")
            with self.assertRaisesRegex(ValueError, "cache_provenance_mismatch"):
                LocalMaskedTensorCache.open(root, partition="train", fold=0,
                                            manifest_sha256="c" * 64,
                                            expected_study_ids=["study-private-12"],
                                            first_ordinal=12,
                                            expected_provenance=wrong)

    def test_archive_size_guard_rejects_before_publishing_final_zip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            root.mkdir()
            (root / "payload.bin").write_bytes(b"123456")
            archive = Path(temporary) / "cache.zip"
            with self.assertRaisesRegex(ValueError, "cache_archive_output_limit_exceeded"):
                write_cache_archive(root, archive, max_output_bytes=32, reserve_bytes=32)
            self.assertFalse(archive.exists())
            self.assertFalse(archive.with_name(archive.name + ".tmp").exists())


if __name__ == "__main__":
    unittest.main()
