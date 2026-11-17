import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.local16_tensor_cache import Local16TensorCache


class Local16TensorCacheTests(unittest.TestCase):
    def setUp(self):
        self.provenance = {
            "schema_version": "local16-cache-provenance-v1",
            "cache_identity": "local16-attention-run01",
            "max_slices_per_series": 16,
            "source_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "folds_sha256": "c" * 64,
            "preprocessing_source_sha256": "d" * 64,
            "gold_included": False,
        }

    def sample(self, slices=16, row_index=0):
        images = [np.ones((slices, 1, 224, 224), dtype=np.float32)]
        masks = [np.ones((slices, 1, 224, 224), dtype=np.bool_)]
        return images, masks, np.zeros(12, dtype=np.float32), np.ones(12, dtype=np.bool_), row_index

    def test_writes_and_reopens_distinct_16_slice_schema_with_exact_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "chunk"
            cache = Local16TensorCache(root, partition="train", fold=0,
                                       manifest_sha256="b" * 64, provenance=self.provenance)
            cache.write_sample(study_id="study-one", series_ids=["series-one"], sample=self.sample(), eligible_slice_counts=[16])
            cache.finalize()
            opened = Local16TensorCache.open(
                root, partition="train", fold=0, manifest_sha256="b" * 64,
                expected_study_ids=["study-one"], expected_series_counts=[1],
                expected_series_ids=[["series-one"]],
                expected_eligible_slice_counts=[[16]],
                expected_provenance=self.provenance)
            self.assertEqual(opened[0][0][0].shape, (16, 1, 224, 224))
            self.assertEqual(opened._records[0]["max_slices_per_series"], 16)

    def test_refuses_eight_slice_payload_and_duplicate_series_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Local16TensorCache(Path(temporary) / "chunk", partition="train", fold=0,
                                       manifest_sha256="b" * 64, provenance=self.provenance)
            with self.assertRaisesRegex(ValueError, "cache_series_order_invalid"):
                cache.write_sample(study_id="study-one", series_ids=["series-one", "series-one"],
                                   sample=self.sample(), eligible_slice_counts=[16, 16])
            with self.assertRaisesRegex(ValueError, "cache_slice_count_invalid"):
                cache.write_sample(study_id="study-two", series_ids=["series-two"], sample=self.sample(17), eligible_slice_counts=[17])
            with self.assertRaisesRegex(ValueError, "cache_eligible_slice_count_mismatch"):
                cache.write_sample(study_id="study-two", series_ids=["series-two"], sample=self.sample(8), eligible_slice_counts=[20])

    def test_open_requires_new_identity_and_hash_checked_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "chunk"
            cache = Local16TensorCache(root, partition="train", fold=0,
                                       manifest_sha256="b" * 64, provenance=self.provenance)
            cache.write_sample(study_id="study-one", series_ids=["series-one"], sample=self.sample(), eligible_slice_counts=[16])
            cache.finalize()
            with self.assertRaisesRegex(ValueError, "cache_manifest_contract_mismatch"):
                Local16TensorCache.open(root, partition="train", fold=0, manifest_sha256="b" * 64,
                                        expected_study_ids=["study-one"], expected_series_counts=[1],
                                        expected_series_ids=[["series-one"]],
                                        expected_eligible_slice_counts=[[16]],
                                        expected_provenance={**self.provenance, "source_sha256": "e" * 64})
            record = cache._records[0]
            (root / record["file"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "cache_file_hash_mismatch"):
                Local16TensorCache.open(root, partition="train", fold=0, manifest_sha256="b" * 64,
                                        expected_study_ids=["study-one"], expected_series_counts=[1],
                                        expected_series_ids=[["series-one"]],
                                        expected_eligible_slice_counts=[[16]],
                                        expected_provenance=self.provenance)


if __name__ == "__main__":
    unittest.main()
