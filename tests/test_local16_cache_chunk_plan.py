import hashlib
import json
import unittest

from scripts.local16_cache_chunk_plan import (
    MAX_SERIES_PER_CHUNK, OUTPUT_LIMIT_BYTES, SERIES_BYTES_UPPER_BOUND,
    create_chunk_plan, validate_chunk_plan,
)


class Local16CacheChunkPlanTests(unittest.TestCase):
    def setUp(self):
        self.records = [{"study_id": f"s{i}", "series": [{"series_id": f"r{i}-{j}"} for j in range(n)]}
                        for i, n in enumerate((2, 1, 3, 2, 1, 4, 2, 1))]
        manifest = {"schema_version": "full-cohort-soft-label-manifest-v1", "records": self.records}
        self.manifest_sha = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n").hexdigest()
        self.manifest = manifest
        self.folds = {"manifest_sha256": self.manifest_sha,
                      "study_assignments": [{"study_id": f"s{i}", "fold": fold}
                                            for i, fold in enumerate((0, 1, 2, 0, 1, 0, 2, 1))]}

    def plan(self, cap=4):
        return create_chunk_plan(self.manifest, self.folds, manifest_sha256=self.manifest_sha,
                                 folds_sha256="b" * 64, source_sha256="c" * 64,
                                 preprocessing_source_sha256="d" * 64,
                                 max_series_per_chunk=cap)

    def test_chunks_preserve_studies_and_bound_sixteen_slice_outputs(self):
        plan = self.plan()
        self.assertTrue(validate_chunk_plan(plan))
        self.assertEqual(plan["max_slices_per_series"], 16)
        self.assertEqual(sum(c["study_count"] for c in plan["chunks"]), len(self.records))
        for chunk in plan["chunks"]:
            self.assertEqual(chunk["max_uncompressed_tensor_bytes"], chunk["series_count"] * SERIES_BYTES_UPPER_BOUND)
            self.assertLess(chunk["max_saved_output_bytes"], OUTPUT_LIMIT_BYTES)

    def test_refuses_historical_eight_slice_identity_and_oversized_chunk_cap(self):
        with self.assertRaisesRegex(ValueError, "series_chunk_cap_invalid"):
            self.plan(MAX_SERIES_PER_CHUNK + 1)
        plan = self.plan()
        plan["cache_identity"] = "legacy-eight-slice"
        with self.assertRaisesRegex(ValueError, "chunk_plan_schema_invalid"):
            validate_chunk_plan(plan)


if __name__ == "__main__":
    unittest.main()
