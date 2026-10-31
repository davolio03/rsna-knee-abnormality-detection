import hashlib
import json
import unittest

from scripts.local_masked_cache_chunk_plan import (
    OUTPUT_LIMIT_BYTES,
    SERIES_BYTES_UPPER_BOUND,
    create_chunk_plan,
    validate_chunk_plan,
)


class LocalMaskedCacheChunkPlanTests(unittest.TestCase):
    def setUp(self):
        self.records = []
        for index, series_count in enumerate((2, 1, 3, 2, 1, 4, 2, 1)):
            self.records.append({
                "study_id": f"private-study-{index}",
                "series": [{"series_id": f"private-series-{index}-{series}"}
                           for series in range(series_count)],
            })
        manifest = {"schema_version": "full-cohort-soft-label-manifest-v1", "records": self.records}
        canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
        self.manifest_sha256 = hashlib.sha256(canonical).hexdigest()
        self.manifest = manifest
        self.assignments = [0, 1, 2, 0, 1, 0, 2, 1]
        self.folds = {
            "manifest_sha256": self.manifest_sha256,
            "study_assignments": [{"study_id": row["study_id"], "fold": fold}
                                  for row, fold in zip(self.records, self.assignments)],
        }
        self.module_hashes = {
            "rsna_knee/data/masked_soft_dataset.py": "a" * 64,
            "rsna_knee/data/dicom_preprocessing.py": "b" * 64,
            "rsna_knee/data/training_dataset.py": "c" * 64,
            "rsna_knee/data/preprocessing.py": "d" * 64,
        }

    def plan(self, *, max_series=4, folds=None, manifest=None):
        return create_chunk_plan(
            self.manifest if manifest is None else manifest,
            self.folds if folds is None else folds,
            manifest_sha256=self.manifest_sha256,
            folds_sha256="e" * 64,
            source_sha256="f" * 64,
            preprocessing_source_members=self.module_hashes,
            max_series_per_chunk=max_series,
        )

    def test_chunks_cover_partition_rows_and_series_without_splitting_studies(self):
        plan = self.plan()
        self.assertTrue(validate_chunk_plan(plan))
        self.assertEqual(plan["partition_counts"], {"train": 5, "validation": 3})
        expected_by_partition = {
            "train": [row for row, fold in zip(self.records, self.assignments) if fold != 0],
            "validation": [row for row, fold in zip(self.records, self.assignments) if fold == 0],
        }
        for partition, records in expected_by_partition.items():
            chunks = [chunk for chunk in plan["chunks"] if chunk["partition"] == partition]
            next_ordinal = 0
            for chunk in chunks:
                self.assertEqual(chunk["start_ordinal"], next_ordinal)
                self.assertEqual(chunk["stop_ordinal_exclusive"] - chunk["start_ordinal"],
                                 chunk["study_count"])
                selected = records[chunk["start_ordinal"]:chunk["stop_ordinal_exclusive"]]
                self.assertEqual(chunk["series_count"], sum(len(row["series"]) for row in selected))
                self.assertLessEqual(chunk["series_count"], plan["max_series_per_chunk"])
                next_ordinal = chunk["stop_ordinal_exclusive"]
            self.assertEqual(next_ordinal, len(records))
        self.assertEqual(sum(chunk["study_count"] for chunk in plan["chunks"]), 8)
        self.assertEqual(sum(chunk["series_count"] for chunk in plan["chunks"]), 16)
        self.assertNotIn("private-study-", json.dumps(plan))
        self.assertNotIn("private-series-", json.dumps(plan))

    def test_plan_caps_worst_case_saved_output_below_notebook_limit(self):
        plan = self.plan(max_series=4000)
        max_chunk = max(plan["chunks"], key=lambda chunk: chunk["series_count"])
        self.assertEqual(plan["preprocessing"]["series_byte_upper_bound"], SERIES_BYTES_UPPER_BOUND)
        self.assertEqual(max_chunk["max_uncompressed_tensor_bytes"],
                         max_chunk["series_count"] * SERIES_BYTES_UPPER_BOUND)
        self.assertLess(max_chunk["max_saved_output_bytes"], OUTPUT_LIMIT_BYTES)
        self.assertEqual(max_chunk["max_saved_output_bytes"],
                         2 * max_chunk["max_uncompressed_tensor_bytes"] + plan["output_reserve_bytes"])

    def test_validator_rejects_missing_duplicate_and_wrong_partition_chunks(self):
        plan = self.plan()
        missing = dict(plan, chunks=plan["chunks"][1:])
        with self.assertRaisesRegex(ValueError, "chunk_plan_coverage_invalid"):
            validate_chunk_plan(missing)
        duplicate = dict(plan, chunks=[*plan["chunks"], plan["chunks"][0]])
        with self.assertRaisesRegex(ValueError, "chunk_plan_coverage_invalid"):
            validate_chunk_plan(duplicate)
        wrong_partition = json.loads(json.dumps(plan))
        wrong_partition["chunks"][0]["partition"] = "validation"
        with self.assertRaisesRegex(ValueError, "chunk_plan_coverage_invalid"):
            validate_chunk_plan(wrong_partition)
        gold_plan = dict(plan, gold_included=True)
        with self.assertRaisesRegex(ValueError, "chunk_plan_schema_invalid"):
            validate_chunk_plan(gold_plan)

    def test_planner_rejects_duplicate_and_missing_fold_assignments(self):
        duplicate = json.loads(json.dumps(self.folds))
        duplicate["study_assignments"].append(duplicate["study_assignments"][0])
        with self.assertRaisesRegex(ValueError, "chunk_plan_assignment_invalid"):
            self.plan(folds=duplicate)
        missing = json.loads(json.dumps(self.folds))
        missing["study_assignments"].pop()
        with self.assertRaisesRegex(ValueError, "chunk_plan_assignment_membership_mismatch"):
            self.plan(folds=missing)

    def test_planner_rejects_study_larger_than_chunk_limit(self):
        manifest = json.loads(json.dumps(self.manifest))
        manifest["records"][0]["series"] = [
            {"series_id": f"private-series-too-many-{index}"} for index in range(5)]
        canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
        folds = json.loads(json.dumps(self.folds))
        folds["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
        with self.assertRaisesRegex(ValueError, "study_exceeds_series_chunk_cap"):
            self.plan(max_series=4, manifest=manifest, folds=folds)


if __name__ == "__main__":
    unittest.main()
