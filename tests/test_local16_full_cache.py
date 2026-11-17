import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from scripts.local16_cache_chunk_plan import create_chunk_plan
from scripts.local16_full_cache import TARGETS, verify_full_cache
from scripts.local16_tensor_cache import Local16TensorCache


class Local16FullCacheTests(unittest.TestCase):
    def test_verifies_exact_folded_study_series_and_tensor_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = []
            for i in range(2):
                records.append({"study_id": f"study-{i}", "series": [{"series_id": f"series-{i}"}],
                                "soft_targets": {name: 0.0 for name in TARGETS},
                                "observed_mask": {name: True for name in TARGETS}})
            manifest = {"schema_version": "full-cohort-soft-label-manifest-v1", "records": records}
            manifest_raw = (json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
            manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
            folds = {"manifest_sha256": manifest_sha,
                     "study_assignments": [{"study_id": "study-0", "fold": 0}, {"study_id": "study-1", "fold": 1}]}
            folds_raw = (json.dumps(folds, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
            (root / "manifest.json").write_bytes(manifest_raw)
            (root / "folds.json").write_bytes(folds_raw)
            source = root / "source.zip"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("src/example.py", "VALUE = 1\n")
            plan = create_chunk_plan(manifest, folds, manifest_sha256=manifest_sha,
                                     folds_sha256=hashlib.sha256(folds_raw).hexdigest(),
                                     source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                                     preprocessing_source_sha256="d" * 64)
            plan_raw = (json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
            (root / "plan.json").write_bytes(plan_raw)
            archive_dir = root / "archives"
            archive_dir.mkdir()
            for chunk in plan["chunks"]:
                partition_rows = [row for row in records if (row["study_id"] == "study-0") == (chunk["partition"] == "validation")]
                provenance = {"schema_version": "local16-cache-provenance-v1", "cache_identity": "local16-attention-run01",
                              "max_slices_per_series": 16, "source_sha256": plan["source_sha256"],
                              "manifest_sha256": manifest_sha, "folds_sha256": plan["folds_sha256"],
                              "preprocessing_source_sha256": "d" * 64, "gold_included": False}
                chunk_root = root / f"chunk-{chunk['chunk_id']}"
                cache = Local16TensorCache(chunk_root, partition=chunk["partition"], fold=0,
                                           manifest_sha256=manifest_sha,
                                           first_ordinal=chunk["start_ordinal"], provenance=provenance)
                for offset, row in enumerate(partition_rows):
                    images = [np.ones((16, 1, 224, 224), dtype=np.float32)]
                    masks = [np.ones((16, 1, 224, 224), dtype=np.bool_)]
                    cache.write_sample(study_id=row["study_id"], series_ids=["series-" + row["study_id"][-1]],
                                       sample=(images, masks, np.zeros(12, np.float32), np.ones(12, np.bool_), offset),
                                       eligible_slice_counts=[20])
                cache.finalize()
                with zipfile.ZipFile(archive_dir / f"local16-cache-{chunk['chunk_id']}.zip", "w") as archive:
                    for path in sorted(chunk_root.rglob("*")):
                        if path.is_file():
                            archive.write(path, path.relative_to(chunk_root).as_posix())
            proof = verify_full_cache(manifest_path=root / "manifest.json", folds_path=root / "folds.json",
                                      plan_path=root / "plan.json", source_archive_path=source,
                                      archive_dir=archive_dir)
            self.assertEqual(proof["status"], "verified")
            self.assertEqual((proof["studies"], proof["series"], proof["slices"]), (2, 2, 32))
            self.assertEqual(proof["partition_study_counts"], {"train": 1, "validation": 1})
            self.assertEqual(proof["series_with_more_than_eight_slices"], 2)
            self.assertFalse(proof["training_started"])

if __name__ == "__main__":
    unittest.main()
