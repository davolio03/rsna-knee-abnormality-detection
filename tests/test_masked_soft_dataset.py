import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

try:
    import numpy as np
    import torch
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("optional NumPy/PyTorch training dependencies are not installed") from exc


class MaskedSoftDatasetTests(unittest.TestCase):
    @property
    def artifact_paths(self):
        root = Path("data/private/full-cohort-prefilter/run01")
        return root / "full-cohort-soft-label-manifest.json", root / "full-cohort-grouped-folds.json"

    def test_pinned_manifest_and_fold_artifact_validate_without_image_root(self):
        from rsna_knee.data.masked_soft_dataset import load_verified_soft_artifacts

        manifest_path, folds_path = self.artifact_paths
        records, assignments, digest = load_verified_soft_artifacts(manifest_path, folds_path)
        self.assertEqual(len(records), 4298)
        self.assertEqual(len(assignments), 4298)
        self.assertEqual(len(digest), 64)
        self.assertEqual(sum(value == 0.5 for record in records
                             for value in record["soft_targets"].values()),
                         sum(not value for record in records for value in record["observed_mask"].values()))

    def test_dataset_preserves_fractional_labels_and_separate_masks(self):
        from rsna_knee.data.masked_soft_dataset import FullCohortSoftDataset
        from rsna_knee.data.masked_soft_dataset import load_verified_soft_artifacts

        manifest_path, folds_path = self.artifact_paths
        records, assignments, _ = load_verified_soft_artifacts(manifest_path, folds_path)
        fold = 0
        partition = "train"
        record = next(row for row in records if assignments[row["study_id"]] != fold)
        with TemporaryDirectory() as temp:
            image_root = Path(temp)
            study_path = image_root / record["study_id"]
            for series in record["series"]:
                (study_path / series["series_id"]).mkdir(parents=True)

            def preprocess(_path, *, max_slices, output_size):
                return SimpleNamespace(
                    images=np.ones((1, 1, output_size, output_size), dtype=np.float32) * 0.25,
                    valid_mask=np.ones((1, 1, output_size, output_size), dtype=bool),
                )

            dataset = FullCohortSoftDataset(
                manifest_path, folds_path, image_root, partition=partition, fold=fold,
                output_size=8, max_slices=1, preprocessor=preprocess,
            )
            images, pixel_masks, targets, observed, row_index = dataset[0]
            self.assertEqual(len(images), len(record["series"]))
            self.assertEqual(images[0].dtype, torch.float32)
            self.assertEqual(pixel_masks[0].dtype, torch.bool)
            self.assertTrue(torch.equal(targets, torch.tensor([
                record["soft_targets"][name] for name in dataset.target_names
            ], dtype=torch.float32)))
            self.assertTrue(torch.equal(observed, torch.tensor([
                record["observed_mask"][name] for name in dataset.target_names
            ], dtype=torch.bool)))
            self.assertEqual(row_index, 0)

    def test_tampered_pinned_bytes_are_rejected_before_image_setup(self):
        from rsna_knee.data.masked_soft_dataset import SoftDatasetError, load_verified_soft_artifacts

        manifest_path, folds_path = self.artifact_paths
        with TemporaryDirectory() as temp:
            changed = Path(temp) / "manifest.json"
            changed.write_bytes(manifest_path.read_bytes() + b" ")
            with self.assertRaisesRegex(SoftDatasetError, "manifest_pin_mismatch"):
                load_verified_soft_artifacts(changed, folds_path)

    def test_tampered_fold_bytes_are_rejected(self):
        from rsna_knee.data.masked_soft_dataset import SoftDatasetError, load_verified_soft_artifacts

        manifest_path, folds_path = self.artifact_paths
        with TemporaryDirectory() as temp:
            changed = Path(temp) / "folds.json"
            changed.write_bytes(folds_path.read_bytes() + b" ")
            with self.assertRaisesRegex(SoftDatasetError, "folds_pin_mismatch"):
                load_verified_soft_artifacts(manifest_path, changed)


if __name__ == "__main__":
    unittest.main()
