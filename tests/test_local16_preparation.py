import tempfile
import unittest
from pathlib import Path

from rsna_knee.data.local16_preparation import (
    Local16PreparationError,
    build_local16_plan,
)


class Local16PreparationTests(unittest.TestCase):
    def test_plan_is_explicit_sixteen_recipe_and_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_local16_plan(root)
            self.assertEqual(plan["max_slices"], 16)
            self.assertEqual(plan["output_size"], 224)
            self.assertEqual(plan["slice_microbatch"], 2)
            self.assertEqual(plan["fold"], 0)
            self.assertEqual(plan["seed"], 42)
            self.assertEqual(plan["output_identity"], "local16-attention-run01")
            self.assertFalse((root / plan["output_identity"]).exists())
            self.assertEqual(plan["cache_policy"], "build_new_from_raw_dicom")
            self.assertFalse(plan["cache_authentication_performed"])
            self.assertFalse(plan["raw_dicom_content_validated"])
            self.assertEqual(plan["comparison"], "target_attention_8_vs_target_attention_16")
            self.assertEqual(plan["arms"], {
                "target_attention_8": {"max_slices": 8, "seed": 42, "epochs": 5},
                "target_attention_16": {"max_slices": 16, "seed": 42, "epochs": 5},
            })
            self.assertTrue(plan["same_fresh_initialization_protocol"])
            self.assertEqual(plan["selected_epoch"], 5)
            self.assertFalse(plan["checkpoint_warm_start"])

    def test_plan_fails_closed_when_raw_root_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "data" / "train_series"
            with self.assertRaises(Local16PreparationError) as caught:
                build_local16_plan(missing)
            self.assertEqual(caught.exception.reason_code, "raw_dicom_root_unavailable")

    def test_plan_rejects_caller_declared_eight_slice_cache_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(Local16PreparationError) as caught:
                build_local16_plan(directory, cache_max_slices=8)
            self.assertEqual(caught.exception.reason_code, "cache_slice_limit_mismatch")

    def test_plan_rejects_non_sixteen_caller_declared_cache_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(Local16PreparationError) as caught:
                build_local16_plan(directory, cache_max_slices=12)
            self.assertEqual(caught.exception.reason_code, "cache_slice_limit_mismatch")


if __name__ == "__main__":
    unittest.main()
