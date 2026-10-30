import unittest
from unittest import mock

try:
    import torch
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("optional PyTorch training dependency is not installed") from exc


class ScratchLearningPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @staticmethod
    def inputs(*, constant=False):
        image = torch.full((2, 1, 16, 16), 0.5) if constant else torch.linspace(
            0.0, 1.0, 512
        ).reshape(2, 1, 16, 16)
        changed = image.clone() if constant else image.flip(-1)
        masks = [torch.ones_like(image, dtype=torch.bool)]
        return [image], masks, [changed], masks

    def test_actual_scratch_model_routes_distinct_inputs_through_learning_path(self):
        from rsna_knee.training.learning_path import diagnose_scratch_learning_path

        images, masks, changed, changed_masks = self.inputs()
        result = diagnose_scratch_learning_path(
            images, masks, changed, changed_masks, torch.zeros(12), steps=2
        )

        self.assertEqual(result["status"], "ok")
        self.assertGreater(result["input_logit_delta"], 0.0)
        for component in ("encoder", "head"):
            self.assertTrue(result["gradients"][component]["finite"])
            self.assertGreater(result["gradients"][component]["norm"], 0.0)
            self.assertGreater(result["updates"][component]["max_abs_delta"], 0.0)
        self.assertGreater(result["post_update_logit_delta"], 0.0)

    def test_constant_inputs_are_not_reported_as_success(self):
        from rsna_knee.training.learning_path import diagnose_scratch_learning_path

        images, masks, changed, changed_masks = self.inputs(constant=True)
        result = diagnose_scratch_learning_path(
            images, masks, changed, changed_masks, torch.zeros(12), steps=1
        )
        self.assertEqual(result["status"], "input_not_diverse")

    def test_nonfinite_inputs_are_rejected_without_passing(self):
        from rsna_knee.training.learning_path import diagnose_scratch_learning_path

        images, masks, changed, changed_masks = self.inputs()
        images[0][0, 0, 0, 0] = float("nan")
        result = diagnose_scratch_learning_path(
            images, masks, changed, changed_masks, torch.zeros(12), steps=1
        )
        self.assertEqual(result["status"], "invalid_input")
        self.assertNotIn("gradients", result)

    def test_invalid_masks_are_rejected_without_passing(self):
        from rsna_knee.training.learning_path import diagnose_scratch_learning_path

        images, masks, changed, changed_masks = self.inputs()
        masks[0].zero_()
        result = diagnose_scratch_learning_path(
            images, masks, changed, changed_masks, torch.zeros(12), steps=1
        )
        self.assertEqual(result["status"], "invalid_input")

    def test_comparison_study_can_have_different_series_and_slice_shapes(self):
        from rsna_knee.training.learning_path import diagnose_scratch_learning_path

        images, masks, _, _ = self.inputs()
        comparison = [torch.rand(1, 1, 24, 24)]
        comparison_masks = [torch.ones_like(comparison[0], dtype=torch.bool)]
        result = diagnose_scratch_learning_path(
            images, masks, comparison, comparison_masks, torch.zeros(12), steps=1
        )
        self.assertNotEqual(result["status"], "invalid_input")
        self.assertTrue(torch.isfinite(torch.tensor(result["input_mean_abs_delta"])))

    def test_probe_preserves_cpu_rng_without_seeding_accelerators(self):
        from rsna_knee.training.learning_path import diagnose_scratch_learning_path

        images, masks, changed, changed_masks = self.inputs()
        caller_state = torch.random.get_rng_state().clone()
        with mock.patch.object(torch.cuda, "manual_seed_all", side_effect=AssertionError("CUDA seeding is forbidden")):
            result = diagnose_scratch_learning_path(
                images, masks, changed, changed_masks, torch.zeros(12), steps=1
            )
        self.assertEqual(result["status"], "ok")
        self.assertTrue(torch.equal(torch.random.get_rng_state(), caller_state))

    def test_non_cpu_masks_return_invalid_input_without_tensor_runtime_error(self):
        from rsna_knee.training.learning_path import diagnose_scratch_learning_path

        images, masks, changed, changed_masks = self.inputs()
        masks = [torch.empty(images[0].shape, dtype=torch.bool, device="meta")]
        result = diagnose_scratch_learning_path(
            images, masks, changed, changed_masks, torch.zeros(12), steps=1
        )
        self.assertEqual(result["status"], "invalid_input")


if __name__ == "__main__":
    unittest.main()
