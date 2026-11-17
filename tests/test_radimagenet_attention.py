"""Contracts for target-specific hierarchical attention over slices and series."""
import unittest

import torch
from torch import nn
from torchvision.models import resnet50

from rsna_knee.models.attention_mil import (
    RadImageNetAttentionStudyModel,
    masked_target_softmax,
    target_weighted_sum,
)
from rsna_knee.models.resnet50 import ResNet50StudyModel


class AttentionMILTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def _models(self):
        base = ResNet50StudyModel(resnet50(weights=None), arm="radimagenet", seed=42)
        attention = RadImageNetAttentionStudyModel.from_base(base)
        return base, attention

    def test_zero_initialized_attention_matches_equal_mean_with_tolerance(self):
        base, attention = self._models()
        images = [torch.rand(2, 1, 32, 32), torch.rand(3, 1, 32, 32)]
        masks = [torch.ones_like(image, dtype=torch.bool) for image in images]
        with torch.no_grad():
            expected = base(images, masks)
            actual = attention(images, masks)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        for scorer in (attention.slice_attention, attention.series_attention):
            self.assertTrue(torch.equal(scorer.weight, torch.zeros_like(scorer.weight)))
            self.assertTrue(torch.equal(scorer.bias, torch.zeros_like(scorer.bias)))

    def test_masked_softmax_sums_to_one_per_target_and_rejects_empty_rows(self):
        scores = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        valid = torch.tensor([True, False, True])
        weights = masked_target_softmax(scores, valid, dim=0)
        self.assertTrue(torch.equal(weights[1], torch.zeros(2)))
        torch.testing.assert_close(weights.sum(0), torch.ones(2))
        with self.assertRaisesRegex(ValueError, "attention_mask_empty"):
            masked_target_softmax(scores, torch.zeros(3, dtype=torch.bool), dim=0)

    def test_weighted_sum_has_known_nonuniform_target_specific_result(self):
        values = torch.tensor([[2.0, 10.0], [8.0, 20.0]])
        weights = torch.tensor([[0.75, 0.1], [0.25, 0.9]])
        actual = target_weighted_sum(values, weights)
        torch.testing.assert_close(actual, torch.tensor([3.5, 19.0]))

    def test_slice_and_series_permutations_preserve_study_logits(self):
        _, model = self._models()
        with torch.no_grad():
            model.slice_attention.weight.normal_(0, 0.002)
            model.series_attention.weight.normal_(0, 0.002)
        images = [torch.rand(3, 1, 32, 32), torch.rand(2, 1, 32, 32)]
        masks = [torch.ones_like(image, dtype=torch.bool) for image in images]
        with torch.no_grad():
            expected = model(images, masks)
            permuted = model([images[1].flip(0), images[0].flip(0)], [m.flip(0) for m in masks[::-1]])
        torch.testing.assert_close(permuted, expected, rtol=1e-5, atol=1e-6)

    def test_attention_and_classifier_receive_finite_gradients(self):
        _, model = self._models()
        images = [torch.rand(2, 1, 32, 32), torch.rand(1, 1, 32, 32)]
        masks = [torch.ones_like(image, dtype=torch.bool) for image in images]
        model.train()
        model(images, masks).sum().backward()
        for scorer in (model.slice_attention, model.series_attention):
            self.assertIsNotNone(scorer.weight.grad)
            self.assertTrue(torch.isfinite(scorer.weight.grad).all())
            self.assertGreater(scorer.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.classifier.weight.grad.abs().sum().item(), 0.0)

    def test_invalid_inputs_fail_closed(self):
        _, model = self._models()
        image = torch.rand(2, 1, 32, 32)
        mask = torch.ones_like(image, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "study_input_invalid"):
            model([image], [])
        with self.assertRaisesRegex(ValueError, "study_mask_invalid"):
            model([image], [torch.zeros_like(mask)])
        with self.assertRaisesRegex(ValueError, "study_input_invalid"):
            model([image * float("nan")], [mask])


if __name__ == "__main__":
    unittest.main()
