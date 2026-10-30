import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import torch
    import torchvision
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("optional PyTorch/TorchVision training dependencies are unavailable") from exc


class ResNet18StudyModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_random_and_pretrained_models_have_matching_head_and_valid_output(self):
        from rsna_knee.models.resnet18 import build_resnet18_study_model

        random_model = build_resnet18_study_model("resnet18_random", head_seed=42)
        pretrained_model = build_resnet18_study_model(
            "resnet18_imagenet1k_v1", head_seed=42,
            weights_path=Path("data/private/resnet18-comparison/weights/resnet18-f37072fd.pth"),
            expected_weights_sha256="f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
            expected_weights_size=46830571,
        )
        self.assertEqual(random_model.classifier.weight.shape, (12, 512))
        self.assertTrue(torch.equal(random_model.classifier.weight, pretrained_model.classifier.weight))
        self.assertTrue(torch.equal(random_model.classifier.bias, pretrained_model.classifier.bias))
        images = [torch.rand(2, 1, 32, 32), torch.rand(1, 1, 32, 32)]
        masks = [torch.ones_like(image, dtype=torch.bool) for image in images]
        with torch.no_grad():
            output = random_model(images, masks)
        self.assertEqual(tuple(output.shape), (1, 12))
        self.assertTrue(torch.isfinite(output).all())
        pretrained_model.train()
        batch_norms = [module for module in pretrained_model.modules()
                       if isinstance(module, torch.nn.BatchNorm2d)]
        bn_before = [(item.running_mean.clone(), item.running_var.clone(), item.num_batches_tracked.clone())
                     for item in batch_norms]
        optimizer = torch.optim.Adam(pretrained_model.parameters(), lr=0.001)
        optimizer.zero_grad(set_to_none=True)
        pretrained_output = pretrained_model(images, masks)
        pretrained_output.square().mean().backward()
        self.assertGreater(float(pretrained_model.backbone.conv1.weight.grad.norm()), 0.0)
        self.assertGreater(float(pretrained_model.classifier.weight.grad.norm()), 0.0)
        optimizer.step()
        self.assertTrue(all(torch.equal(item.running_mean, snapshot[0])
                            and torch.equal(item.running_var, snapshot[1])
                            and torch.equal(item.num_batches_tracked, snapshot[2])
                            for item, snapshot in zip(batch_norms, bn_before)))

    def test_frozen_imagenet_backbone_stays_bitwise_fixed_while_head_trains(self):
        from rsna_knee.models.resnet18 import build_resnet18_study_model

        model = build_resnet18_study_model(
            "resnet18_imagenet1k_v1", head_seed=42,
            weights_path=Path("data/private/resnet18-comparison/weights/resnet18-f37072fd.pth"),
            expected_weights_sha256="f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
            expected_weights_size=46830571, freeze_backbone=True,
        )
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad), 6156)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.backbone.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.classifier.parameters()))
        self.assertFalse(model.checkpoint_trunk)
        model.train()
        batch_norms = [module for module in model.backbone.modules() if isinstance(module, torch.nn.BatchNorm2d)]
        self.assertTrue(all(not layer.training for layer in batch_norms))
        backbone_before = {name: value.detach().clone() for name, value in model.backbone.state_dict().items()}
        head_before = {name: value.detach().clone() for name, value in model.classifier.state_dict().items()}
        images = [torch.rand(2, 1, 32, 32), torch.rand(1, 1, 32, 32)]
        masks = [torch.ones_like(image, dtype=torch.bool) for image in images]
        optimizer = torch.optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad), lr=0.001)
        optimizer.zero_grad(set_to_none=True)
        output = model(images, masks)
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertTrue(all(parameter.grad is None for parameter in model.backbone.parameters()))
        self.assertGreater(float(model.classifier.weight.grad.norm()), 0.0)
        optimizer.step()
        self.assertTrue(all(torch.equal(value, model.backbone.state_dict()[name])
                            for name, value in backbone_before.items()))
        self.assertTrue(any(not torch.equal(value, model.classifier.state_dict()[name])
                            for name, value in head_before.items()))

    def test_invalid_pretrained_file_fails_before_weights_are_deserialized(self):
        from rsna_knee.models.resnet18 import build_resnet18_study_model

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pth"
            path.write_bytes(b"not a checkpoint")
            with mock.patch("torch.load", side_effect=AssertionError("must not deserialize")) as load:
                with self.assertRaises(ValueError):
                    build_resnet18_study_model(
                        "resnet18_imagenet1k_v1", head_seed=42, weights_path=path,
                        expected_weights_sha256="0" * 64, expected_weights_size=path.stat().st_size,
                    )
            load.assert_not_called()

    def test_masks_gate_normalized_pixels_and_pooling_uses_all_series(self):
        from rsna_knee.models.resnet18 import build_resnet18_study_model

        model = build_resnet18_study_model("resnet18_random", head_seed=42)
        captured = []
        original_forward = model.backbone_forward

        def capture(images):
            captured.append(images.detach().clone())
            return original_forward(images)

        model.backbone_forward = capture
        first = torch.full((1, 1, 32, 32), 0.5)
        second = torch.full((1, 1, 32, 32), 0.25)
        mask_first = torch.ones_like(first, dtype=torch.bool)
        mask_first[..., 16, 16] = False
        mask_second = torch.zeros_like(second, dtype=torch.bool)
        mask_second[..., :16, :16] = True
        model.eval()
        with torch.no_grad():
            result = model([first, second], [mask_first, mask_second])
        self.assertEqual(tuple(result.shape), (1, 12))
        self.assertEqual(len(captured), 2)
        self.assertTrue(torch.equal(captured[0][:, :, 16, 16], torch.zeros(1, 3)))
        self.assertTrue(torch.equal(captured[1][:, :, 24, 24], torch.zeros(1, 3)))

    def test_checkpointed_microbatches_update_encoder_and_head_with_frozen_bn_stats(self):
        from rsna_knee.models.resnet18 import build_resnet18_study_model

        model = build_resnet18_study_model("resnet18_random", head_seed=42, slice_microbatch=2)
        model.train()
        norm_layers = [module for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d)]
        self.assertTrue(norm_layers)
        self.assertTrue(all(not layer.training for layer in norm_layers))
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        running_before = [(layer.running_mean.clone(), layer.running_var.clone(), layer.num_batches_tracked.clone())
                          for layer in norm_layers]
        images = [torch.rand(3, 1, 32, 32), torch.rand(1, 1, 32, 32)]
        masks = [torch.ones_like(image, dtype=torch.bool) for image in images]
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.zero_grad(set_to_none=True)
        loss = model(images, masks).square().mean()
        loss.backward()
        encoder_grad = sum(float(parameter.grad.norm()) for parameter in model.encoder_parameters() if parameter.grad is not None)
        head_grad = sum(float(parameter.grad.norm()) for parameter in model.classifier.parameters() if parameter.grad is not None)
        optimizer.step()
        self.assertGreater(encoder_grad, 0.0)
        self.assertGreater(head_grad, 0.0)
        self.assertTrue(any(not torch.equal(before[name], value) for name, value in model.named_parameters()
                            if not name.startswith("classifier.")))
        self.assertTrue(any(not torch.equal(before[name], value) for name, value in model.named_parameters()
                            if name.startswith("classifier.")))
        for layer, snapshots in zip(norm_layers, running_before):
            self.assertTrue(torch.equal(layer.running_mean, snapshots[0]))
            self.assertTrue(torch.equal(layer.running_var, snapshots[1]))
            self.assertTrue(torch.equal(layer.num_batches_tracked, snapshots[2]))

    def test_checkpointed_and_direct_backbone_forward_and_gradients_match(self):
        from rsna_knee.models.resnet18 import build_resnet18_study_model

        direct = build_resnet18_study_model("resnet18_random", head_seed=42, checkpoint_trunk=False)
        checkpointed = build_resnet18_study_model("resnet18_random", head_seed=42, checkpoint_trunk=True)
        checkpointed.load_state_dict(direct.state_dict())
        images = [torch.rand(2, 1, 32, 32), torch.rand(1, 1, 32, 32)]
        masks = [torch.ones_like(image, dtype=torch.bool) for image in images]
        direct.train()
        checkpointed.train()
        out_a = direct(images, masks)
        out_b = checkpointed(images, masks)
        self.assertTrue(torch.allclose(out_a, out_b, rtol=1e-5, atol=1e-6))
        out_a.sum().backward()
        out_b.sum().backward()
        for (name_a, parameter_a), (name_b, parameter_b) in zip(direct.named_parameters(), checkpointed.named_parameters()):
            self.assertEqual(name_a, name_b)
            if parameter_a.grad is not None or parameter_b.grad is not None:
                self.assertTrue(torch.allclose(parameter_a.grad, parameter_b.grad, rtol=2e-4, atol=2e-5), name_a)

    def test_256_pixel_full_resolution_forward_succeeds_without_cropping(self):
        from rsna_knee.models.resnet18 import build_resnet18_study_model

        model = build_resnet18_study_model("resnet18_random", head_seed=42).eval()
        image = torch.rand(1, 1, 256, 256)
        mask = torch.ones_like(image, dtype=torch.bool)
        with torch.no_grad():
            output = model([image], [mask])
        self.assertEqual(tuple(output.shape), (1, 12))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
