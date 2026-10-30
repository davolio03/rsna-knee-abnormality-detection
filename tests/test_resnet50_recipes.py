"""Synthetic ResNet50 recipe and partial-gradient contracts."""
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torchvision.models import resnet50

from rsna_knee.models.resnet50 import ResNet50StudyModel, load_resnet50_recipe, WEIGHT_PINS


class ResNet50RecipeTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1)

    def test_matched_head_native_scaling_and_only_last_block_gradients(self):
        models=[ResNet50StudyModel(resnet50(weights=None),arm=arm,seed=42) for arm in WEIGHT_PINS]
        self.assertTrue(torch.equal(models[0].classifier.weight,models[1].classifier.weight))
        self.assertEqual(models[0].classifier.in_features,2048)
        self.assertEqual(models[1].imagenet_mean.flatten().tolist(),[.5]*3)
        for model in models:
            model.train();before={name:buf.clone() for name,buf in model.named_buffers()}
            image=torch.rand(1,1,32,32);mask=torch.ones_like(image,dtype=torch.bool)
            output=model([image],[mask]);self.assertEqual(tuple(output.shape),(1,12));output.sum().backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.backbone.layer4.parameters()))
            self.assertTrue(model.classifier.weight.grad.abs().sum()>0)
            for name,p in model.backbone.named_parameters():
                if not name.startswith('layer4.') or '.bn' in name or '.downsample.1.' in name:
                    self.assertFalse(p.requires_grad,name)
            for module in model.modules():
                if isinstance(module,nn.modules.batchnorm._BatchNorm):
                    self.assertFalse(module.training)
                    self.assertFalse(any(p.requires_grad for p in module.parameters()))
            for name,buf in model.named_buffers():self.assertTrue(torch.equal(before[name],buf),name)

    def test_invalid_pixels_zero_after_each_native_normalization(self):
        for arm in WEIGHT_PINS:
            model=ResNet50StudyModel(resnet50(weights=None),arm=arm,seed=42)
            image=torch.rand(1,1,32,32);mask=torch.ones_like(image,dtype=torch.bool);mask[:,:,:3,:]=False
            other=image.clone();other[:,:,:3,:]=1.
            with torch.no_grad():self.assertTrue(torch.equal(model([image],[mask]),model([other],[mask])))

    def test_weight_hash_rejected_before_deserialization_and_no_partial_load(self):
        with TemporaryDirectory() as td:
            path=Path(td)/'fake.pt';path.write_bytes(b'not weights')
            with patch('torch.load') as load:
                with self.assertRaises(ValueError):load_resnet50_recipe('radimagenet',path)
                load.assert_not_called()

    def test_rad_mapping_strictly_matches_torchvision_backbone(self):
        from rsna_knee.models.resnet50 import map_radimagenet_state
        model=resnet50(weights=None);model.fc=nn.Identity();state=model.state_dict()
        prefixes={'conv1':'0','bn1':'1','layer1':'4','layer2':'5','layer3':'6','layer4':'7'}
        rad={f'backbone.{prefixes[k.split(".")[0]]}.{k.split(".",1)[1]}':v for k,v in state.items()}
        mapped=map_radimagenet_state(rad)
        self.assertEqual(set(mapped),set(state));model.load_state_dict(mapped,strict=True)
        rad['classifier.weight']=torch.ones(1)
        with self.assertRaises(ValueError):map_radimagenet_state(rad)


class ImageNetCompatibilityTests(unittest.TestCase):
    def test_only_legacy_batchnorm_counters_are_synthesized(self):
        from rsna_knee.models.resnet50 import map_imagenet_state
        from torchvision.models import resnet50
        expected=resnet50(weights=None).state_dict()
        old={k:v for k,v in expected.items() if not k.endswith('.num_batches_tracked')}
        mapped=map_imagenet_state(old,expected)
        self.assertEqual(set(mapped),set(expected))
        self.assertEqual(len(mapped)-len(old),53)
        self.assertTrue(all(v.item()==0 for k,v in mapped.items() if k.endswith('.num_batches_tracked')))
        old.pop('conv1.weight')
        with self.assertRaises(ValueError):map_imagenet_state(old,expected)
