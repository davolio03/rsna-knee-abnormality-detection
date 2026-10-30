"""Pinned ResNet50 pretraining recipes with a matched last-block/head policy."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

from rsna_knee.models.resnet18 import ResNet18StudyModel

WEIGHT_PINS = {
    'imagenet': {'sha256':'0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a',
                 'size':102530333,'filename':'resnet50-imagenet1k-v1.pth',
                 'normalization':'imagenet_mean_std',
                 'url':'https://download.pytorch.org/models/resnet50-0676ba61.pth'},
    'radimagenet': {'sha256':'08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734',
                    'size':94345733,'filename':'resnet50-radimagenet.pt',
                    'normalization':'radimagenet_2x_minus_1',
                    'url':'https://github.com/BMEII-AI/RadImageNet'},
}
BN_POLICY='running_statistics_and_affine_frozen'
TRAINABLE_POLICY='layer4_non_batchnorm_and_seeded_2048_to_12_head'


def map_radimagenet_state(state):
    mapping={'0':'conv1','1':'bn1','4':'layer1','5':'layer2','6':'layer3','7':'layer4'}
    if not isinstance(state,dict) or len(state)!=318:
        raise ValueError('radimagenet_state_invalid')
    converted={}
    for key,value in state.items():
        if type(key) is not str or not isinstance(value,torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError('radimagenet_state_invalid')
        parts=key.split('.',2)
        if len(parts)!=3 or parts[0]!='backbone' or parts[1] not in mapping:
            raise ValueError('radimagenet_state_invalid')
        name=mapping[parts[1]]+'.'+parts[2]
        if name in converted:raise ValueError('radimagenet_state_invalid')
        converted[name]=value
    return converted


def map_imagenet_state(state, expected):
    """The pinned 2017-format V1 weights omit exactly 53 unused BN counters."""
    missing = set(expected) - set(state)
    counters = {k for k in expected if k.endswith('.num_batches_tracked')}
    if set(state) - set(expected) or missing not in (set(), counters) or len(counters) != 53:
        raise ValueError('imagenet_state_invalid')
    return {**state, **{k: torch.zeros_like(expected[k]) for k in missing}}


class ResNet50StudyModel(ResNet18StudyModel):
    """Reuse checked masked pooling; adapt only topology, normalization and gradients."""
    def __init__(self,backbone,*,arm,seed=42,slice_microbatch=2):
        if arm not in WEIGHT_PINS or type(seed) is not int:
            raise ValueError('recipe_invalid')
        backbone.fc=nn.Identity()
        super().__init__(backbone,slice_microbatch=slice_microbatch,checkpoint_trunk=True,freeze_backbone=False)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.classifier=nn.Linear(2048,12)
        self.arm=arm
        if arm=='radimagenet':
            self.imagenet_mean=torch.full((1,3,1,1),.5)
            self.imagenet_std=torch.full((1,3,1,1),.5)
        for name,parameter in self.backbone.named_parameters():
            parameter.requires_grad_(name.startswith('layer4.'))
        for module in self.backbone.modules():
            if isinstance(module,nn.modules.batchnorm._BatchNorm):
                module.eval()
                for parameter in module.parameters():parameter.requires_grad_(False)
        self.train(self.training)

    def backbone_forward(self,rgb):
        # Input-to-layer3 graph is unnecessary: these weights never train.
        with torch.no_grad():
            value=self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(rgb))))
            value=self.backbone.layer3(self.backbone.layer2(self.backbone.layer1(value)))
        return self.backbone.layer4(value)

    def optimizer_groups(self,*,head_lr=1e-3,backbone_lr=1e-5):
        return [{'params':list(self.classifier.parameters()),'lr':head_lr,'name':'head'},
                {'params':[p for p in self.backbone.layer4.parameters() if p.requires_grad],
                 'lr':backbone_lr,'name':'layer4'}]


def load_resnet50_recipe(arm,weights_path,*,seed=42,slice_microbatch=2):
    """Never download; verify published artifact bytes before weights-only loading."""
    if arm not in WEIGHT_PINS:raise ValueError('recipe_invalid')
    pin=WEIGHT_PINS[arm];path=Path(weights_path)
    if not path.is_file() or any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('weights_unavailable')
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if path.stat().st_size!=pin['size'] or digest!=pin['sha256']:
        raise ValueError('weights_pin_mismatch')
    try:
        state=torch.load(path,map_location='cpu',weights_only=True)
        if not isinstance(state,dict) or any(type(k) is not str or not isinstance(v,torch.Tensor)
                or not torch.isfinite(v).all() for k,v in state.items()):raise ValueError
        from torchvision.models import resnet50
        backbone=resnet50(weights=None)
        if arm=='radimagenet':
            backbone.fc=nn.Identity();state=map_radimagenet_state(state)
        # Dtypes as well as names/shapes must match; load_state_dict can otherwise cast.
        expected=backbone.state_dict()
        if arm=='imagenet':state=map_imagenet_state(state,expected)
        if set(expected)!=set(state) or any(expected[k].shape!=state[k].shape or expected[k].dtype!=state[k].dtype for k in expected):
            raise ValueError
        backbone.load_state_dict(state,strict=True)
        return ResNet50StudyModel(backbone,arm=arm,seed=seed,slice_microbatch=slice_microbatch)
    except Exception:
        raise ValueError('weights_state_invalid') from None
