"""Synthetic paired-run selection, atomic continuation and leakage boundaries."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from torch import nn

from rsna_knee.training.resnet50 import RecipeConfig, run_recipe_training, select_benchmark_indices


def metadata():
    names=('ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture')
    records=[{'study_id':str(i),'series':[{'anatomical_plane':'Axial'}]*(i+1),
              'soft_targets':dict.fromkeys(names,.2 if i%2 else .8),'observed_mask':dict.fromkeys(names,True)} for i in range(4)]
    return records,{str(i):i%2 for i in range(4)}


class Data:
    def __init__(self,config,partition):self.partition=partition
    def __len__(self):return 2
    def __getitem__(self,index):
        image=torch.full((1,1,8,8),.25+index*.1);mask=torch.ones_like(image,dtype=torch.bool)
        return [image],[mask],torch.full((12,),.2+index*.3),torch.ones(12,dtype=torch.bool),index


class Toy(nn.Module):
    def __init__(self):super().__init__();self.head=nn.Parameter(torch.ones(12)*.1);self.trunk=nn.Parameter(torch.ones(12)*.1)
    def forward(self,images,masks):return (self.head+self.trunk*images[0].mean()).reshape(1,12)
    def optimizer_groups(self,*,head_lr,backbone_lr):return [{'params':[self.head],'lr':head_lr,'name':'head'},{'params':[self.trunk],'lr':backbone_lr,'name':'layer4'}]


class RecipeTrainingTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_selection_spans_series_quantiles_is_deterministic(self):
        records=[{'series':[{'anatomical_plane':('Axial','Coronal','Sagittal')[i%3]}]*(i+1),'observed_mask':{'ACL':True}} for i in range(30)]
        selected=select_benchmark_indices(records,8)
        self.assertEqual(selected,select_benchmark_indices(records,8));self.assertEqual(len(set(selected)),8)
        self.assertIn(0,selected);self.assertIn(29,selected)
        self.assertLess(max(b-a for a,b in zip(sorted(selected),sorted(selected)[1:])),12)

    def test_continuation_matches_uninterrupted_and_rejects_benchmark_checkpoint(self):
        with TemporaryDirectory() as td,patch('rsna_knee.training.resnet50._metadata',return_value=metadata()):
            root=Path(td);source=root/'source';source.mkdir();(source/'fixture.py').write_text('version=1')
            def cfg(name,**kw):return RecipeConfig(arm='imagenet',weights_path=root/'unused',output_dir=root/name,
                image_root=root,device='cpu',source_root=source,**kw)
            whole=run_recipe_training(cfg('whole',max_epochs_per_run=5),execute=True,dataset_factory=Data,model_factory=lambda c:Toy())
            first=run_recipe_training(cfg('first',max_epochs_per_run=2),execute=True,dataset_factory=Data,model_factory=lambda c:Toy())
            resumed=run_recipe_training(cfg('resumed',max_epochs_per_run=3,resume_checkpoint=Path(first['checkpoint_path']),resume_sha256=first['checkpoint_sha256']),execute=True,dataset_factory=Data,model_factory=lambda c:Toy())
            a=torch.load(whole['checkpoint_path'],weights_only=True);b=torch.load(resumed['checkpoint_path'],weights_only=True)
            for key in a['model_state']:self.assertTrue(torch.equal(a['model_state'][key],b['model_state'][key]))
            without_time=lambda rows:[{k:v for k,v in row.items() if k!='phase_seconds'} for row in rows]
            self.assertEqual(without_time(a['history']),without_time(b['history']))
            self.assertEqual(resumed['epochs'][:2],first['epochs']);self.assertEqual(resumed['completed_epochs'],5)
            bad=cfg('bad',mode='benchmark',resume_checkpoint=Path(first['checkpoint_path']),resume_sha256=first['checkpoint_sha256'])
            with self.assertRaises(ValueError):run_recipe_training(bad,execute=True,dataset_factory=Data,model_factory=lambda c:Toy())
            self.assertFalse((root/'bad').exists())

    def test_plan_and_bad_resume_hash_do_not_construct_model_or_dataset(self):
        with TemporaryDirectory() as td,patch('rsna_knee.training.resnet50._metadata',return_value=metadata()):
            root=Path(td);source=root/'source';source.mkdir();(source/'fixture.py').write_text('x=1')
            config=RecipeConfig(arm='imagenet',weights_path=root/'unused',output_dir=root/'out',image_root=root,device='cpu',source_root=source)
            with patch('rsna_knee.training.resnet50.load_resnet50_recipe') as factory:
                self.assertEqual(run_recipe_training(config)['status'],'planned');factory.assert_not_called()
            config=RecipeConfig(**{**config.__dict__,'max_epochs_per_run':2})
            first=run_recipe_training(config,execute=True,dataset_factory=Data,model_factory=lambda c:Toy())
            bad=RecipeConfig(arm='imagenet',weights_path=root/'unused',output_dir=root/'bad',image_root=root,device='cpu',source_root=source,
                resume_checkpoint=Path(first['checkpoint_path']),resume_sha256='a'*64)
            with patch('rsna_knee.training.resnet50.load_resnet50_recipe') as factory:
                with self.assertRaises(ValueError):run_recipe_training(bad,execute=True)
                factory.assert_not_called()

    def test_benchmark_reports_phase_memory_and_conservative_forecast(self):
        with TemporaryDirectory() as td,patch('rsna_knee.training.resnet50._metadata',return_value=metadata()):
            root=Path(td);source=root/'source';source.mkdir();(source/'fixture.py').write_text('x=1')
            cfg=RecipeConfig('imagenet',root/'unused',root/'out',root,mode='benchmark',device='cpu',source_root=source)
            report=run_recipe_training(cfg,execute=True,dataset_factory=Data,model_factory=lambda c:Toy())
            self.assertEqual(set(report['epochs'][0]['phase_cuda_peak_bytes']),{'train','validation'})
            self.assertGreaterEqual(report['conservative_epoch_seconds'],report['projected_full_epoch_seconds']*1.5)
            self.assertEqual(report['completed_epochs'],1)

    def test_altered_source_and_optimizer_are_rejected_before_output(self):
        with TemporaryDirectory() as td,patch('rsna_knee.training.resnet50._metadata',return_value=metadata()):
            root=Path(td);source=root/'source';source.mkdir();fixture=source/'fixture.py';fixture.write_text('x=1')
            def cfg(name,**kw):return RecipeConfig('imagenet',root/'unused',root/name,root,device='cpu',source_root=source,**kw)
            first=run_recipe_training(cfg('first'),execute=True,dataset_factory=Data,model_factory=lambda c:Toy())
            path=Path(first['checkpoint_path']);payload=torch.load(path,weights_only=True)
            fixture.write_text('x=2')
            with self.assertRaises(ValueError):run_recipe_training(cfg('badsource',resume_checkpoint=path,resume_sha256=first['checkpoint_sha256']),execute=True,dataset_factory=Data,model_factory=lambda c:Toy())
            self.assertFalse((root/'badsource').exists());fixture.write_text('x=1')
            payload['optimizer_state']['state'][0]['step']=torch.tensor(-1.)
            altered=root/'altered.pt';torch.save(payload,altered)
            with self.assertRaises(ValueError):run_recipe_training(cfg('badadam',resume_checkpoint=altered,resume_sha256=hashlib.sha256(altered.read_bytes()).hexdigest()),execute=True,dataset_factory=Data,model_factory=lambda c:Toy())
            self.assertFalse((root/'badadam').exists())

    def test_nonfinite_history_is_rejected_before_model_or_output(self):
        with TemporaryDirectory() as td,patch('rsna_knee.training.resnet50._metadata',return_value=metadata()):
            root=Path(td);source=root/'source';source.mkdir();(source/'fixture.py').write_text('x=1')
            def cfg(name,**kw):return RecipeConfig('imagenet',root/'unused',root/name,root,device='cpu',source_root=source,**kw)
            first=run_recipe_training(cfg('first'),execute=True,dataset_factory=Data,model_factory=lambda c:Toy())
            path=root/'altered.pt';payload=torch.load(first['checkpoint_path'],weights_only=True)
            payload['history'][0]['validation']['brier']=float('nan');torch.save(payload,path)
            with patch('rsna_knee.training.resnet50.load_resnet50_recipe') as factory:
                with self.assertRaises(ValueError):run_recipe_training(cfg('bad',resume_checkpoint=path,resume_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),execute=True)
                factory.assert_not_called()
            self.assertFalse((root/'bad').exists())
