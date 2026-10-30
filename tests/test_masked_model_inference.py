"""Pinned masked checkpoint loading and synthetic inference contracts."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from torch import nn

from rsna_knee.inference.masked import (EXPECTED_REPORT, validate_masked_report,
    load_masked_model, run_masked_inference)
from rsna_knee.inference.fold0 import TARGETS


def report_fixture():
    value = copy.deepcopy(EXPECTED_REPORT)
    value["checkpoint_sha256"] = "a" * 64
    value["runtime_versions"] = {"python": "3.13.15", "torch": "2.11.0+cu128", "numpy": "2.1.3", "cuda": "12.8"}
    metric = {"observed_cell_count": 13195, "masked_bce": .5, "brier": .1,
              "per_target": [{"target": name, "observed_count": count, "masked_bce": .5, "brier": .1}
                             for name, count in zip(TARGETS, (1347,1334,1379,1321,1182,1101,1217,1335,243,753,1130,853))]}
    value["epochs"] = [{"epoch": i, "training_masked_bce": .5, "training_observed_cells": 25740,
                        "training_objective_weighting": "unweighted", "validation_objective_weighting": "unweighted",
                        "validation": copy.deepcopy(metric)} for i in range(1, 6)]
    return value


class ToyModel(nn.Module):
    def __init__(self): super().__init__(); self.weight = nn.Parameter(torch.zeros(12))
    def forward(self, images, masks): return self.weight.reshape(1,12)


class MaskedInferenceTests(unittest.TestCase):
    def test_real_resnet_checkpoint_roundtrip_preserves_logits(self):
        from rsna_knee.inference.masked import _model
        torch.set_num_threads(1)
        original=_model().eval(); image=torch.rand(1,1,32,32); mask=torch.ones_like(image,dtype=torch.bool)
        report=report_fixture()
        with TemporaryDirectory() as td:
            path=Path(td)/'checkpoint.pt'
            torch.save({'schema':'masked-soft-checkpoint-v1','fingerprint':'f'*64,'completed_epochs':5,
                'history':report['epochs'],'model_state':original.state_dict(),'optimizer_state':{},
                'rng_state':{},'last_epoch_seconds':10.},path)
            sha=hashlib.sha256(path.read_bytes()).hexdigest();report['checkpoint_sha256']=sha
            restored=load_masked_model(path,sha,report,'f'*64)
            with torch.inference_mode():
                self.assertTrue(torch.equal(original([image],[mask]),restored([image],[mask])))

    def test_private_runtime_failure_is_sanitized_and_no_csv_remains(self):
        class Data:
            study_ids=('secret-study',);series_count=1;last_recovery=None
            def __init__(self,*a,**kw):pass
            def __len__(self):return 1
            def __getitem__(self,index):raise RuntimeError('SECRET_REPORT_PRIVATE_PATH')
        with TemporaryDirectory() as td:
            root=Path(td);source=root/'source';source.mkdir();raw=json.dumps(report_fixture()).encode()
            (source/'training-report.json').write_bytes(raw);captured=io.StringIO()
            with patch('rsna_knee.inference.masked.TestStudyDataset',Data),patch('rsna_knee.inference.masked.load_masked_model',return_value=ToyModel()),contextlib.redirect_stdout(captured):
                with self.assertRaises(Exception) as caught:
                    run_masked_inference(root,source,hashlib.sha256(raw).hexdigest(),'a'*64,'f'*64,root/'out',device='cpu')
            self.assertNotIn('SECRET',str(caught.exception))
            self.assertNotIn('SECRET',captured.getvalue())
            self.assertFalse((root/'out/submission.csv').exists())

    def test_report_rejects_partial_wrong_provenance_and_bad_history(self):
        validate_masked_report(report_fixture())
        for key, value in (("completed_epochs",4),("training_study_count",333),("manifest_sha256","b"*64)):
            bad=report_fixture(); bad[key]=value
            with self.assertRaises(ValueError): validate_masked_report(bad)
        bad=report_fixture(); bad["epochs"][-1]["validation"]["per_target"].reverse()
        with self.assertRaises(ValueError): validate_masked_report(bad)

    def test_sha_before_torch_load_and_strict_state_history_fingerprint(self):
        with TemporaryDirectory() as td:
            p=Path(td)/"checkpoint.pt"; report=report_fixture()
            checkpoint={"schema":"masked-soft-checkpoint-v1","fingerprint":"f"*64,
                        "completed_epochs":5,"history":report["epochs"], "model_state":ToyModel().state_dict(),
                        "optimizer_state":{},"rng_state":{},"last_epoch_seconds":10.}
            torch.save(checkpoint,p); sha=hashlib.sha256(p.read_bytes()).hexdigest();report["checkpoint_sha256"]=sha
            with patch("torch.load") as load:
                with self.assertRaises(ValueError): load_masked_model(p,"b"*64,report,"f"*64, model_factory=ToyModel)
                load.assert_not_called()
            model=load_masked_model(p,sha,report,"f"*64,model_factory=ToyModel)
            self.assertTrue(torch.equal(model.weight,torch.zeros(12)))
            for key,value in (("fingerprint","e"*64),("history",[]),("model_state",{"weight":torch.full((12,),float('nan'))})):
                bad=copy.deepcopy(checkpoint);bad[key]=value;torch.save(bad,p)
                digest=hashlib.sha256(p.read_bytes()).hexdigest();report["checkpoint_sha256"]=digest
                with self.assertRaises(ValueError):load_masked_model(p,digest,report,"f"*64,model_factory=ToyModel)

    def test_canary_limits_rows_and_full_preserves_all_and_safe_logs(self):
        class Data:
            study_ids=("private-study-a","private-study-b")
            series_count=2
            last_recovery={"series_skipped":0,"slice_skipped":0,"reason_counts":{}}
            def __init__(self,*args,**kwargs):pass
            def __len__(self):return 2
            def __getitem__(self,index):return [torch.zeros(1,1,8,8)],[torch.ones(1,1,8,8,dtype=torch.bool)]
        with TemporaryDirectory() as td:
            root=Path(td);source=root/'source';source.mkdir();raw=json.dumps(report_fixture()).encode();(source/'training-report.json').write_bytes(raw)
            for limit in (1,None):
                out=root/str(limit);captured=io.StringIO()
                with patch('rsna_knee.inference.masked.TestStudyDataset',Data), patch('rsna_knee.inference.masked.load_masked_model',return_value=ToyModel()),contextlib.redirect_stdout(captured):
                    result=run_masked_inference(root,source,hashlib.sha256(raw).hexdigest(),'a'*64,'f'*64,out,device='cpu',max_studies=limit)
                self.assertEqual(result['study_count'],limit or 2)
                self.assertNotIn('private-study',captured.getvalue())
                self.assertTrue((out/'submission.csv').is_file())
                self.assertTrue((out/'recovery-journal.json').is_file())
            with self.assertRaises(ValueError):run_masked_inference(root,source,'b'*64,'a'*64,'f'*64,root/'bad',device='cpu')
