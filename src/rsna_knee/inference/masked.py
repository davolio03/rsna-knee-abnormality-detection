"""Explicit, pinned inference contract for the completed masked-soft model.

The proven legacy dataset, recovery policy and CSV writer are reused unchanged.
No training or optimizer restoration occurs in this adapter.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from rsna_knee.inference.fold0 import (TARGETS, TestStudyDataset, write_submission, InferenceError,
                                      _exception_family, _safe_reason)
from rsna_knee.models.resnet18 import ResNet18StudyModel

EXPECTED_REPORT = {
    "status": "complete", "training_status": "training_completed_no_quality_claim",
    "completed_epochs": 5, "target_epochs": 5, "fold": 0, "seed": 42,
    "training_study_count": 2850, "validation_study_count": 1448,
    "model_kind": "resnet18_imagenet1k_v1", "freeze_backbone": True,
    "optimizer": "Adam", "balance_mode": "unweighted",
    "objective": "sum(mask * soft_bce_cell_loss) / observed_cell_count",
    "validation_metric_weighting": "unweighted_observed_cells",
    "manifest_sha256": "617ea1c267960b1e903b332a78388db3b6ab80d44374721df5a4c5e184e8961b",
    "folds_sha256": "b816c7902a5d0caa68a9e818d3f913943bc61cd2e6db65c4de069f5cd94e15f9",
    "pinned_folds_artifact_sha256": "b816c7902a5d0caa68a9e818d3f913943bc61cd2e6db65c4de069f5cd94e15f9",
    "gold_ids_sha256": "e0e584d475a7e8b0749ffcce70267ac19a2e2549338f98f50ed515f8fabce3a1",
    "source_csv_sha256": "96444cd3c7d396bae4a71454317d9cbe0689759b5278b30939781e04d0774537",
    "official_train_csv_sha256": "8ca2203c0e9d61c080c7a314c7cdb51c1b03a1d9eb4770819f7f34af53ef4e33",
    "train_series_csv_sha256": "573c1d80772bf41211c91b149c95677385a1c22d63f485c347f1b46c0177aef3",
    "prefilter_policy_sha256": "7a1b41d3d96c137859d83d78927fd020afba8ce4034cb0addbed67e730280662",
    "baseline_cohort_sha256": "e88c61c22a463737f16951b7c76db03834034aecd1b31dc61b4ce495080b2be2",
    "baseline_folds_sha256": "b82b376b0086201e3364b11a0203b62890e628442075a52ea4a9e09ee0e778ac",
    "weights_sha256": "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
    "weights_size": 46830571,
}
VALIDATION_COUNTS = (1347,1334,1379,1321,1182,1101,1217,1335,243,753,1130,853)


def _pin(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("artifact_pin_required")


def _file(path):
    path = Path(path)
    if not path.is_file() or any(item.is_symlink() for item in (path,*path.parents)):
        raise ValueError("source_mount_invalid")
    return path


def _hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def _finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def validate_masked_report(report):
    """Reject the old binary cohort, partial training, target/count drift and bad metrics."""
    if not isinstance(report,dict) or any(type(report.get(k)) is not type(v) or report[k] != v
                                          for k,v in EXPECTED_REPORT.items()):
        raise ValueError("source_report_provenance_mismatch")
    _pin(report.get("checkpoint_sha256"))
    history = report.get("epochs")
    if type(history) is not list or len(history) != 5:
        raise ValueError("source_report_provenance_mismatch")
    for epoch,item in enumerate(history,1):
        if (item.get("epoch") != epoch or type(item.get("epoch")) is not int
                or item.get("training_observed_cells") != 25740
                or item.get("training_objective_weighting") != "unweighted"
                or item.get("validation_objective_weighting") != "unweighted"
                or not _finite(item.get("training_masked_bce")) or item["training_masked_bce"] < 0):
            raise ValueError("source_report_provenance_mismatch")
        metric = item.get("validation",{})
        targets = metric.get("per_target")
        if (type(targets) is not list or len(targets) != 12
                or [entry.get("target") for entry in targets] != list(TARGETS)
                or [entry.get("observed_count") for entry in targets] != list(VALIDATION_COUNTS)
                or metric.get("observed_cell_count") != sum(VALIDATION_COUNTS)):
            raise ValueError("source_report_provenance_mismatch")
        for row in [metric,*targets]:
            if (not _finite(row.get("masked_bce")) or row["masked_bce"] < 0
                    or not _finite(row.get("brier")) or not 0 <= row["brier"] <= 1):
                raise ValueError("source_report_provenance_mismatch")
        for key in ("masked_bce","brier"):
            pooled = math.fsum(row[key]*count for row,count in zip(targets,VALIDATION_COUNTS))/sum(VALIDATION_COUNTS)
            if not math.isclose(metric[key],pooled,rel_tol=1e-12,abs_tol=1e-12):
                raise ValueError("source_report_provenance_mismatch")


def _model():
    from torchvision.models import resnet18
    backbone = resnet18(weights=None)
    backbone.fc = nn.Identity()
    return ResNet18StudyModel(backbone,slice_microbatch=2,checkpoint_trunk=False,freeze_backbone=True)


def load_masked_model(checkpoint, expected_sha256, report, expected_fingerprint, *, model_factory=None):
    """Verify SHA before weights-only load, then exact history/fingerprint/tensor shapes."""
    _pin(expected_sha256); _pin(expected_fingerprint)
    validate_masked_report(report)
    path = _file(checkpoint)
    if report["checkpoint_sha256"] != expected_sha256 or _hash(path) != expected_sha256:
        raise ValueError("checkpoint_invalid")
    try:
        value = torch.load(path,map_location='cpu',weights_only=True)
        if (type(value) is not dict or set(value) != {"schema","fingerprint","completed_epochs","history",
                "model_state","optimizer_state","rng_state","last_epoch_seconds"}
                or value["schema"] != "masked-soft-checkpoint-v1"
                or value["fingerprint"] != expected_fingerprint
                or type(value["completed_epochs"]) is not int or value["completed_epochs"] != 5
                or value["history"] != report["epochs"]
                or not _finite(value["last_epoch_seconds"]) or value["last_epoch_seconds"] <= 0):
            raise ValueError
        state = value["model_state"]
        if (not isinstance(state,dict) or not state or any(type(k) is not str or not isinstance(v,torch.Tensor)
                or not torch.isfinite(v).all() for k,v in state.items())):
            raise ValueError
        model = (model_factory or _model)()
        model.load_state_dict(state,strict=True)
        return model.eval()
    except Exception:
        raise ValueError("checkpoint_invalid") from None


def run_masked_inference(competition_root, source_root, report_sha256, checkpoint_sha256,
                         expected_fingerprint, output_dir, *, device='cuda', max_studies=None,
                         recovery_policy_version=1):
    """Explicit canary limit or full test set, retaining proven recovery-v1 semantics."""
    if type(recovery_policy_version) is not int or recovery_policy_version != 1:
        raise ValueError("inference_config_invalid")
    if device not in ('cpu','cuda') or (max_studies is not None and
            (type(max_studies) is not int or not 1 <= max_studies <= 100)):
        raise ValueError("inference_config_invalid")
    _pin(report_sha256)
    output = Path(output_dir)
    if output.exists() or any(item.is_symlink() for item in (output,*output.parents)):
        raise ValueError("output_must_be_new")
    started = time.perf_counter(); completed = 0; stage = 'checkpoint'; journal=[]
    def emit(event,**extra):
        print(json.dumps({'event':event,'stage':stage,'completed':completed,
                          'elapsed_seconds':time.perf_counter()-started,**extra},sort_keys=True),flush=True)
    try:
        report_path = _file(Path(source_root)/'training-report.json')
        if _hash(report_path) != report_sha256: raise ValueError('report_pin_mismatch')
        report = json.loads(report_path.read_text(encoding='utf-8'))
        model = load_masked_model(Path(source_root)/'checkpoint-epoch-05.pt',checkpoint_sha256,
                                  report,expected_fingerprint)
        torch.set_num_threads(1)
        if device == 'cuda' and not torch.cuda.is_available(): raise ValueError('cuda_required')
        model = model.to(device).eval()
        stage='input_metadata'
        competition=Path(competition_root)
        data=TestStudyDataset(competition/'test.csv',competition/'test_series.csv',competition/'test_series',
                              recovery_policy_version=1)
        total=len(data) if max_studies is None else min(max_studies,len(data))
        output.mkdir(parents=True,exist_ok=False)
        values=[]
        with torch.inference_mode():
            for index in range(total):
                stage='preprocess'
                emit('inference_stage',total=total,study_ordinal=index+1)
                try: images,masks=data[index]
                finally:
                    if data.last_recovery: journal.append(data.last_recovery)
                stage='model'
                logits=model([v.to(device) for v in images],[v.to(device) for v in masks])
                if logits.shape != (1,12) or not torch.isfinite(logits).all(): raise ValueError('prediction_invalid')
                values.append(torch.sigmoid(logits[0]).cpu().numpy())
                completed+=1
                emit('inference_progress',total=total,study_ordinal=index+1)
        stage='write'
        write_submission(output/'submission.csv',data.study_ids[:total],np.asarray(values))
        summary={'status':'complete','source_execution':363315909,'fold':0,'study_count':total,
                 'test_study_count':len(data),'canary':max_studies is not None,
                 'checkpoint_sha256':checkpoint_sha256,'report_sha256':report_sha256,
                 'checkpoint_fingerprint':expected_fingerprint,'recovery_policy_version':1,
                 'series_skipped':sum(item['series_skipped'] for item in journal),
                 'slice_skipped':sum(item['slice_skipped'] for item in journal),
                 'pooling':'equal_mean_slice_then_series_logits_then_sigmoid',
                 'output_size':256,'max_slices':8,'slice_microbatch':2}
        (output/'inference-report.json').write_text(json.dumps(summary,sort_keys=True)+'\n',encoding='utf-8')
        emit('inference_completed',**summary)
        return summary
    except Exception as error:
        if output.exists():
            csv=output/'submission.csv'
            if csv.is_file() and not csv.is_symlink(): csv.unlink()
        emit('inference_failed',reason_code=_safe_reason(error),exception_family=_exception_family(error))
        raise InferenceError(_safe_reason(error),study_ordinal=completed+1,
                             exception_family=_exception_family(error)) from None
    finally:
        if output.is_dir() and not output.is_symlink():
            payload={'version':1,'studies':journal}
            try:
                (output/'recovery-journal.json').write_text(json.dumps(payload,sort_keys=True)+'\n',encoding='utf-8')
            except OSError:
                emit('diagnostics_storage_failed',reason_code='unexpected_error',exception_family='io')
