"""Separate, resumable paired ResNet50 recipes using the existing masked data contract."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from rsna_knee.data.masked_soft_dataset import (FullCohortSoftDataset, load_verified_soft_artifacts,
                                              MANIFEST_SHA256,FOLDS_ARTIFACT_SHA256)
from rsna_knee.labels.full_cohort_prefilter import TARGET_COLUMNS
from rsna_knee.models.resnet50 import WEIGHT_PINS, BN_POLICY, TRAINABLE_POLICY, load_resnet50_recipe
from rsna_knee.training.masked_soft import (masked_soft_bce,MaskedSoftMetricAccumulator,
    _write_checkpoint,_capture_rng_state,_restore_rng_state,_runtime_versions,_validate_finite_tensors)
from rsna_knee.training.baseline import _seed_everything


@dataclass(frozen=True)
class RecipeConfig:
    arm:str
    weights_path:Path
    output_dir:Path
    image_root:Path
    manifest_path:Path=Path('data/private/full-cohort-prefilter/run01/full-cohort-soft-label-manifest.json')
    folds_path:Path=Path('data/private/full-cohort-prefilter/run01/full-cohort-grouped-folds.json')
    mode:str='train'
    device:str='cuda'
    source_root:Path|None=None
    max_epochs_per_run:int=2
    max_runtime_seconds:int=28800
    resume_checkpoint:Path|None=None
    resume_sha256:str|None=None
    head_lr:float=1e-3
    backbone_lr:float=1e-5

    def validate(self):
        if (self.arm not in WEIGHT_PINS or self.mode not in ('train','benchmark') or self.device not in ('cpu','cuda')
                or type(self.max_epochs_per_run) is not int or not 1<=self.max_epochs_per_run<=5
                or type(self.max_runtime_seconds) is not int or not 60<=self.max_runtime_seconds<=28800
                or self.head_lr!=1e-3 or self.backbone_lr!=1e-5):raise ValueError('recipe_config_invalid')
        if self.resume_checkpoint is None:
            if self.resume_sha256 is not None:raise ValueError('resume_pin_invalid')
        elif (self.mode=='benchmark' or type(self.resume_sha256) is not str or len(self.resume_sha256)!=64
                or any(c not in '0123456789abcdef' for c in self.resume_sha256)):
            raise ValueError('resume_pin_invalid')


def _metadata(config):
    records,assignment,_=load_verified_soft_artifacts(config.manifest_path,config.folds_path)
    return records,assignment


def select_benchmark_indices(records,count=8):
    """Deterministic series-count quantiles, including shortest and longest studies."""
    if type(count) is not int or count<2 or not records:raise ValueError('selection_invalid')
    ranked=sorted(range(len(records)),key=lambda i:(len(records[i]['series']),i))
    size=min(count,len(records))
    return sorted(ranked[round(i*(len(ranked)-1)/(size-1))] for i in range(size)) if size>1 else ranked


def _fingerprint(config,selection):
    root=Path(config.source_root or Path(__file__).resolve().parents[2])
    files=sorted(root.rglob('*.py'))
    if not files or root.is_symlink() or any(p.is_symlink() for p in files):raise ValueError('source_invalid')
    value={'schema':'resnet50-recipe-checkpoint-v1','arm':config.arm,'mode':config.mode,
           'normalization':WEIGHT_PINS[config.arm]['normalization'],'weights':WEIGHT_PINS[config.arm],
           'manifest':MANIFEST_SHA256,'folds':FOLDS_ARTIFACT_SHA256,'fold':0,'seed':42,
           'output_size':224,'max_slices':8,'pooling':'mean_slice_mean_series_logits',
           'head_lr':config.head_lr,'layer4_lr':config.backbone_lr,'bn':BN_POLICY,'trainable':TRAINABLE_POLICY,
           'epochs':5,'selection':selection,'runtime':_runtime_versions(),
           'source':{p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _validate_history(history):
    """Reject malformed/nonfinite reporting state before constructing a resumed model."""
    def number(value, minimum=0., maximum=float('inf')):
        return type(value) in (int,float) and math.isfinite(value) and minimum<=value<=maximum
    for row in history:
        if not isinstance(row,dict) or set(row)!={'epoch','training_masked_bce','validation','phase_seconds','phase_cuda_peak_bytes'}:
            raise ValueError('checkpoint_history_invalid')
        if not number(row['training_masked_bce']):raise ValueError('checkpoint_history_invalid')
        times=row['phase_seconds'];peaks=row['phase_cuda_peak_bytes'];metrics=row['validation']
        if (not isinstance(times,dict) or set(times)!={'train','validation'}
                or not all(number(v) for v in times.values()) or not isinstance(peaks,dict)
                or set(peaks)!={'train','validation'}):raise ValueError('checkpoint_history_invalid')
        for peak in peaks.values():
            if peak is not None and (not isinstance(peak,dict) or set(peak)!={'allocated','reserved'}
                    or not all(type(v) is int and v>=0 for v in peak.values())):raise ValueError('checkpoint_history_invalid')
        if (not isinstance(metrics,dict) or set(metrics)!={'observed_cell_count','masked_bce','brier','per_target'}
                or type(metrics['observed_cell_count']) is not int or metrics['observed_cell_count']<=0
                or not number(metrics['masked_bce']) or not number(metrics['brier'],maximum=1)
                or not isinstance(metrics['per_target'],list) or len(metrics['per_target'])!=len(TARGET_COLUMNS)):
            raise ValueError('checkpoint_history_invalid')
        counts=[]
        for target,name in zip(metrics['per_target'],TARGET_COLUMNS):
            if (not isinstance(target,dict) or set(target)!={'target','observed_count','masked_bce','brier'}
                    or target['target']!=name or type(target['observed_count']) is not int or target['observed_count']<0):
                raise ValueError('checkpoint_history_invalid')
            count=target['observed_count'];counts.append(count)
            if ((count and (not number(target['masked_bce']) or not number(target['brier'],maximum=1)))
                    or (not count and (target['masked_bce'] is not None or target['brier'] is not None))):
                raise ValueError('checkpoint_history_invalid')
        if sum(counts)!=metrics['observed_cell_count']:raise ValueError('checkpoint_history_invalid')


def _read_resume(config,fingerprint):
    path=Path(config.resume_checkpoint)
    if not path.is_file() or any(p.is_symlink() for p in (path,*path.parents)):raise ValueError('checkpoint_unavailable')
    with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    if digest!=config.resume_sha256:raise ValueError('checkpoint_sha_mismatch')
    try:value=torch.load(path,map_location='cpu',weights_only=True)
    except Exception:raise ValueError('checkpoint_unavailable') from None
    if (not isinstance(value,dict) or set(value)!={'schema','fingerprint','completed_epochs','history','model_state',
            'optimizer_state','rng_state','last_epoch_seconds'} or value['schema']!='resnet50-recipe-checkpoint-v1'
            or value['fingerprint']!=fingerprint or type(value['completed_epochs']) is not int
            or not 1<=value['completed_epochs']<5 or not isinstance(value['history'],list)
            or len(value['history'])!=value['completed_epochs']
            or any(not isinstance(row,dict) for row in value['history'])
            or [row.get('epoch') for row in value['history']]!=list(range(1,value['completed_epochs']+1))
            or type(value['last_epoch_seconds']) not in (int,float) or not math.isfinite(value['last_epoch_seconds'])
            or value['last_epoch_seconds']<=0):raise ValueError('checkpoint_incompatible')
    _validate_history(value['history'])
    return value


def _validate_optimizer(optimizer,config):
    if len(optimizer.param_groups)!=2:raise ValueError('checkpoint_optimizer_invalid')
    for group,name,lr in zip(optimizer.param_groups,('head','layer4'),(config.head_lr,config.backbone_lr)):
        if (group.get('name')!=name or group.get('lr')!=lr or group.get('betas')!=(.9,.999)
                or group.get('eps')!=1e-8 or group.get('weight_decay')!=0 or group.get('amsgrad') is not False):
            raise ValueError('checkpoint_optimizer_invalid')
        for parameter in group['params']:
            state=optimizer.state.get(parameter)
            if (not isinstance(state,dict) or set(state)!={'step','exp_avg','exp_avg_sq'}
                    or not _validate_finite_tensors(state)
                    or not isinstance(state['step'],torch.Tensor) or state['step'].numel()!=1
                    or state['step'].item()<0 or state['step'].item()!=int(state['step'].item())
                    or state['exp_avg'].dtype!=parameter.dtype or state['exp_avg_sq'].dtype!=parameter.dtype
                    or state['exp_avg'].shape!=parameter.shape or state['exp_avg_sq'].shape!=parameter.shape
                    or (state['exp_avg_sq']<0).any()):raise ValueError('checkpoint_optimizer_invalid')


def _constant_metrics(train,validation):
    probabilities=[]
    for name in TARGET_COLUMNS:
        observed=[row['soft_targets'][name] for row in train if row['observed_mask'][name]]
        probabilities.append(min(1-1e-7,max(1e-7,sum(observed)/len(observed))))
    logits=torch.logit(torch.tensor(probabilities,dtype=torch.float64)).reshape(1,-1)
    metrics=MaskedSoftMetricAccumulator(TARGET_COLUMNS)
    for row in validation:
        targets=torch.tensor([[row['soft_targets'][name] for name in TARGET_COLUMNS]],dtype=torch.float64)
        observed=torch.tensor([[row['observed_mask'][name] for name in TARGET_COLUMNS]],dtype=torch.bool)
        metrics.update(logits,targets,observed)
    return metrics.compute()


def run_recipe_training(config,*,execute=False,dataset_factory=None,model_factory=None):
    config.validate()
    records,assignments=_metadata(config)
    train=[row for row in records if assignments[row['study_id']]!=0]
    validation=[row for row in records if assignments[row['study_id']]==0]
    train_indices=select_benchmark_indices(train) if config.mode=='benchmark' else list(range(len(train)))
    valid_indices=select_benchmark_indices(validation) if config.mode=='benchmark' else list(range(len(validation)))
    selected={'train':[train[i]['study_id'] for i in train_indices],
              'validation':[validation[i]['study_id'] for i in valid_indices]}
    selection=hashlib.sha256(json.dumps(selected,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    selected_series={'train':sum(len(train[i]['series']) for i in train_indices),
                     'validation':sum(len(validation[i]['series']) for i in valid_indices)}
    plan={'status':'planned','arm':config.arm,'mode':config.mode,'fold':0,'seed':42,'target_epochs':5,
          'training_study_count':len(train),'validation_study_count':len(validation),
          'selected_training_studies':len(train_indices),'selected_validation_studies':len(valid_indices),
          'selected_series':selected_series,'selection_sha256':selection,'weights_sha256':WEIGHT_PINS[config.arm]['sha256'],
          'normalization':WEIGHT_PINS[config.arm]['normalization'],'batchnorm_policy':BN_POLICY,
          'trainable_policy':TRAINABLE_POLICY,'head_lr':config.head_lr,'backbone_lr':config.backbone_lr,
          'output_size':224,'max_slices':8,'objective':'unweighted_observed_soft_bce',
          'comparison_scope':'pretraining_recipes_not_initialization_only'}
    if not execute:return plan
    output=Path(config.output_dir)
    if output.exists() or any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('output_must_be_new')
    fingerprint=_fingerprint(config,selection)
    checkpoint=_read_resume(config,fingerprint) if config.resume_checkpoint is not None else None
    if config.device=='cuda' and not torch.cuda.is_available():raise ValueError('cuda_required')
    _seed_everything(42,config.device)
    model=(model_factory(config) if model_factory else load_resnet50_recipe(config.arm,config.weights_path)).to(config.device)
    optimizer=torch.optim.Adam(model.optimizer_groups(head_lr=config.head_lr,backbone_lr=config.backbone_lr))
    history=[];completed=0;last_seconds=0.
    if checkpoint:
        expected=model.state_dict();saved=checkpoint['model_state']
        if (not isinstance(saved,dict) or set(saved)!=set(expected) or not _validate_finite_tensors(saved)
                or any(not isinstance(saved[k],torch.Tensor) or saved[k].shape!=expected[k].shape
                       or saved[k].dtype!=expected[k].dtype for k in expected)):
            raise ValueError('checkpoint_model_invalid')
        model.load_state_dict(checkpoint['model_state'],strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state']);_validate_optimizer(optimizer,config)
        _restore_rng_state(checkpoint['rng_state'],device=config.device)
        history=checkpoint['history'];completed=checkpoint['completed_epochs'];last_seconds=checkpoint['last_epoch_seconds']
    factory=dataset_factory or (lambda cfg,part:FullCohortSoftDataset(cfg.manifest_path,cfg.folds_path,cfg.image_root,
                                 partition=part,fold=0,output_size=224,max_slices=8))
    training_data=factory(config,'train');validation_data=factory(config,'validation')
    if len(training_data)!=len(train) or len(validation_data)!=len(validation):raise ValueError('dataset_count_mismatch')
    output.mkdir(parents=True,exist_ok=False)
    if config.device=='cuda':torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter();run_epochs=0;latest=Path(config.resume_checkpoint) if checkpoint else None
    digest=config.resume_sha256
    def sync():
        if config.device=='cuda':torch.cuda.synchronize()
    def emit(event,**kw):print(json.dumps({'event':event,'arm':config.arm,'mode':config.mode,
        'selection_sha256':selection,**kw},sort_keys=True),flush=True)
    emit('recipe_started',completed_epochs=completed,target_epochs=5,training_studies=len(train_indices),validation_studies=len(valid_indices))
    limit=1 if config.mode=='benchmark' else 5
    try:
        for epoch in range(completed,limit):
            if run_epochs>=config.max_epochs_per_run:break
            if completed and time.perf_counter()-started+1.5*last_seconds>=config.max_runtime_seconds:break
            epoch_start=time.perf_counter();phase_times={};phase_peaks={};training_sum=0.;training_cells=0
            for phase,data,indices in [('train',training_data,train_indices),('validation',validation_data,valid_indices)]:
                if config.device=='cuda':torch.cuda.reset_peak_memory_stats()
                model.train(phase=='train');metrics=MaskedSoftMetricAccumulator(TARGET_COLUMNS);sync();phase_start=time.perf_counter()
                for ordinal,index in enumerate(indices,1):
                    images,masks,targets,observed,_=data[index]
                    images=[v.to(config.device,dtype=torch.float32) for v in images]
                    masks=[v.to(config.device,dtype=torch.bool) for v in masks]
                    targets=targets.to(config.device,dtype=torch.float32).reshape(1,-1)
                    observed=observed.to(config.device,dtype=torch.bool).reshape(1,-1)
                    with torch.set_grad_enabled(phase=='train'):
                        logits=model(images,masks)
                        loss=masked_soft_bce(logits,targets,observed)
                        if not torch.isfinite(loss):raise ValueError('loss_nonfinite')
                        if phase=='train':
                            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
                            cells=int(observed.sum().item());training_sum+=float(loss.detach())*cells;training_cells+=cells
                    metrics.update(logits.detach(),targets,observed)
                    if ordinal%100==0 or config.mode=='benchmark':emit('recipe_progress',epoch=epoch+1,partition=phase,completed=ordinal,total=len(indices))
                sync();phase_times[phase]=time.perf_counter()-phase_start
                phase_peaks[phase]={'allocated':torch.cuda.max_memory_allocated(),'reserved':torch.cuda.max_memory_reserved()} if config.device=='cuda' else None
                if phase=='validation':validation_metrics=metrics.compute()
            last_seconds=time.perf_counter()-epoch_start;completed=epoch+1;run_epochs+=1
            history.append({'epoch':completed,'training_masked_bce':training_sum/training_cells,
                            'validation':validation_metrics,'phase_seconds':phase_times,'phase_cuda_peak_bytes':phase_peaks})
            latest=output/f'checkpoint-epoch-{completed:02d}.pt'
            payload={'schema':'resnet50-recipe-checkpoint-v1','fingerprint':fingerprint,'completed_epochs':completed,
                     'history':history,'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),
                     'rng_state':_capture_rng_state(device=config.device),'last_epoch_seconds':last_seconds}
            digest=_write_checkpoint(latest,payload)
            readback=torch.load(latest,map_location='cpu',weights_only=True)
            if (readback['fingerprint']!=fingerprint or readback['completed_epochs']!=completed
                    or hashlib.sha256(latest.read_bytes()).hexdigest()!=digest):raise ValueError('checkpoint_readback_mismatch')
            emit('recipe_checkpointed',epoch=completed,sha256=digest,elapsed_seconds=last_seconds)
        report={**plan,'status':'benchmark_complete' if config.mode=='benchmark' else ('complete' if completed==5 else 'paused_checkpoint_available'),
                'completed_epochs':completed,'epochs':history,'checkpoint_path':str(latest),'checkpoint_sha256':digest,
                'fingerprint':fingerprint,'runtime':_runtime_versions(),
                'cuda_peak_allocated':max(v['allocated'] for row in history for v in row['phase_cuda_peak_bytes'].values()) if config.device=='cuda' else None,
                'cuda_peak_reserved':max(v['reserved'] for row in history for v in row['phase_cuda_peak_bytes'].values()) if config.device=='cuda' else None,
                'constant_baseline':_constant_metrics(train,[validation[i] for i in valid_indices]),
                'quality_scope':'benchmark_not_for_quality_selection' if config.mode=='benchmark' else 'fixed_soft_report_label_validation'}
        if config.mode=='benchmark':
            timings=history[-1]['phase_seconds']
            projected=sum(timings[phase]*sum(len(row['series']) for row in rows)/selected_series[phase]
                          for phase,rows in [('train',train),('validation',validation)])
            report.update(projected_full_epoch_seconds=projected,conservative_epoch_seconds=projected*1.5+300,
                          projected_five_epoch_seconds=projected*5,conservative_five_epoch_seconds=projected*7.5+300,
                          sample_throughput_studies_per_second=(len(train_indices)+len(valid_indices))/sum(timings.values()))
        (output/'training-report.json').write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        emit('recipe_finished',status=report['status'],completed_epochs=completed,cuda_peak_allocated=report['cuda_peak_allocated'])
        return report
    except Exception as error:
        emit('recipe_failed',exception_family=type(error).__name__ if type(error) in (ValueError,RuntimeError,OSError,MemoryError) else 'other')
        raise RuntimeError('recipe_training_failed_checkpoint_preserved') from None
