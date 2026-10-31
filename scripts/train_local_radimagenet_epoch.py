"""Plan or explicitly resume exactly epoch two from the private local cache."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

_REPOSITORY_ROOT=Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path: sys.path.insert(0,str(_REPOSITORY_ROOT))
from scripts import local_masked_full_cache as cache
from scripts import local_radimagenet_checkpoint_migration as migration

EPOCH_TWO_SHA256="bd1abee90e19256292a57d04d31eced4341ad7207c403125d03b150cdeacaf72"
EPOCH_THREE_SHA256="05f0442abf4e3d53713464476b230a6ef45895eb24c58f49fb989a340fc0fcb1"
NATIVE_FINGERPRINT="5c08ab6be89773802ce4f18fa2209a7e85ec88f2703b5caaded9bd940025a920"

def _hash(path: Path) -> str:
    return migration._sha256_file(Path(path))

def _validate_training_device(*, cuda_available: bool, device_count: int, free_memory_bytes: int) -> None:
    if cuda_available is not True or type(device_count) is not int or device_count!=1:
        raise ValueError("target_runtime_mismatch")
    if type(free_memory_bytes) is not int or free_memory_bytes<3*1024**3:
        raise ValueError("gpu_free_memory_below_safety_margin")

def _require_training_cuda(torch_module):
    runtime=migration._current_target_runtime()
    migration.validate_target_runtime(runtime,cuda_device_count=torch_module.cuda.device_count())
    if not torch_module.cuda.is_available(): raise ValueError("target_runtime_mismatch")
    free,total=torch_module.cuda.mem_get_info(0)
    _validate_training_device(cuda_available=True,device_count=torch_module.cuda.device_count(),free_memory_bytes=free)
    return {"free_bytes":free,"total_bytes":total,"device":torch_module.cuda.get_device_name(0)}

def _execute_native_training(native, config, dataset_factory, model_factory):
    """Run native model construction/training inside the exact resumed FP32 policy."""
    import torch
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.set_float32_matmul_precision("highest")
    with torch.autocast(device_type="cuda",enabled=False):
        return native.run_recipe_training(config,execute=True,dataset_factory=dataset_factory,model_factory=model_factory)

def _resolve_resume(root: Path, completed_epochs: int) -> dict:
    """Resolve only the established epoch-one or epoch-two checkpoint lineage."""
    if type(completed_epochs) is not int: raise ValueError("resume_epoch_unsupported")
    root=Path(root).resolve();private=root/"data/private/local-radimagenet-epoch02"
    if completed_epochs==1:
        receipt=private/"checkpoint-migration-run01/migration-receipt.json"
        checkpoint=private/"checkpoint-migration-run01/checkpoint-epoch-01-migrated.pt"
        if not receipt.is_file() or not checkpoint.is_file(): raise ValueError("verified_migrated_checkpoint_required")
        data=json.loads(receipt.read_text(encoding="utf-8"))
        expected=data.get("migrated_fingerprint")
        if data.get("status")!="verified" or data.get("completed_epochs")!=1 or expected!=NATIVE_FINGERPRINT:
            raise ValueError("migration_receipt_invalid")
        checkpoint_sha=migration._sha256_file(checkpoint)
        if checkpoint_sha!=data.get("migrated_checkpoint_sha256"): raise ValueError("migration_checkpoint_pin_mismatch")
        return {"completed_epochs":1,"checkpoint":checkpoint,"checkpoint_sha256":checkpoint_sha,
                "native_fingerprint":expected,"evidence":receipt}
    if completed_epochs==3:
        checkpoint=private.parent/"local-radimagenet-epoch03/epoch03-run01/checkpoint-epoch-03.pt"
        report_path=checkpoint.parent/"training-report.json"
        provenance_path=checkpoint.parent.parent/"epoch03-run01-provenance.json"
        for path in (checkpoint,report_path,provenance_path):
            if migration._has_symlink_component(path) or not path.is_file(): raise ValueError("epoch_three_resume_evidence_unavailable")
        actual_sha=migration._sha256_file(checkpoint)
        if actual_sha!=EPOCH_THREE_SHA256: raise ValueError("epoch_three_checkpoint_pin_mismatch")
        try:
            report=json.loads(report_path.read_text(encoding="utf-8")); provenance=json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError,UnicodeError,json.JSONDecodeError): raise ValueError("epoch_three_resume_evidence_invalid") from None
        _validate_epoch_three_report(report,provenance,checkpoint,actual_sha,
            epoch_two_checkpoint=private/"epoch02-run01/checkpoint-epoch-02.pt",
            epoch_two_checkpoint_sha256=EPOCH_TWO_SHA256)
        return {"completed_epochs":3,"checkpoint":checkpoint,"checkpoint_sha256":actual_sha,
                "native_fingerprint":NATIVE_FINGERPRINT,"evidence":report_path,"provenance":provenance_path,
                "next_epoch":4}
    if completed_epochs!=2: raise ValueError("resume_epoch_unsupported")
    checkpoint=private/"epoch02-run01/checkpoint-epoch-02.pt"
    report_path=private/"epoch02-run01/training-report.json"
    provenance_path=private/"epoch02-run01-provenance.json"
    for path in (checkpoint,report_path,provenance_path):
        if migration._has_symlink_component(path) or not path.is_file(): raise ValueError("epoch_two_resume_evidence_unavailable")
    actual_sha=migration._sha256_file(checkpoint)
    if actual_sha!=EPOCH_TWO_SHA256: raise ValueError("epoch_two_checkpoint_pin_mismatch")
    try:
        report=json.loads(report_path.read_text(encoding="utf-8")); provenance=json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError,UnicodeError,json.JSONDecodeError): raise ValueError("epoch_two_resume_evidence_invalid") from None
    _validate_epoch_two_report(report,provenance,checkpoint,actual_sha,
        migrated_checkpoint_sha256=migration._sha256_file(private/"checkpoint-migration-run01/checkpoint-epoch-01-migrated.pt"))
    return {"completed_epochs":2,"checkpoint":checkpoint,"checkpoint_sha256":actual_sha,
            "native_fingerprint":NATIVE_FINGERPRINT,"evidence":report_path,"provenance":provenance_path,
            "next_epoch":3}

def _output_for_resume(root: Path, completed_epochs: int) -> Path:
    root=Path(root).resolve()
    if completed_epochs==1: return root/"data/private/local-radimagenet-epoch02/epoch02-run01"
    if completed_epochs==2: return root/"data/private/local-radimagenet-epoch03/epoch03-run01"
    if completed_epochs==3: return root/"data/private/local-radimagenet-final-epochs/final-run01"
    raise ValueError("resume_epoch_unsupported")

def _validate_epoch_two_report(report: dict, provenance: dict, checkpoint: Path, actual_sha: str,
                               *, migrated_checkpoint_sha256: str) -> None:
    if type(report) is not dict: raise ValueError("epoch_two_training_report_mismatch")
    epochs=report.get("epochs")
    report_checkpoint=report.get("checkpoint_path")
    if type(report_checkpoint) is not str: raise ValueError("epoch_two_training_report_mismatch")
    if (report.get("status")!="paused_checkpoint_available" or report.get("completed_epochs")!=2
        or report.get("fingerprint")!=NATIVE_FINGERPRINT or report.get("checkpoint_sha256")!=actual_sha
        or Path(report_checkpoint).resolve()!=Path(checkpoint).resolve()
        or type(epochs) is not list or len(epochs)!=2 or any(type(row) is not dict for row in epochs)
        or [row.get("epoch") for row in epochs]!=[1,2]):
        raise ValueError("epoch_two_training_report_mismatch")
    if (type(provenance) is not dict or provenance.get("schema_version")!="local-radimagenet-epoch02-provenance-v1"
        or provenance.get("native_fingerprint")!=NATIVE_FINGERPRINT
        or provenance.get("completed_epochs_before")!=1 or provenance.get("epochs_to_run")!=1
        or provenance.get("migrated_checkpoint_sha256")!=migrated_checkpoint_sha256
        or provenance.get("target_epochs")!=5):
        raise ValueError("epoch_two_provenance_mismatch")

def _validate_epoch_three_report(report: dict, provenance: dict, checkpoint: Path, actual_sha: str,
                                 *, epoch_two_checkpoint: Path, epoch_two_checkpoint_sha256: str) -> None:
    if type(report) is not dict: raise ValueError("epoch_three_training_report_mismatch")
    epochs=report.get("epochs"); report_checkpoint=report.get("checkpoint_path")
    if (report.get("status")!="paused_checkpoint_available" or report.get("completed_epochs")!=3
        or report.get("fingerprint")!=NATIVE_FINGERPRINT or report.get("checkpoint_sha256")!=actual_sha
        or type(report_checkpoint) is not str or Path(report_checkpoint).resolve()!=Path(checkpoint).resolve()
        or type(epochs) is not list or len(epochs)!=3 or any(type(row) is not dict for row in epochs)
        or [row.get("epoch") for row in epochs]!=[1,2,3]):
        raise ValueError("epoch_three_training_report_mismatch")
    resume_path=provenance.get("resume_checkpoint_path") if type(provenance) is dict else None
    if (type(provenance) is not dict or provenance.get("schema_version")!="local-radimagenet-epoch03-provenance-v1"
        or provenance.get("native_fingerprint")!=NATIVE_FINGERPRINT or provenance.get("completed_epochs_before")!=2
        or provenance.get("epochs_to_run")!=1 or provenance.get("target_epochs")!=5
        or provenance.get("resume_checkpoint_sha256")!=epoch_two_checkpoint_sha256
        or type(resume_path) is not str or Path(resume_path).resolve()!=Path(epoch_two_checkpoint).resolve()):
        raise ValueError("epoch_three_provenance_mismatch")

def _validate_run_cap(resume_epoch: int, epochs_this_run: int) -> None:
    if type(epochs_this_run) is not int or epochs_this_run not in (1,2): raise ValueError("run_epoch_cap_unsupported")
    if epochs_this_run==2 and resume_epoch!=3: raise ValueError("two_epoch_run_requires_epoch_three")
    if type(resume_epoch) is not int or resume_epoch not in (1,2,3) or epochs_this_run>5-resume_epoch:
        raise ValueError("run_epoch_cap_exceeds_target")

def make_plan(root: Path, *, train=False, resume_epoch: int = 1, epochs_this_run: int = 1) -> dict:
    _validate_run_cap(resume_epoch,epochs_this_run)
    root=Path(root).resolve(); private=root/"data/private/local-radimagenet-epoch02"
    cohort=cache.validate_full_cache(root)
    original=private/"checkpoint-epoch-01.pt"
    if _hash(original)!=migration.CHECKPOINT_SHA256: raise ValueError("original_checkpoint_pin_mismatch")
    migrated=private/"checkpoint-migration-run01/checkpoint-epoch-01-migrated.pt"
    receipt=private/"checkpoint-migration-run01/migration-receipt.json"
    resume=_resolve_resume(root,resume_epoch)
    migration_data=json.loads(receipt.read_text(encoding="utf-8"))
    if (migration_data.get("status")!="verified" or migration_data.get("original_checkpoint_sha256")!=migration.CHECKPOINT_SHA256
        or migration_data.get("migrated_checkpoint_sha256")!=_hash(migrated)
        or migration_data.get("completed_epochs")!=1 or migration_data.get("migrated_fingerprint")!=NATIVE_FINGERPRINT):
        raise ValueError("migration_receipt_invalid")
    if train and resume["completed_epochs"]+epochs_this_run>5: raise ValueError("training_target_already_complete")
    import torch
    cuda_memory=None
    if train: cuda_memory=_require_training_cuda(torch)
    elif torch.cuda.is_available():
        free,total=torch.cuda.mem_get_info(0);cuda_memory={"free_bytes":free,"total_bytes":total,"device":torch.cuda.get_device_name(0)}
    benchmark_path=private/"cache-fullcohort-one-study-gpu-benchmark-v2.json"
    if not benchmark_path.exists(): benchmark_path=private/"cache-fullcohort-one-study-gpu-benchmark.json"
    benchmark=None
    if benchmark_path.is_file() and not benchmark_path.is_symlink():
        candidate=json.loads(benchmark_path.read_text(encoding="utf-8"))
        if (candidate.get("status")=="pass" and candidate.get("training_epoch_launched") is False
                and candidate.get("original_checkpoint_sha256")==migration.CHECKPOINT_SHA256
                and candidate.get("migrated_checkpoint_sha256")==_hash(migrated)):
            benchmark={k:candidate[k] for k in ("seconds","peak_allocated_mib","peak_reserved_mib","series","slices","observed_targets")}
    output_dir=_output_for_resume(root,resume_epoch)
    if resume_epoch==1:
        payload={k:migration_data.get(k) for k in ("completed_epochs","history_entry_count","optimizer_validation","target_rng_restore_validation","migrated_fingerprint","target_runtime")}
    else:
        payload={"completed_epochs":resume_epoch,"history_entry_count":resume_epoch,
                 "native_fingerprint":NATIVE_FINGERPRINT,"checkpoint_sha256":resume["checkpoint_sha256"]}
    return {"mode":"train" if train else "plan","cohort":cohort,"completed_epochs":resume_epoch,
            "target_epochs":5,"epochs_to_run":epochs_this_run,"next_epoch":resume_epoch+1,
            "final_epoch":resume_epoch+epochs_this_run,"output_dir":str(output_dir),
            "original_checkpoint_sha256":migration.CHECKPOINT_SHA256,
            "migrated_checkpoint_sha256":_hash(migrated) if migrated.is_file() else None,
            "resume_checkpoint":str(resume["checkpoint"]),"resume_checkpoint_sha256":resume["checkpoint_sha256"],
            "resume_payload":payload,
            "resource_observation":{"cuda_device":cuda_memory,"max_runtime_seconds":28800,"epoch_cap":epochs_this_run,"benchmark":benchmark},
            "training_requires_explicit_flag":"--train"}

def run_one_epoch(root: Path, *, resume_epoch: int = 1, epochs_this_run: int = 1) -> dict:
    """Run a bounded continuation with the frozen native recipe and cached dataset seam."""
    plan=make_plan(root,train=True,resume_epoch=resume_epoch,epochs_this_run=epochs_this_run); root=Path(root).resolve(); private=root/"data/private/local-radimagenet-epoch02"
    resume=_resolve_resume(root,resume_epoch)
    receipt=private/"checkpoint-migration-run01/migration-receipt.json"
    output=Path(plan["output_dir"])
    if output.exists() or output.is_symlink(): raise ValueError("training_output_not_fresh")
    if resume_epoch>=2 and (output.parent.exists() or migration._has_symlink_component(output.parent)):
        raise ValueError("training_output_parent_not_fresh")
    archive=private/"resnet50-recipe-source.zip"
    if not archive.exists(): archive=root/"data/private/resnet50-pretraining-preparation/run01/resnet50-recipe-source.zip"
    assets=root/"data/private/resnet50-pretraining-preparation/run01/upload/resnet50-recipe-assets.json"
    source_root=private/"frozen-recipe-source-run01"
    if source_root.is_dir():
        if _hash(archive)!=migration.SOURCE_SHA256: raise ValueError("source_archive_pin_mismatch")
        hashes=json.loads((private/"checkpoint-migration-run01/migration-receipt.json").read_text(encoding="utf-8"))
        # The migration receipt binds this exact extracted inventory; do not rewrite it.
        expected=hashes.get("source_members_sha256")
        inventory={p.relative_to(source_root).as_posix():_hash(p) for p in source_root.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix!=".pyc"}
        actual=hashlib.sha256(json.dumps(inventory,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        if actual!=expected: raise ValueError("frozen_source_inventory_mismatch")
        hashes=inventory
    else:
        hashes=migration.extract_pinned_source(archive,assets,source_root)
    with migration._frozen_import(source_root):
        import importlib
        native=importlib.import_module("rsna_knee.training.resnet50")
        model_module=importlib.import_module("rsna_knee.models.resnet50")
        config=native.RecipeConfig(arm="radimagenet",weights_path=root/"data/private/resnet50-pretraining-assets/run01/resnet50-radimagenet.pt",
            output_dir=output,image_root=private,manifest_path=root/"data/private/full-cohort-prefilter/run01/full-cohort-soft-label-manifest.json",
            folds_path=root/"data/private/full-cohort-prefilter/run01/full-cohort-grouped-folds.json",mode="train",device="cuda",
            source_root=source_root,max_epochs_per_run=epochs_this_run,max_runtime_seconds=28800,
            resume_checkpoint=Path(plan["resume_checkpoint"]),
            resume_sha256=plan["resume_checkpoint_sha256"])
        config.validate()
        records,assignments=native._metadata(config)
        train=[row for row in records if assignments[row["study_id"]]!=0]
        validation=[row for row in records if assignments[row["study_id"]]==0]
        selection_payload={"train":[row["study_id"] for row in train],"validation":[row["study_id"] for row in validation]}
        selection=hashlib.sha256(json.dumps(selection_payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        native_fingerprint=native._fingerprint(config,selection)
        if native_fingerprint!=NATIVE_FINGERPRINT: raise ValueError("native_fingerprint_mismatch")
        native_resume=native._read_resume(config,native_fingerprint)
        if native_resume["completed_epochs"]!=resume_epoch or len(native_resume["history"])!=resume_epoch:
            raise ValueError("native_resume_history_mismatch")
        provenance_path=output.with_name(output.name+"-provenance.json")
        if provenance_path.exists() or provenance_path.is_symlink(): raise ValueError("training_provenance_not_fresh")
        if resume_epoch>=2: output.parent.mkdir(parents=True,exist_ok=False)
        proofs={cid:_hash(root/"data/private/local-radimagenet-epoch02"/f"cache-{cid}-independent-proof.json") for cid in cache.PARTS}
        provenance={"schema_version":"local-radimagenet-final-epochs-provenance-v1" if epochs_this_run==2 else f"local-radimagenet-epoch{resume_epoch+1:02d}-provenance-v1",
            "mode":"two_incremental_epochs" if epochs_this_run==2 else "one_incremental_epoch",
            "source_archive_sha256":migration.SOURCE_SHA256,"source_members_sha256":hashes and hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
            "cache_adapter_sha256":_hash(Path(cache.__file__),),"training_adapter_sha256":_hash(Path(__file__)),
            "cache_plan_sha256":cache.PLAN,"archive_sha256":{k:v[1] for k,v in cache.PARTS.items()},"independent_proof_sha256":proofs,
            "manifest_sha256":migration.MANIFEST_SHA256,"folds_sha256":migration.FOLDS_SHA256,
            "migration_receipt_sha256":_hash(receipt),
            "original_checkpoint_sha256":migration.CHECKPOINT_SHA256,"migrated_checkpoint_sha256":plan["migrated_checkpoint_sha256"],
            "resume_checkpoint_path":plan["resume_checkpoint"],"resume_checkpoint_sha256":plan["resume_checkpoint_sha256"],
            "resume_evidence_sha256":_hash(Path(resume["evidence"])),
            "resume_provenance_sha256":_hash(Path(resume["provenance"])) if resume.get("provenance") else None,
            "native_fingerprint":native_fingerprint,"completed_epochs_before":resume_epoch,"epochs_to_run":epochs_this_run,"target_epochs":5,
            "train_count":len(train),"validation_count":len(validation),"weight_sha256":migration.WEIGHTS_SHA256,
            "device":"cuda:0","precision":"float32_tf32_off_autocast_off"}
        temp=provenance_path.with_name(provenance_path.name+".tmp")
        with temp.open("xb") as stream:
            stream.write((json.dumps(provenance,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode())
            stream.flush();os.fsync(stream.fileno())
        os.replace(temp,provenance_path)
        def factory(cfg,partition): return cache.FullCacheDataset(root,cfg,partition,already_validated=True)
        result=_execute_native_training(native,config,factory,lambda cfg:model_module.load_resnet50_recipe("radimagenet",cfg.weights_path))
    return {"status":result.get("status"),"result":result,"resume_checkpoint_sha256":plan["resume_checkpoint_sha256"],
            "epochs_requested":epochs_this_run,"source_members":len(hashes),"provenance_path":str(provenance_path)}

def benchmark_one_cached_study(root: Path) -> dict:
    """Exercise one real full-cache study and discard all optimizer updates in RAM."""
    import time
    import numpy as np
    import torch
    plan=make_plan(root,train=True); root=Path(root).resolve(); private=root/"data/private/local-radimagenet-epoch02"
    source_root=private/"frozen-recipe-source-run01"
    if not source_root.is_dir(): raise ValueError("verified_frozen_source_required")
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.set_float32_matmul_precision("highest")
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise ValueError("target_runtime_mismatch")
    free,total=torch.cuda.mem_get_info(0)
    if free<3*1024**3: raise ValueError("gpu_free_memory_below_safety_margin")
    with migration._frozen_import(source_root):
        import importlib
        native=importlib.import_module("rsna_knee.training.resnet50")
        model_module=importlib.import_module("rsna_knee.models.resnet50")
        objective=importlib.import_module("rsna_knee.training.masked_soft")
        config=native.RecipeConfig(arm="radimagenet",weights_path=root/"data/private/resnet50-pretraining-assets/run01/resnet50-radimagenet.pt",
            output_dir=private/"epoch02-benchmark-unused",image_root=private,
            manifest_path=root/"data/private/full-cohort-prefilter/run01/full-cohort-soft-label-manifest.json",
            folds_path=root/"data/private/full-cohort-prefilter/run01/full-cohort-grouped-folds.json",mode="train",device="cuda",
            source_root=source_root,max_epochs_per_run=1,max_runtime_seconds=28800,
            resume_checkpoint=private/"checkpoint-migration-run01/checkpoint-epoch-01-migrated.pt",
            resume_sha256=plan["migrated_checkpoint_sha256"])
        config.validate();records,assignments=native._metadata(config)
        chosen={"train":[r["study_id"] for r in records if assignments[r["study_id"]]!=0],
                "validation":[r["study_id"] for r in records if assignments[r["study_id"]]==0]}
        selection=hashlib.sha256(json.dumps(chosen,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        fingerprint=native._fingerprint(config,selection)
        checkpoint=native._read_resume(config,fingerprint)
        dataset=cache.FullCacheDataset(root,config,"train",already_validated=True)
        sample=dataset[0];images,masks,targets,observed,_=sample
        device=torch.device("cuda:0")
        model=model_module.load_resnet50_recipe("radimagenet",config.weights_path).to(device)
        model.load_state_dict(checkpoint["model_state"],strict=True)
        optimizer=torch.optim.Adam(model.optimizer_groups(head_lr=config.head_lr,backbone_lr=config.backbone_lr))
        optimizer.load_state_dict(checkpoint["optimizer_state"]);native._validate_optimizer(optimizer,config)
        before={int(s["step"].item() if isinstance(s["step"],torch.Tensor) else s["step"]) for s in optimizer.state.values()}
        model.train(True);images=[x.to(device,dtype=torch.float32) for x in images];masks=[x.to(device,dtype=torch.bool) for x in masks]
        targets=targets.to(device,dtype=torch.float32).reshape(1,-1);observed=observed.to(device,dtype=torch.bool).reshape(1,-1)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        logits=model(images,masks);loss=objective.masked_soft_bce(logits,targets,observed)
        if logits.shape!=(1,12) or not bool(torch.isfinite(loss)): raise ValueError("benchmark_forward_invalid")
        optimizer.zero_grad(set_to_none=True);loss.backward()
        if (model.classifier.weight.grad is None or model.backbone.layer4[0].conv1.weight.grad is None
            or model.backbone.layer3[0].conv1.weight.grad is not None
            or any(not bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)):
            raise ValueError("benchmark_gradient_policy_invalid")
        optimizer.step();torch.cuda.synchronize()
        elapsed=time.perf_counter()-started
        after={int(s["step"].item() if isinstance(s["step"],torch.Tensor) else s["step"]) for s in optimizer.state.values()}
        if len(before)!=1 or after!={next(iter(before))+1}: raise ValueError("benchmark_adam_step_invalid")
        result={"status":"pass","training_epoch_launched":False,"checkpoint_saved":False,"checkpoint_rng_restored":False,
            "study_index":0,"series":len(images),"slices":sum(int(x.shape[0]) for x in images),"observed_targets":int(observed.sum()),
            "seconds":round(elapsed,3),"peak_allocated_mib":round(torch.cuda.max_memory_allocated()/2**20,1),
            "peak_reserved_mib":round(torch.cuda.max_memory_reserved()/2**20,1),"adam_step_before":next(iter(before)),
            "adam_step_after":next(iter(after)),"precision":"fp32_tf32_off_autocast_off","native_fingerprint":fingerprint,
            "original_checkpoint_sha256":migration.CHECKPOINT_SHA256,"migrated_checkpoint_sha256":plan["migrated_checkpoint_sha256"],
            "migration_receipt_sha256":_hash(private/"checkpoint-migration-run01/migration-receipt.json"),
            "cache_plan_sha256":cache.PLAN,"manifest_sha256":migration.MANIFEST_SHA256,"folds_sha256":migration.FOLDS_SHA256,
            "cache_archive_sha256":{k:v[1] for k,v in cache.PARTS.items()},
            "cache_part_proof_sha256":{cid:_hash(root/"data/private/local-radimagenet-epoch02"/f"cache-{cid}-independent-proof.json") for cid in cache.PARTS},
            "cache_adapter_sha256":_hash(Path(cache.__file__)),"training_adapter_sha256":_hash(Path(__file__)),
            "free_memory_before_bytes":free,"total_memory_bytes":total,
            "torch":torch.__version__,"cuda":torch.version.cuda,"device":torch.cuda.get_device_name(device)}
        proof_path=private/"cache-fullcohort-one-study-gpu-benchmark-v2.json"
        if proof_path.exists() or proof_path.is_symlink(): raise ValueError("benchmark_proof_not_fresh")
        temp=proof_path.with_name(proof_path.name+".tmp")
        with temp.open("xb") as stream:
            stream.write((json.dumps(result,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode())
            stream.flush();os.fsync(stream.fileno())
        os.replace(temp,proof_path)
        return result

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); mode=p.add_mutually_exclusive_group();mode.add_argument("--train",action="store_true",help="explicitly run the bounded epoch count from the selected verified checkpoint");mode.add_argument("--benchmark-study",action="store_true",help="run one real cached study in discarded RAM without saving")
    p.add_argument("--resume-epoch",type=int,choices=(1,2,3),default=1,help="select the prior completed epoch checkpoint; default preserves the original epoch-two plan")
    p.add_argument("--epochs-this-run",type=int,default=1,help="incremental epoch cap (2 is accepted only when resuming epoch three)")
    p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]); args=p.parse_args(argv)
    _validate_run_cap(args.resume_epoch,args.epochs_this_run)
    if args.benchmark_study and (args.resume_epoch!=1 or args.epochs_this_run!=1): raise ValueError("benchmark_selector_unsupported")
    result=run_one_epoch(args.root,resume_epoch=args.resume_epoch,epochs_this_run=args.epochs_this_run) if args.train else (benchmark_one_cached_study(args.root) if args.benchmark_study else make_plan(args.root,resume_epoch=args.resume_epoch,epochs_this_run=args.epochs_this_run))
    print(json.dumps(result,sort_keys=True,allow_nan=False,default=str)); return 0

if __name__=="__main__": raise SystemExit(main())
