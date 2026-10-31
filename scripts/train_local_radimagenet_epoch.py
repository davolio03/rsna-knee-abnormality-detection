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

def make_plan(root: Path, *, train=False) -> dict:
    root=Path(root).resolve(); private=root/"data/private/local-radimagenet-epoch02"
    cohort=cache.validate_full_cache(root)
    original=private/"checkpoint-epoch-01.pt"
    if _hash(original)!=migration.CHECKPOINT_SHA256: raise ValueError("original_checkpoint_pin_mismatch")
    migrated=private/"checkpoint-migration-run01/checkpoint-epoch-01-migrated.pt"
    receipt=private/"checkpoint-migration-run01/migration-receipt.json"
    data=None
    if migrated.is_file() and receipt.is_file():
        data=json.loads(receipt.read_text(encoding="utf-8"))
        if (data.get("status")!="verified" or data.get("original_checkpoint_sha256")!=migration.CHECKPOINT_SHA256
                or data.get("migrated_checkpoint_sha256")!=_hash(migrated)
                or data.get("completed_epochs")!=1 or type(data.get("migrated_fingerprint")) is not str
                or len(data["migrated_fingerprint"])!=64):
            raise ValueError("migration_receipt_invalid")
    if train and data is None: raise ValueError("verified_migrated_checkpoint_required")
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
    return {"mode":"train" if train else "plan","cohort":cohort,"completed_epochs":1,
            "target_epochs":5,"epochs_to_run":1,"output_dir":str(private/"epoch02-run01"),
            "original_checkpoint_sha256":migration.CHECKPOINT_SHA256,
            "migrated_checkpoint_sha256":_hash(migrated) if migrated.is_file() else None,
            "resume_payload":None if data is None else {k:data.get(k) for k in ("completed_epochs","history_entry_count","optimizer_validation","target_rng_restore_validation","migrated_fingerprint","target_runtime")},
            "resource_observation":{"cuda_device":cuda_memory,"max_runtime_seconds":28800,"epoch_cap":1,"benchmark":benchmark},
            "training_requires_explicit_flag":"--train"}

def run_one_epoch(root: Path) -> dict:
    """Run unchanged frozen native recipe with only its documented dataset seam replaced."""
    plan=make_plan(root,train=True); root=Path(root).resolve(); private=root/"data/private/local-radimagenet-epoch02"
    receipt=private/"checkpoint-migration-run01/migration-receipt.json"
    output=Path(plan["output_dir"])
    if output.exists() or output.is_symlink(): raise ValueError("training_output_not_fresh")
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
            source_root=source_root,max_epochs_per_run=1,max_runtime_seconds=28800,
            resume_checkpoint=private/"checkpoint-migration-run01/checkpoint-epoch-01-migrated.pt",
            resume_sha256=plan["migrated_checkpoint_sha256"])
        config.validate()
        records,assignments=native._metadata(config)
        train=[row for row in records if assignments[row["study_id"]]!=0]
        validation=[row for row in records if assignments[row["study_id"]]==0]
        selection_payload={"train":[row["study_id"] for row in train],"validation":[row["study_id"] for row in validation]}
        selection=hashlib.sha256(json.dumps(selection_payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        native_fingerprint=native._fingerprint(config,selection)
        if native_fingerprint!=json.loads(receipt.read_text(encoding="utf-8"))["migrated_fingerprint"]:
            raise ValueError("native_migrated_fingerprint_mismatch")
        provenance_path=output.with_name(output.name+"-provenance.json")
        if provenance_path.exists() or provenance_path.is_symlink(): raise ValueError("training_provenance_not_fresh")
        proofs={cid:_hash(root/"data/private/local-radimagenet-epoch02"/f"cache-{cid}-independent-proof.json") for cid in cache.PARTS}
        provenance={"schema_version":"local-radimagenet-epoch02-provenance-v1","mode":"one_incremental_epoch",
            "source_archive_sha256":migration.SOURCE_SHA256,"source_members_sha256":hashes and hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
            "cache_adapter_sha256":_hash(Path(cache.__file__),),"training_adapter_sha256":_hash(Path(__file__)),
            "cache_plan_sha256":cache.PLAN,"archive_sha256":{k:v[1] for k,v in cache.PARTS.items()},"independent_proof_sha256":proofs,
            "manifest_sha256":migration.MANIFEST_SHA256,"folds_sha256":migration.FOLDS_SHA256,
            "migration_receipt_sha256":_hash(receipt),
            "original_checkpoint_sha256":migration.CHECKPOINT_SHA256,"migrated_checkpoint_sha256":plan["migrated_checkpoint_sha256"],
            "native_fingerprint":native_fingerprint,"completed_epochs_before":1,"epochs_to_run":1,"target_epochs":5,
            "train_count":len(train),"validation_count":len(validation),"weight_sha256":migration.WEIGHTS_SHA256,
            "device":"cuda:0","precision":"float32_tf32_off_autocast_off"}
        temp=provenance_path.with_name(provenance_path.name+".tmp")
        with temp.open("xb") as stream:
            stream.write((json.dumps(provenance,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode())
            stream.flush();os.fsync(stream.fileno())
        os.replace(temp,provenance_path)
        def factory(cfg,partition): return cache.FullCacheDataset(root,cfg,partition,already_validated=True)
        result=_execute_native_training(native,config,factory,lambda cfg:model_module.load_resnet50_recipe("radimagenet",cfg.weights_path))
    return {"status":result.get("status"),"result":result,"migrated_checkpoint_sha256":plan["migrated_checkpoint_sha256"],"source_members":len(hashes),"provenance_path":str(provenance_path)}

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
    p=argparse.ArgumentParser(description=__doc__); mode=p.add_mutually_exclusive_group();mode.add_argument("--train",action="store_true",help="explicitly run one epoch from completed epoch one");mode.add_argument("--benchmark-study",action="store_true",help="run one real cached study in discarded RAM without saving")
    p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]); args=p.parse_args(argv)
    result=run_one_epoch(args.root) if args.train else (benchmark_one_cached_study(args.root) if args.benchmark_study else make_plan(args.root))
    print(json.dumps(result,sort_keys=True,allow_nan=False,default=str)); return 0

if __name__=="__main__": raise SystemExit(main())
