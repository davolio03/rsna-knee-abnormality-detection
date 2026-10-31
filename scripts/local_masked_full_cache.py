"""Strict reader for the seven pinned private selected-slice cache archives."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

SOURCE = "48a64d88c9b5cf72f4fdf6d9ce9839cc2cb3b8632b9fbf4fb7f4db1cdc4f95c8"
MANIFEST = "617ea1c267960b1e903b332a78388db3b6ab80d44374721df5a4c5e184e8961b"
FOLDS = "b816c7902a5d0caa68a9e818d3f913943bc61cd2e6db65c4de069f5cd94e15f9"
PLAN = "7f7e5537b1d18b92a1f129bf0bcac5ae21eea9e44068c30c7542a1dd4a8fb0b5"
PARTS = {
    "train-000": (5125047496, "85ae0be23af507837992172ba0e1e3de16edf1049343a07a2e14277ea2ad7f8c"),
    "train-001": (5138601177, "d76f009fda01d673393ccf5ea9e981cdeb462bb03fff947705819e431e171684"),
    "train-002": (5165667141, "fe72c39518598a768cd5de8b75d59d2cdb40478ac325351b998de24e5b1b2048"),
    "train-003": (4821028360, "965fa20efd672e915395e13b062e0cd68cc75658a291c0232c0a9af81422ceb4"),
    "validation-000": (5102653714, "afb3133a93cf6728ea03a549b701049220633371de87b78f94ac08779fea10f6"),
    "validation-001": (5141967705, "a93be3cda5c63067023713d2ec5a4ad056a3a02d4b355910846a1d2ee13aba43"),
    "validation-002": (50644826, "3587b4b7908a4d4287b774dbfc8841ee135cab13b9074d2deaf2c2521fd750f2"),
}

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _fail(code: str):
    raise ValueError(code)

def validate_receipt(path: Path) -> dict:
    try:
        receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        _fail("cache_receipt_invalid")
    if (type(receipt) is not dict or receipt.get("status") != "pass"
            or receipt.get("schema_version") != "local-masked-full-cache-verification-v1"
            or receipt.get("studies") != 4298 or receipt.get("series") != 23765
            or receipt.get("slices") != 190120 or receipt.get("partition_study_counts") != {"train":2850,"validation":1448}
            or receipt.get("no_overlap") is not True or receipt.get("no_missing_or_extra_studies") is not True
            or receipt.get("gold_intersection_count") != 0 or receipt.get("training_started") is not False
            or receipt.get("source_sha256") != SOURCE or receipt.get("manifest_sha256") != MANIFEST
            or receipt.get("folds_sha256") != FOLDS or receipt.get("plan_sha256") != PLAN
            or receipt.get("all_archive_hashes_rechecked") is not True):
        _fail("cache_receipt_contract_invalid")
    rows=receipt.get("chunks")
    expected_ids=list(PARTS)
    if type(rows) is not list or [r.get("chunk_id") for r in rows if type(r) is dict]!=expected_ids: _fail("cache_receipt_parts_invalid")
    for row in rows:
        cid=row["chunk_id"]; size,digest=PARTS[cid]
        proof_path=Path(path).parent/f"cache-{cid}-independent-proof.json"
        if (row.get("archive_bytes")!=size or row.get("archive_sha256")!=digest
            or row.get("proof_file")!=str(proof_path) and Path(row.get("proof_file","")).name!=proof_path.name
            or not proof_path.is_file() or proof_path.is_symlink() or sha256(proof_path)!=row.get("proof_sha256")):
            _fail("cache_receipt_part_binding_invalid")
        try: proof=json.loads(proof_path.read_text(encoding="utf-8"))
        except (OSError,UnicodeError,json.JSONDecodeError): _fail("cache_part_proof_invalid")
        if (proof.get("status")!="pass" or proof.get("training_started") is not False
            or proof.get("archive_sha256")!=digest or proof.get("archive_bytes")!=size
            or proof.get("studies")!=row.get("study_count") or proof.get("series")!=row.get("series_count")):
            _fail("cache_part_proof_invalid")
    return receipt

def validate_full_cache(root: Path) -> dict:
    """Re-hash pins and prove ordered, disjoint partition coverage from ZIP manifests."""
    root = Path(root)
    private = root / "data/private/local-radimagenet-epoch02"
    validate_receipt(private/"cache-all-seven-chunks-independent-proof.json")
    manifest_path = root / "data/private/full-cohort-prefilter/run01/full-cohort-soft-label-manifest.json"
    folds_path = manifest_path.with_name("full-cohort-grouped-folds.json")
    source_path = root / "data/private/resnet50-pretraining-preparation/run01/resnet50-recipe-source.zip"
    plan_path = private / "cache-chunk-plan.json"
    for path, expected, code in ((source_path,SOURCE,"cache_source_pin_mismatch"),(manifest_path,MANIFEST,"cache_manifest_pin_mismatch"),(folds_path,FOLDS,"cache_folds_pin_mismatch"),(plan_path,PLAN,"cache_plan_pin_mismatch")):
        if path.is_symlink() or not path.is_file() or sha256(path) != expected: _fail(code)
    manifest=json.loads(manifest_path.read_text(encoding="utf-8")); folds=json.loads(folds_path.read_text(encoding="utf-8")); plan=json.loads(plan_path.read_text(encoding="utf-8"))
    assignments={r["study_id"]:r["fold"] for r in folds["study_assignments"]}
    canonical={p:[r for r in manifest["records"] if (assignments.get(r["study_id"]) == 0)==(p=="validation")] for p in ("train","validation")}
    seen=set(); counts={"train":0,"validation":0}; series=0; slices=0
    for chunk in plan.get("chunks",[]):
        cid=chunk.get("chunk_id"); pin=PARTS.get(cid)
        if pin is None: _fail("cache_chunk_inventory_invalid")
        archive=private/"full-cache-archives"/f"local-cache-{cid}.zip"
        if archive.is_symlink() or not archive.is_file() or archive.stat().st_size!=pin[0] or sha256(archive)!=pin[1]: _fail("cache_archive_pin_mismatch")
        try:
            with zipfile.ZipFile(archive) as z:
                if z.testzip() is not None: _fail("cache_archive_crc_invalid")
                cm=json.loads(z.read("cache-manifest.json"))
        except (OSError,KeyError,zipfile.BadZipFile,json.JSONDecodeError): _fail("cache_archive_invalid")
        part=chunk["partition"]; lo=chunk["start_ordinal"]; hi=chunk["stop_ordinal_exclusive"]
        records=cm.get("records")
        if (not isinstance(records,list) or len(records)!=hi-lo or cm.get("sample_count")!=len(records)
            or cm.get("partition")!=part or cm.get("fold")!=0 or cm.get("manifest_sha256")!=MANIFEST
            or cm.get("first_ordinal")!=lo): _fail("cache_chunk_metadata_invalid")
        prov=cm.get("provenance",{})
        if (prov.get("gold_included") is not False or prov.get("source_sha256")!=SOURCE
            or prov.get("manifest_sha256")!=MANIFEST or prov.get("folds_sha256")!=FOLDS): _fail("cache_chunk_provenance_invalid")
        expected=canonical[part][lo:hi]
        if len(expected)!=len(records): _fail("cache_chunk_selection_invalid")
        for i,(rec,row) in enumerate(zip(records,expected)):
            ident=hashlib.sha256(row["study_id"].encode()).hexdigest()
            if rec.get("ordinal")!=lo+i or rec.get("study_sha256")!=ident or ident in seen: _fail("cache_coverage_invalid")
            if rec.get("file")!=f"{part}/sample-{i:06d}.npz": _fail("cache_member_order_invalid")
            seen.add(ident); counts[part]+=1; series+=len(row["series"]); slices+=sum(sh[0] for sh in rec["series_shapes"])
    if len(PARTS)!=len(plan.get("chunks",[])) or counts!={"train":2850,"validation":1448} or len(seen)!=4298 or series!=23765 or slices!=190120: _fail("cache_full_coverage_invalid")
    return {"status":"verified","studies":len(seen),"series":series,"slices":slices,"training_studies":counts["train"],"validation_studies":counts["validation"]}

class FullCacheDataset:
    """dataset_factory-compatible adapter returning the frozen dataset tuple."""
    target_names=("ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture")

    def __init__(self, root: Path, config, partition: str, *, already_validated: bool = False):
        import csv
        import numpy as np
        import torch
        del np, torch
        if partition not in ("train","validation"): _fail("cache_partition_invalid")
        self.root=Path(root); self.partition=partition
        if already_validated is not True: validate_full_cache(self.root)
        base=self.root/"data/private/full-cohort-prefilter/run01"
        manifest=json.loads((base/"full-cohort-soft-label-manifest.json").read_text(encoding="utf-8"))
        folds=json.loads((base/"full-cohort-grouped-folds.json").read_text(encoding="utf-8"))
        assignment={r["study_id"]:r["fold"] for r in folds["study_assignments"]}
        self._records=tuple(r for r in manifest["records"] if (assignment[r["study_id"]]==0)==(partition=="validation"))
        if len(self._records)!=(2850 if partition=="train" else 1448): _fail("cache_dataset_cohort_invalid")
        self._members={}; plan=json.loads((self.root/"data/private/local-radimagenet-epoch02/cache-chunk-plan.json").read_text())
        for ch in plan["chunks"]:
            if ch["partition"]!=partition: continue
            archive=self.root/"data/private/local-radimagenet-epoch02/full-cache-archives"/f"local-cache-{ch['chunk_id']}.zip"
            with zipfile.ZipFile(archive) as z: cm=json.loads(z.read("cache-manifest.json"))
            for i,rec in enumerate(cm["records"]): self._members[ch["start_ordinal"]+i]=(archive,rec)
        self._ordinal_start=0

    def __len__(self): return len(self._records)

    def __getitem__(self,index):
        import io
        import numpy as np
        import torch
        if type(index) is not int or not 0<=index<len(self): raise IndexError("sample_index_invalid")
        archive,rec=self._members[index]
        with zipfile.ZipFile(archive) as z:
            raw=z.read(rec["file"])
        if len(raw)!=rec["size_bytes"] or hashlib.sha256(raw).hexdigest()!=rec["sha256"]: _fail("cache_sample_hash_mismatch")
        try:
            with np.load(io.BytesIO(raw),allow_pickle=False) as data:
                images=[]; masks=[]
                n=rec["series_count"]
                if set(data.files)!={"targets","observed",*(f"images_{j:03d}" for j in range(n)),*(f"masks_{j:03d}" for j in range(n))}: _fail("cache_sample_payload_invalid")
                for j in range(n):
                    im=data[f"images_{j:03d}"]; mask=data[f"masks_{j:03d}"]
                    if (im.dtype!=np.float32 or mask.dtype!=np.bool_ or im.ndim!=4 or im.shape!=mask.shape
                        or im.shape[0]<1 or im.shape[1:]!=(1,224,224) or not np.isfinite(im).all()
                        or np.any(im<0) or np.any(im>1) or np.any(im[~mask]!=0)
                        or np.any(~mask.reshape(mask.shape[0],-1).any(axis=1))): _fail("cache_sample_tensor_invalid")
                    images.append(torch.from_numpy(im.copy())); masks.append(torch.from_numpy(mask.copy()))
                target=data["targets"]; observed=data["observed"]
                if (target.dtype!=np.float32 or target.shape!=(12,) or observed.dtype!=np.bool_ or observed.shape!=(12,)
                    or not np.isfinite(target).all() or np.any(target<0) or np.any(target>1)): _fail("cache_sample_labels_invalid")
                row=self._records[index]
                expected=np.asarray([row["soft_targets"][name] for name in self.target_names],dtype=np.float32)
                expected_mask=np.asarray([row["observed_mask"][name] for name in self.target_names],dtype=np.bool_)
                if not np.array_equal(target,expected) or not np.array_equal(observed,expected_mask): _fail("cache_sample_canonical_labels_mismatch")
                if len(images)!=len(row["series"]): _fail("cache_series_order_mismatch")
                return images,masks,torch.from_numpy(target.copy()),torch.from_numpy(observed.copy()),index
        except (OSError,ValueError,KeyError): _fail("cache_sample_payload_invalid")

