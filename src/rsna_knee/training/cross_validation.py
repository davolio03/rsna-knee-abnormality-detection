"""K-fold cross-validation for the target-attention 16-slice model (E050 recipe).

The verified local16 tensor cache was exported as fold-0 train/validation
archives, but together the archives cover every non-Gold study exactly once.
``FoldCacheDataset`` re-partitions that cache for any fold of the grouped
study-level split, keeping the cohort manifest order, so fold 0 reproduces the
original E050 partition exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import time
from typing import Any
import zipfile

TARGETS = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture")
RECIPE = {"arm": "target_attention", "seed": 42, "epochs": 5, "selected_epoch": 5,
          "head_lr": 1e-3, "layer4_lr": 1e-5, "optimizer": "Adam", "slice_microbatch": 2,
          "max_slices": 16, "output_size": 224, "initialization": "radimagenet_resnet50",
          "objective": "unweighted_observed_soft_bce", "gold_included": False}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@dataclass(frozen=True)
class CachePaths:
    manifest: Path
    folds: Path
    plan: Path
    archives: Path

    @classmethod
    def default(cls, root: Path) -> "CachePaths":
        root = Path(root)
        cohort = root / "data/private/full-cohort-prefilter/run01"
        cache = root / "data/private/attention-sixteen-slices"
        return cls(manifest=cohort / "full-cohort-soft-label-manifest.json",
                   folds=cohort / "full-cohort-grouped-folds.json",
                   plan=cache / "cache-package-run05/local16-cache-chunk-plan.json",
                   archives=cache / "cache-local-run01/archives")


def load_label_table(path: Path) -> dict[str, Any]:
    """Read an external study-level label CSV: ``StudyInstanceUID`` then the 12 targets in order.

    Values are probabilities in [0, 1]. Every cell counts as observed, so such labels replace
    both the cached soft targets and the cached observed mask during training.
    """
    import csv
    import math

    import numpy as np

    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if header != ["StudyInstanceUID", *TARGETS]:
            raise ValueError("label table header must be StudyInstanceUID followed by the 12 targets")
        table: dict[str, Any] = {}
        for row in reader:
            values = [float(value) for value in row[1:]]
            if len(values) != len(TARGETS) or not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in values):
                raise ValueError("label values must be 12 probabilities in [0, 1]")
            if row[0] in table:
                raise ValueError("duplicate study in label table")
            table[row[0]] = np.asarray(values, dtype=np.float32)
    return table


class FoldCacheDataset:
    """Study-level dataset for one partition of one cross-validation fold.

    With ``labels`` (from ``load_label_table``), samples carry those targets with every cell
    observed instead of the cached report labels.
    """

    target_names = TARGETS

    def __init__(self, paths: CachePaths, fold: int, partition: str,
                 labels: dict[str, Any] | None = None):
        if partition not in ("train", "validation"):
            raise ValueError("partition must be 'train' or 'validation'")
        manifest = json.loads(Path(paths.manifest).read_text(encoding="utf-8"))
        folds = json.loads(Path(paths.folds).read_text(encoding="utf-8"))
        plan = json.loads(Path(paths.plan).read_text(encoding="utf-8"))
        n_splits = int(folds["configuration"]["n_splits"])
        if not 0 <= fold < n_splits:
            raise ValueError(f"fold must be in [0, {n_splits})")
        assignment = {row["study_id"]: row["fold"] for row in folds["study_assignments"]}
        records = manifest["records"]
        if set(assignment) != {row["study_id"] for row in records}:
            raise ValueError("fold assignments do not match the cohort manifest")

        cached: dict[str, tuple[Path, dict]] = {}
        for chunk in plan["chunks"]:
            archive_path = Path(paths.archives) / f"local16-cache-{chunk['chunk_id']}.zip"
            with zipfile.ZipFile(archive_path) as archive:
                cache_manifest = json.loads(archive.read("cache-manifest.json"))
            for record in cache_manifest["records"]:
                if record["study_sha256"] in cached:
                    raise ValueError("study appears in more than one cache archive")
                cached[record["study_sha256"]] = (archive_path, record)

        selected = [row for row in records
                    if (assignment[row["study_id"]] == fold) == (partition == "validation")]
        missing = [row for row in selected if _sha256_text(row["study_id"]) not in cached]
        if missing:
            raise ValueError(f"{len(missing)} studies are missing from the cache")
        self.fold, self.partition = fold, partition
        self.study_ids = [row["study_id"] for row in selected]
        self.rows = [cached[_sha256_text(study_id)] for study_id in self.study_ids]
        self.labels = None
        if labels is not None:
            unlabeled = [study_id for study_id in self.study_ids if study_id not in labels]
            if unlabeled:
                raise ValueError(f"{len(unlabeled)} studies are missing from the label table")
            self.labels = [labels[study_id] for study_id in self.study_ids]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        import numpy as np
        import torch
        if not 0 <= index < len(self.rows):
            raise IndexError(index)
        archive_path, record = self.rows[index]
        with zipfile.ZipFile(archive_path) as archive:
            raw = archive.read(record["file"])
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("cache sample hash mismatch")
        with np.load(io.BytesIO(raw), allow_pickle=False) as data:
            count = record["series_count"]
            images = [torch.from_numpy(data[f"images_{i:03d}"].copy()) for i in range(count)]
            masks = [torch.from_numpy(data[f"masks_{i:03d}"].copy()) for i in range(count)]
            targets, observed = data["targets"].copy(), data["observed"].copy()
        if self.labels is not None:
            targets = self.labels[index].copy()
            observed = np.ones_like(observed)
        return images, masks, torch.from_numpy(targets), torch.from_numpy(observed), index


def _configure_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    if free_bytes < 3 * 1024**3:
        raise RuntimeError("less than 3 GiB of free GPU memory")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    return free_bytes, total_bytes


ARMS = ("target_attention", "equal_mean")
# Backbone stages that learn. "layer4" is the E050 recipe; "layer3" trains layer3 and layer4.
TRAINABLE = ("layer4", "layer3")


def _check_arm(arm: str) -> str:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    return arm


def _check_trainable(trainable: str) -> str:
    if trainable not in TRAINABLE:
        raise ValueError(f"unknown trainable backbone setting: {trainable}")
    return trainable


def unfreeze_layer3(model):
    """Let ``model`` also train backbone layer3, at the layer4 learning rate.

    BatchNorm stays frozen, as in the E050 recipe. The pinned model module runs stem-to-layer3
    without gradients, so the slice forward pass is replaced on this instance only. Layer3 joins
    the existing backbone optimizer group, whose ``layer4`` name the checkpoint validators expect.
    """
    import torch
    from torch import nn

    backbone = model.backbone
    for module in backbone.layer3.modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm):
            for parameter in module.parameters(recurse=False):
                parameter.requires_grad_(True)

    def backbone_forward(rgb):
        with torch.no_grad():
            value = backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(rgb))))
            value = backbone.layer2(backbone.layer1(value))
        return backbone.layer4(backbone.layer3(value))

    groups = model.optimizer_groups

    def optimizer_groups(*, head_lr=1e-3, backbone_lr=1e-5):
        result = groups(head_lr=head_lr, backbone_lr=backbone_lr)
        if [group["name"] for group in result] != ["head", "layer4"]:
            raise ValueError("unexpected optimizer groups")
        result[1]["params"] = [p for p in backbone.layer3.parameters() if p.requires_grad] + result[1]["params"]
        return result

    model.backbone_forward = backbone_forward
    model.optimizer_groups = optimizer_groups
    return model


def fold_fingerprint(paths: CachePaths, fold: int, weights: Path,
                     arm: str = "target_attention", trainable: str = "layer4",
                     labels: Path | None = None) -> tuple[str, dict]:
    configuration = {"schema": "attention16-cv-v1", "fold": fold, **RECIPE, "arm": _check_arm(arm),
                     "manifest_sha256": _sha256_file(paths.manifest),
                     "folds_sha256": _sha256_file(paths.folds),
                     "plan_sha256": _sha256_file(paths.plan),
                     "weights_sha256": _sha256_file(weights)}
    if _check_trainable(trainable) != "layer4":  # absent for the baseline, keeping its fingerprint
        configuration["trainable_backbone"] = trainable
    if labels is not None:  # absent for the cached report labels, keeping earlier fingerprints
        configuration["training_labels_sha256"] = _sha256_file(labels)
    raw = json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest(), configuration


def train_fold(*, root: Path, paths: CachePaths, fold: int, weights: Path, output: Path,
               resume: bool = False, arm: str = "target_attention",
               trainable: str = "layer4", labels: Path | None = None) -> dict[str, Any]:
    """Train the fixed five-epoch E050 recipe with ``fold`` held out.

    ``arm`` selects the pooling: ``target_attention`` (default) or ``equal_mean``.
    ``trainable="layer3"`` also trains backbone layer3 (see ``unfreeze_layer3``).
    ``labels`` is an external label CSV used as the training targets (see ``load_label_table``);
    the held-out fold and ``predict_fold`` keep the cached report labels.
    """
    from unittest import mock

    from rsna_knee.training import attention_mil
    from rsna_knee.training.attention_mil import AttentionMILConfig, _train_arm
    from rsna_knee.training.resnet50 import _seed_everything

    output = Path(output)
    fingerprint, configuration = fold_fingerprint(paths, fold, weights, arm, trainable, labels)
    if resume:
        saved = json.loads((output / "run-plan.json").read_text(encoding="utf-8"))
        if saved["fingerprint"] != fingerprint:
            raise ValueError("resume fingerprint mismatch")
    elif output.exists():
        raise ValueError(f"output already exists: {output}")
    free_bytes, total_bytes = _configure_cuda()
    train_data = FoldCacheDataset(paths, fold, "train",
                                  load_label_table(labels) if labels is not None else None)
    validation_data = FoldCacheDataset(paths, fold, "validation")
    if set(train_data.study_ids) & set(validation_data.study_ids):
        raise ValueError("train/validation overlap")
    arm_output = output / arm
    if not resume:
        arm_output.mkdir(parents=True)
        _write_json(output / "run-plan.json", {"fingerprint": fingerprint, "configuration": configuration,
                                               "train_studies": len(train_data),
                                               "validation_studies": len(validation_data)})
    config = AttentionMILConfig(repository_root=Path(root), weights_path=Path(weights),
                                output_dir=output, device="cuda")
    _seed_everything(42, "cuda")
    started = time.time()
    build_model = attention_mil._model_for_arm
    if trainable == "layer3":
        patch = mock.patch.object(attention_mil, "_model_for_arm",
                                  lambda *args, **kwargs: unfreeze_layer3(build_model(*args, **kwargs)))
    else:
        patch = mock.patch.object(attention_mil, "_model_for_arm", build_model)
    with patch:
        report = _train_arm(config, arm=arm, output=arm_output, fingerprint=fingerprint,
                            train_data=train_data, validation_data=validation_data,
                            epochs=RECIPE["epochs"], resume=resume)
    result = {"status": report["status"], "fingerprint": fingerprint, "configuration": configuration,
              "train_studies": len(train_data), "validation_studies": len(validation_data),
              "wall_seconds": time.time() - started, "arm": report,
              "device_memory_before": {"free_bytes": free_bytes, "total_bytes": total_bytes}}
    _write_json(output / "training-report.json", result)
    return result


def predict_fold(*, root: Path, paths: CachePaths, fold: int, weights: Path, checkpoint: Path,
                 output: Path, arm: str = "target_attention") -> dict[str, Any]:
    """Score the held-out fold with a final checkpoint; write private predictions and metrics."""
    import numpy as np
    import torch
    from rsna_knee.training.attention_mil import AttentionMILConfig, _model_for_arm
    from rsna_knee.validation.masked_auc import masked_multilabel_auc

    _configure_cuda()
    config = AttentionMILConfig(repository_root=Path(root), weights_path=Path(weights),
                                output_dir=Path(output), device="cuda")
    model = _model_for_arm(config, _check_arm(arm))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("completed_epochs") != RECIPE["selected_epoch"]:
        raise ValueError("checkpoint is not the preselected final epoch")
    model.load_state_dict(state["model_state"], strict=True)
    model.to("cuda").eval()
    data = FoldCacheDataset(paths, fold, "validation")
    probabilities, soft, observed = [], [], []
    started = time.time()
    with torch.no_grad():
        for index in range(len(data)):
            images, masks, targets, seen, _ = data[index]
            logits = model([image.to("cuda", dtype=torch.float32) for image in images],
                           [mask.to("cuda", dtype=torch.bool) for mask in masks])
            probabilities.append(torch.sigmoid(logits.float()).reshape(-1).cpu().numpy())
            soft.append(targets.numpy())
            observed.append(seen.numpy())
    probabilities, soft, observed = (np.stack(values) for values in (probabilities, soft, observed))
    metrics = masked_multilabel_auc(soft, observed, probabilities, TARGETS)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / f"oof-fold{fold}.npz", study_ids=np.asarray(data.study_ids),
                        probabilities=probabilities, soft_targets=soft, observed=observed)
    result = {"fold": fold, "arm": arm, "studies": len(data), "checkpoint_sha256": _sha256_file(checkpoint),
              "predict_seconds": time.time() - started, **metrics}
    _write_json(output / f"metrics-fold{fold}.json", result)
    return result


def summarize(prediction_dir: Path, folds: tuple[int, ...]) -> dict[str, Any]:
    """Per-fold, mean +/- std and pooled out-of-fold AUC."""
    import numpy as np
    from rsna_knee.validation.masked_auc import masked_multilabel_auc

    per_fold, arrays = [], []
    for fold in folds:
        per_fold.append(json.loads((Path(prediction_dir) / f"metrics-fold{fold}.json").read_text()))
        with np.load(Path(prediction_dir) / f"oof-fold{fold}.npz") as data:
            arrays.append((data["soft_targets"], data["observed"], data["probabilities"]))
    soft, observed, probabilities = (np.concatenate([a[i] for a in arrays]) for i in range(3))
    pooled = masked_multilabel_auc(soft, observed, probabilities, TARGETS)
    macros = np.asarray([row["macro_auc"] for row in per_fold], dtype=np.float64)
    per_target = {}
    for column, name in enumerate(TARGETS):
        values = [row["per_target"][column]["auc"] for row in per_fold]
        values = np.asarray([v for v in values if v is not None], dtype=np.float64)
        per_target[name] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else None,
                            "pooled": pooled["per_target"][column]["auc"],
                            "positives": pooled["per_target"][column]["positives"],
                            "n": pooled["per_target"][column]["n"]}
    per_target_folds = {name: [row["per_target"][column]["auc"] for row in per_fold]
                        for column, name in enumerate(TARGETS)}
    for name in TARGETS:
        per_target[name]["per_fold"] = per_target_folds[name]
    return {"folds": list(folds), "macro_auc_per_fold": macros.tolist(),
            "macro_auc_mean": float(macros.mean()),
            "macro_auc_std": float(macros.std(ddof=1)) if len(macros) > 1 else None,
            "pooled_oof_macro_auc": pooled["macro_auc"], "studies": int(len(soft)),
            "per_target": per_target, "label_definition": "observed LLM soft labels binarized at 0.5"}


def summary_markdown(summary: dict[str, Any]) -> str:
    """Render a cross-validation summary as Markdown tables (aggregate values only)."""
    def fmt(value):
        return "–" if value is None else f"{value:.3f}"

    folds = summary["folds"]
    std = summary["macro_auc_std"]
    lines = [f"**Macro ROC-AUC: {summary['macro_auc_mean']:.3f}"
             + (f" ± {std:.3f}" if std is not None else "")
             + f"** (mean ± std over {len(folds)} folds) · pooled out-of-fold: "
             + f"{fmt(summary['pooled_oof_macro_auc'])} · {summary['studies']:,} studies",
             "",
             "| Target | " + " | ".join(f"Fold {f}" for f in folds) + " | Mean ± std | Pooled OOF | Positives / n |",
             "|---|" + "---:|" * (len(folds) + 3)]
    for name, row in summary["per_target"].items():
        spread = fmt(row["mean"]) + (f" ± {row['std']:.3f}" if row["std"] is not None else "")
        lines.append(f"| {name} | " + " | ".join(fmt(v) for v in row["per_fold"])
                     + f" | {spread} | {fmt(row['pooled'])} | {row['positives']:,} / {row['n']:,} |")
    lines.append("| **Macro** | " + " | ".join(fmt(v) for v in summary["macro_auc_per_fold"])
                 + f" | **{fmt(summary['macro_auc_mean'])}"
                 + (f" ± {std:.3f}" if std is not None else "")
                 + f"** | {fmt(summary['pooled_oof_macro_auc'])} | |")
    return "\n".join(lines) + "\n"
