"""Pinned inference adapter for the completed E048 target-attention model."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from rsna_knee.inference.fold0 import (
    TARGETS, InferenceError, _exception_family, _safe_reason, write_submission,
)
from rsna_knee.inference.radimagenet import (
    OUTPUT_SIZE, MAX_SLICES, NORMALIZATION,
    RadImageNetTestStudyDataset, _file, _pin,
)
from rsna_knee.models.attention_mil import RadImageNetAttentionStudyModel
from rsna_knee.models.resnet50 import ResNet50StudyModel


EXPECTED_REPORT_SHA256 = "f06b3714514fee1adc43462ca2e75c7d1f1dc5581a990877319283537bf0114e"
SOURCE_CHECKPOINT_SHA256 = "4c9d8930f54dcacbb88bcecb7f1f449c498dce83f13929fbcf5bb67fb5152290"
SOURCE_CHECKPOINT_SIZE = 214808723
EXPECTED_CHECKPOINT_SHA256 = SOURCE_CHECKPOINT_SHA256
EXPECTED_CHECKPOINT_SIZE = 214808723
NATIVE_FINGERPRINT = "13658c21bba3f35bb0d9fd392034af852125f5f2639006e3937d6a82f85108cb"
ATTENTION_MODEL_SOURCE_SHA256 = "10367edd912daf18a8cfa2bcde2f60e103e7af224413123c362ce3558e3c01ce"
RESNET_MODEL_SOURCE_SHA256 = "f54085f53c640cb3cb33282f57cea10388b21438e659426405a53c39c3d3e3b0"
MODEL_EXPORT_SCHEMA = "radimagenet-attention-model-only-v1"
POOLING = "target_specific_slice_attention_then_series_attention_logits"
SOURCE_EXECUTION = "disease-attention-mil/e048-run01/target_attention/checkpoint-epoch-05.pt"
VALIDATION_COUNTS = (1347, 1334, 1379, 1321, 1182, 1101, 1217, 1335, 243, 753, 1130, 853)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model():
    from torchvision.models import resnet50

    base = ResNet50StudyModel(resnet50(weights=None), arm="radimagenet", seed=42, slice_microbatch=2)
    return RadImageNetAttentionStudyModel.from_base(base)


def validate_attention_report(report):
    """Validate E048's fixed complete paired-run and attention-arm evidence."""
    if type(report) is not dict:
        raise ValueError("source_report_provenance_mismatch")
    fixed = {
        "status": "complete", "fingerprint": NATIVE_FINGERPRINT,
        "runtime": {"python": "3.11.9", "torch": "2.11.0+cu128", "numpy": "2.1.3", "cuda": "12.8"},
    }
    if any(report.get(key) != value for key, value in fixed.items()):
        raise ValueError("source_report_provenance_mismatch")
    arms = report.get("arms")
    if type(arms) is not dict or set(arms) != {"equal_mean", "target_attention"}:
        raise ValueError("source_report_provenance_mismatch")
    for arm_name, arm in arms.items():
        if (type(arm) is not dict or arm.get("arm") != arm_name or arm.get("status") != "complete"
                or type(arm.get("completed_epochs")) is not int or arm["completed_epochs"] != 5
                or type(arm.get("checkpoint_sha256")) is not str or len(arm["checkpoint_sha256"]) != 64
                or type(arm.get("epochs")) is not list or len(arm["epochs"]) != 5):
            raise ValueError("source_report_provenance_mismatch")
        for epoch_number, epoch in enumerate(arm["epochs"], 1):
            if (type(epoch) is not dict or epoch.get("epoch") != epoch_number
                    or type(epoch.get("training_masked_bce")) not in (int, float)
                    or not math.isfinite(epoch["training_masked_bce"])
                    or type(epoch.get("validation")) is not dict):
                raise ValueError("source_report_provenance_mismatch")
            validation = epoch["validation"]
            if (validation.get("observed_cell_count") != sum(VALIDATION_COUNTS)
                    or type(validation.get("per_target")) is not list
                    or len(validation["per_target"]) != len(TARGETS)
                    or [row.get("target") for row in validation["per_target"]] != list(TARGETS)
                    or [row.get("observed_count") for row in validation["per_target"]] != list(VALIDATION_COUNTS)):
                raise ValueError("source_report_provenance_mismatch")
            for metric in (validation, *validation["per_target"]):
                if (type(metric.get("masked_bce")) not in (int, float)
                        or not math.isfinite(metric["masked_bce"])
                        or type(metric.get("brier")) not in (int, float)
                        or not math.isfinite(metric["brier"])):
                    raise ValueError("source_report_provenance_mismatch")
    attention = arms["target_attention"]
    if (attention["checkpoint_sha256"] != SOURCE_CHECKPOINT_SHA256
            or attention["epochs"][-1]["epoch"] != 5):
        raise ValueError("source_report_provenance_mismatch")
    return None


def _validated_state(state, model):
    expected = model.state_dict()
    if (not isinstance(state, dict) or set(state) != set(expected)
            or any(not isinstance(value, torch.Tensor) or value.shape != expected[key].shape
                   or value.dtype != expected[key].dtype or not torch.isfinite(value).all()
                   for key, value in state.items())):
        raise ValueError("checkpoint_invalid")
    return state


def export_attention_model(source_checkpoint, report_path, output_path, *,
                           source_checkpoint_sha256=SOURCE_CHECKPOINT_SHA256,
                           report_sha256=EXPECTED_REPORT_SHA256):
    """Extract a minimal model-only export after validating the original private checkpoint."""
    _pin(source_checkpoint_sha256)
    _pin(report_sha256)
    source = _file(source_checkpoint)
    report_file = _file(report_path)
    destination = Path(output_path)
    if destination.exists() or any(part.is_symlink() for part in (destination, *destination.parents)):
        raise ValueError("output_must_be_new")
    report_raw = report_file.read_bytes()
    if hashlib.sha256(report_raw).hexdigest() != report_sha256 or report_sha256 != EXPECTED_REPORT_SHA256:
        raise ValueError("report_pin_mismatch")
    report = json.loads(report_raw.decode("utf-8"))
    validate_attention_report(report)
    if (source_checkpoint_sha256 != SOURCE_CHECKPOINT_SHA256
            or source.stat().st_size != SOURCE_CHECKPOINT_SIZE
            or _sha256(source) != source_checkpoint_sha256):
        raise ValueError("checkpoint_invalid")
    try:
        original = torch.load(source, map_location="cpu", weights_only=True)
        arm = report["arms"]["target_attention"]
        if (type(original) is not dict or set(original) != {
                "schema", "status", "fingerprint", "arm", "completed_epochs", "history",
                "model_state", "optimizer_state", "rng_state"}
                or original["schema"] != "disease-attention-mil-checkpoint-v2"
                or original["status"] != "epoch_checkpoint_complete"
                or original["arm"] != "target_attention" or original["completed_epochs"] != 5
                or original["fingerprint"] != NATIVE_FINGERPRINT or original["history"] != arm["epochs"]):
            raise ValueError
        model = _model()
        state = _validated_state(original["model_state"], model)
        header = {
            "schema": MODEL_EXPORT_SCHEMA, "status": "complete", "arm": "target_attention",
            "fingerprint": NATIVE_FINGERPRINT, "completed_epochs": 5,
            "report_sha256": report_sha256, "source_checkpoint_sha256": source_checkpoint_sha256,
            "history_sha256": hashlib.sha256(json.dumps(arm["epochs"], sort_keys=True,
                                                        separators=(",", ":")).encode()).hexdigest(),
            "attention_model_source_sha256": ATTENTION_MODEL_SOURCE_SHA256,
            "resnet_model_source_sha256": RESNET_MODEL_SOURCE_SHA256,
            "output_size": OUTPUT_SIZE, "max_slices": MAX_SLICES, "normalization": NORMALIZATION,
            "slice_microbatch": 2, "pooling": POOLING,
        }
        payload = {**header, "model_state": {key: tensor.detach().cpu().contiguous()
                                               for key, tensor in state.items()}}
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, destination)
        return {"path": str(destination), "sha256": _sha256(destination),
                "size": destination.stat().st_size, "header": header}
    except Exception:
        if destination.exists():
            destination.unlink()
        raise ValueError("checkpoint_invalid") from None


def load_attention_model(checkpoint, report, checkpoint_sha256, *, expected_report_sha256,
                         model_factory=None, checkpoint_size=None):
    """Check pins/schema/state before strict restore; never restore optimizer or RNG state."""
    _pin(checkpoint_sha256)
    _pin(expected_report_sha256)
    validate_attention_report(report)
    if (expected_report_sha256 != EXPECTED_REPORT_SHA256
            or type(checkpoint_size) is not int or checkpoint_size <= 0):
        raise ValueError("checkpoint_invalid")
    path = _file(checkpoint)
    if path.stat().st_size != checkpoint_size or _sha256(path) != checkpoint_sha256:
        raise ValueError("checkpoint_invalid")
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
        expected_header = {
            "schema": MODEL_EXPORT_SCHEMA, "status": "complete", "arm": "target_attention",
            "fingerprint": NATIVE_FINGERPRINT, "completed_epochs": 5,
            "report_sha256": EXPECTED_REPORT_SHA256, "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "history_sha256": hashlib.sha256(json.dumps(
                report["arms"]["target_attention"]["epochs"], sort_keys=True,
                separators=(",", ":")).encode()).hexdigest(),
            "attention_model_source_sha256": ATTENTION_MODEL_SOURCE_SHA256,
            "resnet_model_source_sha256": RESNET_MODEL_SOURCE_SHA256,
            "output_size": OUTPUT_SIZE, "max_slices": MAX_SLICES, "normalization": NORMALIZATION,
            "slice_microbatch": 2, "pooling": POOLING,
        }
        if (type(value) is not dict or set(value) != {*expected_header, "model_state"}
                or any(type(value.get(key)) is not type(expected) or value.get(key) != expected
                       for key, expected in expected_header.items())):
            raise ValueError
        model = (model_factory or _model)()
        model.load_state_dict(_validated_state(value["model_state"], model), strict=True)
        return model.eval()
    except Exception:
        raise ValueError("checkpoint_invalid") from None


def run_attention_inference(competition_root, source_root, report_sha256, checkpoint_sha256,
                            output_dir, *, checkpoint_size=None, device="cuda", max_studies=None,
                            recovery_policy_version=1, model_factory=None, dataset_factory=None):
    """Infer with the exact epoch-five attention model; only full visible coverage can publish CSV."""
    if (recovery_policy_version != 1 or type(recovery_policy_version) is not int
            or device not in ("cpu", "cuda")
            or (max_studies is not None and (type(max_studies) is not int or not 1 <= max_studies <= 3))):
        raise ValueError("inference_configuration_invalid")
    _pin(report_sha256)
    output = Path(output_dir)
    if output.exists() or any(item.is_symlink() for item in (output, *output.parents)):
        raise ValueError("output_must_be_new")
    started, completed, stage, journal = time.perf_counter(), 0, "checkpoint", []

    def emit(event, **extra):
        print(json.dumps({"event": event, "stage": stage, "completed": completed, **extra}, sort_keys=True), flush=True)

    try:
        source = Path(source_root)
        report_path = _file(source / "training-report.json")
        raw = report_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != report_sha256 or report_sha256 != EXPECTED_REPORT_SHA256:
            raise ValueError("report_pin_mismatch")
        report = json.loads(raw.decode("utf-8"))
        model = load_attention_model(source / "attention-model-epoch-05.pt", report, checkpoint_sha256,
                                     expected_report_sha256=report_sha256, model_factory=model_factory,
                                     checkpoint_size=checkpoint_size)
        torch.set_num_threads(1)
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("cuda_required")
        model = model.to(device).eval()
        stage = "input_metadata"
        data = (dataset_factory or RadImageNetTestStudyDataset)(
            Path(competition_root) / "test.csv", Path(competition_root) / "test_series.csv",
            Path(competition_root) / "test_series", recovery_policy_version=1,
        )
        total = len(data) if max_studies is None else min(max_studies, len(data))
        output.mkdir(parents=True, exist_ok=False)
        probabilities = []
        with torch.inference_mode():
            for index in range(total):
                stage = "preprocess"
                emit("inference_stage", total=total, study_ordinal=index + 1)
                try:
                    images, masks = data[index]
                finally:
                    if data.last_recovery:
                        journal.append(data.last_recovery)
                stage = "model"
                logits = model([image.to(device) for image in images], [mask.to(device) for mask in masks])
                if logits.shape != (1, 12) or not torch.isfinite(logits).all():
                    raise ValueError("prediction_invalid")
                probabilities.append(torch.sigmoid(logits[0]).cpu().numpy())
                completed += 1
                emit("inference_progress", total=total, study_ordinal=index + 1)
        stage = "write"
        write_submission(output / "submission.csv", data.study_ids[:total], np.asarray(probabilities))
        summary = {
            "status": "complete", "study_count": total, "test_study_count": len(data),
            "canary": max_studies is not None, "checkpoint_sha256": checkpoint_sha256,
            "report_sha256": report_sha256, "checkpoint_fingerprint": NATIVE_FINGERPRINT,
            "recovery_policy_version": 1, "series_skipped": sum(x["series_skipped"] for x in journal),
            "slice_skipped": sum(x["slice_skipped"] for x in journal), "targets": list(TARGETS),
            "output_size": OUTPUT_SIZE, "max_slices": MAX_SLICES, "slice_microbatch": 2,
            "pooling": POOLING, "elapsed_seconds": time.perf_counter() - started,
        }
        (output / "inference-report.json").write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
        emit("inference_completed", **summary)
        return summary
    except Exception as error:
        if output.exists():
            csv = output / "submission.csv"
            if csv.is_file() and not csv.is_symlink():
                csv.unlink()
        emit("inference_failed", reason_code=_safe_reason(error), exception_family=_exception_family(error))
        raise InferenceError(_safe_reason(error), study_ordinal=completed + 1,
                             exception_family=_exception_family(error)) from None
    finally:
        if output.is_dir() and not output.is_symlink():
            (output / "recovery-journal.json").write_text(
                json.dumps({"version": 1, "studies": journal}, sort_keys=True) + "\n", encoding="utf-8")
