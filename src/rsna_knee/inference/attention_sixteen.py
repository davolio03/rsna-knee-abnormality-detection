"""Pinned sixteen-slice inference for the completed E050 target-attention model."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from rsna_knee.inference.fold0 import (
    TARGETS, InferenceError, _RECOVERABLE_SERIES_REASONS, _contained, _exception_family,
    _safe_reason, write_submission,
)
from rsna_knee.inference.radimagenet import (
    OUTPUT_SIZE, NORMALIZATION, RadImageNetTestStudyDataset, _file, _pin,
)
from rsna_knee.models.attention_mil import RadImageNetAttentionStudyModel
from rsna_knee.models.resnet50 import ResNet50StudyModel
from rsna_knee.data.dicom_preprocessing import DicomPreprocessingError


SOURCE_REPORT_SHA256 = "8c9c83a8d7947f082f4cdb6a0f07f1ed3a956c7b451952c1ceaacf0764e51e38"
SOURCE_CHECKPOINT_SHA256 = "631536a30d0da51c0f6872fb45fac15b5bcb0f6185b5a53708eed1ef2be42f21"
SOURCE_CHECKPOINT_SIZE = 214808723
NATIVE_FINGERPRINT = "477fde596e9abd13e3a6fec49d743f69c27995824fa3546ad79a0a5fe379ade5"
ATTENTION_MODEL_SOURCE_SHA256 = "10367edd912daf18a8cfa2bcde2f60e103e7af224413123c362ce3558e3c01ce"
RESNET_MODEL_SOURCE_SHA256 = "f54085f53c640cb3cb33282f57cea10388b21438e659426405a53c39c3d3e3b0"
MODEL_EXPORT_SCHEMA = "radimagenet-attention16-model-only-v1"
PROVENANCE_SCHEMA = "radimagenet-attention16-provenance-v1"
MAX_SLICES = 16
SLICE_MICROBATCH = 2
POOLING = "target_specific_slice_attention_then_series_attention_logits"
SOURCE_EXECUTION = "attention-sixteen-slices/local16-attention-run01/target_attention/checkpoint-epoch-05.pt"
VALIDATION_COUNTS = (1347, 1334, 1379, 1321, 1182, 1101, 1217, 1335, 243, 753, 1130, 853)
TARGETS_EXPECTED = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
                   "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model():
    from torchvision.models import resnet50

    base = ResNet50StudyModel(resnet50(weights=None), arm="radimagenet", seed=42,
                              slice_microbatch=SLICE_MICROBATCH)
    return RadImageNetAttentionStudyModel.from_base(base)


def validate_attention16_source_report(report):
    """Validate complete E050 source evidence before deriving submission provenance."""
    if (type(report) is not dict or report.get("status") != "complete" or report.get("training_started") is not True
            or report.get("model_constructed") is not True
            or report.get("fingerprint") != NATIVE_FINGERPRINT):
        raise ValueError("source_report_provenance_mismatch")
    configuration = report.get("configuration")
    fixed = {
        "schema": "disease-attention-mil16-run-v1", "cache_identity": "local16-attention-run01",
        "fold": 0, "seed": 42, "epochs": 5, "arm": "target_attention",
        "initialization": "fresh_original_radimagenet_weights", "checkpoint_warm_start": False,
        "optimizer": "Adam", "slice_microbatch": SLICE_MICROBATCH,
        "output_size": OUTPUT_SIZE, "max_slices": MAX_SLICES,
        "normalization": NORMALIZATION, "selected_epoch": 5,
        "trainable": "layer4_non_batchnorm_and_seeded_2048_to_12_head",
        "objective": "unweighted_observed_soft_bce",
    }
    if (type(configuration) is not dict or any(configuration.get(key) != value for key, value in fixed.items())
            or configuration.get("gold_included") is not False
            or configuration.get("checkpoint_warm_start") is not False
            or type(configuration.get("weights_sha256")) is not str
            or configuration["weights_sha256"] != "08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734"):
        raise ValueError("source_report_provenance_mismatch")
    source = configuration.get("source")
    if (type(source) is not dict
            or source.get("src/rsna_knee/models/attention_mil.py") != ATTENTION_MODEL_SOURCE_SHA256
            or source.get("src/rsna_knee/models/resnet50.py") != RESNET_MODEL_SOURCE_SHA256):
        raise ValueError("source_report_provenance_mismatch")
    arm = report.get("arm")
    if (type(arm) is not dict or arm.get("arm") != "target_attention" or arm.get("status") != "complete"
            or arm.get("completed_epochs") != 5 or arm.get("checkpoint_sha256") != SOURCE_CHECKPOINT_SHA256
            or type(arm.get("epochs")) is not list or len(arm["epochs"]) != 5):
        raise ValueError("source_report_provenance_mismatch")
    for epoch_no, epoch in enumerate(arm["epochs"], 1):
        if (type(epoch) is not dict or epoch.get("epoch") != epoch_no
                or type(epoch.get("training_masked_bce")) not in (int, float)
                or not math.isfinite(epoch["training_masked_bce"])
                or type(epoch.get("validation")) is not dict):
            raise ValueError("source_report_provenance_mismatch")
        validation = epoch["validation"]
        rows = validation.get("per_target")
        if (validation.get("observed_cell_count") != 13195 or type(rows) is not list
                or [row.get("target") for row in rows] != list(TARGETS_EXPECTED)
                or [row.get("observed_count") for row in rows] != list(VALIDATION_COUNTS)):
            raise ValueError("source_report_provenance_mismatch")
        for metric in (validation, *rows):
            if (type(metric.get("masked_bce")) not in (int, float) or not math.isfinite(metric["masked_bce"])
                    or type(metric.get("brier")) not in (int, float) or not math.isfinite(metric["brier"])):
                raise ValueError("source_report_provenance_mismatch")
    return None


def sanitized_attention16_provenance(report):
    """Return aggregate, path-free provenance suitable for private model upload."""
    validate_attention16_source_report(report)
    configuration = report["configuration"]
    final = report["arm"]["epochs"][-1]
    safe_config = {key: configuration[key] for key in (
        "arm", "fold", "seed", "epochs", "selected_epoch", "output_size", "max_slices",
        "normalization", "slice_microbatch", "initialization", "checkpoint_warm_start",
        "objective", "optimizer", "trainable", "weights_sha256")}
    safe_validation = final["validation"]
    return {
        "schema": PROVENANCE_SCHEMA, "status": "complete", "fingerprint": NATIVE_FINGERPRINT,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256, "completed_epochs": 5,
        "history_sha256": hashlib.sha256(json.dumps(
            report["arm"]["epochs"], sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "configuration": safe_config,
        "validation": {"observed_cell_count": safe_validation["observed_cell_count"],
                       "masked_bce": safe_validation["masked_bce"], "brier": safe_validation["brier"],
                       "per_target": safe_validation["per_target"]},
    }


def _validated_state(state, model):
    expected = model.state_dict()
    if (not isinstance(state, dict) or set(state) != set(expected)
            or any(not isinstance(value, torch.Tensor) or value.shape != expected[key].shape
                   or value.dtype != expected[key].dtype or not torch.isfinite(value).all()
                   for key, value in state.items())):
        raise ValueError("checkpoint_invalid")
    return state


def export_attention16_model(source_checkpoint, report_path, output_path, *,
                             source_checkpoint_sha256=SOURCE_CHECKPOINT_SHA256,
                             source_report_sha256=SOURCE_REPORT_SHA256):
    """Validate the full private checkpoint and export only strict model tensors."""
    _pin(source_checkpoint_sha256)
    _pin(source_report_sha256)
    source = _file(source_checkpoint)
    report_file = _file(report_path)
    destination = Path(output_path)
    if destination.exists() or any(part.is_symlink() for part in (destination, *destination.parents)):
        raise ValueError("output_must_be_new")
    raw = report_file.read_bytes()
    if hashlib.sha256(raw).hexdigest() != source_report_sha256 or source_report_sha256 != SOURCE_REPORT_SHA256:
        raise ValueError("report_pin_mismatch")
    report = json.loads(raw.decode("utf-8"))
    validate_attention16_source_report(report)
    if (source_checkpoint_sha256 != SOURCE_CHECKPOINT_SHA256 or source.stat().st_size != SOURCE_CHECKPOINT_SIZE
            or _sha256(source) != source_checkpoint_sha256):
        raise ValueError("checkpoint_invalid")
    try:
        original = torch.load(source, map_location="cpu", weights_only=True)
        arm = report["arm"]
        if (type(original) is not dict or set(original) != {
                "schema", "status", "fingerprint", "arm", "completed_epochs", "history",
                "model_state", "optimizer_state", "rng_state"}
                or original["schema"] != "disease-attention-mil-checkpoint-v2"
                or original["status"] != "epoch_checkpoint_complete" or original["arm"] != "target_attention"
                or original["completed_epochs"] != 5 or original["fingerprint"] != NATIVE_FINGERPRINT
                or original["history"] != arm["epochs"]):
            raise ValueError
        model = _model()
        state = _validated_state(original["model_state"], model)
        provenance = sanitized_attention16_provenance(report)
        provenance_raw = (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode("utf-8")
        report_sha = hashlib.sha256(provenance_raw).hexdigest()
        header = {
            "schema": MODEL_EXPORT_SCHEMA, "status": "complete", "arm": "target_attention",
            "fingerprint": NATIVE_FINGERPRINT, "completed_epochs": 5,
            "report_sha256": report_sha, "source_report_sha256": SOURCE_REPORT_SHA256,
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "history_sha256": hashlib.sha256(json.dumps(arm["epochs"], sort_keys=True,
                                                        separators=(",", ":")).encode()).hexdigest(),
            "attention_model_source_sha256": ATTENTION_MODEL_SOURCE_SHA256,
            "resnet_model_source_sha256": RESNET_MODEL_SOURCE_SHA256,
            "output_size": OUTPUT_SIZE, "max_slices": MAX_SLICES, "normalization": NORMALIZATION,
            "slice_microbatch": SLICE_MICROBATCH, "pooling": POOLING,
        }
        payload = {**header, "model_state": {key: tensor.detach().cpu().contiguous()
                                               for key, tensor in state.items()}}
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, destination)
        return {"path": str(destination), "sha256": _sha256(destination), "size": destination.stat().st_size,
                "header": header, "provenance": provenance, "provenance_sha256": report_sha}
    except Exception:
        if destination.exists():
            destination.unlink()
        raise ValueError("checkpoint_invalid") from None


def validate_attention16_provenance(report):
    if (type(report) is not dict or report.get("schema") != PROVENANCE_SCHEMA
            or report.get("status") != "complete" or report.get("fingerprint") != NATIVE_FINGERPRINT
            or report.get("source_checkpoint_sha256") != SOURCE_CHECKPOINT_SHA256
            or report.get("completed_epochs") != 5):
        raise ValueError("source_report_provenance_mismatch")
    config = report.get("configuration")
    if (type(config) is not dict or config.get("arm") != "target_attention" or config.get("fold") != 0
            or config.get("seed") != 42 or config.get("selected_epoch") != 5
            or config.get("output_size") != OUTPUT_SIZE or config.get("max_slices") != MAX_SLICES
            or config.get("normalization") != NORMALIZATION or config.get("slice_microbatch") != SLICE_MICROBATCH):
        raise ValueError("source_report_provenance_mismatch")
    history_hash = report.get("history_sha256")
    validation = report.get("validation")
    if (type(history_hash) is not str or len(history_hash) != 64
            or any(char not in "0123456789abcdef" for char in history_hash)
            or type(validation) is not dict or validation.get("observed_cell_count") != 13195
            or type(validation.get("per_target")) is not list
            or [row.get("target") for row in validation["per_target"]] != list(TARGETS_EXPECTED)
            or [row.get("observed_count") for row in validation["per_target"]] != list(VALIDATION_COUNTS)):
        raise ValueError("source_report_provenance_mismatch")
    return None


def load_attention16_model(checkpoint, report, checkpoint_sha256, *, expected_report_sha256,
                           model_factory=None, checkpoint_size=None):
    _pin(checkpoint_sha256)
    _pin(expected_report_sha256)
    validate_attention16_provenance(report)
    if expected_report_sha256 != hashlib.sha256(
            (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()).hexdigest():
        raise ValueError("checkpoint_invalid")
    path = _file(checkpoint)
    if (type(checkpoint_size) is not int or checkpoint_size <= 0 or path.stat().st_size != checkpoint_size
            or _sha256(path) != checkpoint_sha256):
        raise ValueError("checkpoint_invalid")
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
        history = report.get("validation")
        expected_header = {
            "schema": MODEL_EXPORT_SCHEMA, "status": "complete", "arm": "target_attention",
            "fingerprint": NATIVE_FINGERPRINT, "completed_epochs": 5,
            "report_sha256": expected_report_sha256, "source_report_sha256": SOURCE_REPORT_SHA256,
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "history_sha256": report["history_sha256"],
            "attention_model_source_sha256": ATTENTION_MODEL_SOURCE_SHA256,
            "resnet_model_source_sha256": RESNET_MODEL_SOURCE_SHA256,
            "output_size": OUTPUT_SIZE, "max_slices": MAX_SLICES, "normalization": NORMALIZATION,
            "slice_microbatch": SLICE_MICROBATCH, "pooling": POOLING,
        }
        # The source history digest is independently bound to aggregate provenance by source checkpoint pin.
        if (type(value) is not dict or set(value) != {*expected_header, "model_state"}
                or any(type(value.get(key)) is not type(expected) or value.get(key) != expected
                       for key, expected in expected_header.items())
                or type(history) is not dict):
            raise ValueError
        model = (model_factory or _model)()
        model.load_state_dict(_validated_state(value["model_state"], model), strict=True)
        return model.eval()
    except Exception:
        raise ValueError("checkpoint_invalid") from None


class Attention16TestStudyDataset(RadImageNetTestStudyDataset):
    """The accepted metadata inventory with training-matched sixteen-slice preprocessing."""

    def __getitem__(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError("test_index_invalid")
        self.last_recovery = None
        images, masks, reason_counts = [], [], {}
        skipped_series, skipped_slices = [], []
        for series_index, series in enumerate(self._series[index]):
            try:
                try:
                    path = _contained(self.image_root, self.study_ids[index], series)
                except ValueError as error:
                    target = self.image_root / self.study_ids[index] / series
                    symlink = any(part.is_symlink() for part in (self.image_root / self.study_ids[index], target))
                    if (self.recovery_policy_version == 1 and error.args == ("series_path_invalid",)
                            and not symlink and not target.exists() and not target.is_symlink()):
                        reason_counts["series_path_unavailable"] = reason_counts.get("series_path_unavailable", 0) + 1
                        skipped_series.append({"series_ordinal": series_index + 1,
                                               "reason_code": "series_path_unavailable"})
                        continue
                    raise
                result = self.preprocessor(path, max_slices=MAX_SLICES, output_size=OUTPUT_SIZE,
                                           recovery_policy_version=1)
                image, mask = np.asarray(result.images), np.asarray(result.valid_mask)
                if (image.ndim != 4 or not 1 <= image.shape[0] <= MAX_SLICES
                        or image.shape[1:] != (1, OUTPUT_SIZE, OUTPUT_SIZE) or mask.shape != image.shape
                        or not np.isrealobj(image) or not np.isfinite(image).all()
                        or not np.isin(mask, (0, 1)).all() or (image < 0).any() or (image > 1).any()
                        or not mask.reshape(len(mask), -1).any(axis=1).all()
                        or np.any(image[~mask.astype(bool)] != 0)):
                    raise ValueError("preprocessed_tensor_invalid")
                recovery = getattr(result, "metadata", {}).get("recovery", {})
                if recovery:
                    if type(recovery) is not dict or recovery.get("version") != 1:
                        raise ValueError("preprocessed_tensor_invalid")
                    for reason, count in recovery.get("reason_counts", {}).items():
                        if reason not in _RECOVERABLE_SERIES_REASONS or type(count) is not int or count < 1:
                            raise ValueError("preprocessed_tensor_invalid")
                        reason_counts[reason] = reason_counts.get(reason, 0) + count
                    for detail in recovery.get("slice_reasons", []):
                        ordinal, reason = detail.get("slice_ordinal"), detail.get("reason_code")
                        if type(ordinal) is not int or ordinal < 1 or reason not in _RECOVERABLE_SERIES_REASONS:
                            raise ValueError("preprocessed_tensor_invalid")
                        skipped_slices.append({"series_ordinal": series_index + 1,
                                               "slice_ordinal": ordinal, "reason_code": reason})
                images.append(torch.as_tensor(image.copy(), dtype=torch.float32))
                masks.append(torch.as_tensor(mask.copy(), dtype=torch.bool))
            except DicomPreprocessingError as error:
                if error.reason_code not in _RECOVERABLE_SERIES_REASONS:
                    raise
                reason_counts[error.reason_code] = reason_counts.get(error.reason_code, 0) + 1
                details = error.recovery_details
                if details is not None:
                    if type(details) is not dict or details.get("version") != 1:
                        raise InferenceError("preprocessed_tensor_invalid", study_ordinal=index + 1,
                                             series_ordinal=series_index + 1) from None
                    for reason, count in details.get("reason_counts", {}).items():
                        if reason not in _RECOVERABLE_SERIES_REASONS or type(count) is not int or count < 1:
                            raise InferenceError("preprocessed_tensor_invalid", study_ordinal=index + 1,
                                                 series_ordinal=series_index + 1) from None
                        reason_counts[reason] = reason_counts.get(reason, 0) + count
                    for detail in details.get("slice_reasons", []):
                        ordinal, reason = detail.get("slice_ordinal"), detail.get("reason_code")
                        if type(ordinal) is not int or ordinal < 1 or reason not in _RECOVERABLE_SERIES_REASONS:
                            raise InferenceError("preprocessed_tensor_invalid", study_ordinal=index + 1,
                                                 series_ordinal=series_index + 1) from None
                        skipped_slices.append({"series_ordinal": series_index + 1,
                                               "slice_ordinal": ordinal, "reason_code": reason})
                skipped_series.append({"series_ordinal": series_index + 1,
                                       "reason_code": error.reason_code})
            except ValueError:
                raise
        if not images:
            raise InferenceError("study_series_missing", study_ordinal=index + 1, exception_family="validation")
        self.last_recovery = {
            "version": 1, "study_ordinal": index + 1,
            "series_skipped": len(skipped_series), "slice_skipped": len(skipped_slices),
            "reason_counts": dict(sorted(reason_counts.items())),
            "series_reasons": skipped_series, "slice_reasons": skipped_slices,
        }
        return images, masks


def run_attention16_inference(competition_root, source_root, report_sha256, checkpoint_sha256,
                              output_dir, *, checkpoint_size=None, device="cuda", max_studies=None,
                              recovery_policy_version=1, model_factory=None, dataset_factory=None):
    """Run canary (1-3 rows) or full inference; only full visible test coverage writes submission.csv."""
    if (recovery_policy_version != 1 or type(recovery_policy_version) is not int
            or device not in ("cpu", "cuda")
            or (max_studies is not None and (type(max_studies) is not int or not 1 <= max_studies <= 3))):
        raise ValueError("inference_configuration_invalid")
    _pin(report_sha256)
    output = Path(output_dir)
    if output.exists() or any(part.is_symlink() for part in (output, *output.parents)):
        raise ValueError("output_must_be_new")
    started, completed, stage, journal = time.perf_counter(), 0, "checkpoint", []

    def emit(event, **extra):
        print(json.dumps({"event": event, "stage": stage, "completed": completed, **extra}, sort_keys=True), flush=True)

    try:
        source = Path(source_root)
        report_path = _file(source / "training-report.json")
        raw = report_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != report_sha256:
            raise ValueError("report_pin_mismatch")
        report = json.loads(raw.decode("utf-8"))
        validate_attention16_provenance(report)
        model = load_attention16_model(source / "attention-model-epoch-05.pt", report, checkpoint_sha256,
                                       expected_report_sha256=report_sha256, model_factory=model_factory,
                                       checkpoint_size=checkpoint_size)
        torch.set_num_threads(1)
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("cuda_required")
        model = model.to(device).eval()
        stage = "input_metadata"
        data = (dataset_factory or Attention16TestStudyDataset)(
            Path(competition_root) / "test.csv", Path(competition_root) / "test_series.csv",
            Path(competition_root) / "test_series", recovery_policy_version=1)
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
        full_cohort = max_studies is None and total == len(data)
        if full_cohort:
            write_submission(output / "submission.csv", data.study_ids[:total], np.asarray(probabilities))
        summary = {"status": "complete", "study_count": total, "test_study_count": len(data),
                   "canary": not full_cohort, "submission_written": full_cohort,
                   "checkpoint_sha256": checkpoint_sha256, "report_sha256": report_sha256,
                   "checkpoint_fingerprint": NATIVE_FINGERPRINT, "recovery_policy_version": 1,
                   "series_skipped": sum(row["series_skipped"] for row in journal),
                   "slice_skipped": sum(row["slice_skipped"] for row in journal),
                   "targets": list(TARGETS), "output_size": OUTPUT_SIZE, "max_slices": MAX_SLICES,
                   "slice_microbatch": SLICE_MICROBATCH, "pooling": POOLING,
                   "elapsed_seconds": time.perf_counter() - started}
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

