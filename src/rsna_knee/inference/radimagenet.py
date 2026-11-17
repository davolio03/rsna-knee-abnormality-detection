"""Pinned inference adapter for the completed five-epoch RadImageNet ResNet-50."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from rsna_knee.inference.fold0 import (
    TARGETS,
    InferenceError,
    TestStudyDataset,
    _contained,
    _exception_family,
    _safe_reason,
    _RECOVERABLE_SERIES_REASONS,
    write_submission,
)
from rsna_knee.data.dicom_preprocessing import DicomPreprocessingError, preprocess_dicom_series
from rsna_knee.models.resnet50 import ResNet50StudyModel


EXPECTED_REPORT_SHA256 = "ee6a85e8a01ecf904da8fafad70c89a6b101efe4abda6230055625efa6210f8c"
EXPECTED_CHECKPOINT_SHA256 = "4777e9b5b0e273767444f5d4d7025ab88b394babf56c1c5116b362bc6b0d023a"
EXPECTED_CHECKPOINT_SIZE = 214216115
NATIVE_FINGERPRINT = "5c08ab6be89773802ce4f18fa2209a7e85ec88f2703b5caaded9bd940025a920"
TRAINING_SOURCE_SHA256 = "48a64d88c9b5cf72f4fdf6d9ce9839cc2cb3b8632b9fbf4fb7f4db1cdc4f95c8"
OUTPUT_SIZE = 224
MAX_SLICES = 8
NORMALIZATION = "radimagenet_2x_minus_1"
POOLING = "mean_slice_mean_series_logits"
VALIDATION_COUNTS = (1347, 1334, 1379, 1321, 1182, 1101, 1217, 1335, 243, 753, 1130, 853)


def _pin(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("artifact_pin_required")


def _file(path):
    path = Path(path)
    if not path.is_file() or any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("source_mount_invalid")
    return path


def _checkpoint_file(path):
    return _file(path)


def _checkpoint_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_radimagenet_report(report):
    """Validate the anchored training report schema without assuming current runtime parity."""
    if not isinstance(report, dict):
        raise ValueError("source_report_provenance_mismatch")
    fixed = {
        "status": "complete", "arm": "radimagenet", "completed_epochs": 5, "target_epochs": 5,
        "fold": 0, "seed": 42, "training_study_count": 2850, "validation_study_count": 1448,
        "selected_training_studies": 2850, "selected_validation_studies": 1448,
        "selected_series": {"train": 15733, "validation": 8032},
        "output_size": OUTPUT_SIZE, "max_slices": MAX_SLICES,
        "normalization": NORMALIZATION, "objective": "unweighted_observed_soft_bce",
        "batchnorm_policy": "running_statistics_and_affine_frozen",
        "trainable_policy": "layer4_non_batchnorm_and_seeded_2048_to_12_head",
        "fingerprint": NATIVE_FINGERPRINT, "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "weights_sha256": "08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734",
    }
    if any(type(report.get(key)) is not type(value) or report[key] != value for key, value in fixed.items()):
        raise ValueError("source_report_provenance_mismatch")
    runtime = report.get("runtime")
    if (type(runtime) is not dict or runtime.get("python") != "3.11.9"
            or runtime.get("torch") != "2.11.0+cu128" or runtime.get("numpy") != "2.1.3"
            or runtime.get("cuda") != "12.8"):
        raise ValueError("source_report_provenance_mismatch")
    epochs = report.get("epochs")
    if type(epochs) is not list or len(epochs) != 5:
        raise ValueError("source_report_provenance_mismatch")
    for expected_epoch, item in enumerate(epochs, 1):
        if (type(item) is not dict
                or set(item) != {"epoch", "phase_cuda_peak_bytes", "phase_seconds",
                                 "training_masked_bce", "validation"}
                or type(item.get("epoch")) is not int
                or item["epoch"] != expected_epoch):
            raise ValueError("source_report_provenance_mismatch")
        if type(item["training_masked_bce"]) not in (int, float) or not math.isfinite(item["training_masked_bce"]):
            raise ValueError("source_report_provenance_mismatch")
        phases, peaks = item["phase_seconds"], item["phase_cuda_peak_bytes"]
        if type(phases) is not dict or set(phases) != {"train", "validation"}:
            raise ValueError("source_report_provenance_mismatch")
        if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in phases.values()):
            raise ValueError("source_report_provenance_mismatch")
        if (type(peaks) is not dict or set(peaks) != {"train", "validation"}
                or any(type(peaks[phase]) is not dict or set(peaks[phase]) != {"allocated", "reserved"}
                       or any(type(value) is not int or value <= 0 for value in peaks[phase].values())
                       for phase in peaks)):
            raise ValueError("source_report_provenance_mismatch")
        metrics = item["validation"]
        if (type(metrics) is not dict or metrics.get("observed_cell_count") != sum(VALIDATION_COUNTS)
                or type(metrics.get("per_target")) is not list or len(metrics["per_target"]) != len(TARGETS)
                or any(type(entry) is not dict for entry in metrics["per_target"])
                or [entry.get("target") for entry in metrics["per_target"]] != list(TARGETS)
                or [entry.get("observed_count") for entry in metrics["per_target"]] != list(VALIDATION_COUNTS)):
            raise ValueError("source_report_provenance_mismatch")
        for metric in [metrics, *metrics["per_target"]]:
            if (type(metric.get("masked_bce")) not in (int, float)
                    or not math.isfinite(metric["masked_bce"]) or metric["masked_bce"] < 0
                    or type(metric.get("brier")) not in (int, float)
                    or not math.isfinite(metric["brier"]) or not 0 <= metric["brier"] <= 1):
                raise ValueError("source_report_provenance_mismatch")
    return None


def _model():
    from torchvision.models import resnet50

    backbone = resnet50(weights=None)
    return ResNet50StudyModel(backbone, arm="radimagenet", seed=42, slice_microbatch=2)


def load_radimagenet_model(checkpoint, report, checkpoint_sha256, *, expected_report_sha256,
                           model_factory=None, checkpoint_size=EXPECTED_CHECKPOINT_SIZE):
    """Verify exact report/checkpoint provenance and native state before strict model loading."""
    _pin(checkpoint_sha256)
    _pin(expected_report_sha256)
    validate_radimagenet_report(report)
    if expected_report_sha256 != EXPECTED_REPORT_SHA256 or checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("checkpoint_invalid")
    path = _checkpoint_file(checkpoint)
    if (report["checkpoint_sha256"] != checkpoint_sha256
            or path.stat().st_size != checkpoint_size
            or _checkpoint_sha256(path) != checkpoint_sha256):
        raise ValueError("checkpoint_invalid")
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
        if (type(value) is not dict
                or set(value) != {"schema", "fingerprint", "completed_epochs", "history",
                                  "model_state", "optimizer_state", "rng_state", "last_epoch_seconds"}
                or value["schema"] != "resnet50-recipe-checkpoint-v1"
                or value["fingerprint"] != NATIVE_FINGERPRINT
                or type(value["completed_epochs"]) is not int or value["completed_epochs"] != 5
                or value["history"] != report["epochs"]
                or type(value["last_epoch_seconds"]) not in (int, float)
                or not math.isfinite(value["last_epoch_seconds"])
                or value["last_epoch_seconds"] <= 0):
            raise ValueError
        state = value["model_state"]
        if (not isinstance(state, dict) or not state
                or any(type(key) is not str or not isinstance(tensor, torch.Tensor)
                       or not torch.isfinite(tensor).all() for key, tensor in state.items())):
            raise ValueError
        model = (model_factory or _model)()
        expected = model.state_dict()
        if (set(state) != set(expected)
                or any(state[key].shape != expected[key].shape or state[key].dtype != expected[key].dtype
                       for key in expected)):
            raise ValueError
        model.load_state_dict(state, strict=True)
        return model.eval()
    except Exception:
        raise ValueError("checkpoint_invalid") from None


class RadImageNetTestStudyDataset(TestStudyDataset):
    """Use the proven metadata inventory but run the same PP pipeline at trained 224px."""

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


def run_radimagenet_inference(competition_root, source_root, report_sha256, checkpoint_sha256,
                              output_dir, *, device="cuda", max_studies=None,
                              recovery_policy_version=1, model_factory=None, dataset_factory=None):
    """Infer with the complete trained model; only report/checkpoint failures are fatal."""
    if (recovery_policy_version != 1 or type(recovery_policy_version) is not int
            or device not in ("cpu", "cuda")
            or (max_studies is not None and (type(max_studies) is not int or not 1 <= max_studies <= 100))):
        raise ValueError("device_invalid")
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
        model = load_radimagenet_model(source / "checkpoint-epoch-05.pt", report, checkpoint_sha256,
                                       expected_report_sha256=report_sha256, model_factory=model_factory)
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
            "pooling": "equal_mean_slice_then_series_logits_then_sigmoid",
            "elapsed_seconds": time.perf_counter() - started,
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
