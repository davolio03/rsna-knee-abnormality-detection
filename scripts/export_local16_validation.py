"""Export private fold-0 probabilities for the frozen attention16 validation model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.local16_full_cache import TARGETS, Local16CacheDataset, verify_full_cache
from rsna_knee.inference.attention_sixteen import (
    ATTENTION_MODEL_SOURCE_SHA256, NATIVE_FINGERPRINT, RESNET_MODEL_SOURCE_SHA256,
    SOURCE_CHECKPOINT_SHA256,
    load_attention16_model,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA256 = "95de045ad077691a099bf7426605bfa433eb1a0abcfcd0d321250f43611f67b5"
REPORT_SHA256 = "14c7c5ceb92fe0c2debfceb699eeba006074f62afcd09042979c331fcdcd78f8"
EXPECTED_STUDIES = 1448
EXPECTED_OBSERVED_CELLS = 13195
METRIC_TOLERANCE = 1e-4
PROBABILITY_MATCH_ATOL = 1e-7
SCHEMA = "local16-validation-predictions-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _study_sha(study_id: str) -> str:
    return hashlib.sha256(study_id.encode("utf-8")).hexdigest()


def _is_sha256(value) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _atomic_json(path: Path, value: dict) -> None:
    temp = path.with_name(path.name + f".tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _read_file(path: Path) -> bytes:
    path = Path(path)
    if path.is_symlink() or not path.is_file() or any(part.is_symlink() for part in path.parents):
        raise ValueError("input_unavailable")
    return path.read_bytes()


def _cohort(manifest_path: Path, folds_path: Path, plan_path: Path):
    manifest_raw, folds_raw, plan_raw = (_read_file(path) for path in (manifest_path, folds_path, plan_path))
    try:
        manifest, folds, plan = (json.loads(raw.decode("utf-8")) for raw in (manifest_raw, folds_raw, plan_raw))
        if any(type(value) is not dict for value in (manifest, folds, plan)):
            raise ValueError
        records = manifest["records"]
        assignments = {row["study_id"]: row["fold"] for row in folds["study_assignments"]}
        ids = [row["study_id"] for row in records if assignments.get(row["study_id"]) == 0]
    except (UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        raise ValueError("validation_cohort_invalid") from None
    if (type(records) is not list or not ids or len(ids) != len(set(ids)) or len(assignments) != len(records)
            or set(assignments) != {row.get("study_id") for row in records}
            or any(type(study_id) is not str or not study_id for study_id in ids)
            or plan.get("schema_version") != "local16-cache-chunk-plan-v1"
            or plan.get("fold") != 0 or plan.get("cache_identity") != "local16-attention-run01"
            or plan.get("max_slices_per_series") != 16 or plan.get("output_size") != 224
            or plan.get("gold_included") is not False
            or not _is_sha256(plan.get("source_sha256"))
            or not _is_sha256(plan.get("preprocessing_source_sha256"))
            or type(plan.get("partition_counts")) is not dict
            or plan.get("manifest_sha256") != hashlib.sha256(manifest_raw).hexdigest()
            or plan.get("folds_sha256") != hashlib.sha256(folds_raw).hexdigest()
            or plan.get("partition_counts", {}).get("validation") != len(ids)
            or plan.get("partition_counts", {}).get("train") != len(records) - len(ids)
            or folds.get("manifest_sha256") != hashlib.sha256(
                json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n").hexdigest()):
        raise ValueError("validation_cohort_invalid")
    by_id = {row["study_id"]: row for row in records}
    return manifest_raw, folds_raw, plan_raw, plan, ids, by_id


def _pins(model_path: Path, report_path: Path, model_sha256: str, report_sha256: str):
    if model_sha256 != MODEL_SHA256 or report_sha256 != REPORT_SHA256:
        raise ValueError("model_or_report_pin_mismatch")
    if any(path.is_symlink() or any(parent.is_symlink() for parent in path.parents)
           for path in (Path(model_path), Path(report_path))):
        raise ValueError("model_or_report_pin_mismatch")
    if _sha256(model_path) != model_sha256 or _sha256(report_path) != report_sha256:
        raise ValueError("model_or_report_pin_mismatch")


def _row_path(output: Path, ordinal: int) -> Path:
    return output / f"study-{ordinal:06d}.json"


def _check_row(path: Path, ordinal: int, study_id: str, provenance: dict, expected_record: dict):
    import numpy as np

    try:
        row = json.loads(_read_file(path).decode("utf-8"))
        predictions, logits = row["predictions"], row["logits"]
        labels, observed = row["labels"], row["observed_mask"]
        if any(type(values) is not list or len(values) != 12 for values in
               (predictions, logits, labels, observed)):
            raise ValueError
        if (any(type(mask) is not bool for mask in observed)
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       for values in (predictions, logits, labels) for value in values)):
            raise ValueError
        predicted = np.asarray(predictions, dtype=np.float64)
        saved_logits = np.asarray(logits, dtype=np.float64)
        saved_labels = np.asarray(labels, dtype=np.float32)
        expected_labels = np.asarray([expected_record["soft_targets"][name] for name in TARGETS], dtype=np.float32)
        expected_observed = [bool(expected_record["observed_mask"][name]) for name in TARGETS]
        expected_probabilities = np.empty_like(saved_logits)
        positive = saved_logits >= 0
        expected_probabilities[positive] = 1 / (1 + np.exp(-saved_logits[positive]))
        exp_logits = np.exp(saved_logits[~positive])
        expected_probabilities[~positive] = exp_logits / (1 + exp_logits)
        if (row.get("schema") != SCHEMA or row.get("ordinal") != ordinal or row.get("study_id") != study_id
                or row.get("study_sha256") != _study_sha(study_id) or row.get("provenance") != provenance
                or row.get("targets") != list(TARGETS)
                or not np.array_equal(saved_labels, expected_labels)
                or observed != expected_observed
                or (predicted < 0).any() or (predicted > 1).any()
                or not np.allclose(predicted, expected_probabilities, rtol=1e-6,
                                   atol=PROBABILITY_MATCH_ATOL)):
            raise ValueError
        return row
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        raise ValueError("resume_row_invalid") from None


def _resume_prefix(output: Path, ids: list[str], provenance: dict, by_id: dict):
    meta_path = output / "provenance.json"
    try:
        metadata = json.loads(_read_file(meta_path).decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("resume_provenance_mismatch") from None
    if metadata != {"schema": SCHEMA, "provenance": provenance, "targets": list(TARGETS), "study_count": len(ids)}:
        raise ValueError("resume_provenance_mismatch")
    paths = sorted(output.glob("study-*.json"))
    allowed = {"provenance.json", "progress.json", *(path.name for path in paths)}
    entries = list(output.iterdir())
    stale_temps = {path.name for path in entries if re.fullmatch(
        r"(?:study-\d{6}\.json|provenance\.json|progress\.json)\.tmp-\d+-\d+", path.name)}
    if any(path.is_symlink() for path in entries) or {path.name for path in entries} - allowed - stale_temps:
        raise ValueError("resume_output_inventory_invalid")
    if len(paths) > len(ids) or {path.name for path in paths} != {
            _row_path(output, i).name for i in range(1, len(paths) + 1)}:
        raise ValueError("resume_prefix_invalid")
    rows = [_check_row(_row_path(output, index), index, ids[index - 1], provenance, by_id[ids[index - 1]])
            for index in range(1, len(paths) + 1)]
    return rows


def _masked_metrics(logits, predictions, labels, observed):
    import numpy as np
    logits = np.asarray(logits, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    if logits.shape != predictions.shape or logits.shape != labels.shape or logits.shape != observed.shape:
        raise ValueError("validation_metric_shape_invalid")
    if not observed.any() or not np.isfinite(logits).all() or not np.isfinite(predictions).all():
        raise ValueError("validation_metric_values_invalid")
    bce = np.logaddexp(0.0, logits) - labels * logits
    return {"observed_cell_count": int(observed.sum()), "masked_bce": float(bce[observed].mean()),
            "brier": float(((predictions - labels) ** 2)[observed].mean())}


def _metrics(rows: list[dict]):
    import numpy as np
    logits = np.asarray([row["logits"] for row in rows], dtype=np.float64)
    predictions = np.asarray([row["predictions"] for row in rows], dtype=np.float64)
    labels = np.asarray([row["labels"] for row in rows], dtype=np.float64)
    observed = np.asarray([row["observed_mask"] for row in rows], dtype=bool)
    if predictions.shape != (EXPECTED_STUDIES, 12) or observed.sum() != EXPECTED_OBSERVED_CELLS:
        raise ValueError("validation_coverage_invalid")
    return {"study_count": len(rows), **_masked_metrics(logits, predictions, labels, observed)}


def _configure_runtime(torch, device: str):
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    device_name = torch.cuda.get_device_name() if device == "cuda" else "cpu"
    return {"runtime_python": sys.version.split()[0], "runtime_torch": str(torch.__version__),
            "runtime_cuda": torch.version.cuda, "device": device, "device_name": device_name,
            "precision": "float32", "tf32_enabled": False,
            "matmul_precision": torch.get_float32_matmul_precision()}


def export_validation(*, manifest_path: Path, folds_path: Path, plan_path: Path, archive_dir: Path,
                      output_dir: Path, model_path: Path, report_path: Path, model_sha256: str,
                      report_sha256: str, cache_source: Path, run: bool = False,
                      max_studies: int | None = None, resume: bool = False, device: str = "cuda"):
    import numpy as np
    import torch

    if (type(run) is not bool or (max_studies is not None and
            (type(max_studies) is not int or not 1 <= max_studies <= 3)) or device not in ("cuda", "cpu")
            or (resume and not run)):
        raise ValueError("configuration_invalid")
    manifest_raw, folds_raw, plan_raw, plan, ids, by_id = _cohort(manifest_path, folds_path, plan_path)
    output = Path(output_dir)
    if any(part.is_symlink() for part in (output, *output.parents)):
        raise ValueError("output_path_invalid")
    provenance = {
        "model_sha256": model_sha256, "report_sha256": report_sha256,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256, "fingerprint": NATIVE_FINGERPRINT,
        "cache_identity": plan["cache_identity"], "cache_plan_sha256": hashlib.sha256(plan_raw).hexdigest(),
        "cache_source_sha256": plan["source_sha256"],
        "preprocessing_source_sha256": plan["preprocessing_source_sha256"],
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "folds_sha256": hashlib.sha256(folds_raw).hexdigest(), "fold": 0,
        "cohort_order_sha256": hashlib.sha256("\0".join(ids).encode()).hexdigest(),
        "target_order": list(TARGETS), "output_size": 224, "max_slices": 16,
        "slice_microbatch": 2, "normalization": "radimagenet_2x_minus_1",
        "attention_model_source_sha256": ATTENTION_MODEL_SOURCE_SHA256,
        "resnet_model_source_sha256": RESNET_MODEL_SOURCE_SHA256,
        "exporter_sha256": _sha256(Path(__file__)),
    }
    if not run:
        return {"status": "planned", "validation_studies": len(ids), "targets": list(TARGETS),
                "output_dir": str(output), "inference_started": False, "provenance": provenance}
    if output.exists() and (not resume or output.is_symlink() or not output.is_dir()):
        raise ValueError("output_exists")
    if resume and not output.exists():
        raise ValueError("resume_output_missing")
    if max_studies is None and len(ids) != EXPECTED_STUDIES:
        raise ValueError("validation_cohort_size_mismatch")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda_required")
    provenance.update(_configure_runtime(torch, device))
    _pins(Path(model_path), Path(report_path), model_sha256, report_sha256)
    if max_studies is None:
        cache_proof = verify_full_cache(manifest_path=Path(manifest_path), folds_path=Path(folds_path),
                                        plan_path=Path(plan_path), source_archive_path=Path(cache_source),
                                        archive_dir=Path(archive_dir))
        if (type(cache_proof) is not dict or cache_proof.get("status") != "verified"
                or cache_proof.get("partition_study_counts", {}).get("validation") != len(ids)
                or cache_proof.get("no_missing_or_extra_studies") is not True):
            raise ValueError("full_cache_verification_incomplete")
    dataset = Local16CacheDataset(Path(manifest_path), Path(plan_path), Path(archive_dir), "validation")
    if len(dataset) != len(ids) or len(dataset.rows) != len(ids):
        raise ValueError("cache_study_coverage_mismatch")
    for study_id, (_, record) in zip(ids, dataset.rows):
        if record.get("study_sha256") != _study_sha(study_id):
            raise ValueError("cache_study_order_mismatch")
    report = json.loads(_read_file(Path(report_path)).decode("utf-8"))
    model = load_attention16_model(Path(model_path), report, model_sha256,
                                   expected_report_sha256=report_sha256,
                                   checkpoint_size=Path(model_path).stat().st_size)
    model = model.to(device=device, dtype=torch.float32).eval()
    if output.exists():
        rows = _resume_prefix(output, ids, provenance, by_id)
    else:
        output.mkdir(parents=True, exist_ok=False)
        _atomic_json(output / "provenance.json", {"schema": SCHEMA, "provenance": provenance,
                                                  "targets": list(TARGETS), "study_count": len(ids)})
        rows = []
    end = len(ids) if max_studies is None else min(max_studies, len(ids))
    if len(rows) > end:
        raise ValueError("resume_prefix_exceeds_request")
    for index in range(len(rows), end):
        started = time.perf_counter()
        images, masks, targets, observed, sample_index = dataset[index]
        if sample_index != index:
            raise ValueError("cache_sample_order_mismatch")
        study_id = ids[index]
        expected = by_id[study_id]
        expected_targets = np.asarray([expected["soft_targets"][name] for name in TARGETS], dtype=np.float32)
        expected_mask = np.asarray([expected["observed_mask"][name] for name in TARGETS], dtype=bool)
        actual_targets = np.asarray(targets, dtype=np.float32)
        actual_mask = np.asarray(observed, dtype=bool)
        if (actual_targets.shape != (12,) or actual_mask.shape != (12,) or
                not np.array_equal(actual_targets, expected_targets) or not np.array_equal(actual_mask, expected_mask)):
            raise ValueError("cache_labels_mismatch")
        with torch.inference_mode():
            logits = model([image.to(device=device, dtype=torch.float32) for image in images],
                           [mask.to(device) for mask in masks])
            if (tuple(logits.shape) != (1, 12) or logits.dtype != torch.float32
                    or not torch.isfinite(logits).all()):
                raise ValueError("prediction_invalid")
            probabilities = torch.sigmoid(logits[0]).detach().cpu().numpy().astype(np.float64)
            logit_values = logits[0].detach().cpu().numpy().astype(np.float64)
        if probabilities.shape != (12,) or not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
            raise ValueError("prediction_invalid")
        row = {"schema": SCHEMA, "ordinal": index + 1, "study_id": study_id,
               "study_sha256": _study_sha(study_id), "targets": list(TARGETS),
               "logits": logit_values.tolist(), "predictions": probabilities.tolist(),
               "labels": actual_targets.astype(np.float64).tolist(),
               "observed_mask": actual_mask.tolist(), "provenance": provenance}
        _atomic_json(_row_path(output, index + 1), row)
        rows.append(row)
        _atomic_json(output / "progress.json", {"status": "partial", "completed": len(rows),
                                                 "expected": len(ids), "provenance": provenance,
                                                 "last_row_elapsed_seconds": time.perf_counter() - started})
        print(json.dumps({"event": "study_exported", "ordinal": index + 1,
                          "elapsed_seconds": time.perf_counter() - started}, sort_keys=True), flush=True)
    result = {"status": "partial", "completed": len(rows), "expected": len(ids), "provenance": provenance}
    if len(rows) == len(ids) and max_studies is None:
        result["metrics"] = _metrics(rows)
        historical = report["validation"]
        if any(abs(result["metrics"][key] - historical[key]) > METRIC_TOLERANCE for key in ("masked_bce", "brier")):
            raise ValueError("historical_metric_mismatch")
        result["status"] = "complete"
    _atomic_json(output / "progress.json", result)
    return result


def _defaults():
    cache_root = ROOT / "data/private/attention-sixteen-slices"
    full = ROOT / "data/private/full-cohort-prefilter/run01"
    model = ROOT / "data/private/attention-sixteen-submission/run02"
    return {
        "manifest_path": full / "full-cohort-soft-label-manifest.json",
        "folds_path": full / "full-cohort-grouped-folds.json",
        "plan_path": cache_root / "cache-package-run05/local16-cache-chunk-plan.json",
        "archive_dir": cache_root / "cache-local-run01/archives",
        "cache_source": cache_root / "cache-package-run05/local16-source.zip",
        "model_path": model / "attention-model-epoch-05.pt",
        "report_path": model / "training-report.json",
        "output_dir": ROOT / "data/private/local16-validation-export/run01",
    }


def main(argv=None):
    defaults = _defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    for name, value in defaults.items():
        parser.add_argument("--" + name.replace("_", "-"), type=Path, default=value)
    parser.add_argument("--model-sha256", default=MODEL_SHA256)
    parser.add_argument("--report-sha256", default=REPORT_SHA256)
    parser.add_argument("--run", action="store_true", help="explicitly verify cache and run model inference")
    parser.add_argument("--max-studies", type=int, help="bounded canary (1-3); omit only for full validation")
    parser.add_argument("--resume", action="store_true", help="continue a matching existing output prefix")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = vars(parser.parse_args(argv))
    destination = args["output_dir"].resolve()
    private_root = (ROOT / "data/private").resolve()
    if private_root not in destination.parents:
        parser.error("--output-dir must remain under data/private")
    try:
        result = export_validation(**args)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(json.dumps({"status": "failed", "reason": str(error)}, sort_keys=True), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
