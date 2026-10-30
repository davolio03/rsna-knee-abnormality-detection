"""Create an isolated, bounded Kaggle cost pilot from the pinned run06 notebook."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = Path("data/private/masked-soft-training-preparation/run06/kaggle_masked_soft_training.ipynb")
SOURCE_NOTEBOOK_SHA256 = "9b05364b6df266e0fc603cdd3fe0e5cc93352abd2c78bfe0925a7c113e76e296"
DEFAULT_OUTPUT = Path("data/private/masked-training-cost/run11")
MAX_TRAIN_STUDIES = 64
MAX_VALIDATION_STUDIES = 32
SAMPLE_PER_STRATUM = 4
PILOT_WALL_LIMIT_SECONDS = 900
FULL_RUN_LIMIT_SECONDS = 8 * 60 * 60


def select_stratified_indices(records, *, cap: int, seed: int) -> list[int]:
    """Select up to four records per series-count stratum, covering every stratum."""
    if type(cap) is not int or cap < 1 or type(seed) is not int:
        raise ValueError("pilot_cap_invalid")
    strata: dict[int, list[int]] = {}
    for index, record in enumerate(records):
        series = record.get("series") if isinstance(record, dict) else None
        if not isinstance(series, list) or not series:
            raise ValueError("pilot_series_stratum_invalid")
        strata.setdefault(len(series), []).append(index)
    if len(strata) > cap:
        raise ValueError("pilot_strata_exceed_cap")
    rng = random.Random(seed)
    selected: list[int] = []
    remaining: dict[int, list[int]] = {}
    for count in sorted(strata):
        population = strata[count]
        rng.shuffle(population)
        selected.extend(population[:1])
        remaining[count] = population[1:]
    while len(selected) < cap:
        progressed = False
        for count in sorted(remaining):
            if len(selected) >= cap:
                break
            already = sum(1 for index in selected if len(records[index]["series"]) == count)
            if already < SAMPLE_PER_STRATUM and remaining[count]:
                selected.append(remaining[count].pop())
                progressed = True
        if not progressed:
            break
    return sorted(selected)


def estimate_population_seconds(stats: dict[str, dict[str, float]]) -> float:
    """Weight observed per-study means by the full population in each stratum."""
    estimate = 0.0
    for item in stats.values():
        count = item.get("population_count")
        mean = item.get("mean_seconds")
        if (type(count) is not int or count < 1 or type(mean) not in (int, float)
                or not __import__("math").isfinite(mean) or mean < 0):
            raise ValueError("pilot_timing_invalid")
        estimate += count * float(mean)
        if not __import__("math").isfinite(estimate):
            raise ValueError("pilot_timing_invalid")
    return estimate


def _notebook_cell() -> str:
    return r'''# Cost-only proxy: keeps the verified source and package logic untouched.
import dataclasses, math, os, platform, random, shutil, time, traceback
try: import resource
except ImportError: resource = None

PILOT_SEED = 42
PILOT_TRAIN_CAP = 64
PILOT_VALIDATION_CAP = 32
PILOT_PER_STRATUM = 4
PILOT_WALL_LIMIT_SECONDS = 900
PILOT_OUTPUT_NAME = "masked-cost-pilot-run01"
PILOT_ESTIMATE_SAFETY_FACTOR = 1.5
PILOT_FIXED_OVERHEAD_SECONDS = 300
PILOT_FULL_RUN_LIMIT_SECONDS = 8 * 60 * 60
PILOT_MIN_HOST_AVAILABLE_BYTES = 2 * 1024**3
PILOT_MIN_WORKING_DISK_FREE_BYTES = 1024**3

def _pilot_sync(torch):
    if torch.cuda.is_available(): torch.cuda.synchronize()

def _host_available_bytes():
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None

def _pilot_select(records, cap):
    strata = {}
    for index, record in enumerate(records):
        count = len(record["series"])
        if count < 1: raise RuntimeError("pilot_series_stratum_invalid")
        strata.setdefault(count, []).append(index)
    if len(strata) > cap: raise RuntimeError("pilot_strata_exceed_cap")
    rng = random.Random(PILOT_SEED)
    selected, remaining = [], {}
    for count in sorted(strata):
        rng.shuffle(strata[count])
        selected.append(strata[count][0])
        remaining[count] = strata[count][1:]
    while len(selected) < cap:
        changed = False
        for count in sorted(remaining):
            n = sum(len(records[i]["series"]) == count for i in selected)
            if n < PILOT_PER_STRATUM and remaining[count] and len(selected) < cap:
                selected.append(remaining[count].pop())
                changed = True
        if not changed: break
    return sorted(selected)

def _pilot_estimate(stats):
    estimate = 0.0
    for item in stats.values():
        count, mean = item.get("population_count"), item.get("mean_seconds")
        if type(count) is not int or count < 1 or type(mean) not in (int, float) or not math.isfinite(mean) or mean < 0:
            raise RuntimeError("pilot_timing_invalid")
        estimate += count * float(mean)
        if not math.isfinite(estimate): raise RuntimeError("pilot_timing_invalid")
    return estimate

def _pilot_gate(summary):
    checks = ("finite_metrics", "checkpoint_readback", "report_readback", "memory_gate",
              "host_and_disk_gate", "fold_counts_valid", "finite_measurements",
              "pilot_deadline_gate")
    estimate = summary.get("estimated_five_epoch_seconds_with_safety", float("nan"))
    return (all(summary.get(name) is True for name in checks)
            and summary.get("time_gate") is True
            and type(estimate) in (int, float) and math.isfinite(estimate))

def _pilot_finalize(summary, emit=print):
    summary["acceptable_for_full_run_review"] = _pilot_gate(summary)
    summary["pilot_status"] = "measured_pass" if summary["acceptable_for_full_run_review"] else "measured_gate_failed"
    emit(json.dumps(summary, sort_keys=True, allow_nan=False))
    return summary

def _safe_failure_details(error, stage):
    allowed_types = {"Exception", "RuntimeError", "ValueError", "OSError", "MemoryError",
                     "NameError", "TypeError", "AttributeError", "ImportError",
                     "ModuleNotFoundError", "KeyError", "AssertionError",
                     "FileNotFoundError", "NotImplementedError", "OutOfMemoryError",
                     "SoftDatasetError"}
    error_type = type(error).__name__
    if error_type not in allowed_types:
        error_type = "Exception"
    allowed_reasons = {"pilot_wall_limit_exceeded", "pilot_cuda_unavailable",
                       "pilot_strata_exceed_cap", "pilot_strata_sample_missing",
                       "required_dicom_decoder_unavailable", "bounded_preflight_failed",
                       "metadata_plan_blocked", "prepared_input_missing_or_ambiguous",
                       "prepared_asset_missing", "competition_series_input_missing",
                       "competition_input_ambiguous", "inventory_pin_mismatch",
                       "source_pin_mismatch", "codec_asset_pin_mismatch",
                       "pilot_output_not_fresh", "device_unavailable", "pilot_timing_invalid",
                       "pilot_execution_not_explicit", "pilot_result_missing",
                       "dataset_record_invalid", "dataset_series_missing",
                       "model_weights_missing", "model_weights_pin_mismatch"}
    raw_reason = getattr(error, "reason_code", None)
    if raw_reason not in allowed_reasons:
        raw_reason = "pilot_runtime_failed"
    allowed_files = {"build_masked_cost_pilot.py", "pinned_training_source.py",
                     "resnet18.py", "masked_soft.py", "masked_soft_dataset.py",
                     "training_dataset.py", "dicom_preprocessing.py",
                     "offline_codec_bootstrap.py", "baseline.py"}
    allowed_functions = {"run_masked_cost_pilot", "run_notebook", "execute", "factory",
                         "__getitem__", "run_masked_soft_training", "_build_default_model",
                         "_load_pretrained_weights", "load_pretrained_weights",
                         "run_bounded_decode_preflight", "install_codec_bundle"}
    frames = []
    for frame in traceback.extract_tb(error.__traceback__)[-8:]:
        filename = os.path.basename(frame.filename)
        if filename not in allowed_files:
            filename = "unknown"
        function = frame.name if frame.name in allowed_functions else "unknown"
        frames.append({"file": filename, "function": function, "line": int(frame.lineno)})
    return {"stage": stage, "exception_type": error_type, "reason_code": raw_reason, "frames": frames}

class _TimedSubset:
    def __init__(self, dataset, cap, partition, torch, progress_callback,
                 before_validation=None, deadline=None, stage_callback=None):
        self.dataset, self.torch, self.partition = dataset, torch, partition
        self.progress_callback = progress_callback
        self.before_validation = before_validation
        self.deadline = deadline
        self.stage_callback = stage_callback
        self.indices = _pilot_select(dataset._records, cap)
        self.population = {}
        for row in dataset._records:
            key = str(len(row["series"]))
            self.population[key] = self.population.get(key, 0) + 1
        self.samples = {key: [] for key in self.population}
        self.last_started, self.last_partition = None, None
        self.elapsed = 0.0
        self.finished = None
    def __len__(self): return len(self.indices)
    def __getitem__(self, position):
        if self.partition == "validation" and self.before_validation is not None:
            self.before_validation()
            self.before_validation = None
        if self.deadline is not None and time.perf_counter() > self.deadline:
            raise RuntimeError("pilot_wall_limit_exceeded")
        _pilot_sync(self.torch)
        now = time.perf_counter()
        if self.last_started is not None:
            duration = now - self.last_started
            self.samples[self.last_partition].append(duration)
            self.elapsed += duration
        index = self.indices[position]
        self.last_started = now
        self.last_partition = str(len(self.dataset._records[index]["series"]))
        if self.stage_callback is not None: self.stage_callback(self.partition + "_dataset_access")
        value = self.dataset[index]
        if self.stage_callback is not None: self.stage_callback(self.partition + "_dataset_returned")
        self.progress_callback(self.partition)
        return value
    def finish(self):
        if self.finished is not None: return self.finished
        _pilot_sync(self.torch)
        if self.last_started is not None:
            duration = time.perf_counter() - self.last_started
            self.samples[self.last_partition].append(duration)
            self.elapsed += duration
        result = {}
        for key, population_count in self.population.items():
            values = self.samples[key]
            if not values: raise RuntimeError("pilot_stratum_sample_missing")
            mean = sum(values) / len(values)
            if not math.isfinite(mean): raise RuntimeError("pilot_timing_invalid")
            result[key] = {"sample_count": len(values), "population_count": population_count,
                           "mean_seconds": mean, "max_seconds": max(values)}
        self.finished = result
        return result

def run_masked_cost_pilot(run_notebook):
    run_started = time.perf_counter()
    deadline = run_started + PILOT_WALL_LIMIT_SECONDS
    evidence = WORKING_ROOT / "data" / "private" / "masked-training-cost" / "run01"
    if evidence.exists() or evidence.is_symlink(): fail("pilot_output_not_fresh")
    evidence.mkdir(parents=True)
    progress = {"pilot_only": True, "sample_allocation": {},
                "completed_studies": {"train": 0, "validation": 0}}
    preflight_report = {}
    current_stage = {"value": "pilot_setup"}
    def set_stage(value): current_stage["value"] = value
    safe_codes = {"pilot_wall_limit_exceeded", "pilot_cuda_unavailable",
                  "pilot_strata_exceed_cap", "pilot_strata_sample_missing",
                  "required_dicom_decoder_unavailable", "bounded_preflight_failed",
                  "metadata_plan_blocked", "prepared_input_missing_or_ambiguous",
                  "prepared_asset_missing", "competition_series_input_missing",
                  "competition_input_ambiguous", "inventory_pin_mismatch",
                  "source_pin_mismatch", "codec_asset_pin_mismatch",
                  "pilot_output_not_fresh", "device_unavailable", "pilot_timing_invalid"}
    def record_failure(error):
        details = _safe_failure_details(error, current_stage["value"])
        if details["reason_code"] in safe_codes:
            reason = details["reason_code"]
        else:
            reason = "pilot_runtime_failed"
        (evidence / "pilot-failure.json").write_text(
            json.dumps({"pilot_only": True, "status": "failed",
                        **details, "reason_code": reason,
                        "elapsed_seconds": time.perf_counter() - run_started,
                        "preflight": preflight_report, "progress": progress},
                       sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        return reason
    try:
        import torch
        if not torch.cuda.is_available(): raise RuntimeError("pilot_cuda_unavailable")
        torch.cuda.reset_peak_memory_stats()
        host_available_before = _host_available_bytes()
        disk_free_before = shutil.disk_usage(WORKING_ROOT).free
    except Exception as error:
        raise RuntimeError(record_failure(error)) from None
    train_proxy, validation_proxy = {}, {}
    def in_time(): return time.perf_counter() <= deadline
    def save_progress():
        (evidence / "pilot-progress.json").write_text(
            json.dumps(progress, sort_keys=True, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    def record_allocation(partition, dataset, indices):
        counts = {}
        for index in indices:
            key = str(len(dataset._records[index]["series"]))
            counts[key] = counts.get(key, 0) + 1
        progress["sample_allocation"][partition] = counts
        save_progress()
    def loader(name):
        if name == "rsna_knee.training.masked_preflight":
            real_preflight = importlib.import_module(name)
            def bounded(dataset, *, max_studies):
                set_stage("bounded_preflight")
                started = time.perf_counter()
                value = real_preflight.run_bounded_decode_preflight(dataset, max_studies=max_studies)
                preflight_report.update(value)
                preflight_report["observed_wall_seconds"] = time.perf_counter() - started
                preflight_report["within_pilot_deadline"] = in_time()
                if not in_time(): raise RuntimeError("pilot_wall_limit_exceeded")
                return value
            return type("PreflightProxy", (), {"run_bounded_decode_preflight": staticmethod(bounded)})
        if name != "rsna_knee.training.masked_soft": return importlib.import_module(name)
        # run_notebook extracts the verified source archive and adds it to sys.path
        # before invoking this loader; do not import project modules earlier.
        real_runner = importlib.import_module(name)
        def execute(config, *, execute=False):
            if execute is not True: fail("pilot_execution_not_explicit")
            real_data = importlib.import_module("rsna_knee.data.masked_soft_dataset")
            runner_output = Path(real_runner.PRIVATE_RUN_ROOT) / PILOT_OUTPUT_NAME
            sampled = dataclasses.replace(config, epochs=1, max_steps=None, output_dir=runner_output)
            original_model_factory = getattr(real_runner, "_build_default_model", None)
            if callable(original_model_factory):
                def staged_model_factory(cfg):
                    set_stage("model_construction")
                    model = original_model_factory(cfg)
                    set_stage("runner_training_setup")
                    return model
                real_runner._build_default_model = staged_model_factory
            start = time.perf_counter()
            def factory(cfg, partition):
                dataset = real_data.FullCohortSoftDataset(
                    cfg.manifest_path, cfg.folds_path, cfg.image_root,
                    partition=partition, fold=cfg.fold, output_size=cfg.output_size,
                    max_slices=cfg.max_slices,
                )
                proxy = _TimedSubset(dataset, PILOT_TRAIN_CAP if partition == "train"
                                     else PILOT_VALIDATION_CAP, partition, torch,
                                     lambda part: (progress["completed_studies"].__setitem__(
                                         part, progress["completed_studies"][part] + 1), save_progress()),
                                     before_validation=(lambda: train_proxy.__setitem__(
                                         "stats", train_proxy["dataset"].finish())) if partition == "validation" else None,
                                     deadline=deadline, stage_callback=set_stage)
                record_allocation(partition, dataset, proxy.indices)
                (train_proxy if partition == "train" else validation_proxy)["dataset"] = proxy
                return proxy
            try:
                set_stage("runner_training_setup")
                result = real_runner.run_masked_soft_training(
                    sampled, execute=True, dataset_factory=factory,
                )
            finally:
                if original_model_factory is not None:
                    real_runner._build_default_model = original_model_factory
            elapsed = time.perf_counter() - start
            train_stats = train_proxy.get("stats") or train_proxy["dataset"].finish()
            validation_stats = validation_proxy["dataset"].finish()
            fold_counts_valid = (sum(v["population_count"] for v in train_stats.values()) == 2850
                                 and sum(v["population_count"] for v in validation_stats.values()) == 1448)
            # Recompute the weighted one-epoch estimate from population stratum weights.
            estimated = _pilot_estimate(train_stats) + _pilot_estimate(validation_stats)
            if not math.isfinite(estimated): raise RuntimeError("pilot_timing_invalid")
            total_memory = torch.cuda.get_device_properties(0).total_memory
            peak_allocated = torch.cuda.max_memory_allocated()
            peak_reserved = torch.cuda.max_memory_reserved()
            disk = shutil.disk_usage(WORKING_ROOT)
            report_path = runner_output / "training-report.json"
            model_path = runner_output / "model-state.pt"
            report_readback = json.loads(report_path.read_text(encoding="utf-8"))
            checkpoint_readback = False
            checkpoint_metadata = {}
            if model_path.is_file() and model_path.stat().st_size > 0:
                checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
                checkpoint_metadata = checkpoint.get("metadata", {})
                checkpoint_readback = (isinstance(checkpoint.get("state_dict"), dict)
                                       and checkpoint_metadata.get("model_kind") == "resnet18_imagenet1k_v1"
                                       and checkpoint_metadata.get("freeze_backbone") is True
                                       and checkpoint_metadata.get("optimizer") == "Adam"
                                       and checkpoint_metadata.get("weights_sha256") == real_runner.PINNED_RESNET18_SHA256)
            values = [elapsed, estimated, peak_allocated, peak_reserved, total_memory]
            finite = all(math.isfinite(float(value)) for value in values)
            estimate_5_epochs = estimated * 5 * PILOT_ESTIMATE_SAFETY_FACTOR + PILOT_FIXED_OVERHEAD_SECONDS
            if not math.isfinite(estimate_5_epochs): raise RuntimeError("pilot_timing_invalid")
            memory_ok = peak_allocated <= 0.8 * total_memory and peak_reserved <= 0.8 * total_memory
            host_available_after = _host_available_bytes()
            resource_ok = (host_available_before is not None and host_available_after is not None
                           and host_available_before >= PILOT_MIN_HOST_AVAILABLE_BYTES
                           and host_available_after >= PILOT_MIN_HOST_AVAILABLE_BYTES
                           and disk.free >= PILOT_MIN_WORKING_DISK_FREE_BYTES)
            metric_values = [report_readback["epochs"][0]["training_masked_bce"],
                             report_readback["epochs"][0]["validation"]["masked_bce"],
                             report_readback["epochs"][0]["validation"]["brier"]]
            metrics_finite = all(math.isfinite(float(value)) for value in metric_values)
            acceptable = (finite and metrics_finite and checkpoint_readback and memory_ok and resource_ok
                          and fold_counts_valid and estimate_5_epochs <= PILOT_FULL_RUN_LIMIT_SECONDS
                          and in_time())
            result.update({
                "pilot_only": True, "full_cohort_training": False,
                "sampled_train_count": sum(v["sample_count"] for v in train_stats.values()),
                "sampled_validation_count": sum(v["sample_count"] for v in validation_stats.values()),
                "train_strata": train_stats, "validation_strata": validation_stats,
                "sampled_runner_seconds": elapsed,
                "bounded_preflight": preflight_report,
                "estimated_population_one_epoch_seconds": estimated,
                "estimated_five_epoch_seconds_with_safety": estimate_5_epochs,
                "estimate_is_confidence_interval": False,
                "estimate_population_total": 4298,
                "verified_fold_population": {"train": sum(v["population_count"] for v in train_stats.values()),
                                             "validation": sum(v["population_count"] for v in validation_stats.values()),
                                             "expected_fold0_counts": fold_counts_valid},
                "cuda_total_bytes": total_memory, "cuda_peak_allocated_bytes": peak_allocated,
                "cuda_peak_reserved_bytes": peak_reserved,
                "host_peak_rss_platform_units": (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                                                  if resource is not None else None),
                "host_available_bytes_before": host_available_before,
                "host_available_bytes_after": host_available_after,
                "disk_free_bytes_before": disk_free_before, "disk_free_bytes_after": disk.free,
                "checkpoint_metadata": checkpoint_metadata,
                "checkpoint_readback": checkpoint_readback,
                "report_readback": report_readback["status"] == "complete",
                "finite_metrics": metrics_finite, "memory_gate": memory_ok,
                "finite_measurements": finite,
                "host_and_disk_gate": resource_ok,
                "time_gate": estimate_5_epochs <= PILOT_FULL_RUN_LIMIT_SECONDS,
                "pilot_deadline_gate": in_time(),
                "fold_counts_valid": fold_counts_valid,
                "acceptable_for_full_run_review": acceptable,
                "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                            "device": torch.cuda.get_device_name(0), "accelerator_count": torch.cuda.device_count()},
            })
            result["pilot_status"] = "measured_pass" if _pilot_gate(result) else "measured_gate_failed"
            (evidence / "pilot-summary.json").write_text(
                json.dumps(result, sort_keys=True, allow_nan=False, indent=2) + "\n", encoding="utf-8")
            return result
        return type("RunnerProxy", (), {
            "MaskedSoftConfig": real_runner.MaskedSoftConfig,
            "plan_masked_soft_training": staticmethod(real_runner.plan_masked_soft_training),
            "run_masked_soft_training": staticmethod(execute),
        })
    try:
        result = run_notebook(stage="full_training", full_training=True,
                              input_root=INPUT_ROOT, working_root=WORKING_ROOT,
                              asset_root=PREPARED_ASSET_ROOT,
                              image_root=COMPETITION_IMAGE_ROOT,
                              module_loader=loader, emit=lambda _line: None)
    except Exception as error:
        raise RuntimeError(record_failure(error)) from None
    summary = result.get("training")
    if not summary:
        fail("pilot_result_missing")
    return _pilot_finalize(summary)

if __name__ == "__main__":
    run_masked_cost_pilot(run_notebook)
'''


def build_cost_pilot(repository_root: Path = ROOT, output_dir: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    root = Path(repository_root).resolve(strict=True)
    source = root / SOURCE_NOTEBOOK
    if source.is_symlink() or not source.is_file():
        raise ValueError("pinned_source_not_found")
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_NOTEBOOK_SHA256:
        raise ValueError("pinned_source_mismatch")
    notebook = json.loads(data.decode("utf-8"))
    if any(cell.get("outputs") for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"):
        raise ValueError("pinned_notebook_outputs_present")
    target = Path(output_dir)
    if not target.is_absolute():
        target = root / target
    target = target.resolve(strict=False)
    private = (root / "data" / "private").resolve(strict=True)
    if not target.is_relative_to(private) or target == private or target.exists():
        raise ValueError("pilot_output_path_invalid")
    target.mkdir(parents=True)
    source_code = "".join(notebook["cells"][0]["source"])
    bootstrap = (
        "# Execute the original pinned notebook source in this kernel without its __main__ PLAN autorun.\n"
        f"PINNED_NOTEBOOK_SOURCE = {source_code!r}\n"
        "_outer_name = globals().get('__name__')\n"
        "try:\n"
        "    globals()['__name__'] = 'pinned_training_source'\n"
        "    exec(compile(PINNED_NOTEBOOK_SOURCE, 'pinned_training_source.py', 'exec'), globals())\n"
        "finally:\n"
        "    if _outer_name is not None: globals()['__name__'] = _outer_name\n"
        "del _outer_name\n"
    )
    notebook["cells"] = [
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": bootstrap.splitlines(keepends=True)},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": _notebook_cell().splitlines(keepends=True)},
    ]
    output = target / "kaggle_masked_training_cost.ipynb"
    output.write_text(json.dumps(notebook, ensure_ascii=False, separators=(",", ":")) + "\n",
                      encoding="utf-8")
    readback = json.loads(output.read_text(encoding="utf-8"))
    if len(readback["cells"]) != 2 or any(
            cell.get("outputs") for cell in readback["cells"] if cell["cell_type"] == "code"):
        raise ValueError("pilot_notebook_readback_failed")
    return {"notebook_path": output, "notebook_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "source_notebook_sha256": SOURCE_NOTEBOOK_SHA256,
            "source_cells_preserved": 1,
            "pilot_cell_count": len(notebook["cells"])}


if __name__ == "__main__":
    result = build_cost_pilot()
    print(json.dumps({key: str(value) for key, value in result.items()}, sort_keys=True))
