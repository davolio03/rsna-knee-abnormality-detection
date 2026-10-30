"""Private diagnostic adapter for running the frozen fold-zero runner on public training data."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from zipfile import BadZipFile, ZipFile

FROZEN_SOURCE_PLAN_SHA256 = "5f9fc0f7beea764e975ee3e7f263aeb66db314f43cfb0b5d4dcc69a265abd1f6"
FROZEN_SERIES_METADATA_SHA256 = "573c1d80772bf41211c91b149c95677385a1c22d63f485c347f1b46c0177aef3"
FROZEN_STUDY_COUNT = 100
FROZEN_SERIES_COUNT = 546
FROZEN_PLANE_COUNTS = {"Axial": 131, "Coronal": 197, "Sagittal": 218}
FULL_SOURCE_PLAN_SHA256 = "5f9fc0f7beea764e975ee3e7f263aeb66db314f43cfb0b5d4dcc69a265abd1f6"
FULL_GOLD_SOURCE_SHA256 = "e0e584d475a7e8b0749ffcce70267ac19a2e2549338f98f50ed515f8fabce3a1"
FULL_STUDY_COUNT = 4349
FULL_SERIES_COUNT = 24035
FULL_GOLD_EXCLUDED_SERIES_COUNT = 336
FULL_PLANE_COUNTS = {"Axial": 5817, "Coronal": 8490, "Sagittal": 9728}
FULL_PLAN_MAX_BYTES = 5_000_000
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_PLAN_MEMBER = "public-training-inference-plan.json"


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def validate_public_training_plan(plan, *, expected_source_plan_sha256=FROZEN_SOURCE_PLAN_SHA256,
                                  expected_series_sha256=FROZEN_SERIES_METADATA_SHA256,
                                  expected_studies=FROZEN_STUDY_COUNT,
                                  expected_series=FROZEN_SERIES_COUNT,
                                  expected_plane_counts=FROZEN_PLANE_COUNTS):
    """Validate the compact ID-only projection of the frozen outcome-blind plan."""
    required_v1 = {
        "schema_version", "source_plan_sha256", "series_metadata_sha256",
        "population_study_count", "fresh_study_count", "excluded_study_count",
        "seed", "study_count", "series_count", "plane_counts", "studies",
    }
    required_full = {
        "schema_version", "source_plan_sha256", "series_metadata_sha256",
        "gold_source_sha256", "population_study_count", "gold_excluded_study_count",
        "gold_excluded_series_count", "prior_study_count_included", "study_count",
        "series_count", "plane_counts", "studies",
    }
    if type(plan) is not dict:
        raise ValueError("audit_plan_invalid")
    if plan.get("schema_version") == "public-training-inference-plan-v1":
        if set(plan) != required_v1:
            raise ValueError("audit_plan_invalid")
        counts_valid = (
            plan["source_plan_sha256"] == expected_source_plan_sha256
            and plan["series_metadata_sha256"] == expected_series_sha256
            and plan["population_study_count"] == 4407
            and plan["fresh_study_count"] == 3849
            and plan["excluded_study_count"] == 558
            and plan["seed"] == 42
            and plan["study_count"] == expected_studies
            and plan["series_count"] == expected_series
            and plan["plane_counts"] == expected_plane_counts
            and type(plan["studies"]) is list
            and len(plan["studies"]) == expected_studies
        )
    elif plan.get("schema_version") == "full-public-training-inference-plan-v1":
        if set(plan) != required_full:
            raise ValueError("audit_plan_invalid")
        counts_valid = (
            plan["source_plan_sha256"] == FULL_SOURCE_PLAN_SHA256
            and plan["series_metadata_sha256"] == FROZEN_SERIES_METADATA_SHA256
            and plan["gold_source_sha256"] == FULL_GOLD_SOURCE_SHA256
            and plan["population_study_count"] == 4407
            and plan["gold_excluded_study_count"] == 58
            and plan["gold_excluded_series_count"] == FULL_GOLD_EXCLUDED_SERIES_COUNT
            and plan["prior_study_count_included"] == 500
            and plan["study_count"] == FULL_STUDY_COUNT
            and plan["series_count"] == FULL_SERIES_COUNT
            and plan["plane_counts"] == FULL_PLANE_COUNTS
            and type(plan["studies"]) is list
            and len(plan["studies"]) == FULL_STUDY_COUNT
        )
    else:
        raise ValueError("audit_plan_invalid")
    if not counts_valid:
        raise ValueError("audit_plan_pin_mismatch")
    if (type(plan["study_count"]) is not int or type(plan["series_count"]) is not int
            or type(plan["plane_counts"]) is not dict
            or type(plan["studies"]) is not list):
        raise ValueError("audit_plan_invalid")
    expected_plane_counts = plan["plane_counts"]
    expected_series = plan["series_count"]
    expected_studies = plan["study_count"]
    studies, series_ids, observed_planes = set(), set(), {key: 0 for key in expected_plane_counts}
    for study in plan["studies"]:
        if type(study) is not dict or set(study) != {"study_id", "series"}:
            raise ValueError("audit_plan_invalid")
        study_id = study["study_id"]
        if type(study_id) is not str or _IDENTIFIER.fullmatch(study_id) is None or study_id in studies:
            raise ValueError("audit_plan_invalid")
        studies.add(study_id)
        rows = study["series"]
        if type(rows) is not list or not rows:
            raise ValueError("audit_plan_invalid")
        for row in rows:
            if type(row) is not dict or set(row) != {"series_id", "plane"}:
                raise ValueError("audit_plan_invalid")
            series_id, plane = row["series_id"], row["plane"]
            if (type(series_id) is not str or _IDENTIFIER.fullmatch(series_id) is None
                    or series_id in series_ids or plane not in observed_planes):
                raise ValueError("audit_plan_invalid")
            series_ids.add(series_id)
            observed_planes[plane] += 1
    if len(series_ids) != expected_series or observed_planes != expected_plane_counts:
        raise ValueError("audit_plan_counts_mismatch")
    return tuple(study["study_id"] for study in plan["studies"])


def load_public_training_plan(plan_archive, expected_plan_sha256):
    """Read the pinned JSON payload from a ZIP member or Kaggle-extracted JSON file."""
    path = Path(plan_archive)
    if path.is_symlink() or not path.is_file():
        raise ValueError("audit_plan_archive_invalid")
    try:
        raw = path.read_bytes()
        if raw.startswith(b"PK\x03\x04"):
            with ZipFile(path) as archive:
                names = archive.namelist()
                if names != [_PLAN_MEMBER] or archive.getinfo(_PLAN_MEMBER).file_size > 200_000:
                    raise ValueError("audit_plan_archive_invalid")
                plan_bytes = archive.read(_PLAN_MEMBER)
            if (_digest(raw) != expected_plan_sha256
                    and _digest(plan_bytes) != expected_plan_sha256):
                raise ValueError("audit_plan_archive_pin_mismatch")
        else:
            if len(raw) > FULL_PLAN_MAX_BYTES:
                raise ValueError("audit_plan_archive_invalid")
            if _digest(raw) != expected_plan_sha256:
                raise ValueError("audit_plan_archive_pin_mismatch")
            plan_bytes = raw
        if len(plan_bytes) > FULL_PLAN_MAX_BYTES:
            raise ValueError("audit_plan_archive_invalid")
        plan = json.loads(plan_bytes.decode("utf-8"))
        if len(plan_bytes) > 200_000 and plan.get("schema_version") != "full-public-training-inference-plan-v1":
            raise ValueError("audit_plan_archive_invalid")
        validate_public_training_plan(plan)
        return plan
    except (OSError, UnicodeError, BadZipFile, KeyError, json.JSONDecodeError):
        raise ValueError("audit_plan_archive_invalid") from None


def stage_training_metadata(plan, train_series_csv, staging_dir, *,
                            expected_source_plan_sha256=FROZEN_SOURCE_PLAN_SHA256,
                            expected_series_sha256=FROZEN_SERIES_METADATA_SHA256,
                            expected_studies=FROZEN_STUDY_COUNT,
                            expected_series=FROZEN_SERIES_COUNT,
                            expected_plane_counts=FROZEN_PLANE_COUNTS):
    """Stage only selected training IDs and series IDs; never copy other metadata columns."""
    study_ids = validate_public_training_plan(
        plan, expected_source_plan_sha256=expected_source_plan_sha256,
        expected_series_sha256=expected_series_sha256,
        expected_studies=expected_studies, expected_series=expected_series,
        expected_plane_counts=expected_plane_counts,
    )
    metadata_path = Path(train_series_csv)
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise ValueError("series_metadata_invalid")
    raw = metadata_path.read_bytes()
    if _digest(raw) != expected_series_sha256:
        raise ValueError("series_metadata_pin_mismatch")
    expected = {
        study["study_id"]: {row["series_id"]: row["plane"] for row in study["series"]}
        for study in plan["studies"]
    }
    observed = {study_id: {} for study_id in study_ids}
    try:
        reader = csv.DictReader(raw.decode("utf-8-sig").splitlines())
        if (reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames)
                or not {"StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane"}.issubset(reader.fieldnames)):
            raise ValueError
        all_series = set()
        for row in reader:
            study_id, series_id, plane = (row.get("StudyInstanceUID"), row.get("SeriesInstanceUID"),
                                          row.get("Anatomical_Plane"))
            if (type(study_id) is not str or _IDENTIFIER.fullmatch(study_id) is None
                    or type(series_id) is not str or _IDENTIFIER.fullmatch(series_id) is None
                    or series_id in all_series):
                raise ValueError
            all_series.add(series_id)
            if study_id in observed:
                if plane != expected[study_id].get(series_id):
                    raise ValueError
                observed[study_id][series_id] = plane
    except (UnicodeError, csv.Error, TypeError, ValueError):
        raise ValueError("series_metadata_invalid") from None
    if observed != expected:
        raise ValueError("series_metadata_plan_mismatch")
    destination = Path(staging_dir)
    if destination.exists() or destination.is_symlink():
        raise ValueError("audit_staging_must_be_new")
    destination.mkdir(parents=True, exist_ok=False)
    test_csv, series_csv = destination / "test.csv", destination / "test_series.csv"
    with test_csv.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("StudyInstanceUID",))
        writer.writerows((study_id,) for study_id in study_ids)
    with series_csv.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("StudyInstanceUID", "SeriesInstanceUID"))
        for study in plan["studies"]:
            for row in study["series"]:
                writer.writerow((study["study_id"], row["series_id"]))
    return test_csv, series_csv, study_ids


@contextmanager
def dataset_constructor_injection(fold0_module, competition_root, staged_test_csv,
                                  staged_series_csv, training_image_root):
    """Temporarily redirect only fold0's dataset constructor; always restore it."""
    original = fold0_module.TestStudyDataset
    root = Path(competition_root)
    expected = (root / "test.csv", root / "test_series.csv", root / "test_series")
    staged = (Path(staged_test_csv), Path(staged_series_csv), Path(training_image_root))

    def mapped_dataset(test_csv, series_csv, image_root, **kwargs):
        received = tuple(Path(value) for value in (test_csv, series_csv, image_root))
        if received != expected or kwargs:
            raise ValueError("production_dataset_arguments_unexpected")
        return original(*staged)

    fold0_module.TestStudyDataset = mapped_dataset
    try:
        yield
    finally:
        fold0_module.TestStudyDataset = original


def run_public_training_inference_audit(*, plan_archive, expected_plan_sha256,
                                        train_series_csv, training_image_root,
                                        competition_root, source_root, checkpoint_sha256,
                                        checkpoint_size, report_sha256, output_root,
                                        fold0_module, runner, device="cuda"):
    """Execute unchanged production inference once against the frozen public-training plan."""
    plan = load_public_training_plan(plan_archive, expected_plan_sha256)
    root = Path(output_root)
    if root.exists() or root.is_symlink():
        raise ValueError("audit_output_must_be_new")
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    inference_output = root / "inference"
    with tempfile.TemporaryDirectory(prefix="training-input-", dir=root) as staging_root:
        test_csv, series_csv, study_ids = stage_training_metadata(
            plan, train_series_csv, Path(staging_root) / "metadata",
            expected_source_plan_sha256=plan["source_plan_sha256"],
            expected_series_sha256=plan["series_metadata_sha256"],
            expected_studies=plan["study_count"],
            expected_series=plan["series_count"],
            expected_plane_counts=plan["plane_counts"],
        )
        with dataset_constructor_injection(fold0_module, competition_root, test_csv,
                                           series_csv, training_image_root):
            summary = runner(competition_root, source_root, checkpoint_sha256,
                             checkpoint_size, report_sha256, inference_output, device=device)
    csv_path = inference_output / "submission.csv"
    try:
        try:
            from smoke_submission import validate_submission
        except ModuleNotFoundError:
            from scripts.smoke_submission import validate_submission
        validate_submission(csv_path, expected_ids=study_ids)
    except Exception:
        raise ValueError("audit_submission_validation_failed") from None
    try:
        import torch
        peak_allocated = int(torch.cuda.max_memory_allocated()) if device == "cuda" else None
    except Exception:
        peak_allocated = None
    result = {
        "schema_version": "public-training-inference-audit-v1",
        "status": "complete",
        "study_count": summary["study_count"],
        "series_count": summary["series_count"],
        "plane_counts": plan["plane_counts"],
        "elapsed_seconds": time.perf_counter() - started,
        "inference_elapsed_seconds": summary["elapsed_seconds"],
        "cuda_peak_allocated_bytes": peak_allocated,
        "submission_csv_bytes": csv_path.stat().st_size,
    }
    with (root / "audit-summary.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
    return result
