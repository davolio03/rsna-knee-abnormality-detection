"""Prepare a private, diagnostic-only full public-training inference notebook."""
from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

from scripts.build_kaggle_submission_package import build_submission_package

SOURCE_PLAN = "data/private/training-compatibility-audit/run01-preparation-v2/audit-plan.json"
SOURCE_PLAN_SHA256 = "5f9fc0f7beea764e975ee3e7f263aeb66db314f43cfb0b5d4dcc69a265abd1f6"
SERIES_METADATA_SHA256 = "573c1d80772bf41211c91b149c95677385a1c22d63f485c347f1b46c0177aef3"
CODEC_MANIFEST_SHA256 = "50f39d3f627b3dc06a8080ba51dd6424898ef34fa4c4e21a84593dd326eb776d"
CHECKPOINT_SHA256 = "d85f6531aeb8aa3035f05997ef3dfbb2f88a97c288c07b8786a13795fdf33447"
CHECKPOINT_SIZE = 44811979
REPORT_SHA256 = "5daef84f7ed8ad9aa7173989fe80fc168a900fc196025feeb98ee9c8c1792dc1"
SOURCE_ROOT = "/kaggle/input/notebooks/davidsulleiro/rsna-knee-resnet18-frozen-backbone-run01/resnet18-frozen-backbone/run01"
PLAN_MEMBER = "public-training-inference-plan.json"
GOLD_IDS = "data/private/gold58_study_ids.csv"
FULL_GOLD_SOURCE_SHA256 = "e0e584d475a7e8b0749ffcce70267ac19a2e2549338f98f50ed515f8fabce3a1"
FULL_STUDIES = 4349
FULL_SERIES = 24035
FULL_GOLD_SERIES = 336
FULL_PLAN_COUNTS = {"Axial": 5817, "Coronal": 8490, "Sagittal": 9728}
FULL_PRIOR_COUNT = 500


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _private_plan(source_plan_path):
    raw = Path(source_plan_path).read_bytes()
    if _sha(raw) != SOURCE_PLAN_SHA256:
        raise ValueError("frozen_training_plan_pin_mismatch")
    plan = json.loads(raw)
    studies = plan.get("studies") if type(plan) is dict else None
    if (plan.get("schema_version") != "training-compatibility-v1"
            or plan.get("seed") != 42 or plan.get("sample_size") != 100
            or plan.get("population_study_count") != 4407
            or plan.get("fresh_study_count") != 3849
            or plan.get("source_sha256", {}).get("series") != SERIES_METADATA_SHA256
            or type(studies) is not list or len(studies) != 100
            or len(plan.get("gold_ids", [])) != 58 or len(plan.get("prior_ids", [])) != 500):
        raise ValueError("frozen_training_plan_invalid")
    if set(plan["gold_ids"]) & set(plan["prior_ids"]):
        raise ValueError("frozen_training_plan_invalid")
    excluded = set(plan["gold_ids"]) | set(plan["prior_ids"])
    sample_ids = [row.get("study_id") for row in studies]
    if (len(set(sample_ids)) != 100 or set(sample_ids) & excluded
            or any(type(value) is not str for value in sample_ids)):
        raise ValueError("frozen_training_plan_invalid")
    series_count = 0
    plane_counts = {"Axial": 0, "Coronal": 0, "Sagittal": 0}
    seen_series = set()
    compact_studies = []
    for study in studies:
        if type(study) is not dict or set(study) != {"study_id", "series"}:
            raise ValueError("frozen_training_plan_invalid")
        compact_rows = []
        for row in study["series"]:
            if (type(row) is not dict or set(row) != {"series_id", "plane"}
                    or row["plane"] not in plane_counts or row["series_id"] in seen_series):
                raise ValueError("frozen_training_plan_invalid")
            seen_series.add(row["series_id"])
            plane_counts[row["plane"]] += 1
            series_count += 1
            compact_rows.append({"series_id": row["series_id"], "plane": row["plane"]})
        compact_studies.append({"study_id": study["study_id"], "series": compact_rows})
    if (series_count != 546 or plane_counts != {"Axial": 131, "Coronal": 197, "Sagittal": 218}
            or len(seen_series) != series_count):
        raise ValueError("frozen_training_plan_invalid")
    compact = {
        "schema_version": "public-training-inference-plan-v1",
        "source_plan_sha256": SOURCE_PLAN_SHA256,
        "series_metadata_sha256": SERIES_METADATA_SHA256,
        "population_study_count": 4407,
        "fresh_study_count": 3849,
        "excluded_study_count": 558,
        "seed": 42,
        "study_count": 100,
        "series_count": 546,
        "plane_counts": plane_counts,
        "studies": compact_studies,
    }
    return (json.dumps(compact, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _full_private_plan(repository, source_plan_path=None):
    """Build deterministic ID-only metadata for every training study except Gold-58."""
    source_path = Path(source_plan_path) if source_plan_path else Path(repository) / SOURCE_PLAN
    source_raw = source_path.read_bytes()
    if _sha(source_raw) != SOURCE_PLAN_SHA256:
        raise ValueError("frozen_training_plan_pin_mismatch")
    source_plan = json.loads(source_raw)
    if (type(source_plan) is not dict or source_plan.get("source_sha256", {}).get("series") != SERIES_METADATA_SHA256
            or source_plan.get("source_sha256", {}).get("gold") != FULL_GOLD_SOURCE_SHA256
            or len(source_plan.get("gold_ids", [])) != 58
            or len(source_plan.get("prior_ids", [])) != FULL_PRIOR_COUNT):
        raise ValueError("frozen_training_plan_invalid")

    gold_path = Path(repository) / GOLD_IDS
    gold_raw = gold_path.read_bytes()
    if _sha(gold_raw) != FULL_GOLD_SOURCE_SHA256:
        raise ValueError("frozen_gold_ids_pin_mismatch")
    try:
        with gold_path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["StudyInstanceUID"]:
                raise ValueError
            gold_rows = [row["StudyInstanceUID"] for row in reader]
    except (OSError, UnicodeError, csv.Error, TypeError, ValueError):
        raise ValueError("frozen_gold_ids_invalid") from None
    gold_ids = set(gold_rows)
    if (len(gold_rows) != 58 or len(gold_ids) != 58
            or gold_ids != set(source_plan["gold_ids"])):
        raise ValueError("frozen_gold_ids_mismatch")

    metadata_path = Path(repository) / "data/train_series.csv"
    metadata_raw = metadata_path.read_bytes()
    if _sha(metadata_raw) != SERIES_METADATA_SHA256:
        raise ValueError("series_metadata_pin_mismatch")
    grouped = {}
    seen_series = set()
    all_planes = {"Axial": 0, "Coronal": 0, "Sagittal": 0}
    excluded_series = 0
    try:
        with metadata_path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if (reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames)
                    or not {"StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane"}.issubset(reader.fieldnames)):
                raise ValueError
            for row in reader:
                study_id, series_id, plane = (row.get("StudyInstanceUID"), row.get("SeriesInstanceUID"),
                                              row.get("Anatomical_Plane"))
                if (not study_id or not series_id or series_id in seen_series or plane not in all_planes):
                    raise ValueError
                seen_series.add(series_id)
                all_planes[plane] += 1
                if study_id in gold_ids:
                    excluded_series += 1
                else:
                    grouped.setdefault(study_id, []).append({"series_id": series_id, "plane": plane})
    except (OSError, UnicodeError, csv.Error, TypeError, ValueError):
        raise ValueError("series_metadata_invalid") from None

    prior_ids = set(source_plan["prior_ids"])
    if (len(grouped) != 4349 or len(seen_series) != 24371 or len(all_planes) != 3
            or all_planes != {"Axial": 5898, "Coronal": 8609, "Sagittal": 9864}
            or excluded_series != FULL_GOLD_SERIES or not prior_ids.issubset(grouped)):
        raise ValueError("full_training_population_mismatch")
    for rows in grouped.values():
        rows.sort(key=lambda row: row["series_id"])
    studies = [{"study_id": study_id, "series": grouped[study_id]} for study_id in sorted(grouped)]
    plane_counts = {key: sum(row["plane"] == key for study in studies for row in study["series"])
                    for key in FULL_PLAN_COUNTS}
    series_count = sum(len(study["series"]) for study in studies)
    if len(prior_ids & set(grouped)) != FULL_PRIOR_COUNT or (series_count != FULL_SERIES
                                                            or plane_counts != FULL_PLAN_COUNTS):
        raise ValueError("full_training_population_mismatch")
    compact = {
        "schema_version": "full-public-training-inference-plan-v1",
        "source_plan_sha256": SOURCE_PLAN_SHA256,
        "series_metadata_sha256": SERIES_METADATA_SHA256,
        "gold_source_sha256": FULL_GOLD_SOURCE_SHA256,
        "population_study_count": 4407,
        "gold_excluded_study_count": 58,
        "gold_excluded_series_count": excluded_series,
        "prior_study_count_included": FULL_PRIOR_COUNT,
        "study_count": len(studies),
        "series_count": series_count,
        "plane_counts": plane_counts,
        "studies": studies,
    }
    return (json.dumps(compact, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _zip_bytes(members):
    output = io.BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for name, raw in sorted(members.items()):
            info = ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, raw)
    return output.getvalue()


def build_public_training_audit_package(repository_root, output_dir, *, source_root=SOURCE_ROOT,
                                        checkpoint_sha256=CHECKPOINT_SHA256,
                                        checkpoint_size=CHECKPOINT_SIZE,
                                        report_sha256=REPORT_SHA256,
                                        codec_manifest_sha256=CODEC_MANIFEST_SHA256,
                                        source_plan_path=None, full_training=False):
    repository = Path(repository_root).resolve(strict=True)
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise ValueError("preparation_output_must_be_new")
    plan_raw = (_full_private_plan(repository, source_plan_path) if full_training
                else _private_plan(source_plan_path or repository / SOURCE_PLAN))
    plan_digest = _sha(plan_raw)
    plan_zip = _zip_bytes({PLAN_MEMBER: plan_raw})
    plan_archive_digest = _sha(plan_zip)
    mounted_plan = ("/kaggle/input/datasets/davidsulleiro/rsna-knee-full-training-inference-plan/full-training-inference-plan.json"
                    if full_training else
                    "/kaggle/input/datasets/davidsulleiro/rsna-knee-public-training-inference-plan/public-training-inference-plan/public-training-inference-plan.json")
    output_path = ("/kaggle/working/full-public-training-inference-audit/run01" if full_training
                   else "/kaggle/working/public-training-inference-audit/run01")

    base = build_submission_package(
        repository, output, source_root=source_root,
        checkpoint_sha256=checkpoint_sha256, checkpoint_size=checkpoint_size,
        report_sha256=report_sha256, codec_manifest_sha256=codec_manifest_sha256,
        run_inference=True,
    )
    base_archive_path = output / "fold0-inference-source.zip"
    base_archive_bytes = base_archive_path.read_bytes()
    with ZipFile(base_archive_path) as source_archive:
        members = {name: source_archive.read(name) for name in source_archive.namelist()}
    extra = {
        "rsna_knee/inference/public_training_audit.py": repository.joinpath(
            "src/rsna_knee/inference/public_training_audit.py").read_bytes(),
        "smoke_submission.py": repository.joinpath("scripts/smoke_submission.py").read_bytes(),
    }
    for name, raw in extra.items():
        if name in members:
            raise ValueError("diagnostic_source_member_collision")
        ast.parse(raw, filename=name)
        members[name] = raw
    source_bytes = _zip_bytes(members)
    source_digest = _sha(source_bytes)
    base_archive_path.write_bytes(source_bytes)

    inventory_path = output / "inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory.update({
        "schema_version": "fold0-public-training-inference-audit-source-v1",
        "archive_sha256": source_digest,
        "members": {name: _sha(raw) for name, raw in sorted(members.items())},
        "remote_executed": False,
        "competition_submission": False,
        "audit_only": True,
        "scope": ("all 4,349 non-Gold-58 public-training studies" if full_training
                  else "100 frozen public-training studies / 546 series"),
        "plan_sha256": plan_digest,
        "plan_archive_sha256": plan_archive_digest,
    })
    inventory_path.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")

    notebook_path = Path(base["notebook"])
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook["cells"][0]["source"] = [
        "# RSNA Knee: private full non-Gold public-training inference audit\n",
        "Diagnostic only: outcome-blind plan covers all 4,349 non-Gold-58 studies; no training or competition submission.\n",
        "Keep private, attach official competition inputs, the exact frozen model-output version, and the private ID-only plan dataset.\n",
        "Internet OFF; T4x2. Offline codec wheel dataset is pinned. All outputs and the training-ID CSV remain private.\n",
        "This tests the actual fold-zero dataset/preprocessor/model/runner/CSV path; it does not identify hidden root cause by itself.\n",
        "Never submit this diagnostic notebook to the competition.\n",
    ] if full_training else [
        "# RSNA Knee: private full public-training inference audit\n",
        "Diagnostic only: frozen outcome-blind 100-study / 546-series public-training plan; no training or competition submission.\n",
        "Keep private, attach official competition inputs, the exact frozen model-output version, and the private ID-only plan dataset.\n",
        "Internet OFF; T4x2. Offline codec wheel dataset is pinned. All outputs and the training-ID CSV remain private.\n",
        "This tests the actual fold-zero dataset/preprocessor/model/runner/CSV path; it does not identify hidden root cause by itself.\n",
        "Never submit this diagnostic notebook to the competition.\n",
    ]
    code = "".join(notebook["cells"][1]["source"])
    code = code.replace("RUN_INFERENCE = True", "RUN_AUDIT = True")
    code = code.replace("if RUN_INFERENCE:", "if RUN_AUDIT:")
    code = code.replace("else:\n    print('Inference disabled: verify the exact fold-zero mount and artifact pins before enabling.')",
                        "else:\n    print('Diagnostic audit disabled: verify private inputs and pins before enabling.')")
    code = code.replace(
        f"payload = base64.b64decode('{base64.b64encode(base_archive_bytes).decode()}', validate=True)",
        f"payload = base64.b64decode('{base64.b64encode(source_bytes).decode()}', validate=True)",
    )
    code = code.replace(f"!= '{base['archive_sha256']}'", f"!= '{source_digest}'")
    # The original runner is imported only after the pinned codecs and source archive are ready.
    import_marker = "        from rsna_knee.inference.fold0 import run_inference\n"
    if code.count(import_marker) != 1:
        raise ValueError("inference_import_marker_invalid")
    code = code.replace(import_marker, import_marker +
                        "        from rsna_knee.inference import fold0\n"
                        "        from rsna_knee.inference.public_training_audit import run_public_training_inference_audit\n")
    start_marker = "    summary = run_inference(COMPETITION_ROOT, source, CHECKPOINT_SHA256,"
    end_marker = "\nelse:\n    print('Diagnostic audit disabled: verify private inputs and pins before enabling.')"
    if start_marker not in code or end_marker not in code:
        raise ValueError("inference_call_markers_invalid")
    start = code.index(start_marker)
    end = code.index(end_marker, start)
    diagnostic_call = f'''    try:
        summary = run_public_training_inference_audit(
            plan_archive=Path('{mounted_plan}'),
            expected_plan_sha256='{plan_digest}',
            train_series_csv=Path(COMPETITION_ROOT) / 'train_series.csv',
            training_image_root=Path(COMPETITION_ROOT) / 'train_series',
            competition_root=COMPETITION_ROOT,
            source_root=source,
            checkpoint_sha256=CHECKPOINT_SHA256,
            checkpoint_size=CHECKPOINT_SIZE,
            report_sha256=REPORT_SHA256,
            output_root=Path('{output_path}'),
            fold0_module=fold0,
            runner=run_inference,
            device='cuda',
        )
        print(json.dumps({{'event': 'public_training_inference_audit_completed', **summary}}, sort_keys=True), flush=True)
    except Exception as error:
        from rsna_knee.inference.fold0 import InferenceError
        reason = error.reason_code if isinstance(error, InferenceError) else 'audit_failed'
        ordinal = error.study_ordinal if isinstance(error, InferenceError) else None
        print(json.dumps({{'event': 'public_training_inference_audit_failed',
                          'reason_code': reason, 'study_ordinal': ordinal}}, sort_keys=True), flush=True)
        raise RuntimeError('public_training_inference_audit_failed') from None
'''
    code = code[:start] + diagnostic_call + code[end:]
    compile(code, "public-training-inference-audit-notebook", "exec")
    notebook["cells"][1]["source"] = code.splitlines(keepends=True)
    notebook_path.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    if full_training:
        plan_output = output / "full-training-inference-plan.json"
        plan_output.write_bytes(plan_raw)
        mounted_plan = "/kaggle/input/datasets/davidsulleiro/rsna-knee-full-training-inference-plan/full-training-inference-plan.json"
        result_plan = {"plan_json": str(plan_output), "plan_sha256": plan_digest}
    else:
        plan_output = output / "public-training-inference-plan.zip"
        with plan_output.open("xb") as stream:
            stream.write(plan_zip)
        mounted_plan = "/kaggle/input/datasets/davidsulleiro/rsna-knee-public-training-inference-plan/public-training-inference-plan/public-training-inference-plan.json"
        result_plan = {"plan_archive": str(plan_output), "plan_sha256": plan_archive_digest}
    proof = {
        "schema_version": ("full-public-training-inference-audit-preparation-v1" if full_training
                            else "public-training-inference-audit-preparation-v1"),
        "notebook_sha256": _sha(notebook_path.read_bytes()),
        "source_archive_sha256": source_digest,
        "source_archive_members": len(members),
        "source_plan_sha256": SOURCE_PLAN_SHA256,
        "plan_sha256": plan_digest,
        "plan_archive_sha256": None if full_training else plan_archive_digest,
        "plan_upload_file": plan_output.name,
        "plan_upload_bytes": len(plan_raw),
        "series_metadata_sha256": SERIES_METADATA_SHA256,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size": checkpoint_size,
        "report_sha256": report_sha256,
        "codec_manifest_sha256": codec_manifest_sha256,
        "study_count": 4349 if full_training else 100,
        "series_count": 24035 if full_training else 546,
        "plane_counts": FULL_PLAN_COUNTS if full_training else {"Axial": 131, "Coronal": 197, "Sagittal": 218},
        "remote_executed": False,
        "competition_submission": False,
        "trained_checkpoint_loaded_locally": False,
        "internet_enabled": False,
        "device": "cuda",
        "diagnostic_only": True,
    }
    (output / "preparation-proof.json").write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    return {
        "notebook": str(notebook_path),
        "notebook_sha256": proof["notebook_sha256"],
        "source_archive": str(base_archive_path),
        "source_archive_sha256": source_digest,
        **result_plan,
        "study_count": 4349 if full_training else 100,
        "series_count": 24035 if full_training else 546,
        "competition_submission": False,
    }


def build_full_public_training_audit_package(repository_root, output_dir, **kwargs):
    """Prepare the full non-Gold diagnostic with the same pinned production runner."""
    return build_public_training_audit_package(
        repository_root, output_dir, full_training=True, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--full-training", action="store_true",
                        help="build the deterministic full non-Gold-58 public-training audit")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    builder = build_full_public_training_audit_package if args.full_training else build_public_training_audit_package
    print(json.dumps(builder(repository, args.output_dir), indent=2))
