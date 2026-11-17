"""Build a private, blinded report-review bundle from the frozen Local16 export."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import export_local16_validation as export
from scripts.local16_full_cache import Local16CacheDataset, TARGETS as ALL_TARGETS

TARGETS = ("Contusion", "Lateral Meniscus", "MCL")
SEED = 42
PROTOCOL = {
    "schema": "local16-private-review-sample-v1",
    "targets": list(TARGETS),
    "seed": SEED,
    "high_label": {"minimum": 0.8, "error_count": 10, "control_count": 5},
    "low_label": {"maximum": 0.2, "error_count": 10, "control_count": 5},
    "agreement_absolute_error_maximum": 0.1,
    "error_order": "absolute_error_descending_then_source_ordinal_ascending",
    "control_selection": "numpy.random.default_rng(42).choice without replacement from ordinal-sorted agreements; target order then high then low",
    "observed_only": True,
    "review_order": "deterministic_shuffle_seed_42",
    "case_ids": "case-001 through case-090 assigned after review-order shuffle",
    "source": "immutable E052 validation export; no new inference",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bytes_sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _safe_file(path: Path) -> bytes:
    path = Path(path)
    if path.is_symlink() or not path.is_file() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError("input_unavailable")
    return path.read_bytes()


def _inside(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve(strict=False).relative_to(Path(root).resolve(strict=True))
        return True
    except (OSError, ValueError):
        return False


def validate_private_output(output_dir: Path, private_root: Path) -> Path:
    output = Path(output_dir).absolute()
    private = Path(private_root).resolve(strict=True)
    if (not _inside(output, private) or output == private or output.is_symlink()
            or any(part.is_symlink() for part in output.parents)):
        raise ValueError("output_path_invalid")
    if output.exists():
        raise ValueError("output_exists")
    return output


def load_reports(path: Path, required_ids: set[str] | None = None) -> dict[str, str]:
    path = Path(path)
    if (path.is_symlink() or not path.is_file()
            or any(parent.is_symlink() for parent in path.parents)):
        raise ValueError("input_unavailable")
    if (required_ids is not None and (type(required_ids) is not set
            or any(type(study_id) is not str or not study_id for study_id in required_ids))):
        raise ValueError("report_join_invalid")
    try:
        reports, seen = {}, set()
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if (not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames))
                    or not {"StudyInstanceUID", "Report"}.issubset(reader.fieldnames)):
                raise ValueError
            for row in reader:
                study_id = row.get("StudyInstanceUID")
                if not isinstance(study_id, str) or not study_id.strip() or study_id in seen:
                    raise ValueError
                seen.add(study_id)
                if required_ids is not None and study_id not in required_ids:
                    continue
                report = row.get("Report")
                if not isinstance(report, str) or not report.strip():
                    raise LookupError
                reports[study_id] = report
        if required_ids is not None:
            if set(reports) != required_ids:
                raise KeyError
        elif not reports:
            raise ValueError
        return reports
    except KeyError:
        raise ValueError("report_join_invalid") from None
    except LookupError:
        raise ValueError("report_missing") from None
    except (UnicodeError, csv.Error, TypeError, ValueError):
        raise ValueError("report_id_invalid") from None


def select_review_rows(rows: list[dict], seed: int = SEED) -> list[dict]:
    """Select exactly 30 rows per target, failing closed when any quota is infeasible."""
    if type(seed) is not int or seed != SEED or type(rows) is not list:
        raise ValueError("selection_protocol_invalid")
    ordinals, study_ids = set(), set()
    validated = []
    target_slots = {name: ALL_TARGETS.index(name) for name in TARGETS}
    for row in rows:
        if type(row) is not dict:
            raise ValueError("prediction_row_invalid")
        ordinal, study_id = row.get("ordinal"), row.get("study_id")
        if type(ordinal) is not int or ordinal < 1 or ordinal in ordinals:
            raise ValueError("prediction_row_invalid")
        if type(study_id) is not str or not study_id or study_id in study_ids:
            raise ValueError("prediction_row_invalid")
        ordinals.add(ordinal)
        study_ids.add(study_id)
        predictions, labels, observed = (row.get(key) for key in ("predictions", "labels", "observed_mask"))
        if any(type(values) is not list or len(values) != len(ALL_TARGETS)
               for values in (predictions, labels, observed)):
            raise ValueError("prediction_row_invalid")
        if any(type(flag) is not bool for flag in observed):
            raise ValueError("prediction_row_invalid")
        for values in (predictions, labels):
            if any(type(value) not in (float, int) or not math.isfinite(value) for value in values):
                raise ValueError("prediction_row_invalid")
        validated.append(row)

    selected = []
    import numpy as np

    control_rng = np.random.default_rng(seed)
    for target in TARGETS:
        slot = target_slots[target]
        target_rows = [row for row in validated if row["observed_mask"][slot]]
        for group, threshold, direction in (("high", 0.8, "high"), ("low", 0.2, "low")):
            label_rows = [row for row in target_rows
                          if (row["labels"][slot] >= threshold if direction == "high"
                              else row["labels"][slot] <= threshold)]
            error_rows = sorted(label_rows,
                                key=lambda row: (-abs(row["predictions"][slot] - row["labels"][slot]),
                                                 row["ordinal"]))
            if len(error_rows) < 10:
                raise ValueError("selection_quota_insufficient")
            chosen_errors = error_rows[:10]
            chosen_ids = {row["study_id"] for row in chosen_errors}
            controls = [row for row in label_rows
                        if row["study_id"] not in chosen_ids
                        and abs(row["predictions"][slot] - row["labels"][slot]) <= 0.1]
            controls.sort(key=lambda row: row["ordinal"])
            if len(controls) < 5:
                raise ValueError("selection_quota_insufficient")
            chosen_controls = [controls[int(index)] for index in
                               control_rng.choice(len(controls), size=5, replace=False)]
            for source, stratum in ((chosen_errors, f"{group}_error"),
                                    (chosen_controls, f"{group}_control")):
                for row in source:
                    selected.append({
                        "target": target,
                        "study_id": row["study_id"],
                        "ordinal": row["ordinal"],
                        "prediction": float(row["predictions"][slot]),
                        "soft_label": float(row["labels"][slot]),
                        "observed": bool(row["observed_mask"][slot]),
                        "selection_stratum": stratum,
                        "absolute_error": abs(row["predictions"][slot] - row["labels"][slot]),
                    })
    rng = random.Random(seed)
    rng.shuffle(selected)
    for index, row in enumerate(selected, 1):
        row["case_id"] = f"case-{index:03d}"
    if len(selected) != 90 or len({row["case_id"] for row in selected}) != 90:
        raise ValueError("selection_coverage_invalid")
    return selected


def _atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _preview_path(case_id: str, index: int) -> str:
    return f"previews/{case_id}-{index:03d}.png"


def build_review_sample(*, selected_rows: list[dict], reports: dict[str, str], output_dir: Path,
                        private_root: Path, input_provenance: dict, preview_provider,
                        code_path: Path) -> dict:
    """Atomically publish review artifacts; failed staging remains private for diagnosis."""
    output = validate_private_output(output_dir, private_root)
    if (type(selected_rows) is not list or len(selected_rows) != 90 or type(reports) is not dict
            or type(input_provenance) is not dict or not callable(preview_provider)):
        raise ValueError("review_bundle_input_invalid")
    by_study = {}
    for row in selected_rows:
        if row.get("study_id") not in reports or not reports[row["study_id"]].strip():
            raise ValueError("report_join_invalid")
        by_study.setdefault(row["study_id"], row)
    if not by_study or any(row.get("observed") is not True for row in selected_rows):
        raise ValueError("review_bundle_input_invalid")
    private = Path(private_root).resolve(strict=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not _inside(output.parent, private) or any(part.is_symlink() for part in output.parent.parents):
        raise ValueError("output_path_invalid")
    if output.exists():
        raise ValueError("output_exists")
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=private))
    try:
        for directory in ("cases", "previews"):
            (staging / directory).mkdir()
        preview_links = {}
        for study_id, first_row in by_study.items():
            supplied = preview_provider(study_id, first_row)
            if type(supplied) is not list or not supplied:
                raise ValueError("preview_unavailable")
            links = []
            series_counts = {}
            for index, item in enumerate(supplied, 1):
                if (type(item) is not dict or type(item.get("png_bytes")) is not bytes
                        or not item["png_bytes"] or len(item["png_bytes"]) > 8 * 1024 * 1024):
                    raise ValueError("preview_invalid")
                series_ordinal = item.get("series_ordinal", 1)
                if type(series_ordinal) is not int or series_ordinal < 1:
                    raise ValueError("preview_invalid")
                series_counts[series_ordinal] = series_counts.get(series_ordinal, 0) + 1
                if series_counts[series_ordinal] > 3:
                    raise ValueError("preview_invalid")
                case_id = first_row["case_id"]
                path = _preview_path(case_id, index)
                _atomic_write(staging / path, item["png_bytes"])
                links.append({"path": path, "label": _preview_label(item, series_ordinal)})
            preview_links[study_id] = links

        template_path = staging / "review-template.csv"
        with template_path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["case_id", "target", "decision", "confidence", "note"])
            writer.writeheader()
            for row in selected_rows:
                writer.writerow({"case_id": row["case_id"], "target": row["target"],
                                 "decision": "", "confidence": "", "note": ""})

        case_links = []
        unblinding = []
        for row in selected_rows:
            case_id, study_id = row["case_id"], row["study_id"]
            report = reports[study_id]
            links = preview_links[study_id]
            case_data = {"case_id": case_id, "target": row["target"], "original_report": report,
                         "previews": links, "review": {"decision": "", "confidence": "", "note": ""}}
            _atomic_write(staging / "cases" / f"{case_id}.json", _json_bytes(case_data))
            preview_html = "".join(
                f'<figure><img src="../{html.escape(link["path"], quote=True)}" alt="Cached 224px preview, not diagnostic or full DICOM"><figcaption>{html.escape(link["label"])}</figcaption></figure>'
                for link in links)
            case_html = _case_html(case_id, row["target"], report, preview_html)
            _atomic_write(staging / "cases" / f"{case_id}.html", case_html.encode("utf-8"))
            case_links.append(f'<li><a href="cases/{case_id}.html">{case_id}</a> — {html.escape(row["target"])}</li>')
            unblinding.append({
                "case_id": case_id, "target": row["target"], "source_study_id": study_id,
                "source_ordinal": row["ordinal"], "prediction": row["prediction"],
                "soft_label": row["soft_label"], "observed_mask": row["observed"],
                "selection_stratum": row["selection_stratum"], "absolute_error": row["absolute_error"],
            })

        _atomic_write(staging / "index.html", _index_html(case_links).encode("utf-8"))
        _atomic_write(staging / "unblinding.json", _json_bytes({
            "warning": "DO NOT OPEN UNTIL BLIND REVIEW IS COMPLETE",
            "schema": PROTOCOL["schema"], "cases": unblinding,
        }))
        _atomic_write(staging / "README.txt", _readme().encode("utf-8"))
        inventory = {}
        for path in sorted(staging.rglob("*")):
            if path.is_symlink():
                raise ValueError("output_inventory_invalid")
            if path.is_file():
                inventory[path.relative_to(staging).as_posix()] = _sha256(path)
        receipt = {
            "schema": PROTOCOL["schema"], "selection_protocol": PROTOCOL,
            "input_provenance": input_provenance,
            "builder_sha256": _sha256(code_path),
            "case_count": 90, "unique_study_count": len(by_study),
            "target_case_counts": {target: sum(row["target"] == target for row in selected_rows)
                                   for target in TARGETS},
            "inventory_sha256": inventory,
        }
        _atomic_write(staging / "receipt.json", _json_bytes(receipt))
        if output.exists():
            raise ValueError("output_exists")
        os.rename(staging, output)
        return {"status": "built", "case_count": 90, "unique_study_count": len(by_study),
                "target_case_counts": receipt["target_case_counts"], "output_dir": str(output)}
    except Exception:
        # Keep failed staging private; no partially published output is left behind.
        raise


def _preview_label(item: dict, index: int) -> str:
    series = item.get("series_ordinal")
    if type(series) is not int or series < 1:
        series = index
    plane = item.get("plane")
    suffix = f" · {plane}" if isinstance(plane, str) and plane.strip() else ""
    return f"Series {series:03d}{suffix} · cached 224px preview"


def _case_html(case_id: str, target: str, report: str, preview_html: str) -> str:
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(case_id)} review</title><style>body{{font:16px system-ui;max-width:900px;margin:2rem auto;padding:0 1rem;color:#20242a}}figure{{display:inline-block;margin:.5rem}}img{{width:224px;height:224px;object-fit:contain;background:#111}}.notice{{background:#fff4ce;padding:.7rem}}textarea{{width:100%;min-height:5rem}}label{{display:block;margin:.7rem 0}}</style></head><body><p><a href="../index.html">Back to blinded queue</a></p><h1>{html.escape(case_id)} — {html.escape(target)}</h1><p class="notice">Cached 224px preview, not diagnostic quality or full DICOM. Expert review is required.</p><section aria-label="Cached image previews">{preview_html}</section><h2>Original report</h2><pre>{html.escape(report)}</pre><h2>Review</h2><label>Decision <input name="decision" autocomplete="off"></label><label>Confidence <input name="confidence" autocomplete="off"></label><label>Notes <textarea name="note"></textarea></label></body></html>'''


def _index_html(case_links: list[str]) -> str:
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>Blinded private review queue</title><style>body{font:16px system-ui;max-width:900px;margin:2rem auto;padding:0 1rem}</style></head><body>'
            '<h1>Blinded private review queue</h1><p>This enriched sample is not representative and cannot estimate accuracy or prevalence. '
            'Review reports independently; annotations are not automatically applied.</p><p>Do not open the unblinding key until blind review is complete.</p><ol>'
            + "\n".join(case_links) + '</ol></body></html>')


def _readme() -> str:
    return """Private Local16 report-review bundle

This is a deliberately enriched 90-target-case sample, not a representative cohort and not suitable for estimating accuracy or prevalence. The cached 224px previews are not diagnostic-quality images or full DICOMs. Expert review is required. Review annotations are not automatically applied to labels, reports, training, or inference.

Start with index.html and review-template.csv. Do not open unblinding.json until blind review is complete. All source IDs, predictions, labels, strata, and original reports are private. Keep this directory under ignored data/private and do not publish or transfer it.
"""


def _input_paths():
    export_defaults = export._defaults()
    full = ROOT / "data/private/full-cohort-prefilter/run01"
    return {
        "export_dir": ROOT / "data/private/local16-validation-export/run01",
        "manifest_path": full / "full-cohort-soft-label-manifest.json",
        "folds_path": full / "full-cohort-grouped-folds.json",
        "plan_path": ROOT / "data/private/attention-sixteen-slices/cache-package-run05/local16-cache-chunk-plan.json",
        "archive_dir": ROOT / "data/private/attention-sixteen-slices/cache-local-run01/archives",
        "train_csv": ROOT / "data/train.csv",
        "model_path": export_defaults["model_path"],
        "report_path": export_defaults["report_path"],
        "model_sha256": export.MODEL_SHA256,
        "report_sha256": export.REPORT_SHA256,
        "output_dir": ROOT / "data/private/local16-review-sample/run01",
    }


def _validate_sources(paths: dict) -> tuple[list[dict], dict[str, str], Local16CacheDataset, dict]:
    manifest_raw, folds_raw, plan_raw, plan, ids, by_id = export._cohort(
        paths["manifest_path"], paths["folds_path"], paths["plan_path"])
    if len(ids) != export.EXPECTED_STUDIES:
        raise ValueError("validation_cohort_size_mismatch")
    meta_path = paths["export_dir"] / "provenance.json"
    progress_path = paths["export_dir"] / "progress.json"
    try:
        metadata = json.loads(_safe_file(meta_path).decode("utf-8"))
        progress = json.loads(_safe_file(progress_path).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        raise ValueError("export_receipt_invalid") from None
    if (type(metadata) is not dict or metadata.get("schema") != export.SCHEMA
            or metadata.get("targets") != list(ALL_TARGETS) or metadata.get("study_count") != len(ids)
            or type(metadata.get("provenance")) is not dict
            or progress.get("status") != "complete" or progress.get("completed") != len(ids)
            or progress.get("expected") != len(ids) or progress.get("provenance") != metadata["provenance"]):
        raise ValueError("export_receipt_invalid")
    provenance = metadata["provenance"]
    export._pins(paths["model_path"], paths["report_path"], paths["model_sha256"], paths["report_sha256"])
    rows = export._resume_prefix(paths["export_dir"], ids, provenance, by_id)
    if len(rows) != len(ids) or progress.get("metrics") != export._metrics(rows):
        raise ValueError("export_coverage_invalid")
    reports = load_reports(paths["train_csv"], required_ids=set(ids))
    if set(reports) != set(ids):
        raise ValueError("report_join_invalid")
    cache = Local16CacheDataset(paths["manifest_path"], paths["plan_path"], paths["archive_dir"], "validation")
    if len(cache) != len(ids) or len(cache.rows) != len(ids):
        raise ValueError("cache_study_coverage_mismatch")
    for index, study_id in enumerate(ids):
        cache_record = cache.rows[index][1]
        record = by_id[study_id]
        expected_digest = hashlib.sha256(study_id.encode("utf-8")).hexdigest()
        expected_series = [series["series_id"] for series in record["series"]]
        expected_order = hashlib.sha256(b"\0".join(value.encode() for value in expected_series)).hexdigest()
        if (cache_record.get("ordinal") != index or cache_record.get("study_sha256") != expected_digest
                or cache_record.get("series_count") != len(expected_series)
                or cache_record.get("series_order_sha256") != expected_order):
            raise ValueError("cache_study_order_mismatch")
    input_provenance = {
        "export_provenance": provenance,
        "input_sha256": {
            "manifest": _bytes_sha(manifest_raw), "folds": _bytes_sha(folds_raw),
            "cache_plan": _bytes_sha(plan_raw), "train_csv": _sha256(paths["train_csv"]),
            "export_provenance": _sha256(meta_path), "export_progress": _sha256(progress_path),
            "model": paths["model_sha256"], "training_report": paths["report_sha256"],
        },
        "export_inventory_sha256": _inventory(paths["export_dir"]),
        "protocol": PROTOCOL,
        "cache_limitation": "Only selected accessed NPZ payload hashes are validated; no full 61.2 GB cache verification is repeated.",
    }
    return rows, reports, cache, {"ids": ids, "by_id": by_id, "provenance": input_provenance}


def _inventory(directory: Path) -> dict[str, str]:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("input_unavailable")
    inventory = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("input_inventory_invalid")
        if path.is_file():
            inventory[path.relative_to(directory).as_posix()] = _sha256(path)
    return inventory


def _preview_provider(cache: Local16CacheDataset, info: dict):
    def provide(study_id: str, _selected_row: dict) -> list[dict]:
        import numpy as np
        import torch
        from PIL import Image
        import io

        index = info["ids"].index(study_id)
        images, masks, targets, observed, sample_index = cache[index]
        expected = info["by_id"][study_id]
        expected_targets = np.asarray([expected["soft_targets"][name] for name in ALL_TARGETS], dtype=np.float32)
        expected_observed = np.asarray([expected["observed_mask"][name] for name in ALL_TARGETS], dtype=np.bool_)
        if (sample_index != index or not np.array_equal(targets.cpu().numpy(), expected_targets)
                or not np.array_equal(observed.cpu().numpy(), expected_observed)
                or len(images) != len(expected["series"]) or len(masks) != len(images)):
            raise ValueError("cache_study_payload_invalid")
        result = []
        for series_index, (image_tensor, mask_tensor, series_meta) in enumerate(zip(images, masks, expected["series"]), 1):
            image = image_tensor.detach().cpu().numpy()
            mask = mask_tensor.detach().cpu().numpy()
            if (image.ndim != 4 or image.shape[1:] != (1, 224, 224) or image.shape != mask.shape
                    or image.dtype != np.float32 or mask.dtype != np.bool_ or not np.isfinite(image).all()
                    or np.any(image < 0) or np.any(image > 1) or not mask.reshape(mask.shape[0], -1).any(axis=1).all()
                    or np.any(image[~mask] != 0)):
                raise ValueError("cache_preview_invalid")
            valid = np.flatnonzero(mask.reshape(mask.shape[0], -1).any(axis=1))
            chosen = sorted({int(valid[round(i * (len(valid) - 1) / min(2, len(valid) - 1))])
                             for i in range(min(3, len(valid)))}) if len(valid) > 1 else [int(valid[0])]
            for slice_index in chosen:
                pixels = np.clip(image[slice_index, 0] * 255, 0, 255).astype(np.uint8)
                buffer = io.BytesIO()
                Image.fromarray(pixels, mode="L").save(buffer, format="PNG", optimize=True)
                item = {"png_bytes": buffer.getvalue(), "series_ordinal": series_index}
                if isinstance(series_meta.get("plane"), str):
                    item["plane"] = series_meta["plane"]
                result.append(item)
                # Each series contributes no more than three valid ordered slices.
        if not result:
            raise ValueError("preview_unavailable")
        return result
    return provide


def prepare(paths: dict, *, build: bool = False) -> dict:
    private_root = (ROOT / "data/private").resolve(strict=True)
    output = validate_private_output(paths["output_dir"], private_root)
    rows, reports, cache, info = _validate_sources(paths)
    selected = select_review_rows(rows)
    if not build:
        return {"status": "planned", "target_case_counts": {
                    target: sum(row["target"] == target for row in selected) for target in TARGETS},
                "expected_review_rows": len(selected),
                "unique_study_count": len({row["study_id"] for row in selected}),
                "output_dir": str(output), "build_started": False}
    for row in selected:
        if row["study_id"] not in reports or not reports[row["study_id"]].strip():
            raise ValueError("report_join_invalid")
    return build_review_sample(selected_rows=selected, reports=reports, output_dir=output,
                               private_root=private_root, input_provenance=info["provenance"],
                               preview_provider=_preview_provider(cache, info), code_path=Path(__file__))


def _defaults() -> dict:
    return _input_paths()


def main(argv=None) -> int:
    defaults = _defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    for name, value in defaults.items():
        if name in ("model_sha256", "report_sha256"):
            parser.add_argument("--" + name.replace("_", "-"), default=value)
        else:
            parser.add_argument("--" + name.replace("_", "-"), type=Path, default=value)
    parser.add_argument("--build", action="store_true", help="explicitly build the private review bundle")
    args = vars(parser.parse_args(argv))
    private_root = (ROOT / "data/private").resolve(strict=True)
    args["output_dir"] = validate_private_output(args["output_dir"], private_root)
    try:
        result = prepare(args, build=args.pop("build"))
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except (OSError, ValueError, KeyError, ImportError) as error:
        print(json.dumps({"status": "failed", "reason": str(error)}, sort_keys=True), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
