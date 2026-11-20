"""Build a Kaggle CPU kernel that caches studies missing from the local16 cache.

The local16 cache covers the 4,298 cohort studies. This kernel caches others with the same
preprocessing (``load_series_images`` with ``preprocess_dicom_series``, 16 slices, 224 px): the
58 Gold studies (D037) and the training studies outside the cohort that an external label table
covers. Samples carry no labels (targets 0.5, nothing observed); labels are joined locally. Two
cohort studies are re-cached as controls, so the output can be checked against the existing
cache array by array.

Example:
    python scripts/build_extra_cache_kernel.py --labels data/private/external-labels/songqizhou-7th/8_final_label.csv \
        --out data/private/kaggle-cache-extra/run01
    kaggle kernels push -p data/private/kaggle-cache-extra/run01
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from rsna_knee.data.cohort import _series_by_study  # noqa: E402

DATASETS = ["davidsulleiro/rsna-knee-attention16-cv-inputs", "davidsulleiro/rsna-knee-offline-dicom-codecs"]
CODEC_MANIFEST_SHA256 = "50f39d3f627b3dc06a8080ba51dd6424898ef34fa4c4e21a84593dd326eb776d"

RUNNER = r'''
import hashlib, io, json, os, sys, time, zipfile
from pathlib import Path

CODEC_MANIFEST_SHA256 = __CODEC__
INPUT, WORKING = Path("/kaggle/input"), Path("/kaggle/working")
os.environ["OMP_NUM_THREADS"] = "2"


def one(pattern, root=None):
    # Search attached datasets only; walking the competition DICOM tree would take hours.
    root = root or (INPUT / "datasets" if (INPUT / "datasets").is_dir() else INPUT)
    found = sorted(path for path in root.glob(pattern) if "rsna-knee-abnormality-detection" not in path.parts)
    if len(found) != 1:
        raise SystemExit(f"expected one {pattern}, found {found}")
    return found[0]


studies = json.loads(one("**/extra-cache-studies.json").read_text())
sys.path.insert(0, str(one("**/code/src/rsna_knee/__init__.py").parents[1]))
from rsna_knee.inference.offline_codec_bootstrap import install_codec_bundle
codecs = [path for path in (INPUT / "datasets").glob("*/rsna-knee-offline-dicom-codecs") if path.is_dir()]
if len(codecs) != 1:
    raise SystemExit(f"codec dataset not found: {codecs}")
codecs = codecs[0]
location = codecs if (codecs / "manifest.json").is_file() else codecs / "rsna-knee-offline-dicom-codecs.zip"
installed = install_codec_bundle(location, CODEC_MANIFEST_SHA256, WORKING / "codec-target", working_root=WORKING)
sys.path.insert(0, installed["target"])
import numpy as np
from rsna_knee.data.dicom_preprocessing import preprocess_dicom_series
from rsna_knee.data.training_dataset import load_series_images

series_root = [path for path in (INPUT / "competitions/rsna-knee-abnormality-detection/train_series",
                                 INPUT / "rsna-knee-abnormality-detection/train_series") if path.is_dir()][0]
archive_path = WORKING / "local16-cache-extra-000.zip"
records, failures, started = [], [], time.time()
with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_STORED) as archive:
    for ordinal, study in enumerate(studies):
        payload, shapes = {}, []
        try:
            for index, series in enumerate(study["series"]):
                path = series_root / study["study_id"] / series["series_id"]
                image, mask, eligible = load_series_images(path, preprocessor=preprocess_dicom_series,
                                                           output_size=224, max_slices=16, return_metadata=True)
                image, mask = image.numpy().astype(np.float32), mask.numpy().astype(bool)
                if image.shape[0] != min(eligible, 16) or image.shape[1:] != (1, 224, 224):
                    raise ValueError("shape")
                payload[f"images_{index:03d}"], payload[f"masks_{index:03d}"] = image, mask
                shapes.append(list(image.shape))
        except Exception as error:
            failures.append({"ordinal": ordinal, "error": type(error).__name__})
            continue
        payload["targets"] = np.full(12, 0.5, np.float32)
        payload["observed"] = np.zeros(12, bool)
        buffer = io.BytesIO()
        np.savez(buffer, **payload)
        raw = buffer.getvalue()
        name = f"extra/sample-{len(records):06d}.npz"
        archive.writestr(name, raw)
        records.append({"study_sha256": hashlib.sha256(study["study_id"].encode()).hexdigest(),
                        "role": study["role"], "file": name, "series_count": len(shapes),
                        "series_shapes": shapes, "sha256": hashlib.sha256(raw).hexdigest()})
        if len(records) % 10 == 0:
            print(json.dumps({"event": "progress", "done": len(records), "seconds": round(time.time() - started)}),
                  flush=True)
    archive.writestr("cache-manifest.json", json.dumps({"schema": "local16-extra-cache-v1", "records": records}))
report = {"studies": len(studies), "cached": len(records), "failures": failures,
          "seconds": round(time.time() - started), "archive_bytes": archive_path.stat().st_size}
(WORKING / "extra-cache-report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report))
'''


def study_list(*, train_csv: Path, series_csv: Path, manifest: Path, labels: Path, controls: int = 2) -> list[dict]:
    with train_csv.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    gold = [row["StudyInstanceUID"] for row in rows if row.get("ACL", "").strip() != ""]
    cohort = [record["study_id"] for record in json.loads(manifest.read_text(encoding="utf-8"))["records"]]
    with labels.open(encoding="utf-8", newline="") as stream:
        labelled = [row["StudyInstanceUID"] for row in csv.DictReader(stream)]
    cohort_set, gold_set = set(cohort), set(gold)
    extra = [study for study in labelled if study not in cohort_set and study not in gold_set]
    roles = {**{s: "gold" for s in gold}, **{s: "extra" for s in extra}, **{s: "control" for s in cohort[:controls]}}
    series = _series_by_study(series_csv, set(roles))
    return [{"study_id": s, "role": roles[s], "series": [{"series_id": item["series_id"]} for item in series[s]]}
            for s in [*gold, *extra, *cohort[:controls]]]


def build(*, out: Path, studies: list[dict], slug: str) -> Path:
    out.mkdir(parents=True, exist_ok=False)
    (out / "extra_cache_runner.py").write_text(RUNNER.replace("__CODEC__", repr(CODEC_MANIFEST_SHA256)),
                                               encoding="utf-8", newline="\n")
    data = out / "studies"
    data.mkdir()
    (data / "extra-cache-studies.json").write_text(json.dumps(studies), encoding="utf-8")
    (data / "dataset-metadata.json").write_text(json.dumps({
        "title": "RSNA Knee extra cache study list", "id": slug + "-studies", "licenses": [{"name": "other"}]}))
    metadata = {"id": slug, "title": "RSNA Knee Extra Cache", "code_file": "extra_cache_runner.py",
                "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False,
                "enable_internet": False, "dataset_sources": [*DATASETS, slug + "-studies"],
                "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, required=True, help="external label table that defines extra studies")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--slug", default="davidsulleiro/rsna-knee-extra-cache")
    args = parser.parse_args(argv)
    private = ROOT / "data/private/full-cohort-prefilter/run01"
    studies = study_list(train_csv=ROOT / "data/train.csv", series_csv=ROOT / "data/train_series.csv",
                         manifest=private / "full-cohort-soft-label-manifest.json", labels=args.labels)
    build(out=args.out, studies=studies, slug=args.slug)
    roles = [study["role"] for study in studies]
    print(json.dumps({role: roles.count(role) for role in sorted(set(roles))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
