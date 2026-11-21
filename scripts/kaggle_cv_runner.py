"""Kaggle kernel that runs ``scripts/cv_attention16.py`` jobs, one job per visible GPU.

Generated per run by ``scripts/build_kaggle_cv_kernel.py``, which replaces the ``CONFIG`` line.
Inputs (attached as private datasets):

* ``rsna-knee-attention16-cv-inputs``: ``code/`` (repository ``src`` and ``scripts`` at a pinned
  commit) and ``data/private/...`` (manifest, folds, chunk plan, RadImageNet weights).
* ``rsna-knee-attention16-local16-cache``: the 11 cache chunks, uploaded as ``*.zip.bin`` because
  Kaggle extracts uploaded ``.zip`` files.
* For final models (``final_seeds``): ``rsna-knee-extra-cache-archive`` (the Gold and extra
  studies, ``*.zip.bin``) and the competition data, whose ``train.csv`` holds the Gold labels.
* Optional: the output of an earlier version of this kernel, to resume unfinished jobs.

The kernel rebuilds the repository layout under ``/tmp/root`` with symlinks, then trains and scores
each job (a fold, or a final model per seed) in a subprocess pinned to one GPU. Shortly before
``deadline_hours`` it stops the
subprocesses; completed epoch checkpoints are atomic, so the next version resumes from them.
Outputs go to ``/kaggle/working/cv-attention16`` with the same layout as a local run.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

CONFIG = {"folds": [0], "canary_studies": 0, "deadline_hours": 11.25, "arm": "target_attention", "trainable": "layer4", "labels": None, "final_seeds": []}  # noqa: E501

INPUT = Path("/kaggle/input")
ROOT = Path("/tmp/root")
RUNS = Path("/kaggle/working/cv-attention16")
LOGS = Path("/kaggle/working/logs")
CACHE_DIR = "data/private/attention-sixteen-slices/cache-local-run01/archives"
EXTRA_CACHE = "data/private/attention-sixteen-slices/cache-extra-run01/local16-cache-extra-000.zip"
COMPETITION_DIRS = ("competitions", "rsna-knee-abnormality-detection")

WORKER = r"""
import os, runpy, sys, time
root = sys.argv[1]
sys.path[:0] = [root + "/src", root]
from rsna_knee.training import cross_validation as cv
limit = int(os.environ.get("CANARY_STUDIES", "0"))
state = {"items": 0, "start": time.time()}
def wrap(cls):
    original = cls.__getitem__
    def getitem(self, index):
        state["items"] += 1
        if state["items"] % 200 == 0:
            print(f"[progress] items={state['items']} seconds={time.time() - state['start']:.0f}", flush=True)
        return original(self, index)
    cls.__getitem__ = getitem
    if limit:
        cls.__len__ = lambda self: min(limit, len(self.rows))
for cls in (cv.FoldCacheDataset, cv.FinalCacheDataset):
    wrap(cls)
script = root + "/scripts/cv_attention16.py"
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
"""


def search(pattern: str) -> list[Path]:
    """Glob the attached datasets and notebooks; walking the competition DICOM tree takes hours."""
    return sorted(path for child in INPUT.iterdir() if child.name not in COMPETITION_DIRS
                  for path in child.glob(pattern))


def find_one(pattern: str) -> Path:
    matches = search(pattern)
    if len(matches) != 1:
        raise SystemExit(f"expected one match for {pattern}, found {[str(m) for m in matches]}")
    return matches[0]


def link(target: Path, link_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(target)


def build_root() -> dict:
    code = find_one("**/code/scripts/cv_attention16.py").parents[1]
    data = find_one("**/full-cohort-prefilter/run01/full-cohort-soft-label-manifest.json").parents[3]
    shutil.copytree(code, ROOT)
    for path in sorted(data.rglob("*")):
        if path.is_file():
            link(path, ROOT / "data" / path.relative_to(data))
    chunks = [path for path in search("**/local16-cache-*.zip.bin") if "extra" not in path.name]
    if len(chunks) != 11:
        raise SystemExit(f"expected 11 cache chunks, found {len(chunks)}")
    for chunk in chunks:
        link(chunk, ROOT / CACHE_DIR / chunk.name[: -len(".bin")])
    if CONFIG["final_seeds"]:
        link(find_one("**/local16-cache-extra-000.zip.bin"), ROOT / EXTRA_CACHE)
        train_csv = [INPUT / "competitions/rsna-knee-abnormality-detection/train.csv",
                     INPUT / "rsna-knee-abnormality-detection/train.csv"]
        link(next(path for path in train_csv if path.is_file()), ROOT / "data/train.csv")
    return {"code": str(code), "data": str(data), "chunks": len(chunks),
            "base_commit": (code / "BASE_COMMIT").read_text().strip()}


def restore(job: str) -> bool:
    """Copy an attached earlier output of this job (``fold<k>`` or ``final-seed<s>``) to resume."""
    previous = [path.parent for path in search(f"**/cv-attention16/{job}/run-plan.json")]
    if not previous:
        return False
    if len(previous) > 1:
        raise SystemExit(f"several earlier outputs for {job}: {previous}")
    shutil.copytree(previous[0], RUNS / job)
    for path in (previous[0].parent / "predictions").glob(f"*{job}.*"):
        (RUNS / "predictions").mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, RUNS / "predictions" / path.name)
    return True


def gpus() -> list[str]:
    listing = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
    return [line for line in listing.splitlines() if line.startswith("GPU ")]


def steps(job: str) -> list[list[str]]:
    job_dir = RUNS / job
    finished = (job_dir / CONFIG["arm"] / "checkpoint-epoch-05.pt").exists()
    resume = ["--resume"] if (job_dir / "run-plan.json").exists() else []
    if job.startswith("final-seed"):
        return [] if finished else [["train-final", "--seed", job[len("final-seed"):], *resume]]
    fold = job[len("fold"):]
    commands = []
    if not finished:
        commands.append(["train", "--fold", fold, *resume])
    if not (RUNS / "predictions" / f"metrics-{job}.json").exists():
        commands.append(["predict", "--fold", fold])
    return commands


def main() -> int:
    started = time.time()
    deadline = started + CONFIG["deadline_hours"] * 3600
    RUNS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    report = {"config": CONFIG, "python": platform.python_version(), "gpus": gpus(),
              "layout": build_root(), "folds": {}}
    try:
        import numpy, torch
        report["versions"] = {"torch": torch.__version__, "numpy": numpy.__version__}
    except ImportError as error:
        report["versions"] = {"error": str(error)}
    devices = report["gpus"]
    jobs = [f"fold{fold}" for fold in CONFIG["folds"]] + [f"final-seed{seed}" for seed in CONFIG["final_seeds"]]
    if len(jobs) > len(devices):
        raise SystemExit(f"{len(jobs)} jobs need as many GPUs, found {len(devices)}")
    queues = {}
    for device, fold in enumerate(jobs):
        restored = restore(fold)
        queues[fold] = {"device": device, "restored": restored, "pending": steps(fold),
                        "process": None, "log": None, "results": []}
    env = dict(os.environ, PYTHONUNBUFFERED="1", CANARY_STUDIES=str(CONFIG["canary_studies"]))
    while any(q["pending"] or q["process"] for q in queues.values()):
        for fold, queue in queues.items():
            process = queue["process"]
            if process is not None and process.poll() is not None:
                queue["log"].close()
                queue["results"].append({"step": queue["step"], "returncode": process.returncode,
                                         "seconds": round(time.time() - queue["step_started"], 1)})
                queue["process"] = None
                if process.returncode != 0:
                    queue["pending"] = []
            if queue["process"] is None and queue["pending"] and time.time() < deadline:
                args = queue["pending"].pop(0)
                name = f"{fold}-{args[0]}"
                queue["log"] = (LOGS / f"{name}.log").open("a")
                queue["step"], queue["step_started"] = name, time.time()
                queue["process"] = subprocess.Popen(
                    [sys.executable, "-c", WORKER, str(ROOT), "--data-root", str(ROOT),
                     "--runs", str(RUNS), "--arm", CONFIG["arm"],
                     "--trainable", CONFIG["trainable"],
                     *(["--labels", str(ROOT / CONFIG["labels"])] if CONFIG.get("labels") else []),
                     *args],
                    env=dict(env, CUDA_VISIBLE_DEVICES=str(queue["device"])),
                    stdout=queue["log"], stderr=subprocess.STDOUT)
            if time.time() >= deadline:
                queue["pending"] = []
                if queue["process"] is not None and queue["process"].poll() is None:
                    queue["process"].terminate()
        time.sleep(15)
    for fold, queue in queues.items():
        report["folds"][fold] = {"device": queue["device"], "restored": queue["restored"],
                                 "steps": queue["results"],
                                 "epochs_saved": sorted(p.name for p in
                                                        (RUNS / fold / CONFIG["arm"]).glob("*.pt"))}
    report["wall_seconds"] = round(time.time() - started, 1)
    (Path("/kaggle/working") / "runner-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    for log in sorted(LOGS.glob("*.log")):
        print(f"===== {log.name}\n" + log.read_text()[-3000:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
