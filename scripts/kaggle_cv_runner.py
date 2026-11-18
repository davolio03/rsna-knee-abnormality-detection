"""Kaggle kernel that runs ``scripts/cv_attention16.py`` folds, one fold per visible GPU.

Generated per run by ``scripts/build_kaggle_cv_kernel.py``, which replaces the ``CONFIG`` line.
Inputs (attached as private datasets):

* ``rsna-knee-attention16-cv-inputs``: ``code/`` (repository ``src`` and ``scripts`` at a pinned
  commit) and ``data/private/...`` (manifest, folds, chunk plan, RadImageNet weights).
* ``rsna-knee-attention16-local16-cache``: the 11 cache chunks, uploaded as ``*.zip.bin`` because
  Kaggle extracts uploaded ``.zip`` files.
* Optional: the output of an earlier version of this kernel, to resume unfinished folds.

The kernel rebuilds the repository layout under ``/tmp/root`` with symlinks, then trains and scores
each fold in a subprocess pinned to one GPU. Shortly before ``deadline_hours`` it stops the
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

CONFIG = {"folds": [0], "canary_studies": 0, "deadline_hours": 11.25}

INPUT = Path("/kaggle/input")
ROOT = Path("/tmp/root")
RUNS = Path("/kaggle/working/cv-attention16")
LOGS = Path("/kaggle/working/logs")
CACHE_DIR = "data/private/attention-sixteen-slices/cache-local-run01/archives"

WORKER = r"""
import os, runpy, sys, time
root = sys.argv[1]
sys.path[:0] = [root + "/src", root]
from rsna_knee.training import cross_validation as cv
limit = int(os.environ.get("CANARY_STUDIES", "0"))
if limit:
    cv.FoldCacheDataset.__len__ = lambda self: min(limit, len(self.rows))
original = cv.FoldCacheDataset.__getitem__
state = {"items": 0, "start": time.time()}
def getitem(self, index):
    state["items"] += 1
    if state["items"] % 200 == 0:
        print(f"[progress] items={state['items']} seconds={time.time() - state['start']:.0f}", flush=True)
    return original(self, index)
cv.FoldCacheDataset.__getitem__ = getitem
script = root + "/scripts/cv_attention16.py"
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
"""


def find_one(pattern: str) -> Path:
    matches = sorted(INPUT.glob(pattern))
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
    chunks = sorted(INPUT.glob("**/local16-cache-*.zip.bin"))
    if len(chunks) != 11:
        raise SystemExit(f"expected 11 cache chunks, found {len(chunks)}")
    for chunk in chunks:
        link(chunk, ROOT / CACHE_DIR / chunk.name[: -len(".bin")])
    return {"code": str(code), "data": str(data), "chunks": len(chunks),
            "base_commit": (code / "BASE_COMMIT").read_text().strip()}


def restore(fold: int) -> bool:
    """Copy an attached earlier output of this fold so training can resume."""
    previous = [path.parent for path in INPUT.glob(f"**/cv-attention16/fold{fold}/run-plan.json")]
    if not previous:
        return False
    if len(previous) > 1:
        raise SystemExit(f"several earlier outputs for fold {fold}: {previous}")
    shutil.copytree(previous[0], RUNS / f"fold{fold}")
    for path in (previous[0].parent / "predictions").glob(f"*fold{fold}.*"):
        (RUNS / "predictions").mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, RUNS / "predictions" / path.name)
    return True


def gpus() -> list[str]:
    listing = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
    return [line for line in listing.splitlines() if line.startswith("GPU ")]


def steps(fold: int) -> list[list[str]]:
    fold_dir = RUNS / f"fold{fold}"
    done = (RUNS / "predictions" / f"metrics-fold{fold}.json").exists()
    finished = (fold_dir / "target_attention/checkpoint-epoch-05.pt").exists()
    commands = []
    if not finished:
        resume = ["--resume"] if (fold_dir / "run-plan.json").exists() else []
        commands.append(["train", "--fold", str(fold), *resume])
    if not done:
        commands.append(["predict", "--fold", str(fold)])
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
    if len(CONFIG["folds"]) > len(devices):
        raise SystemExit(f"{len(CONFIG['folds'])} folds need as many GPUs, found {len(devices)}")
    queues = {}
    for device, fold in enumerate(CONFIG["folds"]):
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
                name = f"fold{fold}-{args[0]}"
                queue["log"] = (LOGS / f"{name}.log").open("a")
                queue["step"], queue["step_started"] = name, time.time()
                queue["process"] = subprocess.Popen(
                    [sys.executable, "-c", WORKER, str(ROOT), "--data-root", str(ROOT),
                     "--runs", str(RUNS), *args],
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
                                                        (RUNS / f"fold{fold}/target_attention").glob("*.pt"))}
    report["wall_seconds"] = round(time.time() - started, 1)
    (Path("/kaggle/working") / "runner-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    for log in sorted(LOGS.glob("*.log")):
        print(f"===== {log.name}\n" + log.read_text()[-3000:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
