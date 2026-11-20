"""Write a Kaggle kernel folder that runs cross-validation folds on Kaggle GPUs.

Example (two folds in parallel on a T4 x2 session):
    python scripts/build_kaggle_cv_kernel.py --folds 1 2 --out data/private/kaggle-cv/run01
    kaggle kernels push -p data/private/kaggle-cv/run01

Use ``--canary-studies 60`` for a short timing and wiring check. To resume after a session limit,
rebuild with ``--resume-from <user>/<kernel-slug>`` so the earlier output is attached.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["davidsulleiro/rsna-knee-attention16-cv-inputs",
            "davidsulleiro/rsna-knee-attention16-local16-cache"]


def build(*, folds: list[int], out: Path, slug: str, title: str, canary_studies: int,
          deadline_hours: float, resume_from: str | None,
          arm: str = "target_attention", trainable: str = "layer4", labels: str | None = None) -> Path:
    source = (ROOT / "scripts/kaggle_cv_runner.py").read_text(encoding="utf-8")
    pattern = re.compile(r"^CONFIG = \{.*\}.*$", re.MULTILINE)
    if len(pattern.findall(source)) != 1:
        raise ValueError("runner CONFIG line not found")
    config = {"folds": folds, "canary_studies": canary_studies, "deadline_hours": deadline_hours,
              "arm": arm, "trainable": trainable, "labels": labels}
    out.mkdir(parents=True, exist_ok=False)
    (out / "cv_runner.py").write_text(pattern.sub(lambda _: f"CONFIG = {json.dumps(config)}", source),
                                      encoding="utf-8", newline="\n")
    metadata = {"id": slug, "title": title, "code_file": "cv_runner.py", "language": "python",
                "kernel_type": "script", "is_private": True, "enable_gpu": True,
                "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
                "dataset_sources": DATASETS, "competition_sources": [],
                "kernel_sources": [resume_from] if resume_from else []}
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folds", type=int, nargs="+", required=True, help="one fold per GPU")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--slug", default="davidsulleiro/rsna-knee-attention16-cv-runner")
    parser.add_argument("--title", default="RSNA Knee Attention16 CV Runner")
    parser.add_argument("--canary-studies", type=int, default=0)
    parser.add_argument("--deadline-hours", type=float, default=11.25)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--arm", choices=["target_attention", "equal_mean"], default="target_attention")
    parser.add_argument("--trainable", choices=["layer4", "layer3"], default="layer4")
    parser.add_argument("--labels", default=None,
                        help="training-label CSV path relative to the repository root, shipped in the inputs dataset")
    args = parser.parse_args(argv)
    if len(set(args.folds)) != len(args.folds) or not set(args.folds) <= {0, 1, 2}:
        parser.error("folds must be distinct values in 0..2")
    print(build(folds=args.folds, out=args.out, slug=args.slug, title=args.title,
                canary_studies=args.canary_studies, deadline_hours=args.deadline_hours,
                resume_from=args.resume_from, arm=args.arm,
                trainable=args.trainable, labels=args.labels))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
