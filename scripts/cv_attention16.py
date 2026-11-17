"""Cross-validate the target-attention 16-slice model.

Examples:
    python scripts/cv_attention16.py train --fold 1
    python scripts/cv_attention16.py predict --fold 1
    python scripts/cv_attention16.py summarize --folds 0 1 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from rsna_knee.training.cross_validation import (
    CachePaths, predict_fold, summarize, summary_markdown, train_fold,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=ROOT,
                        help="directory containing data/private (defaults to this checkout)")
    parser.add_argument("--runs", type=Path, default=None,
                        help="run directory (default: <data-root>/data/private/cv-attention16)")
    parser.add_argument("--weights", type=Path, default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="train one fold (5 epochs)")
    train.add_argument("--fold", type=int, required=True)
    train.add_argument("--resume", action="store_true")
    predict = commands.add_parser("predict", help="score a held-out fold with its epoch-5 checkpoint")
    predict.add_argument("--fold", type=int, required=True)
    predict.add_argument("--checkpoint", type=Path, default=None)
    report = commands.add_parser("summarize", help="aggregate per-fold metrics")
    report.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args(argv)

    data_root = args.data_root.resolve()
    paths = CachePaths.default(data_root)
    runs = (args.runs or data_root / "data/private/cv-attention16").resolve()
    weights = (args.weights or data_root / "data/private/resnet50-pretraining-assets/run01/resnet50-radimagenet.pt")
    if args.command == "train":
        result = train_fold(root=ROOT, paths=paths, fold=args.fold, weights=weights,
                            output=runs / f"fold{args.fold}", resume=args.resume)
        result = {key: result[key] for key in ("status", "fingerprint", "train_studies",
                                               "validation_studies", "wall_seconds")}
    elif args.command == "predict":
        checkpoint = args.checkpoint or runs / f"fold{args.fold}/target_attention/checkpoint-epoch-05.pt"
        result = predict_fold(root=ROOT, paths=paths, fold=args.fold, weights=weights,
                              checkpoint=checkpoint, output=runs / "predictions")
    else:
        result = summarize(runs / "predictions", tuple(args.folds))
        (runs / "cv-summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        (runs / "cv-summary.md").write_text(summary_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
