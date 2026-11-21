"""Cross-validate the target-attention 16-slice model.

Examples:
    python scripts/cv_attention16.py train --fold 1
    python scripts/cv_attention16.py predict --fold 1
    python scripts/cv_attention16.py summarize --folds 0 1 2
    python scripts/cv_attention16.py --trainable layer3 --labels <table.csv> train-final --seed 42
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
    ARMS, EXTRA_CACHE, TRAINABLE, CachePaths, predict_fold, summarize, summary_markdown, train_final, train_fold,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=ROOT,
                        help="directory containing data/private (defaults to this checkout)")
    parser.add_argument("--runs", type=Path, default=None,
                        help="run directory (default: <data-root>/data/private/cv-attention16)")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--arm", choices=ARMS, default="target_attention",
                        help="pooling: target-specific attention (default) or equal mean")
    parser.add_argument("--trainable", choices=TRAINABLE, default="layer4",
                        help="backbone stages that learn: layer4 (default) or layer3 (layer3 and layer4)")
    parser.add_argument("--labels", type=Path, default=None,
                        help="external training-label CSV (StudyInstanceUID + 12 targets); "
                             "default: the cached report labels")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="train one fold (5 epochs)")
    train.add_argument("--fold", type=int, required=True)
    train.add_argument("--resume", action="store_true")
    final = commands.add_parser("train-final", help="train a final model on every labelled study, Gold-58 included")
    final.add_argument("--seed", type=int, required=True)
    final.add_argument("--resume", action="store_true")
    final.add_argument("--extra-cache", type=Path, default=None,
                       help="extra cache archive (default: <data-root>/" + EXTRA_CACHE + ")")
    final.add_argument("--gold-labels", type=Path, default=None,
                       help="official train.csv with the Gold-58 expert labels (default: <data-root>/data/train.csv)")
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
                            output=runs / f"fold{args.fold}", resume=args.resume, arm=args.arm,
                            trainable=args.trainable, labels=args.labels)
        result = {key: result[key] for key in ("status", "fingerprint", "train_studies",
                                               "validation_studies", "wall_seconds")}
    elif args.command == "train-final":
        if args.labels is None:
            parser.error("train-final needs --labels")
        result = train_final(root=ROOT, paths=paths, weights=weights, output=runs / f"final-seed{args.seed}",
                             seed=args.seed, labels=args.labels,
                             extra_archive=args.extra_cache or data_root / EXTRA_CACHE,
                             gold_csv=args.gold_labels or data_root / "data/train.csv",
                             resume=args.resume, arm=args.arm, trainable=args.trainable)
        result = {key: result[key] for key in ("status", "fingerprint", "train_studies", "roles", "wall_seconds")}
    elif args.command == "predict":
        checkpoint = args.checkpoint or runs / f"fold{args.fold}/{args.arm}/checkpoint-epoch-05.pt"
        result = predict_fold(root=ROOT, paths=paths, fold=args.fold, weights=weights,
                              checkpoint=checkpoint, output=runs / "predictions", arm=args.arm)
    else:
        result = summarize(runs / "predictions", tuple(args.folds))
        (runs / "cv-summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        (runs / "cv-summary.md").write_text(summary_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
