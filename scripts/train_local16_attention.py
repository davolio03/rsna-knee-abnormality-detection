"""Plan or explicitly launch the isolated target-attention16 training run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rsna_knee.training.local16_attention import Attention16Config
from rsna_knee.training.local16_attention_runtime import (
    attention16_source_closure, run_local16_attention,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", help="explicitly launch five attention16 epochs")
    parser.add_argument("--resume", action="store_true", help="resume the matching attention16 checkpoint")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--cache-root", type=Path,
                        default=Path("data/private/attention-sixteen-slices/cache-local-run01"))
    parser.add_argument("--package-root", type=Path,
                        default=Path("data/private/attention-sixteen-slices/cache-package-run05"))
    parser.add_argument("--weights", type=Path,
                        default=Path("data/private/resnet50-pretraining-assets/run01/resnet50-radimagenet.pt"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/private/attention-sixteen-slices/local16-attention-run01"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.resume and not args.train:
        raise SystemExit("--resume requires --train")
    package = args.package_root if args.package_root.is_absolute() else ROOT / args.package_root
    cache_root = args.cache_root if args.cache_root.is_absolute() else ROOT / args.cache_root
    try:
        paths = attention16_source_closure(ROOT, Path(__file__))
        result = run_local16_attention(
            Attention16Config(repository_root=ROOT, weights_path=args.weights,
                              output_dir=args.output, device=args.device),
            manifest_path=ROOT / "data/private/full-cohort-prefilter/run01/full-cohort-soft-label-manifest.json",
            folds_path=ROOT / "data/private/full-cohort-prefilter/run01/full-cohort-grouped-folds.json",
            plan_path=package / "local16-cache-chunk-plan.json",
            source_archive_path=package / "local16-source.zip",
            archive_dir=cache_root / "archives",
            cache_receipt_path=(cache_root / "cache-verification-receipt.json"
                                if (cache_root / "cache-verification-receipt.json").is_file() else None),
            source_paths=paths, execute=args.train, resume=args.resume)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError) as error:
        reason = str(error)
        if not reason.replace("_", "").isalnum():
            reason = "local16_attention_preparation_blocked"
        print(json.dumps({"status": "blocked", "reason_code": reason,
                          "model_constructed": False, "training_started": False}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
