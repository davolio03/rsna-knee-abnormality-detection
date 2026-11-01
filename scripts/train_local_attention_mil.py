"""Plan or explicitly run the fresh paired RadImageNet attention experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rsna_knee.training.attention_mil import AttentionMILConfig, run_attention_pair


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", help="run both fresh five-epoch arms")
    parser.add_argument("--smoke", action="store_true", help="run one cached study and one optimizer update")
    parser.add_argument("--resume", action="store_true", help="resume the matching isolated run after interruption")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--weights", type=Path,
                        default=Path("data/private/resnet50-pretraining-assets/run01/resnet50-radimagenet.pt"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/private/disease-attention-mil/e048-run01"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.train and args.smoke or args.resume and not args.train:
        raise SystemExit("Choose either --smoke or --train; --resume requires --train.")
    config = AttentionMILConfig(repository_root=_ROOT, weights_path=args.weights,
                               output_dir=args.output, device=args.device)
    result = run_attention_pair(config, execute=args.train, smoke=args.smoke, resume=args.resume)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
