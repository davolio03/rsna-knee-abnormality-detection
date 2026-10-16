"""Plan or explicitly execute the private local provisional training baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from rsna_knee.training.baseline import TrainingConfig, TrainingRunError, run_training  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPOSITORY_ROOT / "configs" / "baseline_training.json",
        help="explicit JSON configuration (default is the repository's local baseline config)",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="fit the local model and write a new private output directory; default is plan-only",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        value = json.loads(args.config.read_text(encoding="utf-8"))
        config = TrainingConfig.from_mapping(value)
        result = run_training(config, execute=args.execute)
    except TrainingRunError as exc:
        print(f"Training command failed ({exc.reason_code}).", file=sys.stderr)
        return 2
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        print("Training command failed (configuration_or_input_invalid).", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
