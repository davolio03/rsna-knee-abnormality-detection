"""Read-only E050 local16 cache validation and attention-only plan command."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.train_local16_attention import main as _training_cli_main


def main(argv: list[str] | None = None) -> int:
    values = sys.argv[1:] if argv is None else argv
    if "--train" in values or "--resume" in values:
        raise SystemExit("This command only prepares a plan; use train_local16_attention.py for an explicit launch.")
    return _training_cli_main(values)


if __name__ == "__main__":
    raise SystemExit(main())
