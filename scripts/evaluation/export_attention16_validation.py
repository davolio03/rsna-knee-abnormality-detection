"""Purpose-grouped convenience entry point for the legacy workflow command."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEGACY_COMMAND = ROOT / "scripts" / "export_local16_validation.py"


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    completed = subprocess.run(
        [sys.executable, str(LEGACY_COMMAND), *arguments],
        cwd=ROOT,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
