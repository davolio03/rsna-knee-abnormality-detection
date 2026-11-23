"""Skip integration tests whose fixtures are git-ignored local artifacts under ``data/private``.

These tests check real checkpoints, manifests and codec bundles. They run on the development
machine, where the artifacts exist, and are skipped elsewhere (for example in CI).
"""
from __future__ import annotations

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def requires(*relative_paths: str):
    missing = [path for path in relative_paths if not (ROOT / path).exists()]
    return unittest.skipIf(bool(missing), "needs local private artifacts: " + ", ".join(missing))
