from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import textwrap
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rsna_knee.inference.attention_sixteen as a16  # noqa: E402


def load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_attention16_ensemble_notebook", ROOT / "scripts" / "build_attention16_ensemble_notebook.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Constant(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, images, masks):
        return torch.full((1, 12), self.value)


class EnsemblePatchTests(unittest.TestCase):
    def test_patch_loads_every_member_with_its_own_pin_and_averages_logits(self):
        builder = load_builder()
        calls = []
        values = iter([1.0, 2.0, 6.0])

        def fake_loader(checkpoint, report, sha, **kwargs):
            calls.append((Path(checkpoint).as_posix(), sha, kwargs))
            return Constant(next(values))

        with mock.patch.object(a16, "load_attention16_model", fake_loader):
            exec(textwrap.dedent(builder.PATCH), {"Path": Path, "json": json})
            ensemble = a16.load_attention16_model("first.pt", {}, "a" * 64, checkpoint_size=5)
            logits = ensemble([], [])
        self.assertTrue(torch.equal(logits, torch.full((1, 12), 3.0)))
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0], ("first.pt", "a" * 64, {"checkpoint_size": 5}))
        self.assertEqual([sha for _, sha, _ in calls[1:]],
                         [sha for _, sha in builder_members(builder)])
        self.assertTrue(all(path.endswith("/attention-model-epoch-05.pt") for path, _, _ in calls[1:]))


def builder_members(builder):
    namespace = {}
    start = builder.PATCH.index("ENSEMBLE_MEMBERS")
    end = builder.PATCH.index("]", start) + 1
    exec(textwrap.dedent(builder.PATCH[start:end]), namespace)
    return namespace["ENSEMBLE_MEMBERS"]


if __name__ == "__main__":
    unittest.main()
