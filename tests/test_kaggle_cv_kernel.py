from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class BuildKernelTests(unittest.TestCase):
    def test_build_injects_config_and_metadata(self):
        builder = load("build_kaggle_cv_kernel")
        with tempfile.TemporaryDirectory() as temp:
            out = builder.build(folds=[1, 2], out=Path(temp) / "k", slug="u/k", title="Kernel title",
                                canary_studies=60, deadline_hours=1.5, resume_from="u/old",
                                arm="equal_mean", trainable="layer3")
            source = (out / "cv_runner.py").read_text(encoding="utf-8")
            self.assertIn('CONFIG = {"folds": [1, 2], "canary_studies": 60, "deadline_hours": 1.5, '
                          '"arm": "equal_mean", "trainable": "layer3"}', source)
            compile(source, "cv_runner.py", "exec")
            metadata = json.loads((out / "kernel-metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["kernel_sources"], ["u/old"])
            self.assertFalse(metadata["enable_internet"])
            self.assertTrue(metadata["is_private"])
            with self.assertRaises(FileExistsError):
                builder.build(folds=[0], out=out, slug="u/k", title="Kernel title", canary_studies=0,
                              deadline_hours=1.0, resume_from=None)


class RunnerLayoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        base = Path(self.temp.name)
        self.runner = load("kaggle_cv_runner")
        self.runner.INPUT = base / "input"
        self.runner.ROOT = base / "root"
        self.runner.RUNS = base / "working/cv-attention16"
        inputs = self.runner.INPUT / "datasets/u/cv-inputs"
        (inputs / "code/scripts").mkdir(parents=True)
        (inputs / "code/scripts/cv_attention16.py").write_text("")
        (inputs / "code/BASE_COMMIT").write_text("abc\n")
        manifest = inputs / "data/private/full-cohort-prefilter/run01/full-cohort-soft-label-manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text("{}")
        cache = self.runner.INPUT / "datasets/u/cache"
        cache.mkdir(parents=True)
        for index in range(11):
            (cache / f"local16-cache-train-{index:03d}.zip.bin").write_bytes(b"x")

    def tearDown(self):
        self.temp.cleanup()

    def test_build_root_links_inputs_with_zip_names(self):
        def copy(target, link_path):  # symlinks need extra privileges on Windows
            link_path.parent.mkdir(parents=True, exist_ok=True)
            link_path.write_bytes(Path(target).read_bytes())
        self.runner.link = copy
        layout = self.runner.build_root()
        self.assertEqual(layout["chunks"], 11)
        self.assertFalse((self.runner.ROOT / "data/code").exists())
        self.assertEqual(layout["base_commit"], "abc")
        archives = self.runner.ROOT / self.runner.CACHE_DIR
        self.assertEqual(len(list(archives.glob("*.zip"))), 11)
        self.assertTrue((self.runner.ROOT / "data/private/full-cohort-prefilter/run01/"
                         "full-cohort-soft-label-manifest.json").exists())

    def test_steps_follow_saved_state(self):
        runs = self.runner.RUNS
        self.assertEqual(self.runner.steps(1), [["train", "--fold", "1"], ["predict", "--fold", "1"]])
        (runs / "fold1/target_attention").mkdir(parents=True)
        (runs / "fold1/run-plan.json").write_text("{}")
        self.assertEqual(self.runner.steps(1)[0], ["train", "--fold", "1", "--resume"])
        (runs / "fold1/target_attention/checkpoint-epoch-05.pt").write_bytes(b"x")
        self.assertEqual(self.runner.steps(1), [["predict", "--fold", "1"]])
        (runs / "predictions").mkdir()
        (runs / "predictions/metrics-fold1.json").write_text("{}")
        self.assertEqual(self.runner.steps(1), [])

    def test_restore_copies_earlier_output(self):
        earlier = self.runner.INPUT / "notebooks/u/runner/cv-attention16"
        (earlier / "fold2/target_attention").mkdir(parents=True)
        (earlier / "fold2/run-plan.json").write_text("{}")
        (earlier / "fold2/target_attention/checkpoint-epoch-03.pt").write_bytes(b"x")
        (earlier / "predictions").mkdir()
        (earlier / "predictions/metrics-fold0.json").write_text("{}")
        self.assertFalse(self.runner.restore(1))
        self.assertTrue(self.runner.restore(2))
        self.assertTrue((self.runner.RUNS / "fold2/target_attention/checkpoint-epoch-03.pt").exists())
        self.assertFalse((self.runner.RUNS / "predictions/metrics-fold0.json").exists())


if __name__ == "__main__":
    unittest.main()
