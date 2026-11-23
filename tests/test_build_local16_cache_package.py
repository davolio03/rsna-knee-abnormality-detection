import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.build_local16_cache_package import build_local16_cache_package
from private_artifacts import requires

ROOT = Path(__file__).resolve().parents[1]


class BuildLocal16CachePackageTests(unittest.TestCase):
    @requires('data/private/full-cohort-prefilter/run01/full-cohort-soft-label-manifest.json',
              'data/private/full-cohort-prefilter/run01/full-cohort-grouped-folds.json',
              'data/private/masked-soft-training-preparation/run09/upload/masked-training-inventory.json',
              'data/private/masked-soft-training-preparation/run09/upload/offline-codecs.zip.b64')
    def test_builds_plan_default_and_bounded_opt_in_artifacts_without_private_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "package"
            result = build_local16_cache_package(ROOT, output)
            self.assertEqual(result["cache_identity"], "local16-attention-run01")
            self.assertEqual(result["max_slices_per_series"], 16)
            self.assertFalse(result["training_started"])
            self.assertGreater(result["chunk_count"], 1)
            self.assertEqual(sum(item["stage"] == "canary" for item in result["notebooks"]), 2)
            import_probe = (
                "import sys, tempfile\n"
                "sys.path.insert(0, sys.argv[1])\n"
                "from rsna_knee.data.masked_soft_dataset import FullCohortSoftDataset, SoftDatasetError\n"
                "from scripts.local16_cache_chunk_plan import validate_chunk_plan\n"
                "from scripts.local16_tensor_cache import Local16TensorCache\n"
                "from pathlib import Path\n"
                "root=Path(tempfile.mkdtemp())\n"
                "try: FullCohortSoftDataset(root/'missing-manifest.json', root/'missing-folds.json', root, partition='train', fold=0, output_size=224, max_slices=16)\n"
                "except SoftDatasetError as exc: assert exc.reason_code != 'slice_limit_invalid', exc.reason_code\n"
                "else: raise AssertionError('expected missing manifest')\n"
            )
            probe = subprocess.run([sys.executable, "-I", "-c", import_probe,
                                    str(output / "local16-source.zip")], capture_output=True, text=True)
            self.assertEqual(probe.returncode, 0, probe.stderr)
            for artifact in result["notebooks"]:
                path = Path(artifact["path"])
                notebook = json.loads(path.read_text(encoding="utf-8"))
                source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
                compile(source, path.name, "exec")
                self.assertIn("cuda_must_remain_uninitialized", source)
                self.assertIn("training_started", source)
                self.assertIn("archive_sha256", source)
                self.assertIn("saved_bytes_with_reserve", source)
                self.assertNotIn("data/private/full-cohort-prefilter/run01/", source)
                if artifact["stage"] == "plan":
                    self.assertIn("STAGE = 'plan'", source)
                    self.assertIn("no export", json.loads((output / "package-manifest.json").read_text())["stage_defaults"]["plan"])
                if artifact["stage"] == "canary":
                    self.assertIn("MAX_STUDIES = 8", source)
                if artifact["stage"] == "full":
                    self.assertIn("STAGE = 'chunk_export'", source)
                if artifact["stage"] == "canary":
                    self.assertLess(source.index("sys.path.insert(0, str(source_path))"),
                                    source.index("from scripts.local16_cache_chunk_plan"))
            input_root = Path(temporary) / "input"
            asset = input_root / "datasets" / "owner" / "assets"
            asset.mkdir(parents=True)
            shutil.copy2(ROOT / "data/private/masked-soft-training-preparation/run09/upload/masked-training-inventory.json",
                         asset / "masked-training-inventory.json")
            cohort = ROOT / "data/private/full-cohort-prefilter/run01"
            shutil.copy2(cohort / "full-cohort-soft-label-manifest.json", asset)
            shutil.copy2(cohort / "full-cohort-grouped-folds.json", asset)
            shutil.copy2(ROOT / "data/private/masked-soft-training-preparation/run09/upload/offline-codecs.zip.b64",
                         asset / "offline-codecs.zip.b64")
            (input_root / "competitions/rsna-knee-abnormality-detection/train_series").mkdir(parents=True)
            working_root = Path(temporary) / "working"
            working_root.mkdir()
            plan_code = "\n".join("".join(cell["source"]) for cell in
                                   json.loads((output / "cache-train-000-plan.ipynb").read_text())["cells"])
            plan_code = plan_code.replace("Path('/kaggle/input')", f"Path({str(input_root).replace(chr(92), '/')!r})")
            plan_code = plan_code.replace("Path('/kaggle/working')", f"Path({str(working_root).replace(chr(92), '/')!r})")
            plan_script = Path(temporary) / "generated-plan.py"
            plan_script.write_text(plan_code, encoding="utf-8")
            execution = subprocess.run([sys.executable, "-I", str(plan_script)], cwd=temporary,
                                       capture_output=True, text=True)
            self.assertEqual(execution.returncode, 0, execution.stderr)
            self.assertIn('"status": "planned"', execution.stdout)
            self.assertIn('"training_started": false', execution.stdout)
            canary_path = output / "cache-train-000-canary.ipynb"
            canary_code = "\n".join("".join(cell["source"]) for cell in
                                     json.loads(canary_path.read_text())["cells"])
            canary_script = Path(temporary) / "generated-canary.py"
            canary_script.write_text(canary_code, encoding="utf-8")
            runner = Path(temporary) / "run-synthetic-canary.py"
            runner.write_text(
                "import json, runpy, sys\n"
                "import numpy as np\n"
                "ns = runpy.run_path(sys.argv[1], run_name='local16_synthetic_test')\n"
                "def factory(manifest_path, folds_path, image_root, **kwargs):\n"
                " from rsna_knee.data.masked_soft_dataset import FullCohortSoftDataset\n"
                " from rsna_knee.labels.full_cohort_prefilter import TARGET_COLUMNS\n"
                " import torch\n"
                " class SyntheticDataset(FullCohortSoftDataset):\n"
                "  def export_sample(self, index):\n"
                "   row = self._records[index]\n"
                "   images = [np.zeros((16,1,224,224),dtype=np.float32) for _ in row['series']]\n"
                "   masks = [np.ones((16,1,224,224),dtype=np.bool_) for _ in row['series']]\n"
                "   targets = torch.tensor([row['soft_targets'][name] for name in TARGET_COLUMNS],dtype=torch.float32)\n"
                "   observed = torch.tensor([row['observed_mask'][name] for name in TARGET_COLUMNS],dtype=torch.bool)\n"
                "   return (images,masks,targets,observed,index), [20 for _ in row['series']]\n"
                " return SyntheticDataset(manifest_path,folds_path,image_root,**kwargs)\n"
                "def installer(bundle, manifest, target, **kwargs):\n"
                " target.mkdir()\n"
                " return {'target':str(target)}\n"
                "result = ns['run'](dataset_factory=factory,input_root=sys.argv[2],working_root=sys.argv[3],codec_installer=installer,decoder_probe=lambda:None)\n"
                "print(json.dumps(result,sort_keys=True))\n", encoding="utf-8")
            canary_execution = subprocess.run([sys.executable, "-I", str(runner), str(canary_script),
                                               str(input_root), str(working_root)], cwd=temporary,
                                              capture_output=True, text=True)
            self.assertEqual(canary_execution.returncode, 0, canary_execution.stderr)
            canary_summary = json.loads(canary_execution.stdout.splitlines()[-1])
            self.assertEqual(canary_summary["status"], "complete")
            self.assertEqual(canary_summary["study_count"], 8)
            self.assertGreater(canary_summary["series_with_more_than_eight_slices"], 0)
            self.assertTrue((working_root / "local16-cache-canary-train.zip").is_file())
            with self.assertRaisesRegex(ValueError, "package_output_not_fresh"):
                build_local16_cache_package(ROOT, output)


if __name__ == "__main__":
    unittest.main()
