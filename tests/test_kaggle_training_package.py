import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from scripts.build_kaggle_training_package import build_first_fold_package, build_package


class KagglePackageTests(unittest.TestCase):
    def test_first_fold_diagnostic_package_pins_current_source_and_preserves_run02(self):
        repository = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as temp:
            root = Path(temp)
            result = build_first_fold_package(
                repository,
                root / "build",
                manifest_path=repository / "data/private/500-study-training-cohort.json",
                folds_path=repository / "data/private/500-study-training-folds.json",
                config_path=repository / "configs/kaggle_first_fold_baseline.json",
                notebook_path=repository / "notebooks/kaggle_first_fold_baseline.ipynb",
            )
            with ZipFile(root / "build/kaggle-first-fold-diagnostic.zip") as package:
                names = set(package.namelist())
                self.assertEqual(len(names), len(package.namelist()))
                self.assertEqual(names, set(result["members"]))
                self.assertIn("src/rsna_knee/training/baseline.py", names)
                self.assertIn("data/private/500-study-training-cohort.json", names)
                self.assertIn("data/private/500-study-training-folds.json", names)
                self.assertNotIn("configs/kaggle_first_fold_baseline.json", names)
                self.assertFalse(any(name.endswith(".ipynb") for name in names))
                self.assertFalse(any("__pycache__" in name or name.endswith((".pyc", ".pt", ".dcm")) for name in names))
                package_members = {name: hashlib.sha256(package.read(name)).hexdigest() for name in sorted(names)}
                self.assertEqual(package_members, result["members"])
                self.assertEqual(package_members, {
                    name: hashlib.sha256((root / "build/package" / name).read_bytes()).hexdigest()
                    for name in sorted(names)
                })
            inventory_bytes = (root / "build/inventory.json").read_bytes()
            inventory = json.loads(inventory_bytes)
            self.assertEqual(inventory["members"], result["members"])
            self.assertEqual(inventory["manifest_sha256"], "e88c61c22a463737f16951b7c76db03834034aecd1b31dc61b4ce495080b2be2")
            self.assertEqual(inventory["folds_sha256"], "b82b376b0086201e3364b11a0203b62890e628442075a52ea4a9e09ee0e778ac")
            self.assertEqual(inventory["package_sha256"], hashlib.sha256((root / "build/kaggle-first-fold-diagnostic.zip").read_bytes()).hexdigest())
            config = json.loads((root / "build/kaggle_first_fold_baseline_run02.json").read_text(encoding="utf-8"))
            self.assertEqual(config["expected_member_count"], len(inventory["members"]))
            self.assertEqual(config["expected_inventory_sha256"], hashlib.sha256(inventory_bytes).hexdigest())
            self.assertEqual(config["expected_archive_provenance"], inventory["package_sha256"])
            self.assertEqual(config["output_dir"], "/kaggle/working/kaggle-first-fold-baseline/run02")
            original_config = json.loads((repository / "configs/kaggle_first_fold_baseline.json").read_text(encoding="utf-8"))
            for key in set(config) - {"output_dir", "expected_member_count", "expected_inventory_sha256", "expected_archive_provenance"}:
                self.assertEqual(config[key], original_config[key], key)
            for key, value in {
                "mode": "fold", "fold": 0, "seed": 42, "epochs": 5, "max_steps": None,
                "overfit_limit": None, "output_size": 256, "max_slices": 8, "device": "cuda",
                "learning_rate": 0.001, "feature_width": 64, "expected_training_count": 333,
                "expected_validation_count": 167, "expected_steps": 1665,
                "private_input_root": "/kaggle/input/datasets/davidsulleiro/rsna-knee-five-study-reader-private",
            }.items():
                self.assertEqual(config[key], value, key)
            notebook = json.loads((root / "build/kaggle_first_fold_baseline_run02.ipynb").read_text(encoding="utf-8"))
            self.assertFalse(any(cell.get("outputs", []) for cell in notebook["cells"]))
            code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
            compile(code, "kaggle_first_fold_baseline_run02.py", "exec")
            self.assertIn("RUN_TRAINING = False", code)
            self.assertIn("run02", code)
            self.assertIn(inventory["package_sha256"], code)
            self.assertIn(hashlib.sha256(inventory_bytes).hexdigest(), code)
            self.assertIn('"expected_member_count": ' + str(len(inventory["members"])), code)
            self.assertNotEqual(inventory["package_sha256"], "fb803d396d6afaf3ad31031b41abd4e8805d961408b1f4377749dfc777ce36af")
            probe = subprocess.run(
                [sys.executable, "-c", (
                    "import json,sys,torch; sys.path.insert(0,sys.argv[1] + '/src'); "
                    "import rsna_knee.training.baseline as b; "
                    "e=torch.cuda.OutOfMemoryError('private sentinel'); "
                    "print(json.dumps({'origin':b.__file__,'family':b._safe_exception_family(e),"
                    "'diagnostics':b._safe_failure_diagnostics(e,'training')}))"
                ), str(root / "build/kaggle-first-fold-diagnostic.zip")],
                cwd=root,
                env={**os.environ, "PYTHONPATH": ""},
                capture_output=True,
                text=True,
                check=True,
            )
            imported = json.loads(probe.stdout)
            self.assertIn("kaggle-first-fold-diagnostic.zip", imported["origin"])
            self.assertEqual(imported["family"], "memory")
            self.assertEqual(imported["diagnostics"]["exception_chain"], ["memory"])
            self.assertNotIn("private sentinel", probe.stdout)

    def test_package_is_import_closed_allowlisted_and_hash_bound(self):
        repository = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "package.zip"
            inventory_path = root / "inventory.json"
            result = build_package(
                repository,
                archive,
                inventory_path,
                manifest_path=repository / "data/private/500-study-training-cohort.json",
                folds_path=repository / "data/private/500-study-training-folds.json",
                notebook_path=repository / "notebooks/kaggle_eight_study_overfit.ipynb",
                config_path=repository / "configs/kaggle_eight_study_overfit.json",
            )
            self.assertEqual(result["manifest_sha256"], "e88c61c22a463737f16951b7c76db03834034aecd1b31dc61b4ce495080b2be2")
            self.assertEqual(result["folds_sha256"], "b82b376b0086201e3364b11a0203b62890e628442075a52ea4a9e09ee0e778ac")
            with ZipFile(archive) as package:
                names = set(package.namelist())
                self.assertEqual(len(names), len(package.namelist()))
                self.assertIn("src/rsna_knee/__init__.py", names)
                self.assertIn("src/rsna_knee/validation/metrics.py", names)
                self.assertIn("src/rsna_knee/labels/siwc_auth.py", names)
                self.assertIn("src/rsna_knee/labels/siwc_responses.py", names)
                self.assertIn("src/rsna_knee/labels/prompt_comparison.py", names)
                self.assertIn("notebooks/kaggle_eight_study_overfit.ipynb", names)
                notebook = json.loads(package.read("notebooks/kaggle_eight_study_overfit.ipynb"))
                code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
                self.assertIn("RUN_TRAINING = False", code)
                self.assertIn("RUN_GPU_PROBE = False", code)
                self.assertIn("/kaggle/input/competitions/rsna-knee-abnormality-detection/train_series", code)
                self.assertIn("configs/kaggle_eight_study_overfit.json", names)
                self.assertFalse(any("__pycache__" in name or name.endswith((".pyc", ".pt", ".dcm")) for name in names))
                self.assertFalse(any("report" in name.lower() or "credential" in name.lower() for name in names))
                members = {name: hashlib.sha256(package.read(name)).hexdigest() for name in sorted(names)}
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
            self.assertEqual(inventory["members"], members)
            self.assertEqual(result["members"], members)
            self.assertEqual(inventory["package_sha256"], hashlib.sha256(archive.read_bytes()).hexdigest())

    def test_package_refuses_modified_or_wrong_private_inputs_without_output(self):
        repository = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as temp:
            root = Path(temp)
            bad_manifest = root / "manifest.json"
            bad_manifest.write_text("{}", encoding="utf-8")
            archive = root / "package.zip"
            with self.assertRaises(ValueError):
                build_package(repository, archive, root / "inventory.json",
                              manifest_path=bad_manifest,
                              folds_path=repository / "data/private/500-study-training-folds.json",
                              notebook_path=repository / "notebooks/kaggle_eight_study_overfit.ipynb",
                              config_path=repository / "configs/kaggle_eight_study_overfit.json")
            self.assertFalse(archive.exists())


if __name__ == "__main__":
    unittest.main()
