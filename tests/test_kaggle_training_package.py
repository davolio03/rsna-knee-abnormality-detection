import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from scripts.build_kaggle_training_package import build_package


class KagglePackageTests(unittest.TestCase):
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
