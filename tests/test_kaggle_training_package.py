import hashlib
import base64
import ast
from io import BytesIO
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
    def test_frozen_resnet_builders_select_pinned_fold_and_protocol_counts(self):
        from scripts.build_kaggle_training_package import (
            build_resnet18_comparison_package,
            build_resnet18_frozen_control_package,
        )

        repository = Path(__file__).resolve().parents[1]
        common = {
            "manifest_path": repository / "data/private/500-study-training-cohort.json",
            "folds_path": repository / "data/private/500-study-training-folds.json",
            "weights_path": repository / "data/private/resnet18-comparison/weights/resnet18-f37072fd.pth",
            "expected_weights_sha256": "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
            "expected_weights_size": 46830571,
        }
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for fold, train_count, validation_count, steps in ((1, 337, 163, 1685), (2, 330, 170, 1650)):
                comparison = build_resnet18_comparison_package(
                    repository, root / f"comparison-fold{fold}", fold=fold, **common,
                )
                frozen = build_resnet18_frozen_control_package(
                    repository, root / f"frozen-fold{fold}", fold=fold, **common,
                )
                comparison_config = json.loads(Path(comparison["config_paths"]["imagenet1k_v1"]).read_text(encoding="utf-8"))
                frozen_config = json.loads(Path(frozen["config_path"]).read_text(encoding="utf-8"))
                frozen_notebook = json.loads(Path(frozen["notebook_path"]).read_text(encoding="utf-8"))
                markdown = "\n".join(
                    "".join(cell["source"])
                    for cell in frozen_notebook["cells"]
                    if cell["cell_type"] == "markdown"
                )
                self.assertIn(f"fold {fold}", markdown)
                self.assertIn("single frozen-backbone control", markdown)
                self.assertNotIn("unchanged fold-0 cohort", markdown)
                self.assertNotIn("one initialization arm of a two-run comparison", markdown)
                for config in (comparison_config, frozen_config):
                    self.assertEqual(config["fold"], fold)
                    self.assertEqual(config["expected_training_count"], train_count)
                    self.assertEqual(config["expected_validation_count"], validation_count)
                    self.assertEqual(config["expected_steps"], steps)
                    self.assertEqual(config["epochs"], 5)
                    self.assertEqual(config["seed"], 42)
                self.assertEqual(frozen_config["output_dir"], f"/kaggle/working/resnet18-frozen-backbone/fold{fold}-run01")
                notebook = json.loads(Path(frozen["notebook_path"]).read_text(encoding="utf-8"))
                self.assertTrue(all(not cell.get("outputs", []) for cell in notebook["cells"]))
                code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
                self.assertIn("RUN_TRAINING = False", code)
                self.assertIn(f"CONFIG['fold'] != {fold}", code)

    def test_frozen_resnet_builders_reject_invalid_fold_before_output_creation(self):
        from scripts.build_kaggle_training_package import (
            build_resnet18_comparison_package,
            build_resnet18_frozen_control_package,
        )

        repository = Path(__file__).resolve().parents[1]
        common = {
            "manifest_path": repository / "data/private/500-study-training-cohort.json",
            "folds_path": repository / "data/private/500-study-training-folds.json",
            "weights_path": repository / "data/private/resnet18-comparison/weights/resnet18-f37072fd.pth",
            "expected_weights_sha256": "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
            "expected_weights_size": 46830571,
        }
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for fold in (-1, 3, True, "1"):
                for builder, name in ((build_resnet18_comparison_package, "comparison"),
                                      (build_resnet18_frozen_control_package, "frozen")):
                    output = root / f"{name}-{type(fold).__name__}-{fold}"
                    with self.subTest(builder=name, fold=fold), self.assertRaises(ValueError):
                        builder(repository, output, fold=fold, **common)
                    self.assertFalse(output.exists())

    def test_frozen_resnet_control_preparation_is_one_arm_disabled_and_pinned(self):
        from scripts.build_kaggle_training_package import build_resnet18_frozen_control_package

        repository = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as temp:
            result = build_resnet18_frozen_control_package(
                repository, Path(temp) / "run01-preparation",
                manifest_path=repository / "data/private/500-study-training-cohort.json",
                folds_path=repository / "data/private/500-study-training-folds.json",
                weights_path=repository / "data/private/resnet18-comparison/weights/resnet18-f37072fd.pth",
                expected_weights_sha256="f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
                expected_weights_size=46830571,
            )
            inventory = json.loads(Path(result["inventory_path"]).read_text(encoding="utf-8"))
            transport = Path(result["transport_dir"])
            archive_bytes = base64.b64decode((transport / "kaggle-resnet18-comparison.zip.b64").read_bytes(), validate=True)
            self.assertEqual(hashlib.sha256(archive_bytes).hexdigest(), result["package_sha256"])
            with ZipFile(BytesIO(archive_bytes)) as archive:
                self.assertEqual(set(archive.namelist()), set(inventory["members"]))
                for name, expected_hash in inventory["members"].items():
                    self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), expected_hash)
                    self.assertEqual(archive.read(name), (transport / "package" / name).read_bytes())
            self.assertEqual(inventory["manifest_sha256"], "e88c61c22a463737f16951b7c76db03834034aecd1b31dc61b4ce495080b2be2")
            self.assertEqual(inventory["folds_sha256"], "b82b376b0086201e3364b11a0203b62890e628442075a52ea4a9e09ee0e778ac")
            self.assertFalse(any(path.suffix.lower() == ".zip" for path in transport.rglob("*")))
            self.assertEqual((transport / "weights/resnet18-f37072fd.pth").stat().st_size, 46830571)
            config = json.loads(Path(result["config_path"]).read_text(encoding="utf-8"))
            self.assertEqual(config["model_kind"], "resnet18_imagenet1k_v1")
            self.assertIs(config["freeze_backbone"], True)
            self.assertEqual(config["expected_training_count"], 333)
            self.assertEqual(config["expected_validation_count"], 167)
            self.assertEqual(config["expected_steps"], 1665)
            self.assertEqual(config["output_dir"], "/kaggle/working/resnet18-frozen-backbone/run01")
            notebook = json.loads(Path(result["notebook_path"]).read_text(encoding="utf-8"))
            self.assertTrue(all(not cell.get("outputs", []) for cell in notebook["cells"]))
            markdown = "\n".join(
                "".join(cell["source"])
                for cell in notebook["cells"]
                if cell["cell_type"] == "markdown"
            )
            self.assertIn("fold 0", markdown)
            self.assertIn("single frozen-backbone control", markdown)
            self.assertNotIn("unchanged fold-0 cohort", markdown)
            self.assertNotIn("one initialization arm of a two-run comparison", markdown)
            code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
            compile(code, "frozen-resnet18.py", "exec")
            self.assertIn("RUN_TRAINING = False", code)
            self.assertTrue(inventory["freeze_backbone"])
            bundled_runner = (Path(result["package_dir"]) / "src/rsna_knee/training/baseline.py").read_text(encoding="utf-8")
            self.assertIn("backbone_frozen", bundled_runner)
            self.assertIn("head_trainable_parameter_count", bundled_runner)
            self.assertIn("/kaggle/input/rsna-knee-five-study-reader-private", code)
            self.assertIn("/kaggle/input/datasets/davidsulleiro/rsna-knee-five-study-reader-private", code)
            self.assertEqual(result["weights_sha256"], config["weights_sha256"])

    def test_resnet_comparison_preparation_pins_two_disabled_matched_arms(self):
        from scripts.build_kaggle_training_package import build_resnet18_comparison_package

        repository = Path(__file__).resolve().parents[1]
        weights = repository / "data/private/resnet18-comparison/weights/resnet18-f37072fd.pth"
        with TemporaryDirectory() as temp:
            output = Path(temp) / "comparison-v1"
            result = build_resnet18_comparison_package(
                repository, output,
                manifest_path=repository / "data/private/500-study-training-cohort.json",
                folds_path=repository / "data/private/500-study-training-folds.json",
                weights_path=weights,
                expected_weights_sha256="f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
                expected_weights_size=46830571,
            )
            inventory = json.loads(Path(result["inventory_path"]).read_text(encoding="utf-8"))
            self.assertEqual(inventory["manifest_sha256"], "e88c61c22a463737f16951b7c76db03834034aecd1b31dc61b4ce495080b2be2")
            self.assertEqual(inventory["folds_sha256"], "b82b376b0086201e3364b11a0203b62890e628442075a52ea4a9e09ee0e778ac")
            self.assertEqual(inventory["weights"], {
                "relative_path": "weights/resnet18-f37072fd.pth",
                "sha256": "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
                "size_bytes": 46830571,
            })
            upload = Path(result["transport_dir"])
            encoded = (upload / "kaggle-resnet18-comparison.zip.b64").read_bytes().decode("ascii")
            archive_bytes = base64.b64decode(encoded, validate=True)
            self.assertEqual(hashlib.sha256(archive_bytes).hexdigest(), result["package_sha256"])
            with ZipFile(BytesIO(archive_bytes)) as archive:
                self.assertEqual(set(archive.namelist()), set(inventory["members"]))
                for name, expected_hash in inventory["members"].items():
                    self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), expected_hash)
                    self.assertEqual(archive.read(name), (upload / "package" / name).read_bytes())
            uploaded_weights = upload / "weights/resnet18-f37072fd.pth"
            self.assertEqual(uploaded_weights.stat().st_size, 46830571)
            self.assertEqual(hashlib.sha256(uploaded_weights.read_bytes()).hexdigest(), inventory["weights"]["sha256"])
            self.assertFalse(any(path.suffix.lower() == ".zip" for path in upload.rglob("*")))

            configurations = []
            for arm, model_kind in (("random", "resnet18_random"), ("imagenet1k_v1", "resnet18_imagenet1k_v1")):
                config_path = Path(result["config_paths"][arm])
                config = json.loads(config_path.read_text(encoding="utf-8"))
                self.assertEqual(config["model_kind"], model_kind)
                self.assertEqual((config["mode"], config["fold"], config["seed"], config["epochs"], config["device"]),
                                 ("fold", 0, 42, 5, "cuda"))
                self.assertEqual(config["expected_training_count"], 333)
                self.assertEqual(config["expected_validation_count"], 167)
                self.assertEqual(config["expected_steps"], 1665)
                self.assertEqual(config["learning_rate"], 0.001)
                configurations.append(config)
                notebook = json.loads(Path(result["notebook_paths"][arm]).read_text(encoding="utf-8"))
                self.assertTrue(all(not cell.get("outputs", []) for cell in notebook["cells"]))
                code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
                compile(code, f"resnet18-{arm}.py", "exec")
                self.assertIn("RUN_TRAINING = False", code)
                self.assertIn("expected_inventory_sha256", code)
                self.assertIn("base64.b64decode", code)
                self.assertIn("resnet18_run_start", code)
                self.assertIn("torchvision_version", code)
                self.assertNotIn("download", code.lower())
                self.assertNotIn("extractall", code)
            self.assertEqual(configurations[0]["expected_inventory_sha256"], configurations[1]["expected_inventory_sha256"])
            self.assertEqual(configurations[0]["expected_archive_provenance"], configurations[1]["expected_archive_provenance"])
            self.assertEqual(configurations[0]["output_dir"], "/kaggle/working/resnet18-comparison/run01-random")
            self.assertEqual(configurations[1]["output_dir"], "/kaggle/working/resnet18-comparison/run01-imagenet1k-v1")

    def test_resnet_comparison_rejects_weight_pin_mismatch_before_output_creation(self):
        from scripts.build_kaggle_training_package import build_resnet18_comparison_package

        repository = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as temp:
            root = Path(temp)
            fake = root / "weights.pth"
            fake.write_bytes(b"wrong")
            output = root / "must-not-exist"
            with self.assertRaises(ValueError):
                build_resnet18_comparison_package(
                    repository, output,
                    manifest_path=repository / "data/private/500-study-training-cohort.json",
                    folds_path=repository / "data/private/500-study-training-folds.json",
                    weights_path=fake,
                    expected_weights_sha256="0" * 64,
                    expected_weights_size=5,
                )
            self.assertFalse(output.exists())

    def test_overfit_sanity_upload_transport_uses_pinned_base64_archive(self):
        from scripts.build_kaggle_training_package import build_overfit_sanity_package

        repository = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as temp:
            output = Path(temp) / "sanity-v3"
            result = build_overfit_sanity_package(
                repository, output,
                manifest_path=repository / "data/private/500-study-training-cohort.json",
                folds_path=repository / "data/private/500-study-training-folds.json",
                config_path=repository / "configs/kaggle_eight_study_overfit.json",
                notebook_path=repository / "notebooks/kaggle_eight_study_overfit.ipynb",
            )
            transport = Path(result["transport_dir"])
            self.assertEqual({path.name for path in transport.iterdir()}, {
                "package", "inventory.json", "kaggle_training_learning_diagnostics_sanity.json",
                "kaggle-training-learning-diagnostics-sanity.zip.b64",
            })
            transport_files = [path for path in transport.rglob("*") if path.is_file()]
            self.assertEqual(len(transport_files), len(result["members"]) + 3)
            self.assertFalse(any(path.suffix.lower() == ".zip" for path in transport_files))

            encoded_path = transport / "kaggle-training-learning-diagnostics-sanity.zip.b64"
            encoded = encoded_path.read_bytes().decode("ascii")
            archive_bytes = base64.b64decode(encoded, validate=True)
            self.assertEqual(hashlib.sha256(archive_bytes).hexdigest(), result["package_sha256"])
            with ZipFile(BytesIO(archive_bytes)) as package:
                self.assertEqual(set(package.namelist()), set(result["members"]))
                for member, expected_hash in result["members"].items():
                    payload = package.read(member)
                    self.assertEqual(hashlib.sha256(payload).hexdigest(), expected_hash)
                    self.assertEqual(payload, (transport / "package" / member).read_bytes())

            notebook = json.loads(Path(result["notebook_path"]).read_text(encoding="utf-8"))
            self.assertFalse(any(cell.get("outputs", []) for cell in notebook["cells"]))
            code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
            compile(code, "kaggle_training_learning_diagnostics_sanity.py", "exec")
            self.assertIn("RUN_TRAINING = False", code)
            self.assertIn("INPUT_ROOT = Path('/kaggle/input/rsna-knee-five-study-reader-private')", code)
            self.assertIn("base64.b64decode(ARCHIVE_B64_PATH.read_bytes().decode('ascii'), validate=True)", code)
            self.assertIn("io.BytesIO(archive_bytes)", code)
            self.assertIn("sha256(archive_bytes) != EXPECTED_ARCHIVE_SHA256", code)
            self.assertNotIn("ARCHIVE_PATH.read_bytes()", code)

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

    def test_overfit_sanity_package_pins_protocol_source_and_private_inputs(self):
        from scripts.build_kaggle_training_package import build_overfit_sanity_package

        repository = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "sanity-run01-preparation"
            result = build_overfit_sanity_package(
                repository,
                output,
                manifest_path=repository / "data/private/500-study-training-cohort.json",
                folds_path=repository / "data/private/500-study-training-folds.json",
                config_path=repository / "configs/kaggle_eight_study_overfit.json",
                notebook_path=repository / "notebooks/kaggle_eight_study_overfit.ipynb",
            )
            with ZipFile(result["archive_path"]) as package:
                names = set(package.namelist())
                self.assertIn("src/rsna_knee/training/baseline.py", names)
                self.assertIn("data/private/500-study-training-cohort.json", names)
                self.assertIn("data/private/500-study-training-folds.json", names)
                self.assertFalse(any(name.endswith((".dcm", ".pt", ".pth", ".ckpt")) for name in names))
                for name in names:
                    self.assertEqual(hashlib.sha256(package.read(name)).hexdigest(), result["members"][name])
            inventory_bytes = Path(result["inventory_path"]).read_bytes()
            inventory = json.loads(inventory_bytes)
            config = json.loads(Path(result["config_path"]).read_text(encoding="utf-8"))
            self.assertEqual(inventory["manifest_sha256"], "e88c61c22a463737f16951b7c76db03834034aecd1b31dc61b4ce495080b2be2")
            self.assertEqual(inventory["folds_sha256"], "b82b376b0086201e3364b11a0203b62890e628442075a52ea4a9e09ee0e778ac")
            self.assertEqual(config["mode"], "overfit")
            self.assertEqual((config["fold"], config["seed"], config["epochs"], config["max_steps"], config["overfit_limit"]), (0, 42, 30, 240, 8))
            self.assertEqual((config["output_size"], config["max_slices"], config["feature_width"], config["learning_rate"]), (256, 8, 64, 0.001))
            self.assertEqual(config["diagnostic_target_bce"], 0.10)
            self.assertEqual(config["expected_manifest_sha256"], inventory["manifest_sha256"])
            self.assertEqual(config["expected_folds_sha256"], inventory["folds_sha256"])
            self.assertEqual(config["expected_source_sha256"], inventory["members"]["src/rsna_knee/training/baseline.py"])
            self.assertEqual(config["expected_inventory_sha256"], hashlib.sha256(inventory_bytes).hexdigest())
            self.assertEqual(config["expected_archive_provenance"], inventory["package_sha256"])
            notebook = json.loads(Path(result["notebook_path"]).read_text(encoding="utf-8"))
            self.assertFalse(any(cell.get("outputs", []) for cell in notebook["cells"]))
            code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
            compile(code, "kaggle_training_learning_diagnostics_sanity.ipynb", "exec")
            self.assertIn("RUN_TRAINING = False", code)
            self.assertIn(config["expected_inventory_sha256"], code)
            self.assertIn(config["expected_source_sha256"], code)
            self.assertNotIn("study_id", code)


            existing = output / "existing-sentinel.txt"
            existing.write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be new"):
                build_overfit_sanity_package(
                    repository, output,
                    manifest_path=repository / "data/private/500-study-training-cohort.json",
                    folds_path=repository / "data/private/500-study-training-folds.json",
                    config_path=repository / "configs/kaggle_eight_study_overfit.json",
                    notebook_path=repository / "notebooks/kaggle_eight_study_overfit.ipynb",
                )
            self.assertEqual(existing.read_text(encoding="utf-8"), "preserve")
    def test_learning_path_bundle_adds_helper_without_changing_existing_members(self):
        from scripts.build_kaggle_training_package import (
            build_learning_path_diagnostic_package,
            build_overfit_sanity_package,
        )

        repository = Path(__file__).resolve().parents[1]
        manifest = repository / "data/private/500-study-training-cohort.json"
        folds = repository / "data/private/500-study-training-folds.json"
        config = repository / "configs/kaggle_eight_study_overfit.json"
        template = repository / "notebooks/kaggle_eight_study_overfit.ipynb"
        with TemporaryDirectory() as temp:
            root = Path(temp)
            base = build_overfit_sanity_package(
                repository, root / "base", manifest_path=manifest, folds_path=folds,
                config_path=config, notebook_path=template,
            )
            result = build_learning_path_diagnostic_package(
                repository, root / "learning-path", manifest_path=manifest,
                folds_path=folds, config_path=config, notebook_path=template,
            )
            with ZipFile(base["archive_path"]) as old_package, ZipFile(result["archive_path"]) as package:
                old_names = set(old_package.namelist())
                names = set(package.namelist())
                self.assertEqual(names, old_names | {"src/rsna_knee/training/learning_path.py"})
                for name in old_names:
                    self.assertEqual(package.read(name), old_package.read(name))
                self.assertEqual(
                    hashlib.sha256(package.read("src/rsna_knee/training/learning_path.py")).hexdigest(),
                    result["members"]["src/rsna_knee/training/learning_path.py"],
                )
            inventory = json.loads(Path(result["inventory_path"]).read_text(encoding="utf-8"))
            prepared_config = json.loads(Path(result["config_path"]).read_text(encoding="utf-8"))
            notebook = json.loads(Path(result["notebook_path"]).read_text(encoding="utf-8"))
            self.assertEqual(inventory["manifest_sha256"], "e88c61c22a463737f16951b7c76db03834034aecd1b31dc61b4ce495080b2be2")
            self.assertEqual(inventory["folds_sha256"], "b82b376b0086201e3364b11a0203b62890e628442075a52ea4a9e09ee0e778ac")
            self.assertEqual(prepared_config["expected_member_count"], len(names))
            self.assertEqual(prepared_config["max_diagnostic_steps"], 2)
            self.assertEqual(prepared_config["expected_inventory_sha256"], hashlib.sha256(Path(result["inventory_path"]).read_bytes()).hexdigest())
            self.assertEqual(len([path for path in Path(result["transport_dir"]).rglob("*") if path.is_file()]), len(names) + 3)
            self.assertFalse(any(cell.get("outputs", []) for cell in notebook["cells"]))
            code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
            compile(code, "learning_path_diagnostic.ipynb", "exec")
            self.assertLess(code.index("RUN_DIAGNOSTIC = False"), code.index("TrainingStudyDataset"))
            self.assertIn("bounded_training_subset(8)", code)
            self.assertIn('"input_audit_study_count": 8', code)
            self.assertIn('"probe_study_count": 2', code)
            self.assertIn('"optimizer_steps_total": 2', code)
            self.assertNotIn("study_id", code)
            self.assertNotIn("RUN_DIAGNOSTIC = True", code)
            exec(compile(code, "learning_path_diagnostic.ipynb", "exec"), {})
            expected_roots = (
                "/kaggle/input/datasets/davidsulleiro/rsna-knee-five-study-reader-private",
                "/kaggle/input/rsna-knee-five-study-reader-private",
            )
            self.assertTrue(all(root in code for root in expected_roots))
            self.assertLess(code.index("resolve_input_root(INPUT_CANDIDATES"), code.index("inventory_raw ="))
            self.assertNotIn("rglob", code)
            module = ast.parse(code)
            guard = next(node for node in module.body if isinstance(node, ast.If))
            resolver_node = next(node for node in guard.orelse if isinstance(node, ast.FunctionDef) and node.name == "resolve_input_root")
            namespace = {"hashlib": hashlib}
            exec(compile(ast.Module(body=[resolver_node], type_ignores=[]), "resolver.py", "exec"), namespace)
            resolve_input_root = namespace["resolve_input_root"]
            with TemporaryDirectory() as mount_temp:
                long_root, short_root = Path(mount_temp) / "long", Path(mount_temp) / "short"
                long_root.mkdir()
                short_root.mkdir()
                raw = b'{"schema_version":"mount-test"}\n'
                (short_root / "inventory.json").write_bytes(raw)
                expected = hashlib.sha256(raw).hexdigest()
                self.assertEqual(resolve_input_root((long_root, short_root), expected), short_root)
                (long_root / "inventory.json").write_bytes(raw)
                self.assertEqual(resolve_input_root((long_root, short_root), expected), long_root)
                (long_root / "inventory.json").write_bytes(b"wrong")
                (short_root / "inventory.json").write_bytes(b"also-wrong")
                with self.assertRaisesRegex(RuntimeError, "input_mount_unavailable"):
                    resolve_input_root((long_root, short_root), expected)

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
