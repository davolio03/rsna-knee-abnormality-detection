"""Offline-only tests for the private masked-training package preparation."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock
from zipfile import ZipFile

from scripts.build_kaggle_masked_training_package import (
    INVENTORY_SCHEMA,
    SOURCE_MODULE_ALLOWLIST,
    build_masked_training_package,
)


ROOT = Path(__file__).resolve().parents[1]


class KaggleMaskedTrainingPackageTests(unittest.TestCase):
    def test_authorized_launch_variant_is_distinct_and_bounded_with_resume_explicit(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = build_masked_training_package(ROOT, root / "launch", private_root=root,
                                                  authorized_full_training=True)
            self.assertEqual(Path(result["notebook_path"]).name,
                             "kaggle_masked_soft_training_authorized.ipynb")
            code = "\n".join("".join(cell["source"]) for cell in
                              json.loads(Path(result["notebook_path"]).read_text())["cells"])
            self.assertIn('RUN_STAGE = "full_training"', code)
            self.assertIn("RUN_FULL_TRAINING = True", code)
            self.assertIn('OUTPUT_RUN_NAME = "run01"', code)
            self.assertIn("RESUME_CHECKPOINT = None", code)
            self.assertIn("RESUME_CHECKPOINT_SHA256 = None", code)
            self.assertIn("MAX_EPOCHS_PER_RUN = 2", code)
            self.assertIn("MAX_RUNTIME_SECONDS = 28800", code)
            self.assertIn('"epochs": 5, "max_steps": None', code)
            self.assertIn('"resume_checkpoint": RESUME_CHECKPOINT', code)
            self.assertIn("run_bounded_decode_preflight(dataset, max_studies=8)", code)

    def test_archive_imports_dynamic_codec_bootstrap_in_isolated_python(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = build_masked_training_package(ROOT, root / "package", private_root=root)
            extracted = root / "extracted"
            extracted.mkdir()
            with ZipFile(result["archive_path"]) as archive:
                archive.extractall(extracted)

            script = (
                "import importlib, pathlib, sys, types; "
                "torch=types.ModuleType('torch'); torch.Tensor=object; "
                "nn=types.ModuleType('torch.nn'); nn.Module=type('Module',(),{}); "
                "torch.nn=nn; functional=types.ModuleType('torch.nn.functional'); "
                "nn.functional=functional; sys.modules.update({'torch':torch,'torch.nn':nn,'torch.nn.functional':functional}); "
                "root=pathlib.Path(sys.argv[1]).resolve(); sys.path.insert(0, str(root)); "
                "names=('rsna_knee.training.masked_soft', 'rsna_knee.inference.offline_codec_bootstrap', "
                "'rsna_knee.data.masked_soft_dataset', 'rsna_knee.training.masked_preflight', "
                "'rsna_knee.models.resnet18', 'rsna_knee.models.baseline'); "
                "modules=[importlib.import_module(name) for name in names]; "
                "assert all(pathlib.Path(module.__file__).resolve().is_relative_to(root) for module in modules); "
                "assert callable(modules[1].install_codec_bundle); print('isolated-bootstrap-ok')"
            )
            completed = subprocess.run(
                [sys.executable, "-I", "-c", script, str(extracted)],
                cwd=root, capture_output=True, text=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("isolated-bootstrap-ok", completed.stdout)

    def test_generated_plan_uses_packaged_source_and_pinned_metadata_in_isolated_python(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = build_masked_training_package(ROOT, root / "package", private_root=root)
            input_root = root / "input" / "competitions" / "rsna-knee-abnormality-detection" / "train_series"
            input_root.mkdir(parents=True)
            working_root = root / "working"
            working_root.mkdir()
            script = (
                "import json, pathlib, sys; "
                "from pathlib import Path; "
                "root=Path(sys.argv[1]); upload=Path(sys.argv[2]); notebook=Path(sys.argv[3]); "
                "input_root=Path(sys.argv[4]); working_root=Path(sys.argv[5]); "
                "data=json.loads(notebook.read_text(encoding='utf-8')); "
                "code='\\n'.join(''.join(c['source']) for c in data['cells'] if c['cell_type']=='code'); "
                "ns={'__name__':'isolated_generated_notebook'}; exec(compile(code,str(notebook),'exec'),ns); "
                "result=ns['run_notebook'](asset_root=upload,input_root=input_root,working_root=working_root,emit=lambda _:None); "
                "assert result['plan']['status']=='planned'; "
                "assert (result['plan']['training_study_count'],result['plan']['validation_study_count'])==(2850,1448); "
                "assert 'pydicom' not in sys.modules; "
                "assert not (working_root/'masked-codec-install').exists(); "
                "print('isolated-plan-ok')"
            )
            completed = subprocess.run(
                [sys.executable, "-I", "-c", script, str(root), result["upload_dir"],
                 result["notebook_path"], str(input_root.parents[2]), str(working_root)],
                cwd=root, capture_output=True, text=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("isolated-plan-ok", completed.stdout)

    def test_builds_deterministic_allowlisted_package_and_disabled_notebook(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = build_masked_training_package(ROOT, root / "run01", private_root=root)
            second = build_masked_training_package(ROOT, root / "run02", private_root=root)

            first_archive = Path(first["archive_path"]).read_bytes()
            second_archive = Path(second["archive_path"]).read_bytes()
            self.assertEqual(first_archive, second_archive)
            self.assertEqual(hashlib.sha256(first_archive).hexdigest(), first["archive_sha256"])
            prior_archive = ROOT / "data/private/masked-soft-training-preparation/run05/masked-training-source.zip"
            if prior_archive.is_file():
                with ZipFile(prior_archive) as previous, ZipFile(first["archive_path"]) as current:
                    prior_names, current_names = set(previous.namelist()), set(current.namelist())
                    self.assertEqual(current_names - prior_names, {"rsna_knee/models/baseline.py"})
                    self.assertEqual(prior_names - current_names, set())
                    for name in prior_names:
                        if name != "rsna_knee/training/masked_soft.py":
                            self.assertEqual(current.read(name), previous.read(name), name)
            inventory = json.loads(Path(first["inventory_path"]).read_text(encoding="utf-8"))
            self.assertEqual(inventory["schema_version"], INVENTORY_SCHEMA)
            self.assertEqual(inventory["source_archive_sha256"], first["archive_sha256"])
            self.assertEqual(inventory["model"], {
                "kind": "resnet18_imagenet1k_v1", "freeze_backbone": True,
                "optimizer": "Adam", "learning_rate": 0.001,
                "weights_sha256": "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
                "weights_size": 46830571,
            })
            self.assertEqual(inventory["training_config"], {
                "stage_default": "plan", "fold": 0, "seed": 42, "epochs": 5,
                "output_size": 256, "max_slices": 8, "balance_mode": "unweighted",
                "max_steps": None, "learning_rate": 0.001,
            })
            self.assertEqual(inventory["cohort"]["manifest_sha256"],
                             "617ea1c267960b1e903b332a78388db3b6ab80d44374721df5a4c5e184e8961b")
            self.assertEqual(inventory["cohort"]["folds_sha256"],
                             "b816c7902a5d0caa68a9e818d3f913943bc61cd2e6db65c4de069f5cd94e15f9")
            self.assertEqual(inventory["cohort"]["eligible_study_count"], 4298)
            self.assertEqual(inventory["cohort"]["eligible_series_count"], 23765)
            self.assertEqual(inventory["cohort"]["full_non_gold_series_count"], 24035)
            self.assertEqual(set(inventory["assets"]), {
                "full-cohort-soft-label-manifest.json", "full-cohort-grouped-folds.json",
                "resnet18-f37072fd.pth", "offline-codecs.zip",
            })
            with ZipFile(first["archive_path"]) as archive:
                names = set(archive.namelist())
                self.assertEqual(names, set(inventory["source_members"]))
                self.assertEqual(len(names), len(archive.namelist()))
                self.assertTrue(all(name.endswith(".py") for name in names))
                self.assertFalse(any(name.endswith((".csv", ".dcm", ".pth", ".ipynb")) for name in names))
                self.assertTrue(all(name.split("/", 1)[0] == "rsna_knee" for name in names))
                self.assertIn("rsna_knee/models/baseline.py", names)
                for name, digest in inventory["source_members"].items():
                    payload = archive.read(name)
                    self.assertEqual(hashlib.sha256(payload).hexdigest(), digest)

            upload = Path(first["upload_dir"])
            self.assertFalse((upload / "fold0-inference-source.zip").exists())
            self.assertFalse((upload / "fold0-inference-notebook.ipynb").exists())
            self.assertFalse((upload / "baseline-cohort.json").exists())
            source_b64 = upload / "masked-training-source.zip.b64"
            self.assertEqual(base64.b64decode(source_b64.read_bytes(), validate=True), first_archive)
            self.assertEqual(hashlib.sha256((upload / "full-cohort-soft-label-manifest.json").read_bytes()).hexdigest(),
                             inventory["cohort"]["manifest_sha256"])
            self.assertEqual(hashlib.sha256((upload / "full-cohort-grouped-folds.json").read_bytes()).hexdigest(),
                             inventory["cohort"]["folds_sha256"])
            self.assertEqual(hashlib.sha256((upload / "resnet18-f37072fd.pth").read_bytes()).hexdigest(),
                             inventory["model"]["weights_sha256"])

            notebook = json.loads(Path(first["notebook_path"]).read_text(encoding="utf-8"))
            self.assertTrue(all(not cell.get("outputs", []) for cell in notebook["cells"]))
            code = "\n".join("".join(cell["source"]) for cell in notebook["cells"]
                              if cell["cell_type"] == "code")
            compile(code, "masked_training_notebook.py", "exec")
            self.assertIn('RUN_STAGE = "plan"', code)
            self.assertIn('RUN_FULL_TRAINING = False', code)
            self.assertIn("install_codec_bundle", code)
            self.assertIn("plan_masked_soft_training", code)
            self.assertNotIn("submission.csv", code)
            self.assertNotIn("extractall", code)
            self.assertEqual(Path(first["notebook_path"]).read_bytes(),
                             (upload / "kaggle_masked_soft_training.ipynb").read_bytes())

    def test_bootstrap_and_decoder_preflight_precede_training_import(self):
        with TemporaryDirectory() as temporary:
            result = build_masked_training_package(ROOT, Path(temporary) / "run01", private_root=Path(temporary))
            notebook = json.loads(Path(result["notebook_path"]).read_text(encoding="utf-8"))
            code = "\n".join("".join(cell["source"]) for cell in notebook["cells"]
                              if cell["cell_type"] == "code")
            plan_return = code.index('if stage == "plan":')
            codec_install = code.index("install_result = installer(")
            pydicom_import = code.index('pydicom = load("pydicom")')
            decoder_checks = code.index("_check_required_decoders(pydicom, uid_module.UID)")
            dataset_load = code.index('dataset_module = load("rsna_knee.data.masked_soft_dataset")')
            preflight_call = code.index("run_bounded_decode_preflight(dataset, max_studies=8)")
            training_call = code.index("runner.run_masked_soft_training(config, execute=True)")
            self.assertLess(plan_return, codec_install)
            self.assertLess(codec_install, pydicom_import)
            self.assertLess(pydicom_import, decoder_checks)
            self.assertLess(decoder_checks, dataset_load)
            self.assertLess(preflight_call, training_call)
            self.assertIn('getattr(decoder, "is_available", False) is not True', code)

    def test_rejects_bad_weight_pin_and_existing_output_before_writing(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            bad_weights = root / "bad.pth"
            bad_weights.write_bytes(b"not-the-pinned-weights")
            output = root / "must-not-exist"
            with self.assertRaises(ValueError):
                build_masked_training_package(ROOT, output, weights_path=bad_weights, private_root=root)
            self.assertFalse(output.exists())

            occupied = root / "occupied"
            occupied.mkdir()
            with self.assertRaises(ValueError):
                build_masked_training_package(ROOT, occupied, private_root=root)
            self.assertEqual(list(occupied.iterdir()), [])

    def test_rejects_unapproved_source_closure_before_output_creation(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "must-not-exist"
            with mock.patch(
                "scripts.build_kaggle_masked_training_package.SOURCE_MODULE_ALLOWLIST",
                frozenset({"rsna_knee.training.masked_soft"}),
            ):
                with self.assertRaises(ValueError):
                    build_masked_training_package(ROOT, output, private_root=Path(temporary))
            self.assertFalse(output.exists())


class BoundedDecodePreflightTests(unittest.TestCase):
    def test_preflight_is_bounded_and_reports_only_aggregate_counts(self):
        import numpy as np
        from rsna_knee.training.masked_preflight import run_bounded_decode_preflight

        class SyntheticDataset:
            def __init__(self):
                self.calls = []

            def __len__(self):
                return 100

            def __getitem__(self, index):
                self.calls.append(index)
                images = [np.zeros((2, 1, 8, 8), dtype=np.float32),
                          np.zeros((3, 1, 8, 8), dtype=np.float32)]
                pixel_masks = [np.ones_like(images[0], dtype=bool),
                               np.ones_like(images[1], dtype=bool)]
                return images, pixel_masks, np.zeros(12), np.ones(12, dtype=bool), index

        dataset = SyntheticDataset()
        result = run_bounded_decode_preflight(dataset, max_studies=8)
        self.assertEqual(dataset.calls, list(range(8)))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["attempted_study_count"], 8)
        self.assertEqual(result["decoded_series_count"], 16)
        self.assertEqual(result["decoded_slice_count"], 40)
        self.assertNotIn("study_id", result)
        self.assertNotIn("path", result)
        self.assertNotIn("targets", result)

    def test_preflight_failure_is_categorical_and_never_contains_private_exception_text(self):
        import numpy as np
        from rsna_knee.data.masked_soft_dataset import SoftDatasetError
        from rsna_knee.training.masked_preflight import run_bounded_decode_preflight

        class SyntheticDataset:
            def __len__(self):
                return 2

            def __getitem__(self, index):
                if index:
                    raise SoftDatasetError("study_load_failed")
                images = [np.zeros((1, 1, 8, 8), dtype=np.float32)]
                masks = [np.ones_like(images[0], dtype=bool)]
                return images, masks, np.zeros(12), np.ones(12, dtype=bool), index

        result = run_bounded_decode_preflight(SyntheticDataset(), max_studies=2)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason_code"], "study_load_failed")
        self.assertNotIn("study_id", result)

    def test_preflight_rejects_unbounded_or_empty_limits(self):
        from rsna_knee.training.masked_preflight import run_bounded_decode_preflight

        for limit in (0, 9, True, "8"):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                run_bounded_decode_preflight([], max_studies=limit)


class GeneratedNotebookExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.package = build_masked_training_package(
            ROOT, cls.root / "package", private_root=cls.root,
        )
        cls.upload = Path(cls.package["upload_dir"])
        notebook = json.loads(Path(cls.package["notebook_path"]).read_text(encoding="utf-8"))
        cls.code = "\n".join("".join(cell["source"]) for cell in notebook["cells"]
                              if cell["cell_type"] == "code")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def namespace(self):
        namespace = {"__name__": "masked_notebook_test"}
        exec(compile(self.code, "generated_masked_notebook.py", "exec"), namespace)
        self.assertIn("run_notebook", namespace)
        return namespace

    def competition_root(self, root: Path, relative="competitions/rsna-knee-abnormality-detection/train_series"):
        path = root.joinpath(*Path(relative).parts)
        path.mkdir(parents=True)
        return path

    def fake_modules(self, *, decoder_available=True, preflight_status="complete", path_list=None):
        events = []
        calls = {"installer": 0, "pydicom": 0, "dataset": 0, "preflight": 0, "runner": 0}
        work_codec = self.root / "codec-target"
        work_codec.mkdir(exist_ok=True)

        class FakeConfig:
            @classmethod
            def from_mapping(cls, values):
                return types.SimpleNamespace(**values)

        def loader(name):
            events.append(name)
            if name == "rsna_knee.training.masked_soft":
                def plan(config):
                    events.append("plan")
                    return {"status": "planned", "training_study_count": 2850,
                            "validation_study_count": 1448}

                def run(config, *, execute=False):
                    calls["runner"] += 1
                    events.append(("training", execute, config.output_dir,
                                   config.max_epochs_per_run, config.max_runtime_seconds,
                                   config.resume_checkpoint, config.resume_checkpoint_sha256))
                    return {"status": "complete"}

                return types.SimpleNamespace(MaskedSoftConfig=FakeConfig,
                                             plan_masked_soft_training=plan,
                                             run_masked_soft_training=run)
            if name == "rsna_knee.inference.offline_codec_bootstrap":
                def install(*args):
                    calls["installer"] += 1
                    events.append("codec_install")
                    return {"target": str(work_codec)}
                return types.SimpleNamespace(install_codec_bundle=install)
            if name == "pydicom":
                calls["pydicom"] += 1
                events.append(("pydicom_import", path_list[0] if path_list else None))
                return types.SimpleNamespace(pixels=types.SimpleNamespace(
                    get_decoder=lambda uid: types.SimpleNamespace(is_available=decoder_available)))
            if name == "pydicom.uid":
                return types.SimpleNamespace(UID=lambda value: value)
            if name == "rsna_knee.data.masked_soft_dataset":
                def dataset(*args, **kwargs):
                    calls["dataset"] += 1
                    events.append("dataset")
                    return object()
                return types.SimpleNamespace(FullCohortSoftDataset=dataset)
            if name == "rsna_knee.training.masked_preflight":
                def preflight(dataset, *, max_studies):
                    calls["preflight"] += 1
                    events.append(("preflight", max_studies))
                    return {"status": preflight_status, "attempted_study_count": 8}
                return types.SimpleNamespace(run_bounded_decode_preflight=preflight)
            raise AssertionError("unexpected module import: " + name)

        return loader, events, calls, work_codec

    def test_generated_plan_executes_without_codec_install_or_pydicom_import(self):
        namespace = self.namespace()
        input_root = self.root / "plan-input"
        self.competition_root(input_root)
        working_root = self.root / "plan-working"
        working_root.mkdir()
        path_list = []
        loader, events, calls, _ = self.fake_modules(path_list=path_list)
        def forbidden_install(*args):
            raise AssertionError("codec installation must not run in default PLAN")
        result = namespace["run_notebook"](
            asset_root=self.upload, input_root=input_root, working_root=working_root,
            module_loader=loader, codec_installer=forbidden_install, path_list=path_list,
        )
        self.assertEqual(namespace["RUN_STAGE"], "plan")
        self.assertIs(namespace["RUN_FULL_TRAINING"], False)
        self.assertEqual(result["plan"]["status"], "planned")
        self.assertEqual(calls, {"installer": 0, "pydicom": 0, "dataset": 0,
                                 "preflight": 0, "runner": 0})
        self.assertFalse((working_root / "masked-codec-install").exists())

    def test_nested_assets_and_flat_and_nested_competition_mounts_are_bounded(self):
        namespace = self.namespace()
        marker_bytes = (self.upload / "masked-training-inventory.json").read_bytes()
        for relative in ("dataset-slug", "datasets/owner/dataset-slug"):
            root = self.root / ("input-" + str(len(relative)))
            marker = root.joinpath(*Path(relative).parts, "masked-training-inventory.json")
            marker.parent.mkdir(parents=True)
            marker.write_bytes(marker_bytes)
            for name in ("masked-training-source.zip.b64", "full-cohort-soft-label-manifest.json",
                         "full-cohort-grouped-folds.json", "resnet18-f37072fd.pth",
                         "offline-codecs.zip.b64"):
                (marker.parent / name).write_bytes(b"fixture")
            self.assertEqual(namespace["find_prepared_asset_root"](root), marker.parent)
        for relative in (
            "rsna-knee-abnormality-detection/train_images",
            "competitions/rsna-knee-abnormality-detection/train_series",
        ):
            root = self.root / ("competition-" + str(len(relative)))
            expected = self.competition_root(root, relative)
            self.assertEqual(namespace["find_competition_root"](root), expected)
            self.assertEqual(namespace["find_competition_root"](root, expected), expected)

    def test_missing_or_ambiguous_competition_mount_fails_closed(self):
        namespace = self.namespace()
        missing = self.root / "competition-missing"
        missing.mkdir()
        with self.assertRaisesRegex(RuntimeError, "competition_series_input_missing"):
            namespace["find_competition_root"](missing)
        ambiguous = self.root / "competition-ambiguous"
        self.competition_root(ambiguous, "rsna-knee-abnormality-detection/train_series")
        self.competition_root(ambiguous, "competitions/rsna-knee-abnormality-detection/train_images")
        with self.assertRaisesRegex(RuntimeError, "competition_input_ambiguous"):
            namespace["find_competition_root"](ambiguous)

    def test_ambiguous_input_assets_fail_before_source_extraction_or_import(self):
        namespace = self.namespace()
        input_root = self.root / "ambiguous-input"
        marker_bytes = (self.upload / "masked-training-inventory.json").read_bytes()
        for relative in ("dataset-one", "datasets/owner/dataset-two"):
            marker = input_root.joinpath(*Path(relative).parts, "masked-training-inventory.json")
            marker.parent.mkdir(parents=True)
            marker.write_bytes(marker_bytes)
        working_root = self.root / "ambiguous-working"
        working_root.mkdir()
        path_list = []
        loader, _, calls, _ = self.fake_modules(path_list=path_list)
        with self.assertRaisesRegex(RuntimeError, "prepared_input_missing_or_ambiguous"):
            namespace["run_notebook"](input_root=input_root, working_root=working_root,
                                      stage="plan", full_training=False,
                                      module_loader=loader, path_list=path_list)
        self.assertEqual(calls["pydicom"], 0)
        self.assertFalse((working_root / "masked-soft-source").exists())

    def test_preflight_installs_codecs_and_exposes_target_before_available_decoder_checks(self):
        namespace = self.namespace()
        input_root = self.root / "preflight-input"
        self.competition_root(input_root)
        working_root = self.root / "preflight-working"
        working_root.mkdir()
        path_list = []
        loader, events, calls, codec_target = self.fake_modules(path_list=path_list)
        result = namespace["run_notebook"](
            asset_root=self.upload, input_root=input_root, working_root=working_root,
            stage="decode_preflight", full_training=False, module_loader=loader,
            path_list=path_list,
        )
        self.assertEqual(result["preflight"]["status"], "complete")
        self.assertEqual(calls["runner"], 0)
        self.assertEqual(calls["preflight"], 1)
        self.assertLess(events.index("codec_install"), events.index("pydicom"))
        pydicom_event = next(event for event in events if isinstance(event, tuple) and event[0] == "pydicom_import")
        self.assertEqual(Path(pydicom_event[1]), codec_target)
        self.assertEqual(path_list[0], str(codec_target))

    def test_unavailable_decoder_fails_before_dataset_preflight(self):
        namespace = self.namespace()
        input_root = self.root / "decoder-input"
        self.competition_root(input_root)
        working_root = self.root / "decoder-working"
        working_root.mkdir()
        path_list = []
        loader, _, calls, _ = self.fake_modules(decoder_available=False, path_list=path_list)
        with self.assertRaisesRegex(RuntimeError, "required_dicom_decoder_unavailable"):
            namespace["run_notebook"](asset_root=self.upload, input_root=input_root,
                                      working_root=working_root, stage="decode_preflight",
                                      full_training=False, module_loader=loader,
                                      path_list=path_list)
        self.assertEqual(calls["dataset"], 0)
        self.assertEqual(calls["runner"], 0)

    def test_full_training_waits_for_successful_preflight_and_uses_persistent_private_root(self):
        namespace = self.namespace()
        input_root = self.root / "full-input"
        self.competition_root(input_root)
        working_root = self.root / "full-working"
        working_root.mkdir()
        path_list = []
        loader, events, calls, _ = self.fake_modules(path_list=path_list)
        result = namespace["run_notebook"](
            asset_root=self.upload, input_root=input_root, working_root=working_root,
            stage="full_training", full_training=True, module_loader=loader,
            path_list=path_list,
        )
        expected_output = working_root / "data/private/masked-soft-training/run01"
        self.assertEqual(result["training"]["status"], "complete")
        self.assertEqual(events[-1], ("training", True, expected_output, 2, 28800, None, None))
        self.assertEqual(calls["preflight"], 1)

        namespace["OUTPUT_RUN_NAME"] = "run02"
        namespace["RESUME_CHECKPOINT"] = "/kaggle/input/prior-output/checkpoint-epoch-02.pt"
        namespace["RESUME_CHECKPOINT_SHA256"] = "a" * 64
        resumed_working_root = self.root / "resume-working"
        resumed_working_root.mkdir()
        resumed = namespace["run_notebook"](
            asset_root=self.upload, input_root=input_root, working_root=resumed_working_root,
            stage="full_training", full_training=True, module_loader=loader, path_list=path_list,
        )
        self.assertEqual(resumed["training"]["status"], "complete")
        self.assertEqual(events[-1], ("training", True,
            resumed_working_root / "data/private/masked-soft-training/run02", 2, 28800,
            "/kaggle/input/prior-output/checkpoint-epoch-02.pt", "a" * 64))

        source_root = working_root / "masked-soft-source"
        script = (
            "import sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); "
            "import rsna_knee.training.masked_soft as m; "
            "expected=Path(sys.argv[2]); assert m.PRIVATE_RUN_ROOT == expected; "
            "assert m._check_soft_output(type('Config',(),{'output_dir':expected/'run01'})()) == expected/'run01'; "
            "print('persistent-root-ok')"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(source_root)
        completed = subprocess.run([sys.executable, "-c", script,
                                    str(source_root),
                                    str(working_root / "data/private/masked-soft-training")],
                                   cwd=ROOT, env=env, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("persistent-root-ok", completed.stdout)

    def test_full_training_does_not_start_when_bounded_preflight_fails(self):
        namespace = self.namespace()
        input_root = self.root / "full-fail-input"
        self.competition_root(input_root)
        working_root = self.root / "full-fail-working"
        working_root.mkdir()
        path_list = []
        loader, _, calls, _ = self.fake_modules(preflight_status="failed", path_list=path_list)
        with self.assertRaisesRegex(RuntimeError, "bounded_preflight_failed"):
            namespace["run_notebook"](asset_root=self.upload, input_root=input_root,
                                      working_root=working_root, stage="full_training",
                                      full_training=True, module_loader=loader,
                                      path_list=path_list)
        self.assertEqual(calls["runner"], 0)


if __name__ == "__main__":
    unittest.main()
