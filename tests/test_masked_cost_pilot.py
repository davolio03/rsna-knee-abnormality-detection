import unittest
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
import sys
import types
from dataclasses import dataclass
from unittest import mock

from scripts.build_masked_cost_pilot import (
    SOURCE_NOTEBOOK,
    SOURCE_NOTEBOOK_SHA256,
    build_cost_pilot,
    estimate_population_seconds,
    select_stratified_indices,
)


class StratifiedSamplingTests(unittest.TestCase):
    def test_selection_is_deterministic_bounded_and_covers_every_stratum(self):
        records = [
            {"study_id": str(index), "series": [{}] * series_count}
            for series_count, count in ((1, 10), (2, 8), (3, 3), (4, 1))
            for index in range(count)
        ]
        first = select_stratified_indices(records, cap=12, seed=42)
        second = select_stratified_indices(records, cap=12, seed=42)
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 12)
        self.assertEqual({len(records[index]["series"]) for index in first}, {1, 2, 3, 4})

    def test_selection_fails_closed_when_strata_cannot_fit_in_cap(self):
        records = [{"series": [{}] * count} for count in range(1, 5)]
        with self.assertRaisesRegex(ValueError, "pilot_strata_exceed_cap"):
            select_stratified_indices(records, cap=3, seed=42)

    def test_population_estimate_is_stratum_weighted_and_rejects_nonfinite_values(self):
        stats = {
            "1": {"sample_count": 2, "population_count": 10, "mean_seconds": 1.0},
            "2": {"sample_count": 2, "population_count": 2, "mean_seconds": 3.0},
        }
        self.assertEqual(estimate_population_seconds(stats), 16.0)
        stats["2"]["mean_seconds"] = float("nan")
        with self.assertRaisesRegex(ValueError, "pilot_timing_invalid"):
            estimate_population_seconds(stats)
        with self.assertRaisesRegex(ValueError, "pilot_timing_invalid"):
            estimate_population_seconds({
                "overflow": {"sample_count": 1, "population_count": 4298,
                             "mean_seconds": 1e308},
            })


class NotebookGenerationTests(unittest.TestCase):
    def test_generation_preserves_pinned_source_cells_and_adds_only_a_pilot_cell(self):
        repository = Path(__file__).resolve().parents[1]
        source = repository / SOURCE_NOTEBOOK
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), SOURCE_NOTEBOOK_SHA256)
        original = json.loads(source.read_text(encoding="utf-8"))
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / SOURCE_NOTEBOOK).parent.mkdir(parents=True)
            shutil.copyfile(source, root / SOURCE_NOTEBOOK)
            (root / "data/private").mkdir(parents=True, exist_ok=True)
            result = build_cost_pilot(root, Path("data/private/pilot"))
            generated = json.loads(Path(result["notebook_path"]).read_text(encoding="utf-8"))
        self.assertEqual(len(generated["cells"]), 2)
        bootstrap = "".join(generated["cells"][0]["source"])
        self.assertIn("PINNED_NOTEBOOK_SOURCE", bootstrap)
        self.assertIn('RUN_STAGE = "plan"', bootstrap)
        pilot = "".join(generated["cells"][-1]["source"])
        compile(pilot, "pilot.py", "exec")
        self.assertIn('"pilot_only": True', pilot)
        self.assertIn('"full_cohort_training": False', pilot)
        self.assertIn('epochs=1, max_steps=None', pilot)
        self.assertIn('"masked-cost-pilot-run01"', pilot)
        self.assertIn('== 2850', pilot)
        self.assertIn('== 1448', pilot)
        self.assertIn('_pilot_estimate(train_stats) + _pilot_estimate(validation_stats)', pilot)
        self.assertNotIn('"study_id":', pilot)

    def test_generation_rejects_changed_source_without_creating_output(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / SOURCE_NOTEBOOK
            source.parent.mkdir(parents=True)
            source.write_text("{}", encoding="utf-8")
            private = root / "data/private"
            private.mkdir(parents=True, exist_ok=True)
            with self.assertRaisesRegex(ValueError, "pinned_source_mismatch"):
                build_cost_pilot(root, Path("data/private/pilot"))
            self.assertFalse((private / "pilot").exists())

    def test_embedded_proxy_times_fake_dataset_with_cuda_synchronization(self):
        from scripts.build_masked_cost_pilot import _notebook_cell

        namespace = {"__name__": "pilot_test", "json": json}
        exec(compile(_notebook_cell(), "pilot.py", "exec"), namespace)
        syncs = []

        class FakeCuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def synchronize():
                syncs.append(True)

        class FakeTorch:
            cuda = FakeCuda

        class FakeDataset:
            _records = [
                {"series": [{}]}, {"series": [{}]},
                {"series": [{}, {}]}, {"series": [{}, {}]},
            ]

            def __getitem__(self, index):
                return (index, None)

        proxy = namespace["_TimedSubset"](FakeDataset(), cap=4, partition="train", torch=FakeTorch,
                                           progress_callback=lambda _partition: None)
        with mock.patch("time.perf_counter", side_effect=[1.0, 3.0, 5.0, 7.0, 9.0]):
            for index in range(4):
                proxy[index]
            stats = proxy.finish()
        self.assertEqual(syncs, [True] * 5)
        self.assertEqual(stats["1"]["sample_count"], 2)
        self.assertEqual(stats["1"]["mean_seconds"], 2.0)
        self.assertEqual(stats["2"]["sample_count"], 2)
        self.assertEqual(namespace["_pilot_estimate"](stats), 8.0)

    def test_failed_cost_gate_still_emits_measured_summary_without_exception(self):
        from scripts.build_masked_cost_pilot import _notebook_cell

        namespace = {"__name__": "pilot_test", "json": json}
        exec(compile(_notebook_cell(), "pilot.py", "exec"), namespace)
        lines = []
        measured = {
            "finite_metrics": True, "checkpoint_readback": True, "report_readback": True,
            "memory_gate": True, "host_and_disk_gate": True, "fold_counts_valid": True,
            "finite_measurements": True, "pilot_deadline_gate": True,
            "time_gate": False, "estimated_five_epoch_seconds_with_safety": 40000,
        }
        result = namespace["_pilot_finalize"](measured, lines.append)
        self.assertEqual(len(lines), 1)
        self.assertEqual(result["pilot_status"], "measured_gate_failed")
        self.assertFalse(result["acceptable_for_full_run_review"])

    def test_failure_details_keep_safe_code_type_stage_and_sanitized_frames(self):
        from scripts.build_masked_cost_pilot import _notebook_cell

        namespace = {"__name__": "pilot_test", "json": json}
        exec(compile(_notebook_cell(), "pilot.py", "exec"), namespace)
        try:
            raise KeyError("private-study-123 C:/restricted/token")
        except KeyError as error:
            details = namespace["_safe_failure_details"](error, "train_dataset_access")
        self.assertEqual(details["stage"], "train_dataset_access")
        self.assertEqual(details["exception_type"], "KeyError")
        self.assertEqual(details["reason_code"], "pilot_runtime_failed")
        self.assertTrue(details["frames"])
        self.assertNotIn("private-study-123", json.dumps(details))
        self.assertNotIn("restricted", json.dumps(details))
        self.assertTrue(all("/" not in frame["file"] and "\\" not in frame["file"]
                            for frame in details["frames"]))

    def test_failure_details_allowlist_prevents_identifiers_in_type_reason_and_frames(self):
        from scripts.build_masked_cost_pilot import _notebook_cell

        namespace = {"__name__": "pilot_test", "json": json}
        exec(compile(_notebook_cell(), "pilot.py", "exec"), namespace)
        private_error = type("PRIVATESTUDY123", (Exception,), {})
        scope = {"PRIVATESTUDY123": private_error}
        exec(compile("def PRIVATESTUDY123(error):\n    raise error\n",
                     "PRIVATESTUDY123.dicom", "exec"), scope)
        try:
            scope["PRIVATESTUDY123"](private_error("private"))
        except Exception as error:
            captured_error = error
        error = captured_error
        error.reason_code = "private_study123"
        details = namespace["_safe_failure_details"](error, "train_dataset_access")
        serialized = json.dumps(details)
        self.assertEqual(details["exception_type"], "Exception")
        self.assertEqual(details["reason_code"], "pilot_runtime_failed")
        self.assertNotIn("PRIVATESTUDY123", serialized)
        self.assertNotIn("private_study123", serialized)
        self.assertNotIn("dicom", serialized)
        self.assertTrue(all(frame["file"] == "unknown" and frame["function"] == "unknown"
                            for frame in details["frames"]))

    def test_generated_cells_run_plan_once_with_pinned_mounts_and_no_autorun(self):
        repository = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / SOURCE_NOTEBOOK
            source.parent.mkdir(parents=True)
            shutil.copyfile(repository / SOURCE_NOTEBOOK, source)
            (root / "data/private").mkdir(parents=True, exist_ok=True)
            artifact = build_cost_pilot(root, Path("data/private/pilot"))
            notebook = json.loads(Path(artifact["notebook_path"]).read_text(encoding="utf-8"))
            namespace = {"__name__": "__main__"}
            exec(compile("".join(notebook["cells"][0]["source"]),
                         "generated-bootstrap.py", "exec"), namespace)
            self.assertEqual(namespace["PINNED_NOTEBOOK_SOURCE"],
                             "".join(json.loads((repository / SOURCE_NOTEBOOK).read_text(
                                 encoding="utf-8"))["cells"][0]["source"]))
            namespace["__name__"] = "pilot_test"
            exec(compile("".join(notebook["cells"][1]["source"]),
                         "generated-pilot.py", "exec"), namespace)

            calls = []
            class Config:
                @classmethod
                def from_mapping(cls, values):
                    return values

            class FakeRunner:
                MaskedSoftConfig = Config
                @staticmethod
                def plan_masked_soft_training(_config):
                    calls.append("plan")
                    return {"status": "planned", "training_study_count": 2850,
                            "validation_study_count": 1448}

            def module_loader(name):
                self.assertEqual(name, "rsna_knee.training.masked_soft")
                return FakeRunner

            image_root = root / "images"
            image_root.mkdir()
            working_root = root / "working"
            working_root.mkdir()
            result = namespace["run_notebook"](
                stage="plan", full_training=False,
                asset_root=repository / "data/private/masked-soft-training-preparation/run06/upload",
                image_root=image_root, working_root=working_root,
                module_loader=module_loader, emit=lambda _value: None,
            )
            self.assertEqual(calls, ["plan"])
            self.assertEqual(result["plan"]["training_study_count"], 2850)
            self.assertFalse(namespace["_pilot_gate"]({"time_gate": False}))

    def test_full_pilot_bootstrap_lazily_loads_runtime_then_uses_fake_training(self):
        repository = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / SOURCE_NOTEBOOK
            source.parent.mkdir(parents=True)
            shutil.copyfile(repository / SOURCE_NOTEBOOK, source)
            (root / "data/private").mkdir(parents=True, exist_ok=True)
            artifact = build_cost_pilot(root, Path("data/private/pilot"))
            notebook = json.loads(Path(artifact["notebook_path"]).read_text(encoding="utf-8"))
            namespace = {"__name__": "__main__"}
            exec(compile("".join(notebook["cells"][0]["source"]), "bootstrap.py", "exec"), namespace)
            namespace["__name__"] = "pilot_test"
            exec(compile("".join(notebook["cells"][1]["source"]), "pilot.py", "exec"), namespace)

            working = root / "working"
            input_root = root / "input"
            images = input_root / "competition" / "train_series"
            images.mkdir(parents=True)
            input_root.mkdir(exist_ok=True)
            working.mkdir()
            namespace["WORKING_ROOT"] = working
            namespace["INPUT_ROOT"] = input_root
            namespace["PREPARED_ASSET_ROOT"] = repository / "data/private/masked-soft-training-preparation/run06/upload"
            namespace["COMPETITION_IMAGE_ROOT"] = images
            namespace["_host_available_bytes"] = lambda: 10 * 1024**3

            @dataclass(frozen=True)
            class FakeConfig:
                manifest_path: Path
                folds_path: Path
                image_root: Path
                output_dir: Path
                fold: int
                epochs: int
                max_steps: object
                output_size: int
                max_slices: int
                device: str
                learning_rate: float
                balance_mode: str
                seed: int
                weights_path: Path
                weights_sha256: str
                weights_size: int

                @classmethod
                def from_mapping(cls, values):
                    return cls(**values)

            class FakeDataset:
                def __init__(self, _manifest, _folds, _image_root, *, partition, **_kwargs):
                    self._records = [{"series": [{}]}] * (2850 if partition == "train" else 1448)

                def __getitem__(self, _index):
                    return None

            expected_weights = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
            original_default_factory = lambda _config: "fake-model"
            fake_runner = types.SimpleNamespace(
                PRIVATE_RUN_ROOT=working / "masked-soft-source/data/private/masked-soft-training",
                PINNED_RESNET18_SHA256=expected_weights,
                MaskedSoftConfig=FakeConfig,
                _build_default_model=original_default_factory,
                plan_masked_soft_training=lambda _config: {
                    "status": "planned", "training_study_count": 2850, "validation_study_count": 1448,
                },
            )

            def fake_training(config, *, execute, dataset_factory):
                self.assertIs(execute, True)
                staged_factory = fake_runner._build_default_model
                self.assertIsNot(staged_factory, original_default_factory)
                self.assertEqual(staged_factory(config), "fake-model")
                self.assertEqual(config.epochs, 1)
                self.assertIsNone(config.max_steps)
                train = dataset_factory(config, "train")
                validation = dataset_factory(config, "validation")
                for index in range(len(train)):
                    train[index]
                for index in range(len(validation)):
                    validation[index]
                config.output_dir.mkdir(parents=True)
                (config.output_dir / "model-state.pt").write_bytes(b"fixture")
                report = {"status": "complete", "epochs": [{
                    "training_masked_bce": 0.5,
                    "validation": {"masked_bce": 0.5, "brier": 0.2},
                }]}
                (config.output_dir / "training-report.json").write_text(json.dumps(report), encoding="utf-8")
                return report

            fake_runner.run_masked_soft_training = fake_training
            class FakeCuda:
                @staticmethod
                def is_available(): return True
                @staticmethod
                def reset_peak_memory_stats(): pass
                @staticmethod
                def synchronize(): pass
                @staticmethod
                def get_device_properties(_index): return types.SimpleNamespace(total_memory=10**9)
                @staticmethod
                def max_memory_allocated(): return 10**6
                @staticmethod
                def max_memory_reserved(): return 2 * 10**6
                @staticmethod
                def get_device_name(_index): return "fake-gpu"
                @staticmethod
                def device_count(): return 1
            fake_torch = types.SimpleNamespace(
                cuda=FakeCuda,
                load=lambda *_args, **_kwargs: {
                    "state_dict": {"fixture": 1},
                    "metadata": {"model_kind": "resnet18_imagenet1k_v1", "freeze_backbone": True,
                                 "optimizer": "Adam", "weights_sha256": expected_weights},
                },
            )
            fake_pydicom = types.SimpleNamespace(pixels=types.SimpleNamespace(
                get_decoder=lambda _uid: types.SimpleNamespace(is_available=True)))

            def fake_install(_bundle, _manifest, target):
                Path(target).mkdir()
                return {"target": str(target)}

            fake_preflight = types.SimpleNamespace(run_bounded_decode_preflight=lambda dataset, **_kwargs: {
                "status": "complete", "attempted_study_count": 8,
                "decoded_series_count": 8, "decoded_slice_count": 8,
            })
            fake_modules = {
                "rsna_knee.training.masked_soft": fake_runner,
                "rsna_knee.data.masked_soft_dataset": types.SimpleNamespace(
                    FullCohortSoftDataset=FakeDataset),
                "rsna_knee.training.masked_preflight": fake_preflight,
                "rsna_knee.inference.offline_codec_bootstrap": types.SimpleNamespace(
                    install_codec_bundle=fake_install),
                "pydicom": fake_pydicom,
                "pydicom.uid": types.SimpleNamespace(UID=lambda value: value),
            }
            imported = []
            original_import = __import__("importlib").import_module
            def fake_import(name, *args, **kwargs):
                imported.append(name)
                if name in fake_modules:
                    if name.startswith("rsna_knee."):
                        self.assertTrue((working / "masked-soft-source").is_dir(),
                                        "runtime imports must occur only after source extraction")
                    return fake_modules[name]
                if name.startswith("rsna_knee."):
                    self.assertTrue((working / "masked-soft-source").is_dir(),
                                    "runtime imports must occur only after source extraction")
                    raise AssertionError("unexpected project module import")
                return original_import(name, *args, **kwargs)
            with mock.patch("importlib.import_module", side_effect=fake_import), \
                    mock.patch.dict(sys.modules, {"torch": fake_torch}):
                try:
                    result = namespace["run_masked_cost_pilot"](namespace["run_notebook"])
                except RuntimeError as error:
                    failure = working / "data/private/masked-training-cost/run01/pilot-failure.json"
                    raise AssertionError(f"{error}: imports={imported}; {failure.read_text() if failure.exists() else 'no failure record'}")
            self.assertEqual(result["pilot_status"], "measured_pass")
            self.assertEqual(result["verified_fold_population"], {
                "train": 2850, "validation": 1448, "expected_fold0_counts": True,
            })
            self.assertTrue((working / "data/private/masked-training-cost/run01/pilot-summary.json").is_file())
            self.assertIs(fake_runner._build_default_model, original_default_factory)


if __name__ == "__main__":
    unittest.main()
