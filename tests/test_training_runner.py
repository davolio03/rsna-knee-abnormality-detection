import json
import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

try:
    import numpy as np
    import torch
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("optional NumPy/PyTorch training dependencies are not installed") from exc

from rsna_knee.labels.subscription import LABEL_COLUMNS


class BaselineRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_config(self, root, **overrides):
        from rsna_knee.training.baseline import TrainingConfig
        values = {
            "mode": "overfit",
            "manifest_path": str(root / "manifest.json"),
            "folds_path": str(root / "folds.json"),
            "image_root": str(root / "images"),
            "output_dir": str(root / "new-output"),
            "fold": 0,
            "seed": 73,
            "epochs": 1,
            "max_steps": 2,
            "overfit_limit": 2,
            "output_size": 16,
            "max_slices": 1,
            "device": "cpu",
            "learning_rate": 0.001,
            "feature_width": 8,
        }
        values.update(overrides)
        return TrainingConfig.from_mapping(values)

    def write_inputs(self, root):
        (root / "manifest.json").write_text("{}", encoding="utf-8")
        (root / "folds.json").write_text("{}", encoding="utf-8")
        (root / "images").mkdir()

    def test_config_is_checked_and_rejects_invalid_bounds(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_inputs(root)
            with self.assertRaises(ValueError):
                self.make_config(root, mode="cross_validation")
            with self.assertRaises(ValueError):
                self.make_config(root, mode=[])
            with self.assertRaises(ValueError):
                self.make_config(root, overfit_limit=17)
            with self.assertRaises(ValueError):
                self.make_config(root, max_steps=0)
            with self.assertRaises(ValueError):
                self.make_config(root, epochs=True)
            with self.assertRaises(ValueError):
                self.make_config(root, device="auto")

    def test_resnet_training_config_is_explicit_and_keeps_scratch_defaults(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_inputs(root)
            scratch = self.make_config(root)
            self.assertEqual(scratch.model_kind, "scratch_study_cnn")
            self.assertEqual(scratch.model_seed, 73)
            self.assertEqual(scratch.slice_microbatch, 2)
            configured = self.make_config(
                root, model_kind="resnet18_imagenet1k_v1", model_seed=101,
                weights_path=str(root / "offline.pth"),
                weights_sha256="f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
                weights_size=46830571, slice_microbatch=2,
            )
            self.assertEqual(configured.model_kind, "resnet18_imagenet1k_v1")
            self.assertEqual(configured.model_seed, 101)
            with self.assertRaises(ValueError):
                self.make_config(root, model_kind="resnet18_imagenet1k_v1")
            with self.assertRaises(ValueError):
                self.make_config(root, model_kind="resnet18_random", weights_path=str(root / "offline.pth"))
            with self.assertRaises(ValueError):
                self.make_config(root, slice_microbatch=0)
            frozen = self.make_config(
                root, model_kind="resnet18_imagenet1k_v1", freeze_backbone=True,
                weights_path=str(root / "offline.pth"),
                weights_sha256="f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
                weights_size=46830571,
            )
            self.assertTrue(frozen.freeze_backbone)
            self.assertFalse(scratch.freeze_backbone)
            for invalid in (1, "true", None):
                with self.subTest(freeze_backbone=invalid), self.assertRaises(ValueError):
                    self.make_config(root, freeze_backbone=invalid)
            with self.assertRaises(ValueError):
                self.make_config(root, model_kind="resnet18_random", freeze_backbone=True)

    def test_frozen_resnet_runner_reports_only_trainable_head_and_actual_checkpointing(self):
        from rsna_knee.training.baseline import run_training

        class Dataset:
            def __len__(self):
                return 1
            def bounded_training_subset(self, _limit):
                return self
            def __getitem__(self, index):
                if index != 0:
                    raise IndexError
                image = torch.rand(1, 1, 32, 32)
                return [image], [torch.ones_like(image, dtype=torch.bool)], torch.tensor([0, 1] * 6, dtype=torch.float32), index

        class TinyFrozenModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model_kind = "resnet18_imagenet1k_v1"
                self.weights_sha256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
                self.weights_size = 46830571
                self.head_seed = 73
                self.freeze_backbone = True
                self.checkpoint_trunk = False
                self.activation_checkpointing = False
                self.backbone = torch.nn.Linear(1, 1)
                self.classifier = torch.nn.Linear(1, 12)
                for parameter in self.backbone.parameters():
                    parameter.requires_grad_(False)
            def forward(self, *_args):
                return self.classifier(torch.ones(1, 1))

        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_inputs(root)
            config = self.make_config(
                root, model_kind="resnet18_imagenet1k_v1", freeze_backbone=True,
                weights_path=str(root / "offline.pth"),
                weights_sha256="f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
                weights_size=46830571, model_seed=73,
            )
            report = run_training(config, execute=True, dataset_factory=lambda *_: Dataset(),
                                  model_factory=lambda _config: TinyFrozenModel())
        self.assertEqual(report["model_initialization"]["trainable_parameter_count"], 24)
        self.assertTrue(report["model_initialization"]["freeze_backbone"])
        self.assertEqual(report["model_initialization"]["batchnorm_running_statistics"], "frozen_eval_affine_frozen")
        self.assertFalse(report["model_initialization"]["activation_checkpointing"])

    def test_frozen_runner_start_event_uses_actual_official_model_parameter_counts(self):
        from rsna_knee.models.resnet18 import build_resnet18_study_model
        from rsna_knee.training.baseline import run_training

        class Dataset:
            def __len__(self):
                return 1
            def bounded_training_subset(self, _limit):
                return self
            def __getitem__(self, index):
                if index != 0:
                    raise IndexError
                image = torch.rand(1, 1, 32, 32)
                return [image], [torch.ones_like(image, dtype=torch.bool)], torch.tensor([0, 1] * 6, dtype=torch.float32), index

        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_inputs(root)
            config = self.make_config(
                root, model_kind="resnet18_imagenet1k_v1", freeze_backbone=True,
                weights_path="data/private/resnet18-comparison/weights/resnet18-f37072fd.pth",
                weights_sha256="f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
                weights_size=46830571, model_seed=42, slice_microbatch=1,
            )
            output = io.StringIO()
            with redirect_stdout(output):
                report = run_training(
                    config, execute=True, dataset_factory=lambda *_: Dataset(),
                    model_factory=lambda cfg: build_resnet18_study_model(
                        cfg.model_kind, head_seed=cfg.model_seed, weights_path=cfg.weights_path,
                        expected_weights_sha256=cfg.weights_sha256, expected_weights_size=cfg.weights_size,
                        slice_microbatch=cfg.slice_microbatch, freeze_backbone=cfg.freeze_backbone,
                    ),
                )
        start_event = json.loads(output.getvalue().splitlines()[0])
        self.assertEqual(start_event["event"], "resnet18_training_start")
        self.assertTrue(start_event["backbone_frozen"])
        self.assertEqual(start_event["trainable_parameter_count"], 6156)
        self.assertEqual(start_event["head_trainable_parameter_count"], 6156)
        self.assertFalse(start_event["activation_checkpointing"])
        self.assertEqual(start_event["batchnorm_running_statistics"], "frozen_eval_affine_frozen")
        self.assertEqual(report["model_initialization"]["trainable_parameter_count"], 6156)

    def test_runner_selects_resnet_arm_and_reports_initialization_provenance(self):
        from rsna_knee.training.baseline import run_training

        class Dataset:
            def __len__(self):
                return 1
            def bounded_training_subset(self, _limit):
                return self
            def __getitem__(self, index):
                if index != 0:
                    raise IndexError
                image = torch.rand(1, 1, 32, 32)
                mask = torch.ones_like(image, dtype=torch.bool)
                labels = torch.tensor([0, 1] * 6, dtype=torch.float32)
                return [image], [mask], labels, index

        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_inputs(root)
            config = self.make_config(
                root, model_kind="resnet18_random", model_seed=73,
                slice_microbatch=1, output_size=32, max_steps=1, overfit_limit=1,
            )
            progress = io.StringIO()
            with redirect_stdout(progress):
                report = run_training(config, execute=True, dataset_factory=lambda *_: Dataset())
            serialized = json.loads((config.output_dir / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["model_kind"], "resnet18_random")
        self.assertEqual(report["model_initialization"]["weights_sha256"], None)
        self.assertEqual(report["model_initialization"]["model_seed"], 73)
        self.assertGreater(report["model_initialization"]["trainable_parameter_count"], 11_000_000)
        self.assertEqual(serialized["model_kind"], "resnet18_random")
        self.assertNotIn("weights_path", serialized["config"])
        self.assertIn("torchvision_version", report["runtime"])
        events = [json.loads(line) for line in progress.getvalue().splitlines()]
        self.assertEqual([event["event"] for event in events], ["resnet18_training_start", "resnet18_training_progress"])
        self.assertEqual(events[0]["model_kind"], "resnet18_random")
        self.assertEqual(events[0]["trainable_parameter_count"], report["model_initialization"]["trainable_parameter_count"])
        self.assertFalse(events[0]["backbone_frozen"])
        self.assertEqual(events[1]["model_kind"], "resnet18_random")
        self.assertEqual(events[1]["epoch"], 1)
        self.assertEqual(events[1]["steps_completed"], 1)
        self.assertEqual(events[1]["planned_steps"], 1)
        self.assertGreaterEqual(events[1]["online_optimization_step_loss_mean"], 0.0)
        self.assertEqual(events[1]["peak_cuda_memory_bytes"], None)
        self.assertNotIn("labels", progress.getvalue())
        self.assertNotIn("probabilities", progress.getvalue())
        self.assertNotIn("row_index", progress.getvalue())

    def test_overfit_sanity_protocol_bounds_allow_30_epochs_and_240_steps(self):
        from rsna_knee.training.baseline import TrainingConfig

        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_inputs(root)
            config = self.make_config(root, epochs=30, max_steps=240, overfit_limit=8)
            self.assertEqual(config.epochs, 30)
            self.assertEqual(config.max_steps, 240)
            self.assertEqual(config.overfit_limit, 8)

            defaults = self.make_config(root)
            self.assertEqual(defaults.epochs, 1)
            self.assertEqual(defaults.max_steps, 2)
            configured_defaults = TrainingConfig.from_mapping({
                "mode": "overfit",
                "manifest_path": str(root / "manifest.json"),
                "folds_path": str(root / "folds.json"),
                "image_root": str(root / "images"),
                "output_dir": str(root / "default-output"),
            })
            self.assertEqual(configured_defaults.epochs, 1)
            self.assertEqual(configured_defaults.max_steps, 20)
            self.assertEqual(configured_defaults.overfit_limit, 8)
            for invalid in (
                {"epochs": 31, "max_steps": 240},
                {"epochs": 30, "max_steps": 241},
            ):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    self.make_config(root, **invalid)
    def test_run_revalidates_dataclass_replacements_before_factory_or_output(self):
        from rsna_knee.training.baseline import TrainingConfig, TrainingRunError, run_training

        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_inputs(root)
            config = self.make_config(root)
            factory_calls = []
            invalid_configs = (
                replace(config, epochs=3, overfit_limit=17, max_steps=241),
                replace(config, epochs=True),
                replace(config, manifest_path=17),
            )
            for invalid in invalid_configs:
                with self.subTest(config=invalid):
                    with self.assertRaises(TrainingRunError):
                        run_training(invalid, execute=True,
                                     dataset_factory=lambda *args: factory_calls.append(args))
                    self.assertEqual(factory_calls, [])
                    self.assertFalse((root / "new-output").exists())

    def test_cli_preserves_safe_runtime_reason_code_for_unavailable_gpu(self):
        from scripts.train_baseline import main

        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_inputs(root)
            config = self.make_config(root, device="cuda")
            config_path = root / "config.json"
            config_path.write_text(json.dumps({
                "mode": config.mode,
                "manifest_path": str(config.manifest_path),
                "folds_path": str(config.folds_path),
                "image_root": str(config.image_root),
                "output_dir": str(config.output_dir),
                "fold": config.fold,
                "seed": config.seed,
                "epochs": config.epochs,
                "max_steps": config.max_steps,
                "overfit_limit": config.overfit_limit,
                "device": config.device,
                "feature_width": config.feature_width,
            }), encoding="utf-8")
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                exit_code = main(["--config", str(config_path), "--execute"])
            self.assertEqual(exit_code, 2)
            self.assertIn("cuda_unavailable", stderr.getvalue())
            self.assertNotIn(str(root), stderr.getvalue())
            self.assertFalse(Path(config.output_dir).exists())


    def test_unexpected_training_failure_has_bounded_safe_diagnostics(self):
        from rsna_knee.training.baseline import TrainingRunError, run_training

        class Dataset:
            def __len__(self):
                return 1
            def bounded_training_subset(self, _limit):
                return self

        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_inputs(root)
            config = self.make_config(root)
            secret = f"{root}/private/study-identifier"
            def fail_model(_config):
                try:
                    raise OSError(f"decoder path {secret}")
                except OSError as cause:
                    raise RuntimeError(f"CUDA operation failed at {secret}") from cause

            with self.assertRaises(TrainingRunError) as caught:
                run_training(config, execute=True,
                             dataset_factory=lambda *_: Dataset(),
                             model_factory=fail_model)

        error = caught.exception
        self.assertEqual(error.reason_code, "training_failed")
        self.assertEqual(error.diagnostics["stage"], "model_initialization")
        self.assertEqual(error.diagnostics["exception_family"], "runtime")
        self.assertEqual(error.diagnostics["exception_chain"], ["runtime", "operating_system"])
        self.assertEqual(set(error.diagnostics), {"stage", "exception_family", "exception_chain", "source"})
        self.assertNotIn(secret, repr(error.diagnostics))
        source = error.diagnostics["source"]
        self.assertEqual(source["module"], "training.baseline")
        self.assertEqual(source["function"], "run_training")
        self.assertIsInstance(source["line"], int)
        self.assertNotIn(secret, str(error))

    def test_actual_torch_cuda_oom_type_is_classified_as_memory_without_gpu_work(self):
        from rsna_knee.training.baseline import _safe_exception_family, _safe_failure_diagnostics

        error = torch.cuda.OutOfMemoryError("sensitive sentinel")
        self.assertEqual(_safe_exception_family(error), "memory")
        diagnostics = _safe_failure_diagnostics(error, "training")
        self.assertEqual(diagnostics["exception_family"], "memory")
        self.assertEqual(diagnostics["exception_chain"], ["memory"])
        self.assertNotIn("sensitive sentinel", repr(diagnostics))

        spoofed_type = type("OutOfMemoryError", (RuntimeError,), {"__module__": "torch.cuda"})
        self.assertEqual(_safe_exception_family(spoofed_type("spoof sentinel")), "runtime")

    def test_safe_source_location_prefers_deepest_trusted_internal_frame(self):
        from rsna_knee.training.baseline import _safe_failure_diagnostics

        namespace = {}
        source = "def run_training():\n    _sample_loss()\ndef _sample_loss():\n    raise RuntimeError('private sentinel')\n"
        exec(compile(source, "C:/workspace/src/rsna_knee/training/baseline.py", "exec"), namespace)
        try:
            namespace["run_training"]()
        except RuntimeError as error:
            diagnostics = _safe_failure_diagnostics(error, "training")
        self.assertEqual(diagnostics["source"]["function"], "_sample_loss")
        self.assertNotIn("private sentinel", repr(diagnostics))

    def test_cli_emits_only_allowlisted_training_diagnostics(self):
        from scripts.train_baseline import main
        from rsna_knee.training.baseline import TrainingRunError

        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_inputs(root)
            config_path = root / "config.json"
            config_path.write_text("{}", encoding="utf-8")
            safe = {
                "stage": "training", "exception_family": "runtime",
                "exception_chain": ["runtime", "operating_system"],
                "source": {"module": "training.baseline", "function": "run_training", "line": 12},
            }
            stderr = io.StringIO()
            with mock.patch("scripts.train_baseline.TrainingConfig.from_mapping", return_value=object()), \
                 mock.patch("scripts.train_baseline.run_training", side_effect=TrainingRunError(
                     "training_failed", diagnostics=safe)), \
                 redirect_stderr(stderr):
                exit_code = main(["--config", str(config_path), "--execute"])
        self.assertEqual(exit_code, 2)
        event = json.loads(stderr.getvalue())
        self.assertEqual(event["reason_code"], "training_failed")
        self.assertEqual(event["diagnostics"], safe)
        self.assertNotIn(str(root), stderr.getvalue())

    def test_cuda_device_requires_available_cuda_and_returns_cuda_device(self):
        from rsna_knee.training.baseline import TrainingRunError, _check_device

        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(TrainingRunError) as caught:
                _check_device("cuda")
        self.assertEqual(caught.exception.reason_code, "cuda_unavailable")
        with mock.patch("torch.cuda.is_available", side_effect=RuntimeError("device probe failed")):
            with self.assertRaises(TrainingRunError) as probe_error:
                _check_device("cuda")
        self.assertEqual(probe_error.exception.reason_code, "cuda_unavailable")
        with mock.patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(_check_device("cuda"), torch.device("cuda"))
        self.assertEqual(_check_device("cpu"), torch.device("cpu"))

    def test_cuda_peak_memory_resets_before_work_and_reads_after_synchronizing(self):
        from rsna_knee.training.baseline import _reset_peak_cuda_memory, _runtime_metadata

        calls = []
        device = torch.device("cuda")
        config = self.make_config(Path("."), device="cuda")
        with mock.patch("torch.cuda.synchronize", side_effect=lambda d: calls.append(("sync", d))), \
             mock.patch("torch.cuda.reset_peak_memory_stats", side_effect=lambda d: calls.append(("reset", d))), \
             mock.patch("torch.cuda.max_memory_allocated", side_effect=lambda d: calls.append(("peak", d)) or 123456), \
             mock.patch("torch.cuda.is_available", return_value=True):
            _reset_peak_cuda_memory(device)
            metadata = _runtime_metadata(config, 1.25, device=device)
        self.assertEqual(calls, [("sync", device), ("reset", device), ("sync", device), ("peak", device)])
        self.assertEqual(metadata["peak_cuda_memory_bytes"], 123456)
        self.assertTrue(metadata["cuda_available"])

    def test_cpu_runtime_metadata_keeps_cuda_peak_memory_empty(self):
        from rsna_knee.training.baseline import _runtime_metadata

        config = self.make_config(Path("."), device="cpu")
        metadata = _runtime_metadata(config, 0.25, device=torch.device("cpu"))
        self.assertIsNone(metadata["peak_cuda_memory_bytes"])

    def test_overfit_uses_only_bounded_train_subset_and_trains_one_study_per_step(self):
        from rsna_knee.training.baseline import run_training

        labels = torch.tensor([i % 2 for i in range(12)], dtype=torch.float32)
        class Dataset:
            partition = "train"
            def __len__(self): return 10
            def bounded_training_subset(self, limit):
                self.limit = limit
                return [([torch.ones(1, 1, 8, 8)], [torch.ones(1, 1, 8, 8, dtype=torch.bool)], labels, i)
                        for i in range(limit)]
        dataset = Dataset()
        made = []
        def factory(config, partition):
            made.append(partition)
            return dataset
        with TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            self.write_inputs(root)
            config = self.make_config(root, max_steps=2, overfit_limit=2)
            output = io.StringIO()
            with redirect_stdout(output):
                result = run_training(config, execute=True, dataset_factory=factory)
            self.assertEqual(made, ["train"])
            self.assertEqual(output.getvalue(), "")
            self.assertEqual(dataset.limit, 2)
            self.assertEqual(result["steps_completed"], 2)
            self.assertEqual(len(result["train_loss_by_step"]), 2)
            self.assertNotIn("validation", result)
            self.assertNotIn("macro_auc", result)
            self.assertLessEqual(result["train_loss_end"], result["train_loss_start"])
            self.assertGreaterEqual(result["runtime"]["elapsed_seconds"], 0.0)
            self.assertIsNone(result["runtime"]["peak_cuda_memory_bytes"])
            self.assertTrue((root / "new-output" / "checkpoint.pt").is_file())
            self.assertTrue((root / "new-output" / "report.json").is_file())

    def test_fold_mode_evaluates_all_validation_rows_and_writes_12_probabilities(self):
        from rsna_knee.training.baseline import run_training

        labels = [torch.tensor([i % 2 for i in range(12)], dtype=torch.float32) for _ in range(4)]
        class Dataset:
            def __init__(self, partition): self.partition = partition
            def __len__(self): return 2
            def __getitem__(self, index):
                return [torch.full((1, 1, 8, 8), float(index + 1))], [torch.ones((1, 1, 8, 8), dtype=torch.bool)], labels[index], index
        made = []
        def factory(config, partition):
            made.append(partition)
            return Dataset(partition)
        with TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            self.write_inputs(root)
            config = self.make_config(root, mode="fold", epochs=1, max_steps=None, overfit_limit=None)
            result = run_training(config, execute=True, dataset_factory=factory)
            self.assertEqual(made, ["train", "validation"])
            self.assertEqual(result["validation_study_count"], 2)
            predictions = json.loads((root / "new-output" / "validation_predictions.json").read_text())
            self.assertEqual([row["row_index"] for row in predictions["rows"]], [0, 1])
            self.assertEqual(len(predictions["rows"][0]["probabilities"]), 12)
            self.assertTrue(all(0 <= value <= 1 for value in predictions["rows"][0]["probabilities"]))
            self.assertEqual(predictions["target_names"], list(LABEL_COLUMNS))
            self.assertEqual(predictions["input_sha256"].keys(), {"manifest", "folds"})
            self.assertIn("runtime", predictions)
            self.assertEqual(len(result["per_target_auc"]), 12)
            self.assertIn("not clinical", result["interpretation_limit"].lower())

    def test_rejects_existing_output_and_unavailable_cuda_without_touching_it(self):
        from rsna_knee.training.baseline import run_training

        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_inputs(root)
            output = root / "new-output"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("keep")
            with self.assertRaises(ValueError):
                run_training(self.make_config(root), execute=True)
            self.assertEqual(sentinel.read_text(), "keep")
            config = self.make_config(root, output_dir=str(root / "cuda-output"), device="cuda")
            with mock.patch("torch.cuda.is_available", return_value=False):
                with self.assertRaises(ValueError):
                    run_training(config, execute=True)
            self.assertFalse((root / "cuda-output").exists())

    def test_plan_does_not_decode_or_create_output(self):
        from rsna_knee.training.baseline import run_training

        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_inputs(root)
            def factory(config, partition):
                class PlanDataset:
                    def __len__(self): return 4
                    def bounded_training_subset(self, limit): return list(range(limit))
                return PlanDataset()
            plan = run_training(self.make_config(root), execute=False, dataset_factory=factory)
            self.assertEqual(plan["status"], "plan_only")
            self.assertFalse((root / "new-output").exists())

    def test_epoch_loss_telemetry_aggregates_optimization_steps_for_both_modes(self):
        from rsna_knee.training.baseline import run_training

        labels = torch.tensor([i % 2 for i in range(12)], dtype=torch.float32)
        class Dataset:
            def __init__(self, partition): self.partition = partition
            def __len__(self): return 2
            def __getitem__(self, index):
                return ([torch.full((1, 1, 8, 8), float(index + 1))],
                        [torch.ones((1, 1, 8, 8), dtype=torch.bool)], labels, index)
            def bounded_training_subset(self, limit): return [self[i] for i in range(limit)]

        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_inputs(root)
            overfit = run_training(
                self.make_config(root, epochs=2, max_steps=4, output_dir=str(root / "overfit-output")),
                execute=True,
                dataset_factory=lambda _config, partition: Dataset(partition),
            )
            fold = run_training(
                self.make_config(root, mode="fold", epochs=2, max_steps=None, overfit_limit=None,
                                 output_dir=str(root / "fold-output")),
                execute=True,
                dataset_factory=lambda _config, partition: Dataset(partition),
            )

            expected_keys = {
                "epoch", "planned_study_count", "completed_study_count",
                "completed_step_count", "status", "optimization_step_loss_mean",
            }
            for report in (overfit, fold):
                telemetry = report["training_loss_by_epoch"]
                self.assertEqual(len(telemetry), 2)
                self.assertEqual([row["epoch"] for row in telemetry], [1, 2])
                self.assertEqual([row["planned_study_count"] for row in telemetry], [2, 2])
                self.assertEqual([row["completed_study_count"] for row in telemetry], [2, 2])
                self.assertEqual([row["completed_step_count"] for row in telemetry], [2, 2])
                self.assertEqual([row["status"] for row in telemetry], ["complete", "complete"])
                self.assertTrue(all(set(row) == expected_keys for row in telemetry))
                self.assertTrue(all(np.isfinite(row["optimization_step_loss_mean"]) for row in telemetry))

            self.assertEqual(overfit["steps_completed"], 4)
            self.assertEqual(len(overfit["train_loss_by_step"]), 4)
            self.assertIn("train_loss_start", overfit)
            self.assertIn("train_loss_end", overfit)
            self.assertEqual(fold["steps_completed"], 4)
            self.assertNotIn("train_loss_by_step", fold)
            self.assertNotIn("train_loss_start", fold)
            self.assertNotIn("train_loss_end", fold)
            serialized = (root / "fold-output" / "report.json").read_text(encoding="utf-8")
            self.assertNotIn("study-identifier", serialized)
            self.assertNotIn("manifest.json", serialized)
            self.assertNotIn("optimization_step_losses", serialized)

    def test_epoch_loss_telemetry_marks_a_capped_partial_epoch(self):
        from rsna_knee.training.baseline import run_training

        labels = torch.tensor([i % 2 for i in range(12)], dtype=torch.float32)
        class Dataset:
            def __len__(self): return 2
            def __getitem__(self, index):
                return ([torch.ones(1, 1, 8, 8)], [torch.ones(1, 1, 8, 8, dtype=torch.bool)], labels, index)
            def bounded_training_subset(self, limit): return [self[i] for i in range(limit)]

        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_inputs(root)
            report = run_training(
                self.make_config(root, epochs=3, max_steps=3), execute=True,
                dataset_factory=lambda *_: Dataset(),
            )
        telemetry = report["training_loss_by_epoch"]
        self.assertEqual(len(telemetry), 2)
        self.assertEqual([row["completed_study_count"] for row in telemetry], [2, 1])
        self.assertEqual([row["completed_step_count"] for row in telemetry], [2, 1])
        self.assertEqual([row["status"] for row in telemetry], ["complete", "partial"])
        self.assertTrue(all(np.isfinite(row["optimization_step_loss_mean"]) for row in telemetry))
        self.assertEqual(report["steps_completed"], 3)
    def test_overfit_final_subset_diagnostics_are_computed_during_existing_final_pass(self):
        from rsna_knee.training.baseline import run_training

        class Dataset:
            def __init__(self):
                self.read_count = 0
                self.labels = []
                for index in range(8):
                    row = [0] * 12
                    row[0] = index % 2
                    row[1] = 1
                    self.labels.append(torch.tensor(row, dtype=torch.float32))
            def __len__(self): return 8
            def __getitem__(self, index):
                if not 0 <= index < len(self):
                    raise IndexError
                self.read_count += 1
                return ([torch.ones(1, 1, 8, 8)], [torch.ones(1, 1, 8, 8, dtype=torch.bool)],
                        self.labels[index], index)
            def bounded_training_subset(self, limit): return self

        class SyntheticModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # The same target logits are emitted for every study, but vary by target.
                self.logits = torch.nn.Parameter(torch.linspace(-1.0, 1.0, 12))
            def forward(self, images, masks): return self.logits.reshape(1, 12)

        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_inputs(root)
            dataset = Dataset()
            report = run_training(
                self.make_config(root, epochs=1, max_steps=2, overfit_limit=8), execute=True,
                dataset_factory=lambda *_: dataset,
                model_factory=lambda _config: SyntheticModel(),
            )
        diagnostics = report["final_subset_diagnostics"]
        self.assertEqual(diagnostics["training_study_count"], 8)
        self.assertEqual(dataset.read_count, 18)  # 8 start + 2 steps + 8 final; no extra evaluation pass.
        target_records = diagnostics["per_target"]
        self.assertEqual([item["target"] for item in target_records], list(LABEL_COLUMNS))
        self.assertTrue(all(np.isfinite(item["probability_std"]) for item in target_records))
        self.assertTrue(all(item["probability_std"] == 0.0 for item in target_records))
        self.assertTrue(all(0.0 <= item["probability_min"] <= item["probability_mean"] <= item["probability_max"] <= 1.0 for item in target_records))
        self.assertGreater(target_records[-1]["probability_mean"], target_records[0]["probability_mean"])
        self.assertEqual(target_records[0]["positive_count"], 4)
        self.assertEqual(target_records[0]["negative_count"], 4)
        self.assertEqual(target_records[0]["auc"], 0.5)
        self.assertEqual(target_records[1]["positive_count"], 8)
        self.assertEqual(target_records[1]["negative_count"], 0)
        self.assertIsNone(target_records[1]["auc"])
        self.assertNotIn("probabilities", diagnostics)
        self.assertNotIn("labels", diagnostics)
        self.assertNotIn("row_index", diagnostics)
        self.assertIn("training-subset diagnostic", diagnostics["interpretation_limit"].lower())
    def test_deterministic_initialization_and_checkpoint_is_state_dict(self):
        from rsna_knee.training.baseline import run_training

        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_inputs(root)
            config = self.make_config(root, max_steps=1)
            class OverfitRows:
                def __len__(self): return 1
                def __getitem__(self, index):
                    return ([torch.ones(1, 1, 8, 8)], [torch.ones(1, 1, 8, 8, dtype=torch.bool)],
                            torch.tensor([0, 1] * 6, dtype=torch.float32), index)
                def bounded_training_subset(self, limit):
                    return [self[0]]
            # The configured seed is included in each private report; separate identical runs
            # have deterministic sample/model behavior even though output locations differ.
            for name in ("out-a", "out-b"):
                cfg = self.make_config(root, output_dir=str(root / name), max_steps=1)
                run_training(cfg, execute=True, dataset_factory=lambda c, p: OverfitRows())
            first = torch.load(root / "out-a" / "checkpoint.pt", map_location="cpu", weights_only=True)
            second = torch.load(root / "out-b" / "checkpoint.pt", map_location="cpu", weights_only=True)
            self.assertEqual(set(first), set(second))
            self.assertTrue(all(torch.equal(first[key], second[key]) for key in first))

    def test_cli_defaults_to_plan_and_notebook_is_output_free_and_disabled(self):
        import ast

        from scripts.train_baseline import build_parser

        args = build_parser().parse_args([])
        self.assertFalse(args.execute)
        notebook = json.loads(Path("notebooks/baseline_training.ipynb").read_text(encoding="utf-8"))
        self.assertTrue(all(cell.get("outputs", []) == [] for cell in notebook["cells"]))
        source = "\n".join(notebook["cells"][1]["source"])
        ast.parse(source)
        self.assertIn("RUN_TRAINING = False", source)
        self.assertIn("private_input_update_pending", source)


if __name__ == "__main__":
    unittest.main()
