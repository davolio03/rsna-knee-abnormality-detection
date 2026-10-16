import json
import io
import unittest
from contextlib import redirect_stderr
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

    def test_run_revalidates_dataclass_replacements_before_factory_or_output(self):
        from rsna_knee.training.baseline import TrainingConfig, TrainingRunError, run_training

        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_inputs(root)
            config = self.make_config(root)
            factory_calls = []
            invalid_configs = (
                replace(config, epochs=3, overfit_limit=17, max_steps=41),
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
            result = run_training(config, execute=True, dataset_factory=factory)
            self.assertEqual(made, ["train"])
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
