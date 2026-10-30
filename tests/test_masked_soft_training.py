import unittest
import json
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock
from contextlib import redirect_stderr, redirect_stdout

try:
    import numpy as np
    import torch
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("optional NumPy/PyTorch training dependencies are not installed") from exc


class MaskedSoftObjectiveTests(unittest.TestCase):
    def test_loss_uses_fractional_targets_and_observed_cell_denominator(self):
        from rsna_knee.training.masked_soft import masked_soft_bce

        logits = torch.tensor([[0.0, 1.0, -1.0]], requires_grad=True)
        targets = torch.tensor([[0.25, 0.75, 0.5]])
        observed = torch.tensor([[True, True, False]])
        actual = masked_soft_bce(logits, targets, observed)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[:, :2], targets[:, :2], reduction="none"
        ).mean()
        self.assertAlmostEqual(actual.item(), expected.item(), places=7)
        actual.backward()
        self.assertAlmostEqual(logits.grad[0, 0].item(), 0.125, places=7)
        self.assertAlmostEqual(logits.grad[0, 1].item(), -0.0094707012, places=7)
        self.assertEqual(logits.grad[0, 2].item(), 0.0)

    def test_symmetric_class_weights_match_soft_label_formula(self):
        from rsna_knee.training.masked_soft import masked_soft_bce

        logits = torch.tensor([[0.4, -0.7]])
        targets = torch.tensor([[0.25, 0.8]])
        mask = torch.ones_like(targets, dtype=torch.bool)
        weights = {"positive": [2.0, 1.5], "negative": [1.25, 3.0]}
        actual = masked_soft_bce(logits, targets, mask, class_weights=weights)
        expected_cells = (
            2.0 * 0.25 * torch.nn.functional.softplus(-logits[0, 0])
            + 1.25 * 0.75 * torch.nn.functional.softplus(logits[0, 0])
            + 1.5 * 0.8 * torch.nn.functional.softplus(-logits[0, 1])
            + 3.0 * 0.2 * torch.nn.functional.softplus(logits[0, 1])
        )
        self.assertAlmostEqual(actual.item(), (expected_cells / 2).item(), places=7)

    def test_metrics_pool_observed_cells_not_per_row_means(self):
        from rsna_knee.training.masked_soft import MaskedSoftMetricAccumulator

        accumulator = MaskedSoftMetricAccumulator(target_names=("a", "b"))
        accumulator.update(torch.tensor([[0.0, 0.0]]), torch.tensor([[0.25, 0.5]]),
                           torch.tensor([[True, False]]))
        accumulator.update(torch.tensor([[0.0, 1.0]]), torch.tensor([[0.75, 1.0]]),
                           torch.tensor([[True, True]]))
        report = accumulator.compute()
        self.assertEqual(report["observed_cell_count"], 3)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor([0.0, 0.0, 1.0]), torch.tensor([0.25, 0.75, 1.0]), reduction="mean"
        ).item()
        self.assertAlmostEqual(report["masked_bce"], expected, places=7)
        self.assertEqual(report["per_target"][0]["observed_count"], 2)
        self.assertEqual(report["per_target"][1]["observed_count"], 1)

    def test_zero_observation_mask_and_invalid_values_fail_closed(self):
        from rsna_knee.training.masked_soft import masked_soft_bce

        logits = torch.zeros((1, 2))
        targets = torch.full((1, 2), 0.5)
        with self.assertRaisesRegex(ValueError, "no_observed_targets"):
            masked_soft_bce(logits, targets, torch.zeros((1, 2), dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, "target_out_of_range"):
            masked_soft_bce(logits, torch.tensor([[1.1, 0.2]]), torch.ones((1, 2), dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, "mask_invalid"):
            masked_soft_bce(logits, targets, torch.ones((1, 2)))

    def test_extreme_observed_weighted_logit_fails_before_nonfinite_loss_escapes(self):
        from rsna_knee.training.masked_soft import masked_soft_bce

        with self.assertRaisesRegex(ValueError, "loss_nonfinite"):
            masked_soft_bce(
                torch.tensor([[2.0e38]]), torch.tensor([[0.25]]), torch.tensor([[True]]),
                class_weights={"positive": [3.0], "negative": [3.0]},
            )

    def test_extreme_masked_cell_cannot_poison_observed_loss_or_gradient(self):
        from rsna_knee.training.masked_soft import masked_soft_bce

        logits = torch.tensor([[0.0, 3.4e38]], requires_grad=True)
        loss = masked_soft_bce(
            logits, torch.tensor([[0.25, 0.25]]), torch.tensor([[True, False]]),
            class_weights={"positive": [3.0, 3.0], "negative": [3.0, 3.0]},
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertEqual(logits.grad[0, 1].item(), 0.0)

    def test_metrics_mark_undefined_target_explicitly(self):
        from rsna_knee.training.masked_soft import MaskedSoftMetricAccumulator

        accumulator = MaskedSoftMetricAccumulator(target_names=("a", "b", "c"))
        accumulator.update(torch.zeros((1, 3)), torch.tensor([[0.25, 0.5, 0.5]]),
                           torch.tensor([[True, False, False]]))
        report = accumulator.compute()
        self.assertIsNone(report["per_target"][1]["masked_bce"])
        self.assertIsNone(report["per_target"][2]["brier"])


class MaskedSoftRunnerTests(unittest.TestCase):
    def config_values(self, root, **overrides):
        values = {
            "manifest_path": "data/private/full-cohort-prefilter/run01/full-cohort-soft-label-manifest.json",
            "folds_path": "data/private/full-cohort-prefilter/run01/full-cohort-grouped-folds.json",
            "image_root": str(root / "images"), "fold": 0, "epochs": 1, "max_steps": 1,
            "output_size": 8, "max_slices": 1, "device": "cpu", "learning_rate": 0.001,
            "balance_mode": "unweighted",
        }
        values.update(overrides)
        return values

    def test_default_path_only_plans_and_does_not_make_output_or_model(self):
        from rsna_knee.training.masked_soft import MaskedSoftConfig, run_masked_soft_training

        with TemporaryDirectory() as temp:
            root = Path(temp)
            config = MaskedSoftConfig.from_mapping(self.config_values(root, output_dir=str(root / "new")))
            result = run_masked_soft_training(
                config, model_factory=lambda *_: self.fail("plan must not construct model"),
                dataset_factory=lambda *_: self.fail("plan must not inspect images"),
            )
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["reason_code"], "offline_weights_path_required")
            self.assertFalse((root / "new").exists())
            self.assertEqual(result["training_study_count"], 2850)

    def test_cli_defaults_to_pinned_plan_without_output_or_images(self):
        from scripts.train_masked_soft import main

        with TemporaryDirectory() as temp:
            config_path = Path(temp) / "config.json"
            output = Path(temp) / "not-created"
            config_path.write_text(json.dumps(self.config_values(Path(temp), output_dir=str(output))),
                                   encoding="utf-8")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(["--config", str(config_path)])
            result = json.loads(stdout.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["reason_code"], "offline_weights_path_required")
            self.assertFalse(output.exists())

    def test_cli_sanitizes_unexpected_runtime_exception(self):
        from scripts.train_masked_soft import main

        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch("rsna_knee.training.masked_soft.run_masked_soft_training",
                        side_effect=RuntimeError("SYNTHETIC_PRIVATE_MARKER")):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main([])
        self.assertEqual(code, 2)
        payload = stdout.getvalue() + stderr.getvalue()
        self.assertNotIn("SYNTHETIC_PRIVATE_MARKER", payload)
        self.assertNotIn("Traceback", payload)

    def test_default_model_factory_uses_frozen_imagenet_v1_and_offline_pin(self):
        from rsna_knee.training.masked_soft import MaskedSoftConfig, _build_default_model

        with TemporaryDirectory() as temp:
            weight_path = Path(temp) / "offline.pth"
            config = MaskedSoftConfig.from_mapping(self.config_values(
                Path(temp), weights_path=str(weight_path),
                weights_sha256="f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
                weights_size=46830571,
            ))
            with mock.patch("rsna_knee.models.resnet18.build_resnet18_study_model") as builder:
                builder.return_value = object()
                with mock.patch("rsna_knee.training.masked_soft._validate_offline_weights", return_value={
                    "weights_sha256": "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
                    "weights_size": 46830571,
                }):
                    _build_default_model(config)
            builder.assert_called_once_with(
                "resnet18_imagenet1k_v1", head_seed=config.seed,
                weights_path=weight_path,
                expected_weights_sha256="f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
                expected_weights_size=46830571, slice_microbatch=2,
                checkpoint_trunk=True, freeze_backbone=True,
            )

    def test_nonfinite_weighted_loss_fails_before_optimizer_step(self):
        from rsna_knee.training.masked_soft import MaskedSoftConfig, run_masked_soft_training

        class ExtremeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logit = torch.nn.Parameter(torch.tensor(2.0e38))

            def forward(self, _images, _masks):
                return self.logit.expand(1, 12)

        sample = ([torch.zeros((1, 1, 8, 8))], [torch.ones((1, 1, 8, 8), dtype=torch.bool)],
                  torch.full((12,), 0.25), torch.ones((12,), dtype=torch.bool), 0)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            config = MaskedSoftConfig.from_mapping(self.config_values(
                root, output_dir=str(root / "runs" / "overflow"), output_size=8,
                max_slices=1, balance_mode="training_only_class_weights",
            ))
            model = ExtremeModel()
            initial_value = model.logit.detach().clone()
            with mock.patch("rsna_knee.training.masked_soft.PRIVATE_RUN_ROOT", root / "runs"):
                with mock.patch.object(torch.optim.Adam, "step", autospec=True) as step:
                    with self.assertRaisesRegex(ValueError, "loss_nonfinite"):
                        run_masked_soft_training(
                            config, execute=True,
                            dataset_factory=lambda _cfg, _part: [sample],
                            model_factory=lambda _cfg: model,
                        )
                    step.assert_not_called()
            self.assertTrue(torch.equal(model.logit.detach(), initial_value))
            self.assertFalse(config.output_dir.exists())

    def test_offline_weight_missing_or_mismatched_fails_before_factories(self):
        from rsna_knee.training.masked_soft import MaskedSoftConfig, run_masked_soft_training

        with TemporaryDirectory() as temp:
            root = Path(temp)
            missing = MaskedSoftConfig.from_mapping(self.config_values(
                root, output_dir=str(root / "runs" / "missing"),
                weights_path=str(root / "missing.pth"),
                weights_sha256="f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
                weights_size=46830571,
            ))
            with self.assertRaisesRegex(ValueError, "offline_weights_unavailable"):
                run_masked_soft_training(
                    missing, execute=True,
                    dataset_factory=lambda *_: self.fail("missing weights reached dataset"),
                    model_factory=lambda *_: self.fail("missing weights reached model"),
                )
            wrong_path = root / "wrong.pth"
            wrong_path.write_bytes(b"not the trusted weights")
            mismatched = MaskedSoftConfig.from_mapping(self.config_values(
                root, output_dir=str(root / "runs" / "mismatch"),
                weights_path=str(wrong_path),
                weights_sha256="f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
                weights_size=46830571,
            ))
            with self.assertRaisesRegex(ValueError, "offline_weights_pin_mismatch"):
                run_masked_soft_training(
                    mismatched, execute=True,
                    dataset_factory=lambda *_: self.fail("bad weights reached dataset"),
                    model_factory=lambda *_: self.fail("bad weights reached model"),
                )

    def test_execution_rejects_tampered_folds_before_model_or_dataset_factory(self):
        from rsna_knee.data.masked_soft_dataset import SoftDatasetError
        from rsna_knee.training.masked_soft import MaskedSoftConfig, run_masked_soft_training

        with TemporaryDirectory() as temp:
            root = Path(temp)
            original = Path("data/private/full-cohort-prefilter/run01/full-cohort-grouped-folds.json")
            changed = root / "folds.json"
            changed.write_bytes(original.read_bytes() + b" ")
            config = MaskedSoftConfig.from_mapping(self.config_values(
                root, folds_path=str(changed), output_dir=str(root / "runs" / "unused"),
            ))
            with self.assertRaisesRegex(SoftDatasetError, "folds_pin_mismatch"):
                run_masked_soft_training(
                    config, execute=True,
                    dataset_factory=lambda *_: self.fail("tampered input reached image factory"),
                    model_factory=lambda *_: self.fail("tampered input reached model factory"),
                )

    def test_injected_cpu_training_uses_real_soft_dataset_contract_and_writes_private_report(self):
        from rsna_knee.data.masked_soft_dataset import FullCohortSoftDataset, load_verified_soft_artifacts
        from rsna_knee.training.masked_soft import MaskedSoftConfig, run_masked_soft_training

        manifest = Path("data/private/full-cohort-prefilter/run01/full-cohort-soft-label-manifest.json")
        folds = Path("data/private/full-cohort-prefilter/run01/full-cohort-grouped-folds.json")
        records, assignments, _ = load_verified_soft_artifacts(manifest, folds)
        with TemporaryDirectory(dir=Path.cwd()) as temp:
            root = Path(temp)
            image_root = root / "images"
            ids = [next(row["study_id"] for row in records if assignments[row["study_id"]] != 0),
                   next(row["study_id"] for row in records if assignments[row["study_id"]] == 0)]
            for study_id in ids:
                row = next(value for value in records if value["study_id"] == study_id)
                for series in row["series"]:
                    (image_root / study_id / series["series_id"]).mkdir(parents=True)

            def preprocess(_path, *, max_slices, output_size):
                return SimpleNamespace(images=np.full((1, 1, output_size, output_size), 0.3, np.float32),
                                       valid_mask=np.ones((1, 1, output_size, output_size), dtype=bool))

            def datasets(config, partition):
                dataset = FullCohortSoftDataset(
                    config.manifest_path, config.folds_path, image_root,
                    partition=partition, fold=config.fold, output_size=config.output_size,
                    max_slices=config.max_slices, preprocessor=preprocess,
                )
                return torch.utils.data.Subset(dataset, [0])

            sample_config = __import__("rsna_knee.training.masked_soft", fromlist=["MaskedSoftConfig"]).MaskedSoftConfig.from_mapping(
                self.config_values(root, image_root=str(image_root), output_size=8, max_slices=1)
            )
            datasets(sample_config, "train")[0]
            datasets(sample_config, "validation")[0]

            class TinyModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.scale = torch.nn.Parameter(torch.tensor(0.1))

                def forward(self, images, masks):
                    value = images[0][masks[0]].mean() * self.scale
                    return value.reshape(1, 1).expand(1, 12)

            for mode in ("unweighted", "training_only_class_weights"):
                output = root / "runs" / f"run-{mode}"
                config = MaskedSoftConfig.from_mapping(self.config_values(
                    root, image_root=str(image_root), output_dir=str(output), balance_mode=mode,
                ))
                with mock.patch("rsna_knee.training.masked_soft.PRIVATE_RUN_ROOT", root / "runs"):
                    with mock.patch("torch.optim.Adam", wraps=torch.optim.Adam) as adam:
                        result = run_masked_soft_training(
                            config, execute=True, dataset_factory=datasets,
                            model_factory=lambda _config: TinyModel(),
                        )
                    adam.assert_called_once()
                self.assertEqual(result["status"], "complete")
                validation = result["epochs"][0]["validation"]
                self.assertEqual(validation["observed_cell_count"], sum(
                    entry["observed_count"] for entry in validation["per_target"]
                ))
                report = json.loads((output / "training-report.json").read_text(encoding="utf-8"))
                self.assertEqual(report["balance_mode"], mode)
                self.assertEqual(report["epochs"][0]["training_objective_weighting"], mode)
                report_text = (output / "training-report.json").read_text(encoding="utf-8")
                self.assertTrue(all(study_id not in report_text for study_id in ids))
                self.assertTrue((output / "model-state.pt").is_file())

    def test_zero_mask_stops_before_optimizer_update_and_cleans_only_new_run(self):
        from rsna_knee.training.masked_soft import MaskedSoftConfig, run_masked_soft_training

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(2.0))

            def forward(self, _images, _masks):
                return self.weight.expand(1, 12)

        model = TinyModel()
        sample = ([torch.zeros((1, 1, 8, 8))], [torch.ones((1, 1, 8, 8), dtype=torch.bool)],
                  torch.full((12,), 0.5), torch.zeros((12,), dtype=torch.bool), 0)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "runs" / "zero-mask"
            config = MaskedSoftConfig.from_mapping(self.config_values(
                root, output_dir=str(output), output_size=8, max_slices=1,
            ))
            with mock.patch("rsna_knee.training.masked_soft.PRIVATE_RUN_ROOT", root / "runs"):
                with self.assertRaisesRegex(ValueError, "no_observed_targets"):
                    run_masked_soft_training(
                        config, execute=True,
                        dataset_factory=lambda _cfg, _part: [sample],
                        model_factory=lambda _cfg: model,
                    )
            self.assertEqual(model.weight.item(), 2.0)
            self.assertFalse(output.exists())

    def test_interrupted_resume_matches_uninterrupted_training(self):
        from rsna_knee.training.masked_soft import MaskedSoftConfig, run_masked_soft_training

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.2))

            def forward(self, images, _masks):
                return (images[0].mean() * self.weight).reshape(1, 1).expand(1, 12)

        class RandomTinyDataset:
            def __len__(self):
                return 1
            def __getitem__(self, _index):
                image = torch.rand((1, 1, 8, 8))
                return ([image], [torch.ones((1, 1, 8, 8), dtype=torch.bool)],
                        torch.full((12,), 0.25), torch.ones((12,), dtype=torch.bool), 0)
        def datasets(_config, _partition):
            return RandomTinyDataset()
        with TemporaryDirectory(dir=Path.cwd()) as temp:
            root = Path(temp)
            base = self.config_values(root, epochs=2, max_steps=1, output_size=8, max_slices=1)
            whole_config = MaskedSoftConfig.from_mapping({**base, "output_dir": str(root / "runs" / "whole")})
            split1 = MaskedSoftConfig.from_mapping({**base, "output_dir": str(root / "runs" / "part1"),
                                                    "max_epochs_per_run": 1})
            with mock.patch("rsna_knee.training.masked_soft.PRIVATE_RUN_ROOT", root / "runs"):
                whole = run_masked_soft_training(whole_config, execute=True,
                                                 dataset_factory=datasets, model_factory=lambda _: TinyModel())
                first = run_masked_soft_training(split1, execute=True,
                                                 dataset_factory=datasets, model_factory=lambda _: TinyModel())
                resume_config = MaskedSoftConfig.from_mapping({**base,
                    "output_dir": str(root / "runs" / "part2"),
                    "resume_checkpoint": first["checkpoint_path"],
                    "resume_checkpoint_sha256": first["checkpoint_sha256"],
                    "max_epochs_per_run": 2})
                resumed = run_masked_soft_training(resume_config, execute=True,
                                                   dataset_factory=datasets, model_factory=lambda _: TinyModel())
            self.assertEqual(first["status"], "paused_checkpoint_available")
            self.assertEqual(first["completed_epochs"], 1)
            self.assertEqual(resumed["status"], "complete")
            self.assertEqual(resumed["epochs"], whole["epochs"])
            whole_state = torch.load(whole_config.output_dir / "model-state.pt", weights_only=True)
            resumed_state = torch.load(resume_config.output_dir / "model-state.pt", weights_only=True)
            self.assertTrue(torch.equal(whole_state["state_dict"]["weight"],
                                        resumed_state["state_dict"]["weight"]))
            whole_checkpoint = torch.load(whole["checkpoint_path"], weights_only=True)
            resumed_checkpoint = torch.load(resumed["checkpoint_path"], weights_only=True)
            self.assertEqual(whole_checkpoint["history"], resumed_checkpoint["history"])
            for key, whole_value in whole_checkpoint["optimizer_state"]["state"][0].items():
                resumed_value = resumed_checkpoint["optimizer_state"]["state"][0][key]
                if isinstance(whole_value, torch.Tensor):
                    self.assertTrue(torch.equal(whole_value, resumed_value))
                else:
                    self.assertEqual(whole_value, resumed_value)
            self.assertEqual(whole_checkpoint["rng_state"]["numpy"],
                             resumed_checkpoint["rng_state"]["numpy"])
            self.assertTrue(torch.equal(whole_checkpoint["rng_state"]["torch_cpu"],
                                        resumed_checkpoint["rng_state"]["torch_cpu"]))
            incompatible = MaskedSoftConfig.from_mapping({**base,
                "output_dir": str(root / "runs" / "mismatch"),
                "resume_checkpoint": first["checkpoint_path"],
                "resume_checkpoint_sha256": first["checkpoint_sha256"], "learning_rate": 0.002})
            with mock.patch("rsna_knee.training.masked_soft.PRIVATE_RUN_ROOT", root / "runs"):
                with self.assertRaisesRegex(ValueError, "checkpoint_incompatible"):
                    run_masked_soft_training(incompatible, execute=True,
                        dataset_factory=lambda *_: self.fail("invalid checkpoint reached dataset"),
                        model_factory=lambda *_: self.fail("invalid checkpoint reached model"))
            bad_hash = MaskedSoftConfig.from_mapping({**base,
                "output_dir": str(root / "runs" / "bad-hash"),
                "resume_checkpoint": first["checkpoint_path"],
                "resume_checkpoint_sha256": "0" * 64})
            with mock.patch("rsna_knee.training.masked_soft.PRIVATE_RUN_ROOT", root / "runs"):
                with mock.patch("torch.load") as torch_load:
                    with self.assertRaisesRegex(ValueError, "checkpoint_sha_mismatch"):
                        run_masked_soft_training(bad_hash, execute=True,
                            dataset_factory=lambda *_: self.fail("bad hash reached dataset"),
                            model_factory=lambda *_: self.fail("bad hash reached model"))
                    torch_load.assert_not_called()
            corrupted_path = root / "bad-state.pt"
            bad_state = torch.load(first["checkpoint_path"], weights_only=True)
            bad_state["optimizer_state"]["state"][0]["exp_avg"].fill_(float("nan"))
            torch.save(bad_state, corrupted_path)
            corrupted = MaskedSoftConfig.from_mapping({**base,
                "output_dir": str(root / "runs" / "bad-state"),
                "resume_checkpoint": str(corrupted_path),
                "resume_checkpoint_sha256": __import__("hashlib").sha256(corrupted_path.read_bytes()).hexdigest()})
            with mock.patch("rsna_knee.training.masked_soft.PRIVATE_RUN_ROOT", root / "runs"):
                with mock.patch.object(torch.optim.Adam, "step", autospec=True) as step:
                    with self.assertRaisesRegex(ValueError, "checkpoint_state_invalid"):
                        run_masked_soft_training(corrupted, execute=True,
                            dataset_factory=datasets, model_factory=lambda _: TinyModel())
                    step.assert_not_called()
            interrupted = MaskedSoftConfig.from_mapping({**base, "epochs": 3,
                "output_dir": str(root / "runs" / "interrupted")})
            from rsna_knee.training.masked_soft import MaskedSoftMetricAccumulator
            original_update = MaskedSoftMetricAccumulator.update
            calls = 0
            def fail_on_second_validation(instance, *args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("synthetic interruption")
                return original_update(instance, *args, **kwargs)
            with mock.patch("rsna_knee.training.masked_soft.PRIVATE_RUN_ROOT", root / "runs"):
                with mock.patch.object(MaskedSoftMetricAccumulator, "update", fail_on_second_validation):
                    with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                        run_masked_soft_training(interrupted, execute=True,
                            dataset_factory=datasets, model_factory=lambda _: TinyModel())
            preserved = interrupted.output_dir / "checkpoint-epoch-01.pt"
            self.assertTrue(preserved.is_file())
            self.assertEqual(torch.load(preserved, weights_only=True)["completed_epochs"], 1)

    def test_checkpoint_fingerprint_binds_steps_and_all_packaged_source_bytes_not_paths(self):
        import shutil
        from rsna_knee.training.masked_soft import MaskedSoftConfig, _checkpoint_fingerprint

        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            source_a = root / "one" / "src"
            source_b = root / "two" / "src"
            shutil.copytree(Path("src"), source_a)
            shutil.copytree(source_a, source_b)
            config = MaskedSoftConfig.from_mapping(self.config_values(root, max_steps=1))
            plan = {"manifest_sha256": "a" * 64, "folds_sha256": "b" * 64,
                    "model_kind": "resnet18_imagenet1k_v1", "weights_sha256": "c" * 64,
                    "weights_size": 46830571}
            baseline = _checkpoint_fingerprint(config, plan, source_root=source_a)
            relocated = _checkpoint_fingerprint(config, plan, source_root=source_b)
            self.assertEqual(baseline, relocated)
            changed_source = source_b / "rsna_knee" / "data" / "training_dataset.py"
            changed_source.write_bytes(changed_source.read_bytes() + b"\n")
            self.assertNotEqual(baseline, _checkpoint_fingerprint(config, plan, source_root=source_b))
            changed_steps = MaskedSoftConfig.from_mapping(self.config_values(root, max_steps=2))
            self.assertNotEqual(baseline, _checkpoint_fingerprint(changed_steps, plan, source_root=source_a))

    def test_resume_requires_expected_checkpoint_sha_before_load(self):
        from rsna_knee.training.masked_soft import MaskedSoftConfig

        with TemporaryDirectory() as temporary:
            values = self.config_values(Path(temporary), resume_checkpoint=str(Path(temporary) / "checkpoint.pt"))
            with self.assertRaisesRegex(ValueError, "checkpoint_sha_required"):
                MaskedSoftConfig.from_mapping(values)

    def test_cuda_rng_mismatch_is_rejected_before_rng_mutation(self):
        from rsna_knee.training.masked_soft import _restore_rng_state

        state = {"python": __import__("random").getstate(),
                 "numpy": ["MT19937", np.random.get_state()[1].tolist(), 0, 0, 0.0],
                 "torch_cpu": torch.get_rng_state(), "torch_cuda": []}
        with mock.patch("torch.cuda.is_available", return_value=True), \
                mock.patch("torch.cuda.device_count", return_value=1), \
                mock.patch("torch.cuda.set_rng_state_all") as set_cuda, \
                mock.patch("torch.set_rng_state") as set_cpu:
            with self.assertRaisesRegex(ValueError, "checkpoint_device_mismatch"):
                _restore_rng_state(state, device="cuda")
        set_cuda.assert_not_called()
        set_cpu.assert_not_called()

    def test_training_emits_safe_start_and_100_study_progress(self):
        import json
        from contextlib import redirect_stdout
        from io import StringIO
        from rsna_knee.training.masked_soft import MaskedSoftConfig, run_masked_soft_training

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.1))
            def forward(self, images, _masks):
                return (images[0].mean() * self.weight).reshape(1, 1).expand(1, 12)

        sample = ([torch.ones((1, 1, 8, 8))], [torch.ones((1, 1, 8, 8), dtype=torch.bool)],
                  torch.full((12,), 0.5), torch.ones((12,), dtype=torch.bool), "PRIVATE_STUDY_MARKER")
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            config = MaskedSoftConfig.from_mapping(self.config_values(root,
                output_dir=str(root / "runs" / "progress"), max_steps=100, max_epochs_per_run=1))
            output = StringIO()
            with mock.patch("rsna_knee.training.masked_soft.PRIVATE_RUN_ROOT", root / "runs"):
                with redirect_stdout(output):
                    run_masked_soft_training(config, execute=True,
                        dataset_factory=lambda *_: [sample] * 100,
                        model_factory=lambda _: TinyModel())
            events = [json.loads(line) for line in output.getvalue().splitlines()]
            start = next(event for event in events if event["event"] == "training_started")
            self.assertEqual((start["training_study_count"], start["validation_study_count"],
                              start["target_epochs"], start["max_epochs_per_run"]), (2850, 1448, 1, 1))
            self.assertTrue(any(event.get("partition") == "train" and event.get("completed") == 100
                                for event in events))
            self.assertTrue(any(event.get("partition") == "validation" and event.get("completed") == 100
                                for event in events))
            self.assertNotIn("PRIVATE_STUDY_MARKER", output.getvalue())


if __name__ == "__main__":
    unittest.main()
