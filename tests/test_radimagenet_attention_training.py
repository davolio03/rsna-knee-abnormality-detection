"""Safe preparation, provenance and masked-metric checks for E048."""
import tempfile
from pathlib import Path
import json
import hashlib
import unittest
from unittest.mock import patch

import torch
from torch import nn

from rsna_knee.training.attention_mil import (AttentionMILConfig, _train_arm, make_pair_plan,
                                              run_attention_pair)
from rsna_knee.training.masked_soft import MaskedSoftMetricAccumulator


class AttentionMILTrainingTests(unittest.TestCase):
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Parameter(torch.zeros(12))
            self.trunk = nn.Parameter(torch.ones(12) * 0.1)

        def forward(self, images, masks):
            return (self.head + self.trunk).reshape(1, 12)

        def optimizer_groups(self, *, head_lr, backbone_lr):
            return [{"params": [self.head], "lr": head_lr, "name": "head"},
                    {"params": [self.trunk], "lr": backbone_lr, "name": "layer4"}]

    class OneStudy:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            return ([torch.ones(1, 1, 8, 8)], [torch.ones(1, 1, 8, 8, dtype=torch.bool)],
                    torch.full((12,), 0.25), torch.ones(12, dtype=torch.bool), index)

    def _make_checkpoint(self, root, *, epochs=1):
        config = AttentionMILConfig(root, root / "weights", root / "output", device="cpu")
        root.mkdir(parents=True, exist_ok=True)
        with patch("rsna_knee.training.attention_mil._model_for_arm", side_effect=lambda *_: self.ToyModel()):
            _train_arm(config, arm="target_attention", output=root, fingerprint="f" * 64,
                       train_data=self.OneStudy(), validation_data=self.OneStudy(), epochs=epochs, resume=False)
        return config

    def _rewrite_checkpoint(self, root, epoch, mutate):
        path = root / f"checkpoint-epoch-{epoch:02d}.pt"
        sidecar = path.with_suffix(".sha256")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        mutate(payload)
        torch.save(payload, path)
        proof = json.loads(sidecar.read_text(encoding="utf-8"))
        proof["checkpoint_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        sidecar.write_text(json.dumps(proof), encoding="utf-8")

    def _write_pair_checkpoint(self, output, arm, epoch, fingerprint):
        arm_root = output / arm
        arm_root.mkdir(parents=True, exist_ok=True)
        path = arm_root / f"checkpoint-epoch-{epoch:02d}.pt"
        path.write_bytes(b"verified-checkpoint-fixture")
        proof = {"schema": "disease-attention-mil-checkpoint-proof-v1", "arm": arm,
                 "fingerprint": fingerprint, "completed_epochs": epoch,
                 "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        path.with_suffix(".sha256").write_text(json.dumps(proof), encoding="utf-8")

    def test_plan_only_has_no_output_or_model_dataset_construction(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "data/private/full-cohort-prefilter/run01/full-cohort-soft-label-manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("manifest", encoding="utf-8")
            (manifest.parent / "full-cohort-grouped-folds.json").write_text("folds", encoding="utf-8")
            receipt = root / "data/private/local-radimagenet-epoch02/cache-all-seven-chunks-independent-proof.json"
            receipt.parent.mkdir(parents=True)
            receipt.write_text("receipt", encoding="utf-8")
            weights = root / "weights.pt"
            weights.write_bytes(b"weights")
            output = root / "data/private/disease-attention-mil/new-output"
            config = AttentionMILConfig(root, weights, output)
            with patch("rsna_knee.training.attention_mil.WEIGHT_PINS", {
                    "radimagenet": {"sha256": "a" * 64, "size": 7, "normalization": "radimagenet_2x_minus_1"}}), \
                    patch("rsna_knee.training.attention_mil._digest", return_value="a" * 64), \
                    patch("rsna_knee.training.attention_mil._source_fingerprint", return_value={"model.py": "b" * 64}), \
                    patch("rsna_knee.training.attention_mil._runtime_versions", return_value={"torch": "test"}), \
                    patch("scripts.local_masked_full_cache.PARTS", {"train-000": (1, "c" * 64)}):
                plan = run_attention_pair(config, cache_validator=lambda _: {
                    "status": "verified", "studies": 4298, "series": 23765,
                    "training_studies": 2850, "validation_studies": 1448})
            self.assertEqual(plan["status"], "planned")
            self.assertEqual(plan["configuration"]["arms"], ["equal_mean", "target_attention"])
            self.assertEqual(plan["configuration"]["epochs"], 5)
            self.assertFalse(output.exists())

    def test_plan_fails_without_verified_cache_and_never_constructs_model(self):
        config = AttentionMILConfig(Path("."), Path("weights"),
                                    Path("data/private/disease-attention-mil/out"))
        with patch("rsna_knee.training.attention_mil._model_for_arm") as model:
            with self.assertRaisesRegex(ValueError, "cache_unavailable"):
                run_attention_pair(config, cache_validator=lambda _: (_ for _ in ()).throw(ValueError("cache_unavailable")))
            model.assert_not_called()

    def test_output_path_must_be_isolated_under_private_experiment_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = AttentionMILConfig(root, root / "weights", root / "outputs/escape")
            with self.assertRaisesRegex(ValueError, "output_must_be_private"):
                run_attention_pair(config, cache_validator=lambda _: self.fail("cache should not be inspected"))

    def test_masked_metrics_report_soft_target_counts_not_auc(self):
        metric = MaskedSoftMetricAccumulator(("ACL", "MCL"))
        metric.update(torch.tensor([[0.0, 1.0]]), torch.tensor([[0.25, 0.75]]),
                      torch.tensor([[True, False]]))
        result = metric.compute()
        self.assertEqual(result["observed_cell_count"], 1)
        self.assertEqual([row["observed_count"] for row in result["per_target"]], [1, 0])
        self.assertIsNone(result["per_target"][1]["brier"])
        self.assertNotIn("auc", result)

    def test_atomic_epoch_checkpoint_resumes_from_matching_sha_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = AttentionMILConfig(root, root / "weights", root / "output", device="cpu")
            with patch("rsna_knee.training.attention_mil._model_for_arm", side_effect=lambda *_: self.ToyModel()):
                first = _train_arm(config, arm="target_attention", output=root, fingerprint="f" * 64,
                                   train_data=self.OneStudy(), validation_data=self.OneStudy(), epochs=1, resume=False)
                self.assertEqual(first["completed_epochs"], 1)
                checkpoint = root / "checkpoint-epoch-01.pt"
                proof = checkpoint.with_suffix(".sha256")
                self.assertTrue(checkpoint.is_file())
                self.assertTrue(proof.is_file())
                with patch("rsna_knee.training.attention_mil._model_for_arm", side_effect=lambda *_: self.ToyModel()):
                    resumed = _train_arm(config, arm="target_attention", output=root, fingerprint="f" * 64,
                                         train_data=self.OneStudy(), validation_data=self.OneStudy(), epochs=2, resume=True)
                self.assertEqual(resumed["completed_epochs"], 2)
                self.assertTrue((root / "checkpoint-epoch-02.pt").is_file())

                latest_proof = (root / "checkpoint-epoch-02.sha256")
                payload = json.loads(latest_proof.read_text(encoding="utf-8"))
                payload["fingerprint"] = "0" * 64
                latest_proof.write_text(json.dumps(payload), encoding="utf-8")
                with patch("rsna_knee.training.attention_mil._model_for_arm", side_effect=lambda *_: self.ToyModel()), \
                        patch("torch.load") as load:
                    with self.assertRaisesRegex(ValueError, "checkpoint_fingerprint_mismatch"):
                        _train_arm(config, arm="target_attention", output=root, fingerprint="f" * 64,
                                   train_data=self.OneStudy(), validation_data=self.OneStudy(), epochs=3, resume=True)
                    load.assert_not_called()

    def test_resume_rejects_latest_checkpoint_without_proof_instead_of_falling_back(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._make_checkpoint(root, epochs=1)
            with patch("rsna_knee.training.attention_mil._model_for_arm", side_effect=lambda *_: self.ToyModel()):
                _train_arm(config, arm="target_attention", output=root, fingerprint="f" * 64,
                           train_data=self.OneStudy(), validation_data=self.OneStudy(), epochs=2, resume=True)
            (root / "checkpoint-epoch-02.sha256").unlink()
            with patch("rsna_knee.training.attention_mil._model_for_arm") as factory, patch("torch.load") as load:
                with self.assertRaisesRegex(ValueError, "checkpoint_proof_missing"):
                    _train_arm(config, arm="target_attention", output=root, fingerprint="f" * 64,
                               train_data=self.OneStudy(), validation_data=self.OneStudy(), epochs=3, resume=True)
                factory.assert_not_called()
                load.assert_not_called()

    def test_resume_without_any_checkpoint_refuses_implicit_fresh_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            root.mkdir(exist_ok=True)
            config = AttentionMILConfig(root, root / "weights", root / "output", device="cpu")
            with patch("rsna_knee.training.attention_mil._model_for_arm") as factory:
                with self.assertRaisesRegex(ValueError, "checkpoint_missing"):
                    _train_arm(config, arm="target_attention", output=root, fingerprint="f" * 64,
                               train_data=self.OneStudy(), validation_data=self.OneStudy(), epochs=1, resume=True)
                factory.assert_not_called()

    def test_started_arm_missing_checkpoint_fails_before_data_or_other_arm_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "data/private/disease-attention-mil/run01"
            output.mkdir(parents=True)
            fingerprint = "f" * 64
            (output / "equal_mean").mkdir()
            (output / "run-plan.json").write_text(json.dumps({"status": "planned", "fingerprint": fingerprint,
                "arm_states": {"equal_mean": "started", "target_attention": "not_started"}}), encoding="utf-8")
            config = AttentionMILConfig(root, root / "weights", output, device="cuda")
            plan = {"status": "planned", "fingerprint": fingerprint, "output_dir": str(output),
                    "arm_states": {"equal_mean": "not_started", "target_attention": "not_started"}}
            with patch("rsna_knee.training.attention_mil.make_pair_plan", return_value=plan), \
                    patch("torch.cuda.is_available", return_value=True), \
                    patch("torch.cuda.mem_get_info", return_value=(4 * 1024**3, 6 * 1024**3)), \
                    patch("rsna_knee.training.attention_mil._cache_dataset") as dataset, \
                    patch("rsna_knee.training.attention_mil._model_for_arm") as model:
                with self.assertRaisesRegex(ValueError, "checkpoint_missing"):
                    run_attention_pair(config, execute=True, resume=True)
                dataset.assert_not_called()
                model.assert_not_called()
            self.assertTrue((output / "equal_mean").is_dir())
            self.assertFalse((output / "target_attention").exists())

    def _run_interrupted_pair(self, *, control_state, control_epoch):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        output = root / "data/private/disease-attention-mil/run01"
        output.mkdir(parents=True)
        fingerprint = "f" * 64
        states = {"equal_mean": control_state, "target_attention": "not_started"}
        run_plan = {"status": "planned", "fingerprint": fingerprint, "arm_states": states}
        (output / "run-plan.json").write_text(json.dumps(run_plan), encoding="utf-8")
        self._write_pair_checkpoint(output, "equal_mean", control_epoch, fingerprint)
        config = AttentionMILConfig(root, root / "weights", output, device="cuda")
        plan = {"status": "planned", "fingerprint": fingerprint, "output_dir": str(output),
                "arm_states": {"equal_mean": "not_started", "target_attention": "not_started"}}
        calls = []

        def pretend_train(_config, *, arm, output, fingerprint, train_data, validation_data, epochs,
                          resume, checkpoint_callback):
            calls.append((arm, resume))
            return {"arm": arm, "completed_epochs": 5, "epochs": [], "checkpoint_sha256": "d" * 64,
                    "status": "complete"}

        with patch("rsna_knee.training.attention_mil.make_pair_plan", return_value=plan), \
                patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.mem_get_info", return_value=(4 * 1024**3, 6 * 1024**3)), \
                patch("rsna_knee.training.attention_mil._seed_everything"), \
                patch("rsna_knee.training.attention_mil._train_arm", side_effect=pretend_train):
            result = run_attention_pair(config, execute=True, resume=True,
                                        dataset_factory=lambda _partition: object())
        return result, calls, output

    def test_pair_resume_allows_checkpointed_control_and_proven_pending_attention(self):
        result, calls, output = self._run_interrupted_pair(control_state="started", control_epoch=1)
        self.assertEqual(calls, [("equal_mean", True), ("target_attention", False)])
        self.assertEqual(result["status"], "complete")
        stored = json.loads((output / "run-plan.json").read_text(encoding="utf-8"))
        self.assertEqual(stored["arm_states"], {"equal_mean": "complete", "target_attention": "complete"})

    def test_pair_resume_after_complete_control_can_start_pending_attention(self):
        result, calls, output = self._run_interrupted_pair(control_state="complete", control_epoch=5)
        self.assertEqual(calls, [("equal_mean", True), ("target_attention", False)])
        self.assertEqual(result["status"], "complete")
        self.assertEqual(json.loads((output / "run-plan.json").read_text(encoding="utf-8"))["arm_states"],
                         {"equal_mean": "complete", "target_attention": "complete"})

    def test_resume_rejects_corrupt_history_model_optimizer_and_rng_before_restore(self):
        cases = (
            ("history_nan", lambda p: p["history"][0].update(training_masked_bce=float("nan")), "checkpoint_history_invalid"),
            ("history_gap", lambda p: p.update(history=[]), "checkpoint_history_invalid"),
            ("history_cell_sum", lambda p: p["history"][0]["validation"].update(observed_cell_count=99), "checkpoint_history_invalid"),
            ("checkpoint_status", lambda p: p.update(status="incomplete"), "checkpoint_incompatible"),
            ("model_nan", lambda p: p["model_state"].update(head=torch.full((12,), float("nan"))), "checkpoint_model_invalid"),
            ("model_dtype", lambda p: p["model_state"].update(head=p["model_state"]["head"].double()), "checkpoint_model_invalid"),
            ("model_shape", lambda p: p["model_state"].update(head=p["model_state"]["head"][:11]), "checkpoint_model_invalid"),
            ("optimizer_lr", lambda p: p["optimizer_state"]["param_groups"][0].update(lr=2.0), "checkpoint_optimizer_invalid"),
            ("optimizer_group", lambda p: p["optimizer_state"]["param_groups"][1].update(name="head"), "checkpoint_optimizer_invalid"),
            ("optimizer_nan", lambda p: p["optimizer_state"]["state"].__setitem__(0, {"step": torch.tensor(1.), "exp_avg": torch.full((12,), float("nan")), "exp_avg_sq": torch.ones(12)}), "checkpoint_optimizer_invalid"),
            ("optimizer_shape", lambda p: p["optimizer_state"]["state"][0].update(exp_avg=torch.ones(11)), "checkpoint_optimizer_invalid"),
            ("optimizer_step", lambda p: p["optimizer_state"]["state"][0].update(step=torch.tensor(0.)), "checkpoint_optimizer_invalid"),
            ("optimizer_option", lambda p: p["optimizer_state"]["param_groups"][0].update(foreach=True), "checkpoint_optimizer_invalid"),
            ("rng_missing", lambda p: p["rng_state"].pop("torch_cpu"), "checkpoint_rng_invalid"),
            ("rng_shape", lambda p: p["rng_state"].update(torch_cpu=p["rng_state"]["torch_cpu"][:10]), "checkpoint_rng_invalid"),
        )
        for name, mutate, expected in cases:
            with self.subTest(case=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                config = self._make_checkpoint(root)
                self._rewrite_checkpoint(root, 1, mutate)
                model = self.ToyModel()
                with patch("rsna_knee.training.attention_mil._model_for_arm", return_value=model), \
                        patch.object(model, "load_state_dict", wraps=model.load_state_dict) as load_model, \
                        patch("torch.optim.Adam.load_state_dict", autospec=True) as load_optimizer:
                    with self.assertRaisesRegex(ValueError, expected):
                        _train_arm(config, arm="target_attention", output=root, fingerprint="f" * 64,
                                   train_data=self.OneStudy(), validation_data=self.OneStudy(), epochs=2, resume=True)
                    load_model.assert_not_called()
                    load_optimizer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
