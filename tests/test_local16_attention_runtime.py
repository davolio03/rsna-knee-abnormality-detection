import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rsna_knee.training.local16_attention import Attention16Config
from rsna_knee.training.local16_attention_runtime import (
    attention16_source_closure,
    run_local16_attention,
)


class Local16AttentionRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.weights = self.root / "weights.pt"
        self.weights.write_bytes(b"test")
        self.receipt = self.root / "cache-verification.json"
        self.output = Path("data/private/attention-sixteen-slices/local16-attention-run01")
        self.config = Attention16Config(self.root, self.weights, self.output, device="cuda")
        self.proof = {"status": "verified", "cache_identity": "local16-attention-run01",
                      "max_slices_per_series": 16, "studies": 4298, "series": 23765, "slices": 380000,
                      "partition_study_counts": {"train": 2850, "validation": 1448},
                      "no_overlap": True, "no_missing_or_extra_studies": True, "gold_included": False,
                      "all_archive_hashes_rechecked": True, "training_started": False}
        self.receipt.write_bytes((json.dumps(self.proof, sort_keys=True, indent=2) + "\n").encode())
        self.paths = []
        for name in ("trainer.py", "model.py", "cache.py"):
            path = self.root / name
            path.write_text(name, encoding="utf-8")
            self.paths.append(path)

    def tearDown(self):
        self.temporary.cleanup()

    def test_source_fingerprint_closure_includes_reexported_seed_helper(self):
        paths = attention16_source_closure(Path.cwd(), Path("scripts/train_local16_attention.py"))
        self.assertIn(Path.cwd() / "src/rsna_knee/training/baseline.py", paths)

    def test_default_plan_validates_cache_and_fingerprint_without_model_or_output(self):
        with patch("rsna_knee.training.local16_attention.WEIGHT_PINS",
                   {"radimagenet": {"size": 4, "sha256": hashlib.sha256(b"test").hexdigest()}}):
            result = run_local16_attention(
                self.config, manifest_path=self.root / "manifest.json", folds_path=self.root / "folds.json",
                plan_path=self.root / "plan.json", source_archive_path=self.root / "source.zip",
                archive_dir=self.root / "archives", cache_receipt_path=self.receipt,
                cache_verifier=lambda **kwargs: self.proof, source_paths=tuple(self.paths))
        self.assertEqual(result["status"], "planned")
        self.assertEqual(result["configuration"]["arm"], "target_attention")
        self.assertFalse(result["model_constructed"])
        self.assertFalse(result["training_started"])
        self.assertFalse((self.root / self.output).exists())

    def test_execution_dispatches_one_attention_arm_with_fresh_16_fingerprint(self):
        calls = []
        def fake_train(*args, **kwargs):
            calls.append((kwargs["arm"], kwargs["fingerprint"], kwargs["epochs"], kwargs["resume"]))
            return {"arm": kwargs["arm"], "completed_epochs": 5, "status": "complete", "epochs": []}

        class FakeDataset:
            def __init__(self, count):
                self.count = count
            def __len__(self):
                return self.count

        with patch("rsna_knee.training.local16_attention.WEIGHT_PINS",
                   {"radimagenet": {"size": 4, "sha256": hashlib.sha256(b"test").hexdigest()}}), \
             patch("rsna_knee.training.resnet50._seed_everything"), \
             patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.mem_get_info", return_value=(4 * 1024**3, 8 * 1024**3)):
            kwargs = dict(
                config=self.config, manifest_path=self.root / "manifest.json", folds_path=self.root / "folds.json",
                plan_path=self.root / "plan.json", source_archive_path=self.root / "source.zip",
                archive_dir=self.root / "archives", cache_receipt_path=self.receipt,
                cache_verifier=lambda **kwargs: self.proof, source_paths=tuple(self.paths), execute=True,
                dataset_factory=lambda partition: FakeDataset(2850 if partition == "train" else 1448),
                train_arm=fake_train)
            result = run_local16_attention(**kwargs)
            resumed = run_local16_attention(**{**kwargs, "resume": True})
        self.assertEqual(result["status"], "complete")
        self.assertEqual(resumed["status"], "complete")
        self.assertEqual(len(calls), 2)
        self.assertEqual((calls[0][0], calls[0][2], calls[0][3]), ("target_attention", 5, False))
        self.assertTrue(calls[1][3])
        self.assertEqual(result["configuration"]["max_slices"], 16)
        self.assertFalse(result["configuration"]["checkpoint_warm_start"])


if __name__ == "__main__":
    unittest.main()
