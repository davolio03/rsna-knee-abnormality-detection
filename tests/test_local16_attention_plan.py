import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rsna_knee.training.local16_attention import Attention16Config, make_attention16_plan

ROOT = Path(__file__).resolve().parents[1]


class Local16AttentionPlanTests(unittest.TestCase):
    def test_plan_is_single_fresh_attention_arm_and_requires_verified_new_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = root / "weights.pt"
            weights.write_bytes(b"test")
            receipt = root / "cache-verification.json"
            receipt.write_text("verified", encoding="utf-8")
            source = root / "trainer.py"
            source.write_text("TRAIN = False\n", encoding="utf-8")
            config = Attention16Config(repository_root=root, weights_path=weights,
                                       output_dir="data/private/attention-sixteen-slices/local16-attention-run01")
            good = {"status": "verified", "cache_identity": "local16-attention-run01",
                    "max_slices_per_series": 16, "studies": 4298, "series": 23765, "slices": 380000,
                    "partition_study_counts": {"train": 2850, "validation": 1448},
                    "no_overlap": True, "no_missing_or_extra_studies": True, "gold_included": False,
                    "all_archive_hashes_rechecked": True, "training_started": False}
            pin = {"size": 4, "sha256": hashlib.sha256(b"test").hexdigest()}
            with patch("rsna_knee.training.local16_attention.WEIGHT_PINS", {"radimagenet": pin}):
                plan = make_attention16_plan(config, good, cache_receipt_path=receipt,
                                             source_paths=(source,))
                self.assertEqual(plan["status"], "planned")
                frozen = plan["configuration"]
                self.assertEqual(frozen["arm"], "target_attention")
                self.assertEqual(frozen["max_slices"], 16)
                self.assertEqual(frozen["epochs"], 5)
                self.assertEqual(frozen["slice_microbatch"], 2)
                self.assertFalse(frozen["checkpoint_warm_start"])
                self.assertFalse(plan["model_constructed"])
                self.assertFalse(plan["training_started"])
                old_cache = {**good, "cache_identity": "legacy-eight-slice"}
                with self.assertRaisesRegex(ValueError, "attention16_cache_verification_required"):
                    make_attention16_plan(config, old_cache, cache_receipt_path=receipt,
                                          source_paths=(source,))

    def test_default_cli_stops_without_verified_cache_and_never_starts_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = subprocess.run(
                [sys.executable, str(ROOT / "scripts/prepare_local16_attention.py"),
                 "--cache-root", str(root / "missing-cache"),
                 "--package-root", str(root / "missing-package")],
                cwd=ROOT, capture_output=True, text=True,
                env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
            self.assertEqual(run.returncode, 2)
            result = json.loads(run.stdout)
            self.assertEqual(result["status"], "blocked")
            self.assertFalse(result["model_constructed"])
            self.assertFalse(result["training_started"])


if __name__ == "__main__":
    unittest.main()
