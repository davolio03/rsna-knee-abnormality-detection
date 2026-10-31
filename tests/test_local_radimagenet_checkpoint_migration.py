import hashlib
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from scripts.local_radimagenet_checkpoint_migration import (
    SOURCE_SHA256,
    OLD_RUNTIME,
    TARGET_RUNTIME,
    CheckpointMigrationError,
    main,
    validate_target_runtime,
    migrate_epoch_one_checkpoint,
    _repo_input_paths,
)


ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data/private"
SOURCE_ZIP = PRIVATE / "resnet50-pretraining-preparation/run01/resnet50-recipe-source.zip"
SOURCE_ASSETS = PRIVATE / "resnet50-pretraining-preparation/run01/upload/resnet50-recipe-assets.json"
CHECKPOINT = PRIVATE / "local-radimagenet-epoch02/checkpoint-epoch-01.pt"
MANIFEST = PRIVATE / "full-cohort-prefilter/run01/full-cohort-soft-label-manifest.json"
FOLDS = PRIVATE / "full-cohort-prefilter/run01/full-cohort-grouped-folds.json"
WEIGHTS = PRIVATE / "resnet50-pretraining-assets/run01/resnet50-radimagenet.pt"


class LocalRadImageNetCheckpointMigrationTests(unittest.TestCase):
    def test_cli_defaults_to_read_only_plan(self):
        real_paths = _repo_input_paths()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {**real_paths, "source_root": root / "source", "output_dir": root / "migration"}
            self.assertFalse(paths["source_root"].exists())
            self.assertFalse(paths["output_dir"].exists())
            output = StringIO()
            with patch("scripts.local_radimagenet_checkpoint_migration._repo_input_paths", return_value=paths):
                with patch("sys.stdout", output):
                    self.assertEqual(main([]), 0)
            plan = json.loads(output.getvalue())
            self.assertEqual(plan["status"], "planned")
            self.assertEqual(plan["execution_requires_explicit_flag"], "--execute")
            self.assertFalse(paths["source_root"].exists())
            self.assertFalse(paths["output_dir"].exists())

    def test_target_runtime_requires_exact_versions_and_one_visible_cuda_device(self):
        self.assertTrue(validate_target_runtime(TARGET_RUNTIME, cuda_device_count=1))
        for runtime, device_count in (
            (dict(TARGET_RUNTIME, python="3.13.15"), 1),
            (dict(TARGET_RUNTIME, torch="2.11.0+cpu"), 1),
            (dict(TARGET_RUNTIME, numpy="2.2.0"), 1),
            (dict(TARGET_RUNTIME, cuda="12.4"), 1),
            (TARGET_RUNTIME, 2),
            (TARGET_RUNTIME, 0),
        ):
            with self.subTest(runtime=runtime, cuda_device_count=device_count):
                with self.assertRaisesRegex(CheckpointMigrationError, "target_runtime_mismatch"):
                    validate_target_runtime(runtime, cuda_device_count=device_count)

    @unittest.skipUnless(
        all(path.is_file() for path in (SOURCE_ZIP, SOURCE_ASSETS, CHECKPOINT, MANIFEST, FOLDS, WEIGHTS)),
        "private pinned migration inputs are unavailable",
    )
    def test_migrates_one_gpu_rng_and_only_the_two_authorized_checkpoint_fields(self):
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            self.skipTest("the pinned migration requires exactly one visible CUDA device")
        original_sha = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
        self.assertEqual(original_sha, "83518ebe59f8a3ecf39b2d8da35aa4c618afeeb04a78b315b5cda13ff56014a8")
        precision_before = (torch.backends.cuda.matmul.allow_tf32,
                            torch.backends.cudnn.allow_tf32,
                            torch.get_float32_matmul_precision())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "frozen-source"
            output_dir = root / "migration"
            result = migrate_epoch_one_checkpoint(
                checkpoint_path=CHECKPOINT,
                source_archive=SOURCE_ZIP,
                source_assets=SOURCE_ASSETS,
                manifest_path=MANIFEST,
                folds_path=FOLDS,
                weights_path=WEIGHTS,
                source_root=source_root,
                output_dir=output_dir,
            )

            self.assertEqual(hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(), original_sha)
            self.assertEqual(result["source_archive_sha256"], SOURCE_SHA256)
            self.assertEqual(len(list(source_root.rglob("*.py"))), 29)
            self.assertEqual(sum(path.is_file() for path in source_root.rglob("*")), 29)
            self.assertEqual(result["original_runtime"], OLD_RUNTIME)
            self.assertEqual(result["target_runtime"], TARGET_RUNTIME)
            self.assertEqual(result["cuda_device_map"], {"0": "0"})
            self.assertEqual(result["unused_original_cuda_rng_state_count"], 1)
            self.assertEqual(result["optimizer_validation"], "passed")
            self.assertEqual(result["target_rng_restore_validation"], "passed")
            self.assertNotEqual(result["original_fingerprint"], result["migrated_fingerprint"])

            receipt_path = output_dir / "migration-receipt.json"
            checkpoint_path = output_dir / "checkpoint-epoch-01-migrated.pt"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt, result)
            self.assertEqual(hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                             result["migrated_checkpoint_sha256"])
            old = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
            new = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.assertEqual(set(old), set(new))
            for key in ("schema", "completed_epochs", "history", "last_epoch_seconds"):
                self.assertEqual(old[key], new[key])
            self.assertEqual(old["fingerprint"], result["original_fingerprint"])
            self.assertEqual(new["fingerprint"], result["migrated_fingerprint"])
            self.assertEqual(len(new["rng_state"]["torch_cuda"]), 1)
            self.assertTrue(torch.equal(new["rng_state"]["torch_cuda"][0],
                                        old["rng_state"]["torch_cuda"][0]))
            for key in ("python", "numpy", "torch_cpu"):
                if isinstance(old["rng_state"][key], torch.Tensor):
                    self.assertTrue(torch.equal(old["rng_state"][key], new["rng_state"][key]))
                else:
                    self.assertEqual(old["rng_state"][key], new["rng_state"][key])
            self.assertTrue(torch.equal(old["model_state"]["backbone.conv1.weight"],
                                        new["model_state"]["backbone.conv1.weight"]))
            self.assertEqual(old["optimizer_state"]["param_groups"],
                             new["optimizer_state"]["param_groups"])
            self.assertEqual((torch.backends.cuda.matmul.allow_tf32,
                              torch.backends.cudnn.allow_tf32,
                              torch.get_float32_matmul_precision()), precision_before)

    @unittest.skipUnless(SOURCE_ZIP.is_file(), "private frozen recipe archive is unavailable")
    def test_rejects_modified_source_archive_before_publishing_any_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            altered = root / "altered.zip"
            with zipfile.ZipFile(SOURCE_ZIP) as original, zipfile.ZipFile(altered, "w") as changed:
                for info in original.infolist():
                    changed.writestr(info, original.read(info.filename))
                changed.writestr("unexpected.py", b"extra")
            with self.assertRaisesRegex(CheckpointMigrationError, "source_archive_pin_mismatch"):
                migrate_epoch_one_checkpoint(
                    checkpoint_path=CHECKPOINT,
                    source_archive=altered,
                    source_assets=SOURCE_ASSETS,
                    manifest_path=MANIFEST,
                    folds_path=FOLDS,
                    weights_path=WEIGHTS,
                    source_root=root / "frozen-source",
                    output_dir=root / "migration",
                )
            self.assertFalse((root / "migration").exists())

    @unittest.skipUnless(CHECKPOINT.is_file(), "private epoch-one checkpoint is unavailable")
    def test_refuses_existing_bundle_before_extracting_source_or_overwriting_files(self):
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            self.skipTest("the pinned migration requires exactly one visible CUDA device")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "frozen-source"
            output_dir = root / "migration"
            output_dir.mkdir()
            marker = output_dir / "keep.txt"
            marker.write_text("unchanged", encoding="utf-8")
            with self.assertRaisesRegex(CheckpointMigrationError, "migration_output_not_fresh"):
                migrate_epoch_one_checkpoint(
                    checkpoint_path=CHECKPOINT,
                    source_archive=SOURCE_ZIP,
                    source_assets=SOURCE_ASSETS,
                    manifest_path=MANIFEST,
                    folds_path=FOLDS,
                    weights_path=WEIGHTS,
                    source_root=source_root,
                    output_dir=output_dir,
                )
            self.assertFalse(source_root.exists())
            self.assertEqual(marker.read_text(encoding="utf-8"), "unchanged")


if __name__ == "__main__":
    unittest.main()
