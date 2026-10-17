import hashlib
import io
import json
import math
import ast
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("optional NumPy image dependency is not installed") from exc

from rsna_knee.data.dicom_preprocessing import DicomPreprocessingError
from scripts.build_kaggle_preprocessing_canary import build_canary_notebook
from scripts.kaggle_preprocessing_canary_runtime import (
    CanaryFailure,
    run_preprocessing_canary,
    select_target_series,
    verify_package,
)


MANIFEST_REL = "data/private/500-study-training-cohort.json"
FOLDS_REL = "data/private/500-study-training-folds.json"


def package_fixture(root: Path, manifest: bytes, folds: bytes):
    members = {
        MANIFEST_REL: manifest,
        FOLDS_REL: folds,
        "src/payload.py": b"trusted-package-member\n",
    }
    for name, raw in members.items():
        target = root.joinpath(*Path(name).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    inventory = {
        "schema_version": "kaggle-training-package-inventory-v1",
        "package_sha256": "a" * 64,
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "folds_sha256": hashlib.sha256(folds).hexdigest(),
        "members": {name: hashlib.sha256(raw).hexdigest() for name, raw in members.items()},
    }
    raw_inventory = (json.dumps(inventory, sort_keys=True) + "\n").encode()
    config = {
        "expected_member_count": len(members),
        "expected_inventory_sha256": hashlib.sha256(raw_inventory).hexdigest(),
        "expected_archive_provenance": inventory["package_sha256"],
        "expected_manifest_sha256": inventory["manifest_sha256"],
        "expected_folds_sha256": inventory["folds_sha256"],
        "manifest_relative_path": MANIFEST_REL,
        "folds_relative_path": FOLDS_REL,
        "fold": 0,
        "expected_training_count": 1,
        "expected_validation_count": 0,
        "target_series_ordinal": 1,
        "expected_target_file_count": 1,
        "output_size": 256,
        "max_slices": 8,
        "max_seconds": 600.0,
        "output_dir": str(root.parent / "output"),
    }
    return raw_inventory, config


class KagglePreprocessingCanaryTests(unittest.TestCase):
    def test_target_series_uses_manifest_training_order_and_fixed_ordinal(self):
        with tempfile.TemporaryDirectory() as temp:
            image_root = Path(temp)
            records = [
                ("validation", ("not-selected",)),
                ("train-a", ("first", "target")),
                ("train-b", ("later",)),
            ]
            assignments = {"validation": 0, "train-a": 1, "train-b": 2}
            for study_id, series_ids in records:
                for series_id in series_ids:
                    folder = image_root / study_id / series_id
                    folder.mkdir(parents=True)
                    (folder / "slice-1.dcm").write_bytes(b"synthetic")
            target = image_root / "train-a" / "target"
            (target / "slice-2.dcm").write_bytes(b"synthetic")
            (target / "slice-3.dcm").write_bytes(b"synthetic")

            folder, file_count = select_target_series(
                records, assignments, image_root,
                fold=0, expected_training_count=2, expected_validation_count=1,
                target_ordinal=2, expected_file_count=3,
            )

            self.assertEqual(folder, target)
            self.assertEqual(file_count, 3)

    def test_missing_target_ordinal_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "train" / "series"
            folder.mkdir(parents=True)
            (folder / "one.dcm").write_bytes(b"synthetic")
            with self.assertRaises(CanaryFailure) as caught:
                select_target_series(
                    [("train", ("series",))], {"train": 1}, root,
                    fold=0, expected_training_count=1, expected_validation_count=0,
                    target_ordinal=2, expected_file_count=1,
                )
            self.assertEqual(caught.exception.reason_code, "target_series_unavailable")

    def test_package_preflight_checks_inventory_members_before_processing(self):
        manifest = b'{"records": []}'
        folds = b'{"study_assignments": []}'
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "input"
            root.mkdir()
            inventory, config = package_fixture(root, manifest, folds)
            self.assertEqual(len(verify_package(root, inventory, config)["inventory"]["members"]), 3)
            (root / "src/payload.py").write_bytes(b"tampered")
            with self.assertRaises(CanaryFailure) as caught:
                verify_package(root, inventory, config)
            self.assertEqual(caught.exception.reason_code, "package_member_hash_mismatch")

    def test_package_mismatch_blocks_before_target_or_preprocessor_access(self):
        manifest = b'{"records": []}'
        folds = b'{"study_assignments": []}'
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root = base / "input"
            root.mkdir()
            inventory, config = package_fixture(root, manifest, folds)
            calls = []
            with self.assertRaises(CanaryFailure) as caught:
                run_preprocessing_canary(
                    config, inventory, lambda _raw: calls.append("manifest"),
                    input_root=base / "missing-mount", image_root=base / "missing-images",
                    preprocessor=lambda *_args, **_kwargs: calls.append("preprocess"),
                )
            self.assertEqual(caught.exception.reason_code, "package_inventory_invalid")
            self.assertEqual(calls, [])

    def test_preprocessing_summary_is_aggregate_only_and_output_is_new(self):
        manifest = b'{"records":[{"study_id":"opaque-study","series":[{"series_id":"opaque-series"}]}]}'
        folds = b'{"study_assignments":[{"study_id":"opaque-study","fold":1}]}'
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root = base / "input"
            root.mkdir()
            image_root = base / "images"
            target = image_root / "opaque-study" / "opaque-series"
            target.mkdir(parents=True)
            (target / "slice.dcm").write_bytes(b"synthetic")
            inventory, config = package_fixture(root, manifest, folds)
            config["output_dir"] = str(base / "output")
            called = []

            def preprocessor(series_path, *, max_slices, output_size, dcmread=None):
                called.append((series_path, max_slices, output_size))
                return SimpleNamespace(images=np.full((1, 1, 256, 256), 0.5, dtype=np.float32))

            def manifest_parser(raw):
                self.assertEqual(raw, manifest)
                return [("opaque-study", ("opaque-series",))]

            summary = run_preprocessing_canary(
                config, inventory, manifest_parser,
                input_root=root, image_root=image_root,
                preprocessor=preprocessor,
            )

            self.assertEqual(len(called), 1)
            self.assertEqual(called[0][1:], (8, 256))
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["series_ordinal"], 1)
            self.assertEqual(summary["input_file_count"], 1)
            self.assertEqual(summary["image_shape"], [1, 1, 256, 256])
            self.assertTrue(summary["finite_values"])
            self.assertEqual(summary["normalized_value_range"], [0.5, 0.5])
            serialized = json.dumps(summary)
            self.assertNotIn("opaque-study", serialized)
            self.assertNotIn("opaque-series", serialized)
            self.assertNotIn(str(target), serialized)
            saved = json.loads((base / "output" / "preprocessing-summary.json").read_text())
            self.assertEqual(saved, summary)

    def test_preprocessing_failure_exposes_only_safe_reason_and_no_success_output(self):
        manifest = b'{"records":[{"study_id":"opaque-study","series":[{"series_id":"opaque-series"}]}]}'
        folds = b'{"study_assignments":[{"study_id":"opaque-study","fold":1}]}'
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root = base / "input"
            root.mkdir()
            image_root = base / "images"
            target = image_root / "opaque-study" / "opaque-series"
            target.mkdir(parents=True)
            (target / "slice.dcm").write_bytes(b"synthetic")
            inventory, config = package_fixture(root, manifest, folds)
            config["output_dir"] = str(base / "output")

            def failing_preprocessor(*args, **kwargs):
                raise DicomPreprocessingError("pixel_decode_failed", f"private {target}")

            with self.assertRaises(CanaryFailure) as caught:
                run_preprocessing_canary(
                    config, inventory, lambda _raw: [("opaque-study", ("opaque-series",))],
                    input_root=root, image_root=image_root,
                    preprocessor=failing_preprocessor,
                )

            self.assertEqual(caught.exception.reason_code, "pixel_decode_failed")
            self.assertNotIn(str(target), str(caught.exception))
            self.assertFalse((base / "output").exists())

    def test_notebook_is_disabled_output_free_and_embeds_exact_helper_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inventory = json.dumps({
                "schema_version": "kaggle-training-package-inventory-v1",
                "package_sha256": "a" * 64,
                "manifest_sha256": "b" * 64,
                "folds_sha256": "c" * 64,
                "members": {
                    "data/private/500-study-training-cohort.json": "d" * 64,
                    "data/private/500-study-training-folds.json": "e" * 64,
                    **{f"src/module_{index}.py": "f" * 64 for index in range(20)},
                },
            }, sort_keys=True).encode()
            config = {
                "expected_member_count": 22, "expected_manifest_sha256": "b" * 64,
                "expected_folds_sha256": "c" * 64, "fold": 0,
                "target_series_ordinal": 355,
            }
            paths = build_canary_notebook(
                root, inventory, config,
                runtime_helper_path=Path("scripts/kaggle_preprocessing_canary_runtime.py"),
                manifest_helper_path=Path("src/rsna_knee/data/orientation_audit.py"),
            )
            notebook = json.loads(paths["notebook_path"].read_text(encoding="utf-8"))
            self.assertTrue(all(not cell.get("outputs", []) for cell in notebook["cells"]))
            code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
            self.assertEqual(len(code_cells), 1)
            code = "".join(code_cells[0]["source"])
            self.assertIn("RUN_CANARY = False", code)
            self.assertNotIn("RUN_CANARY = True", code)
            self.assertNotIn("torch", code.lower())
            self.assertNotIn("run_training", code)
            self.assertNotIn("train_dataset", code)
            self.assertNotIn("RUN_TRAINING", code)
            compile(code, "preprocessing_canary.py", "exec")
            namespace = {"__name__": "canary_notebook_test"}
            exec(compile(code, "preprocessing_canary.py", "exec"), namespace)
            self.assertFalse(namespace["RUN_CANARY"])
            helper_match = re.search(r'^    CANARY_HELPER_B64 = (".*")$', code, re.MULTILINE)
            helper_hash_match = re.search(r'^    CANARY_HELPER_B64_SHA256 = "([0-9a-f]{64})"$', code, re.MULTILINE)
            self.assertIsNotNone(helper_match)
            self.assertEqual(
                hashlib.sha256(ast.literal_eval(helper_match.group(1)).encode()).hexdigest(),
                helper_hash_match.group(1),
            )
            self.assertTrue(math.isfinite(float(config["target_series_ordinal"])))

    def test_enabled_notebook_fails_closed_without_mounted_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            members = {
                "data/private/500-study-training-cohort.json": "d" * 64,
                "data/private/500-study-training-folds.json": "e" * 64,
                **{f"src/module_{index}.py": "f" * 64 for index in range(20)},
            }
            inventory = json.dumps({
                "schema_version": "kaggle-training-package-inventory-v1",
                "package_sha256": "a" * 64,
                "manifest_sha256": "b" * 64,
                "folds_sha256": "c" * 64,
                "members": members,
            }, sort_keys=True).encode()
            paths = build_canary_notebook(
                root, inventory,
                {
                    "expected_member_count": 22, "expected_manifest_sha256": "b" * 64,
                    "expected_folds_sha256": "c" * 64, "fold": 0,
                    "target_series_ordinal": 355, "private_input_root": str(root / "absent"),
                    "image_root": str(root / "images"),
                },
                runtime_helper_path=Path("scripts/kaggle_preprocessing_canary_runtime.py"),
                manifest_helper_path=Path("src/rsna_knee/data/orientation_audit.py"),
            )
            notebook = json.loads(paths["notebook_path"].read_text(encoding="utf-8"))
            code = "".join(notebook["cells"][0]["source"]).replace(
                "RUN_CANARY = False", "RUN_CANARY = True", 1,
            )
            stdout = io.StringIO()
            with self.assertRaisesRegex(RuntimeError, "preprocessing_failed"), redirect_stdout(stdout):
                exec(compile(code, "canary.ipynb", "exec"), {"__name__": "enabled_canary_test"})
            event = json.loads(stdout.getvalue())
            self.assertEqual(event, {"reason_code": "preprocessing_failed", "status": "failed"})
            self.assertNotIn(str(root), stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
