import hashlib
import io
import json
import math
import ast
import re
import tempfile
import unittest
import os
import subprocess
import sys
import zipfile
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
from scripts.build_kaggle_training_package import _local_import_closure


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

    def test_uninventoried_files_fail_closed_without_echoing_their_names(self):
        manifest = b'{"records": []}'
        folds = b'{"study_assignments": []}'
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "input"
            root.mkdir()
            inventory, config = package_fixture(root, manifest, folds)
            secret_name = "reports/private-report.txt"
            unexpected = root / secret_name
            unexpected.parent.mkdir()
            unexpected.write_text("synthetic sentinel", encoding="utf-8")
            with self.assertRaises(CanaryFailure) as caught:
                verify_package(root, inventory, config)
            self.assertEqual(caught.exception.reason_code, "package_inventory_invalid")
            self.assertNotIn(secret_name, str(caught.exception))

    def test_symlinked_input_root_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            target = base / "target"
            target.mkdir()
            inventory, config = package_fixture(target, b"{}", b"{}")
            linked_root = base / "linked-input"
            try:
                linked_root.symlink_to(target, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("directory symlinks are unavailable in this environment")
            with self.assertRaises(CanaryFailure) as caught:
                verify_package(linked_root, inventory, config)
            self.assertEqual(caught.exception.reason_code, "package_inventory_invalid")

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
            with self.assertRaisesRegex(RuntimeError, "package_inventory_invalid"), redirect_stdout(stdout):
                exec(compile(code, "canary.ipynb", "exec"), {"__name__": "enabled_canary_test"})
            event = json.loads(stdout.getvalue())
            self.assertEqual(event, {"reason_code": "package_inventory_invalid", "status": "failed"})
            self.assertNotIn(str(root), stdout.getvalue())

    def test_config_hash_domain_matches_notebook_decode_check(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inventory = json.dumps({
                "schema_version": "kaggle-training-package-inventory-v1",
                "package_sha256": "a" * 64, "manifest_sha256": "b" * 64,
                "folds_sha256": "c" * 64,
                "members": {
                    "data/private/500-study-training-cohort.json": "d" * 64,
                    "data/private/500-study-training-folds.json": "e" * 64,
                    **{f"src/module_{index}.py": "f" * 64 for index in range(20)},
                },
            }, sort_keys=True).encode()
            paths = build_canary_notebook(
                root, inventory,
                {"expected_member_count": 22, "expected_manifest_sha256": "b" * 64,
                 "expected_folds_sha256": "c" * 64, "private_input_root": str(root),
                 "image_root": str(root), "fold": 0, "target_series_ordinal": 355},
                runtime_helper_path=Path("scripts/kaggle_preprocessing_canary_runtime.py"),
                manifest_helper_path=Path("src/rsna_knee/data/orientation_audit.py"),
            )
            code = "".join(json.loads(paths["notebook_path"].read_text())["cells"][0]["source"])
            config_b64 = ast.literal_eval(re.search(r'^    CANARY_CONFIG_B64 = (".*")$', code, re.MULTILINE).group(1))
            config_hash = re.search(r'^    CANARY_CONFIG_SHA256 = "([0-9a-f]{64})"$', code, re.MULTILINE).group(1)
            self.assertEqual(hashlib.sha256(config_b64.encode("ascii")).hexdigest(), config_hash)

    def test_enabled_notebook_preprocesses_only_synthetic_packaged_series(self):
        try:
            import pydicom
            from pydicom.dataset import FileDataset, FileMetaDataset
            from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid
        except ModuleNotFoundError as exc:
            self.skipTest(f"optional synthetic DICOM dependency is unavailable: {exc}")

        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            input_root = base / "input"
            source_root = input_root / "src"
            manifest = json.dumps({"records": [{"study_id": "synthetic-study", "series": [
                {"series_id": "synthetic-series"},
            ]}]}).encode()
            folds = json.dumps({"study_assignments": [{"study_id": "synthetic-study", "fold": 1}]}).encode()
            members = {
                path.relative_to(Path.cwd()).as_posix(): path.read_bytes()
                for path in _local_import_closure(Path.cwd() / "src")
            }
            members[MANIFEST_REL] = manifest
            members[FOLDS_REL] = folds
            package_bytes_path = base / "synthetic-package.zip"
            with zipfile.ZipFile(package_bytes_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, raw in sorted(members.items()):
                    archive.writestr(name, raw)
            inventory = {
                "schema_version": "kaggle-training-package-inventory-v1",
                "package_sha256": hashlib.sha256(package_bytes_path.read_bytes()).hexdigest(),
                "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
                "folds_sha256": hashlib.sha256(folds).hexdigest(),
                "members": {name: hashlib.sha256(raw).hexdigest() for name, raw in members.items()},
            }
            inventory_bytes = (json.dumps(inventory, indent=2, sort_keys=True) + "\n").encode()
            for name, raw in members.items():
                target = input_root.joinpath(*Path(name).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)

            image_root = base / "images"
            series_path = image_root / "synthetic-study" / "synthetic-series"
            series_path.mkdir(parents=True)
            for index, z in enumerate((0, 1), start=1):
                file_meta = FileMetaDataset()
                file_meta.MediaStorageSOPClassUID = CTImageStorage
                file_meta.MediaStorageSOPInstanceUID = generate_uid()
                file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
                dataset = FileDataset(str(series_path / f"slice-{index}.dcm"), {}, file_meta=file_meta, preamble=b"\0" * 128)
                dataset.SOPClassUID = CTImageStorage
                dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
                dataset.StudyInstanceUID = generate_uid()
                dataset.SeriesInstanceUID = generate_uid()
                dataset.Modality = "CT"
                dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
                dataset.ImagePositionPatient = [0, 0, z]
                dataset.PixelSpacing = [0.7, 0.8]
                dataset.Rows = 8
                dataset.Columns = 8
                dataset.SamplesPerPixel = 1
                dataset.PhotometricInterpretation = "MONOCHROME2"
                dataset.BitsAllocated = 16
                dataset.BitsStored = 16
                dataset.HighBit = 15
                dataset.PixelRepresentation = 0
                dataset.PixelData = np.arange(64, dtype=np.uint16).tobytes()
                dataset.save_as(series_path / f"slice-{index}.dcm", enforce_file_format=True)

            artifacts_dir = base / "notebook-artifacts"
            artifacts_dir.mkdir()
            config = {
                "expected_member_count": len(members),
                "expected_manifest_sha256": inventory["manifest_sha256"],
                "expected_folds_sha256": inventory["folds_sha256"],
                "private_input_root": str(input_root), "manifest_relative_path": MANIFEST_REL,
                "folds_relative_path": FOLDS_REL, "image_root": str(image_root),
                "output_dir": str(base / "canary-output"), "fold": 0,
                "expected_training_count": 1, "expected_validation_count": 0,
                "target_series_ordinal": 1, "expected_target_file_count": 2,
                "output_size": 256, "max_slices": 8, "max_seconds": 60,
            }
            paths = build_canary_notebook(
                artifacts_dir, inventory_bytes, config,
                runtime_helper_path=Path("scripts/kaggle_preprocessing_canary_runtime.py"),
                manifest_helper_path=Path("src/rsna_knee/data/orientation_audit.py"),
            )
            code = "".join(json.loads(paths["notebook_path"].read_text())["cells"][0]["source"])
            enabled_path = base / "enabled-canary.py"
            enabled_path.write_text(code.replace("RUN_CANARY = False", "RUN_CANARY = True", 1), encoding="utf-8")
            env = dict(os.environ)
            env.pop("PYTHONPATH", None)
            result = subprocess.run(
                [sys.executable, str(enabled_path)], cwd=base, env=env,
                capture_output=True, text=True, timeout=60, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            event = json.loads(result.stdout.strip().splitlines()[-1])
            self.assertEqual(event["status"], "completed")
            self.assertEqual(event["series_ordinal"], 1)
            self.assertEqual(event["input_file_count"], 2)
            self.assertTrue(event["finite_values"])
            saved = json.loads((base / "canary-output" / "preprocessing-summary.json").read_text())
            self.assertEqual(saved, event)
            self.assertNotIn("synthetic-study", result.stdout + result.stderr)
            self.assertNotIn("synthetic-series", result.stdout + result.stderr)

            changed = pydicom.dcmread(series_path / "slice-2.dcm")
            orientation = list(changed.ImageOrientationPatient)
            orientation[4] += 1.0001e-5
            changed.ImageOrientationPatient = orientation
            changed.save_as(series_path / "slice-2.dcm", enforce_file_format=True)
            failure_config = dict(config)
            failure_config["output_dir"] = str(base / "failed-output")
            failure_artifacts = base / "failure-notebook-artifacts"
            failure_artifacts.mkdir()
            failure_paths = build_canary_notebook(
                failure_artifacts, inventory_bytes, failure_config,
                runtime_helper_path=Path("scripts/kaggle_preprocessing_canary_runtime.py"),
                manifest_helper_path=Path("src/rsna_knee/data/orientation_audit.py"),
            )
            failure_code = "".join(json.loads(failure_paths["notebook_path"].read_text())["cells"][0]["source"])
            failure_path = base / "enabled-canary-failure.py"
            failure_path.write_text(failure_code.replace("RUN_CANARY = False", "RUN_CANARY = True", 1), encoding="utf-8")
            failure_result = subprocess.run(
                [sys.executable, str(failure_path)], cwd=base, env=env,
                capture_output=True, text=True, timeout=60, check=False,
            )
            self.assertNotEqual(failure_result.returncode, 0)
            failure_event = json.loads(failure_result.stdout.strip().splitlines()[-1])
            self.assertEqual(failure_event, {"reason_code": "inconsistent_orientation", "status": "failed"})
            self.assertFalse((base / "failed-output").exists())
            self.assertNotIn("synthetic-study", failure_result.stdout + failure_result.stderr)


if __name__ == "__main__":
    unittest.main()
