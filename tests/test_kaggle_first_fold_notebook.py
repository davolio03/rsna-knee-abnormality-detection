import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
import sys
from types import SimpleNamespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch


REPOSITORY = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPOSITORY / "notebooks/kaggle_first_fold_baseline.ipynb"
CONFIG_PATH = REPOSITORY / "configs/kaggle_first_fold_baseline.json"
IGNORED_LAUNCH_PATH = REPOSITORY / "data/private/kaggle-first-fold-baseline/launch_cell.py"


def load_notebook_code():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    if len(cells) != 1:
        raise AssertionError("expected one standalone launch cell")
    code = "".join(cells[0]["source"])
    namespace = {"__name__": "first_fold_test_module"}
    exec(compile(code, str(NOTEBOOK_PATH), "exec"), namespace)
    return notebook, code, namespace


class KaggleFirstFoldNotebookTests(unittest.TestCase):
    def make_package(self, root):
        members = {}
        for index in range(22):
            relative = f"src/package/module_{index:02d}.py"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"value={index}\n", encoding="utf-8")
            members[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        private_hashes = {}
        for relative, value in (
            ("data/private/500-study-training-cohort.json", b"manifest"),
            ("data/private/500-study-training-folds.json", b"folds"),
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
            private_hashes[relative] = hashlib.sha256(value).hexdigest()
            members[relative] = private_hashes[relative]
        inventory = {
            "schema_version": "kaggle-training-package-inventory-v1",
            "package_sha256": "a" * 64,
            "manifest_sha256": private_hashes["data/private/500-study-training-cohort.json"],
            "folds_sha256": private_hashes["data/private/500-study-training-folds.json"],
            "members": members,
        }
        raw_inventory = json.dumps(inventory, sort_keys=True).encode("utf-8")
        return raw_inventory, private_hashes

    def test_notebook_is_clean_output_free_and_defaults_to_plan_only(self):
        notebook, code, _ = load_notebook_code()
        self.assertEqual(notebook["nbformat"], 4)
        self.assertEqual(notebook["nbformat_minor"], 5)
        self.assertTrue(all(not cell.get("outputs") for cell in notebook["cells"]))
        self.assertTrue(all(cell.get("execution_count") is None for cell in notebook["cells"]
                            if cell["cell_type"] == "code"))
        self.assertIn("RUN_TRAINING = False", code)
        self.assertNotIn("RUN_GPU_PROBE", code)
        self.assertNotIn("RUN_TRAINING = True", code)
        self.assertNotIn("pip install", code.lower())
        self.assertNotIn("wget ", code.lower())
        self.assertNotIn("requests.", code.lower())
        self.assertNotIn("kaggle_eight_study_overfit", code)
        self.assertNotIn("train_loss_by_step", code)
        self.assertLess(code.index("audit = audit_package(input_root)"), code.index("sys.path.insert"))
        self.assertNotIn('report["rows"]', code)
        self.assertNotIn('report["labels"]', code)
        self.assertIn("raise RuntimeError(reason) from None", code)
        self.assertIn("SAFE_REASON_CODES = frozenset", code)

    def test_ignored_launch_cell_is_the_exact_notebook_code_cell(self):
        _, code, _ = load_notebook_code()
        launch = IGNORED_LAUNCH_PATH.read_text(encoding="utf-8")
        self.assertTrue(launch.startswith('"""Paste this complete cell'))
        self.assertEqual(launch.split("\n", 1)[1], code)

    def test_public_config_matches_notebook_frozen_trial(self):
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        _, _, namespace = load_notebook_code()
        self.assertEqual(config, namespace["RUN_CONFIG"])
        self.assertEqual(config["mode"], "fold")
        self.assertEqual((config["fold"], config["seed"]), (0, 42))
        self.assertEqual((config["epochs"], config["max_steps"]), (5, None))
        self.assertIsNone(config["overfit_limit"])
        self.assertEqual(config["output_size"], 256)
        self.assertEqual(config["max_slices"], 8)
        self.assertEqual(config["feature_width"], 64)
        self.assertEqual(config["learning_rate"], 0.001)
        self.assertEqual(config["device"], "cuda")
        self.assertEqual(config["expected_training_count"], 333)
        self.assertEqual(config["expected_validation_count"], 167)
        self.assertEqual(config["expected_steps"], 1665)
        self.assertEqual(config["output_dir"], "/kaggle/working/kaggle-first-fold-baseline/run01")

    def test_embedded_inventory_audits_all_members_without_remote_inventory(self):
        _, _, namespace = load_notebook_code()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inventory_bytes, private_hashes = self.make_package(root)
            result = namespace["audit_package"](
                root,
                inventory_bytes=inventory_bytes,
                expected_inventory_sha256=hashlib.sha256(inventory_bytes).hexdigest(),
                expected_manifest_sha256=private_hashes["data/private/500-study-training-cohort.json"],
                expected_folds_sha256=private_hashes["data/private/500-study-training-folds.json"],
                expected_archive_provenance="a" * 64,
            )
        self.assertEqual(result["validated_member_count"], 24)
        self.assertTrue(result["manifest_matches"])
        self.assertTrue(result["folds_match"])
        self.assertFalse(result["remote_archive_bytes_verified"])

    def test_unlisted_remote_file_fails_membership_guard(self):
        _, _, namespace = load_notebook_code()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inventory_bytes, private_hashes = self.make_package(root)
            (root / "unlisted.txt").write_text("not part of the pinned package", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "package_membership_mismatch"):
                namespace["audit_package"](
                    root,
                    inventory_bytes=inventory_bytes,
                    expected_inventory_sha256=hashlib.sha256(inventory_bytes).hexdigest(),
                    expected_manifest_sha256=private_hashes["data/private/500-study-training-cohort.json"],
                    expected_folds_sha256=private_hashes["data/private/500-study-training-folds.json"],
                    expected_archive_provenance="a" * 64,
                )

    def test_member_hash_and_fixed_partition_guards_fail_closed(self):
        _, _, namespace = load_notebook_code()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inventory_bytes, private_hashes = self.make_package(root)
            (root / "src/package/module_00.py").write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "package_member_hash_mismatch"):
                namespace["audit_package"](
                    root,
                    inventory_bytes=inventory_bytes,
                    expected_inventory_sha256=hashlib.sha256(inventory_bytes).hexdigest(),
                    expected_manifest_sha256=private_hashes["data/private/500-study-training-cohort.json"],
                    expected_folds_sha256=private_hashes["data/private/500-study-training-folds.json"],
                    expected_archive_provenance="a" * 64,
                )
        valid_plan = {
            "status": "plan_only", "mode": "fold", "fold": 0,
            "training_study_count": 333, "validation_study_count": 167,
            "training_steps_planned": 1665,
            "input_sha256": {
                "manifest": namespace["RUN_CONFIG"]["expected_manifest_sha256"],
                "folds": namespace["RUN_CONFIG"]["expected_folds_sha256"],
            },
        }
        with self.assertRaisesRegex(RuntimeError, "plan_count_mismatch"):
            namespace["validate_plan"]({**valid_plan, "validation_study_count": 166})

    def test_plan_gate_never_fits_before_full_count_and_hash_validation(self):
        _, _, namespace = load_notebook_code()
        expected_hashes = {
            "manifest": namespace["RUN_CONFIG"]["expected_manifest_sha256"],
            "folds": namespace["RUN_CONFIG"]["expected_folds_sha256"],
        }
        plan = {
            "status": "plan_only", "mode": "fold", "fold": 0,
            "training_study_count": 333, "validation_study_count": 167,
            "input_sha256": expected_hashes,
        }
        calls = []

        def runner(_config, *, execute):
            calls.append(execute)
            if not execute:
                return plan
            return {"status": "completed", "steps_completed": 1665}

        checked_plan, result = namespace["run_with_plan_gate"](
            runner, namespace["RUN_CONFIG"], run_enabled=False)
        self.assertEqual(checked_plan, plan)
        self.assertIsNone(result)
        self.assertEqual(calls, [False])

        calls.clear()
        with self.assertRaisesRegex(RuntimeError, "plan_count_mismatch"):
            namespace["run_with_plan_gate"](
                lambda _config, *, execute: calls.append(execute) or {
                    **plan, "training_study_count": 332,
                },
                namespace["RUN_CONFIG"], run_enabled=True)
        self.assertEqual(calls, [False])

    def test_preflight_failure_is_reported_and_raises_for_batch_failure(self):
        _, _, namespace = load_notebook_code()
        output = io.StringIO()
        with patch.object(Path, "is_dir", return_value=True), \
                patch.dict(namespace, {"audit_package": unittest.mock.Mock(
                    side_effect=RuntimeError("package_member_hash_mismatch"))}), \
                redirect_stdout(output):
            with self.assertRaisesRegex(RuntimeError, "package_member_hash_mismatch"):
                namespace["main"]()
        diagnostic = output.getvalue()
        self.assertIn('"reason_code": "package_member_hash_mismatch"', diagnostic)
        self.assertNotIn("Traceback", diagnostic)
        self.assertNotIn("/kaggle/input", diagnostic)

        output = io.StringIO()
        with patch.object(Path, "is_dir", return_value=True), \
                patch.dict(namespace, {"audit_package": unittest.mock.Mock(
                    side_effect=RuntimeError("private/path/study-id"))}), \
                redirect_stdout(output):
            with self.assertRaisesRegex(RuntimeError, "runtime_failed"):
                namespace["main"]()
        self.assertIn('"reason_code": "runtime_failed"', output.getvalue())
        self.assertNotIn("study-id", output.getvalue())

    def test_training_failure_emits_only_allowlisted_diagnostics(self):
        _, _, namespace = load_notebook_code()
        error = RuntimeError("private path and study identifier must not escape")
        error.reason_code = "training_failed"
        error.diagnostics = {
            "stage": "model_initialization",
            "exception_family": "runtime", "exception_chain": ["runtime", "operating_system"],
            "source": {"module": "training.baseline", "function": "run_training", "line": 321},
        }
        output = io.StringIO()
        torch_stub = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        with patch.object(Path, "is_dir", return_value=True), \
                patch.dict(sys.modules, {"torch": torch_stub}), \
                patch.dict(namespace, {
                    "audit_package": lambda _root: {
                        "validated_member_count": 24, "manifest_matches": True,
                        "folds_match": True, "remote_archive_bytes_verified": False,
                    },
                    "run_with_plan_gate": Mock(side_effect=error),
                }), redirect_stdout(output):
            with self.assertRaisesRegex(RuntimeError, "training_failed"):
                namespace["main"]()
        event = json.loads(output.getvalue())
        self.assertEqual(event["reason_code"], "training_failed")
        self.assertEqual(event["diagnostics"]["stage"], "model_initialization")
        self.assertEqual(event["diagnostics"]["exception_chain"], ["runtime", "operating_system"])
        self.assertEqual(event["diagnostics"]["source"]["line"], 321)
        self.assertNotIn("private path", output.getvalue())
        self.assertNotIn("study identifier", output.getvalue())

    def test_malformed_failure_diagnostics_are_omitted_not_echoed(self):
        _, _, namespace = load_notebook_code()
        error = RuntimeError("private path and study identifier must not escape")
        error.reason_code = "training_failed"
        error.diagnostics = {
            "stage": "training", "exception_family": "runtime",
            "exception_chain": ["runtime"], "unsafe": "private/path/study-id",
        }
        output = io.StringIO()
        torch_stub = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        with patch.object(Path, "is_dir", return_value=True), \
                patch.dict(sys.modules, {"torch": torch_stub}), \
                patch.dict(namespace, {
                    "audit_package": lambda _root: {
                        "validated_member_count": 24, "manifest_matches": True,
                        "folds_match": True, "remote_archive_bytes_verified": False,
                    },
                    "run_with_plan_gate": Mock(side_effect=error),
                }), redirect_stdout(output):
            with self.assertRaisesRegex(RuntimeError, "training_failed"):
                namespace["main"]()
        event = json.loads(output.getvalue())
        self.assertNotIn("diagnostics", event)
        self.assertNotIn("private path", output.getvalue())
        self.assertNotIn("study identifier", output.getvalue())
        self.assertNotIn("private/path/study-id", output.getvalue())

    def test_partial_execution_report_is_not_treated_as_success(self):
        _, _, namespace = load_notebook_code()
        plan = {
            "status": "plan_only", "mode": "fold", "fold": 0,
            "training_study_count": 333, "validation_study_count": 167,
            "input_sha256": {
                "manifest": namespace["RUN_CONFIG"]["expected_manifest_sha256"],
                "folds": namespace["RUN_CONFIG"]["expected_folds_sha256"],
            },
        }
        calls = []

        def runner(_config, *, execute):
            calls.append(execute)
            return plan if not execute else {
                "status": "partial_training", "steps_completed": 1664,
            }

        with self.assertRaisesRegex(RuntimeError, "training_report_incomplete"):
            namespace["run_with_plan_gate"](
                runner, namespace["RUN_CONFIG"], run_enabled=True)
        self.assertEqual(calls, [False, True])

    def test_complete_report_contract_matches_runner_fold_result(self):
        _, _, namespace = load_notebook_code()
        plan = {
            "status": "plan_only", "mode": "fold", "fold": 0,
            "training_study_count": 333, "validation_study_count": 167,
            "input_sha256": {
                "manifest": namespace["RUN_CONFIG"]["expected_manifest_sha256"],
                "folds": namespace["RUN_CONFIG"]["expected_folds_sha256"],
            },
        }
        report = {
            **plan,
            "status": "completed",
            "training_steps_planned": 1665,
            "steps_completed": 1665,
            "model_kind": "scratch_study_cnn",
            "per_target_auc": {f"target_{index}": 0.5 for index in range(12)},
            "macro_auc": 0.5,
            "missing_class_targets": [],
            "runtime": {"peak_cuda_memory_bytes": 1},
            "interpretation_limit": "Single development fold.",
        }
        calls = []

        def runner(_config, *, execute):
            calls.append(execute)
            return plan if not execute else report

        checked_plan, checked_report = namespace["run_with_plan_gate"](
            runner, namespace["RUN_CONFIG"], run_enabled=True)
        self.assertEqual(checked_plan, plan)
        self.assertEqual(checked_report, report)
        self.assertEqual(calls, [False, True])


if __name__ == "__main__":
    unittest.main()
