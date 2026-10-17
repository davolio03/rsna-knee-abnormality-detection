import base64
import hashlib
import json
import contextlib
import io
import unittest
from pathlib import Path
import tempfile


REPOSITORY = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (REPOSITORY / "data/private/kaggle-first-fold-background/"
                 "orientation-audit-run01/kaggle_first_fold_orientation_audit_run01.ipynb")
HELPER_PATH = REPOSITORY / "src/rsna_knee/data/orientation_audit.py"


class KaggleOrientationAuditNotebookTests(unittest.TestCase):
    def test_notebook_is_output_free_disabled_and_syntax_valid(self):
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(len(code_cells), 1)
        self.assertTrue(all(not cell.get("outputs") for cell in notebook["cells"]))
        self.assertTrue(all(cell.get("execution_count") is None for cell in code_cells))
        code = "".join(code_cells[0]["source"])
        self.assertIn("RUN_AUDIT = False", code)
        self.assertNotIn("RUN_AUDIT = True", code)
        self.assertIn("/kaggle/working/orientation-delta-audit/run01", code)
        self.assertNotIn("torch", code.lower())
        self.assertNotIn("cuda", code.lower())
        self.assertNotIn("train_dataset", code)
        self.assertNotIn("PixelData", code)
        self.assertNotIn("pixel_array", code)
        compile(code, str(NOTEBOOK_PATH), "exec")

    def test_embedded_audit_helper_matches_tested_source_exactly(self):
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        code = "".join("".join(cell["source"]) for cell in notebook["cells"]
                       if cell["cell_type"] == "code")
        namespace = {}
        exec(compile(code, str(NOTEBOOK_PATH), "exec"), namespace)
        helper_bytes = base64.b64decode(namespace["AUDIT_HELPER_B64"], validate=True)
        expected_bytes = HELPER_PATH.read_bytes()
        self.assertEqual(helper_bytes, expected_bytes)
        self.assertEqual(hashlib.sha256(helper_bytes).hexdigest(),
                         namespace["EXPECTED_AUDIT_HELPER_SHA256"])

    def test_disabled_notebook_does_not_create_output_or_read_private_input(self):
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        code = "".join("".join(cell["source"]) for cell in notebook["cells"]
                       if cell["cell_type"] == "code")
        namespace = {"__name__": "audit_notebook_test"}
        exec(compile(code, str(NOTEBOOK_PATH), "exec"), namespace)
        self.assertFalse(namespace["RUN_AUDIT"])
        self.assertNotIn("RESULT", namespace)

    def test_series_order_is_manifest_series_then_training_preprocessor_filename_order(self):
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        code = "".join("".join(cell["source"]) for cell in notebook["cells"]
                       if cell["cell_type"] == "code")
        namespace = {"__name__": "audit_order_test"}
        exec(compile(code, str(NOTEBOOK_PATH), "exec"), namespace)
        namespace["AUDIT_CONFIG"]["expected_training_count"] = 3
        namespace["AUDIT_CONFIG"]["expected_validation_count"] = 1
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for study, series, names in (
                ("study-b", "series-2", ("b.DCM", "a.dcm", "note.txt")),
                ("study-b", "series-1", ("c.dcm",)),
                ("study-a", "series-3", ("d.dcm",)),
                ("study-c", "series-4", ("e.dcm",)),
            ):
                folder = root / study / series
                folder.mkdir(parents=True)
                for name in names:
                    (folder / name).write_bytes(b"not a DICOM; paths only")
            records = [
                ("study-b", ("series-2", "series-1")),
                ("study-a", ("series-3",)),
                ("study-c", ("series-4",)),
                ("validation-only", ("series-v",)),
            ]
            assignments = {"study-b": 1, "study-a": 2, "study-c": 1, "validation-only": 0}
            groups = list(namespace["_series_groups"](records, assignments, root))
            self.assertEqual([[p.name for p in group] for group in groups],
                             [["a.dcm", "b.DCM"], ["c.dcm"], ["d.dcm"], ["e.dcm"]])

    def test_notebook_input_preflight_precedes_embedded_helper_execution(self):
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        code = "".join("".join(cell["source"]) for cell in notebook["cells"]
                       if cell["cell_type"] == "code")
        self.assertLess(code.index("member_count=_verify_input(root)"),
                        code.index('exec(compile(source.decode("utf-8")'))

    def test_enabled_failures_emit_only_sanitized_reason_then_raise(self):
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        code = "".join("".join(cell["source"]) for cell in notebook["cells"]
                       if cell["cell_type"] == "code")
        namespace = {"__name__": "audit_failure_test"}
        exec(compile(code, str(NOTEBOOK_PATH), "exec"), namespace)

        for error, expected in ((ValueError("series_files_missing"), "series_files_missing"),
                                (ValueError("/private/path/patient-123"), "audit_failed"),
                                (RuntimeError("reader failed for /private/path/patient-123"), "audit_failed")):
            with self.subTest(expected=expected):
                def fail(error=error):
                    raise error
                namespace["_run_audit"] = fail
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    with self.assertRaisesRegex(RuntimeError, f"^{expected}$"):
                        namespace["_run_enabled_audit"]()
                self.assertEqual(json.loads(output.getvalue()),
                                 {"status": "blocked", "reason_code": expected})
                self.assertNotIn("/private/path", output.getvalue())
                self.assertNotIn("patient-123", output.getvalue())

    def test_absent_private_mount_prints_safe_failure_then_raises(self):
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        code = "".join("".join(cell["source"]) for cell in notebook["cells"]
                       if cell["cell_type"] == "code")
        namespace = {"__name__": "audit_absent_mount_test"}
        exec(compile(code, str(NOTEBOOK_PATH), "exec"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            absent = Path(directory) / "not-mounted"
            namespace["AUDIT_CONFIG"]["private_input_root"] = str(absent)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                with self.assertRaisesRegex(RuntimeError, "^input_unavailable$"):
                    namespace["_run_enabled_audit"]()
        self.assertEqual(json.loads(output.getvalue()),
                         {"status": "blocked", "reason_code": "input_unavailable"})
        self.assertNotIn(str(absent), output.getvalue())

    def test_empty_series_directory_is_not_reported_as_complete(self):
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        code = "".join("".join(cell["source"]) for cell in notebook["cells"]
                       if cell["cell_type"] == "code")
        namespace = {"__name__": "audit_empty_series_test"}
        exec(compile(code, str(NOTEBOOK_PATH), "exec"), namespace)
        namespace["AUDIT_CONFIG"]["expected_training_count"] = 1
        namespace["AUDIT_CONFIG"]["expected_validation_count"] = 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty = root / "study" / "series"
            empty.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "^series_files_missing$"):
                list(namespace["_series_groups"]([("study", ("series",))], {"study": 1}, root))


if __name__ == "__main__":
    unittest.main()
