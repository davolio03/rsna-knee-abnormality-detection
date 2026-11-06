"""Tests for the read-only workflow inventory validator."""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.check_workflow_inventory import InventoryError, validate_inventory


class WorkflowInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "scripts").mkdir()
        (self.root / "notebooks").mkdir()
        (self.root / "src").mkdir()
        (self.root / "tests").mkdir()
        (self.root / "scripts" / "run.py").write_text("print('not executed')\n")
        (self.root / "notebooks" / "demo.ipynb").write_text('{"cells": []}\n')
        (self.root / "src" / "helper.py").write_text("VALUE = 1\n")
        (self.root / "tests" / "test_example.py").write_text("pass\n")
        self.catalog = self.root / "inventory.csv"
        self.rows = [
            {
                "path": "scripts/run.py",
                "role": "current",
                "lifecycle": "current",
                "purpose": "A test workflow.",
                "known_consumers": "",
                "evidence": "scripts/run.py#main",
                "compatibility_notes": "Preserved for the fixture.",
            },
            {
                "path": "notebooks/demo.ipynb",
                "role": "historical",
                "lifecycle": "historical",
                "purpose": "A test notebook.",
                "known_consumers": "scripts/run.py",
                "evidence": "src/helper.py",
                "compatibility_notes": "Retained as fixture evidence.",
            },
        ]
        self.sources = ["scripts/run.py", "notebooks/demo.ipynb"]
        self.write_rows()

    def tearDown(self):
        self.temp.cleanup()

    def write_rows(self, rows=None):
        with self.catalog.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows if rows is None else rows)

    def test_accepts_exact_coverage_and_references(self):
        summary = validate_inventory(self.root, self.catalog, source_paths=self.sources)
        self.assertEqual(summary["sources"], 2)
        self.assertEqual(summary["roles"]["current"], 1)
        self.assertEqual(summary["roles"]["historical"], 1)

    def test_rejects_missing_duplicate_and_extra_catalog_rows(self):
        self.write_rows(self.rows[:1])
        with self.assertRaisesRegex(InventoryError, "missing"):
            validate_inventory(self.root, self.catalog, source_paths=self.sources)
        self.write_rows(self.rows + [self.rows[0]])
        with self.assertRaisesRegex(InventoryError, "duplicate"):
            validate_inventory(self.root, self.catalog, source_paths=self.sources)
        extra = dict(self.rows[0], path="scripts/other.py")
        (self.root / "scripts" / "other.py").write_text("pass\n")
        self.write_rows(self.rows + [extra])
        with self.assertRaisesRegex(InventoryError, "extra"):
            validate_inventory(self.root, self.catalog, source_paths=self.sources)

    def test_rejects_invalid_fields_paths_and_missing_references(self):
        bad = dict(self.rows[0], role="unknown")
        self.write_rows([bad, self.rows[1]])
        with self.assertRaisesRegex(InventoryError, "role"):
            validate_inventory(self.root, self.catalog, source_paths=self.sources)
        bad = dict(self.rows[0], lifecycle="unknown")
        self.write_rows([bad, self.rows[1]])
        with self.assertRaisesRegex(InventoryError, "lifecycle"):
            validate_inventory(self.root, self.catalog, source_paths=self.sources)
        bad = dict(self.rows[0], compatibility_notes=" ")
        self.write_rows([bad, self.rows[1]])
        with self.assertRaisesRegex(InventoryError, "nonblank"):
            validate_inventory(self.root, self.catalog, source_paths=self.sources)
        bad = dict(self.rows[0], purpose="")
        self.write_rows([bad, self.rows[1]])
        with self.assertRaisesRegex(InventoryError, "nonblank"):
            validate_inventory(self.root, self.catalog, source_paths=self.sources)
        bad = dict(self.rows[0], evidence="")
        self.write_rows([bad, self.rows[1]])
        with self.assertRaisesRegex(InventoryError, "nonblank"):
            validate_inventory(self.root, self.catalog, source_paths=self.sources)
        bad = dict(self.rows[0], evidence=" ; ")
        self.write_rows([bad, self.rows[1]])
        with self.assertRaisesRegex(InventoryError, "at least one"):
            validate_inventory(self.root, self.catalog, source_paths=self.sources)
        bad = dict(self.rows[0], path="scripts/../src/helper.py")
        self.write_rows([bad, self.rows[1]])
        with self.assertRaisesRegex(InventoryError, "canonical"):
            validate_inventory(self.root, self.catalog, source_paths=self.sources)
        bad = dict(self.rows[0], evidence="src/missing.py")
        self.write_rows([bad, self.rows[1]])
        with self.assertRaisesRegex(InventoryError, "does not exist"):
            validate_inventory(self.root, self.catalog, source_paths=self.sources)
        bad = dict(self.rows[0], known_consumers="tests/test_example.py")
        self.write_rows([bad, self.rows[1]])
        with self.assertRaisesRegex(InventoryError, "consumer"):
            validate_inventory(self.root, self.catalog, source_paths=self.sources)

    def test_rejects_schema_drift(self):
        self.catalog.write_text("path,role\nscripts/run.py,current\n", encoding="utf-8")
        with self.assertRaisesRegex(InventoryError, "columns"):
            validate_inventory(self.root, self.catalog, source_paths=self.sources)

    def test_git_source_discovery_failure_is_not_empty_success(self):
        with mock.patch("scripts.check_workflow_inventory.subprocess.run", side_effect=OSError("git unavailable")):
            with self.assertRaisesRegex(InventoryError, "git ls-files"):
                validate_inventory(self.root, self.catalog)


if __name__ == "__main__":
    unittest.main()
