import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsna_knee.labels.subscription import LABEL_COLUMNS
from rsna_knee.validation.cohort import build_grouped_training_folds
from scripts.prepare_training_folds import main as prepare_main


class TrainingFoldTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = self._manifest()

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def _manifest():
        records = []
        for group_index in range(12):
            studies = (f"study-{group_index}-a", f"study-{group_index}-b") if group_index < 4 else (f"study-{group_index}",)
            for study_index, study_id in enumerate(studies):
                positive = group_index < 4 and study_index == 0
                records.append({
                    "study_id": study_id,
                    "labels": {column: int(positive) for column in LABEL_COLUMNS},
                    "series": [{"series_id": f"series-{study_id}", "fluid_sensitive": 1,
                                "fat_suppression": 1, "anatomical_plane": "Sagittal"}],
                    "prompt_version": "c2-literal-evidence-v1",
                    "normalized_report_sha256": hashlib.sha256(
                        f"report-{group_index}".encode("utf-8")
                    ).hexdigest(),
                })
        return {
            "schema_version": "baseline-training-cohort-v1",
            "configuration": {
                "cohort_size": len(records),
                "report_group_normalization": "python-casefold-collapse-whitespace-retain-punctuation-v1",
            },
            "source_sha256": {name: "a" * 64 for name in (
                "train_csv", "labels_jsonl", "series_csv", "gold_exclusions", "prompt_journal",
            )},
            "prompt_provenance_counts": {"legacy": 0, "c2-literal-evidence-v1": len(records),
                                          "c2-literal-evidence-v2": 0,
                                          "c2-literal-evidence-v3": 0},
            "records": records,
        }

    @staticmethod
    def _manifest_sha(manifest):
        return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()

    def test_allocates_groups_deterministically_and_audits_study_level_coverage(self):
        digest = self._manifest_sha(self.manifest)
        first = build_grouped_training_folds(self.manifest, manifest_sha256=digest)
        second = build_grouped_training_folds(self.manifest, manifest_sha256=digest)
        self.assertEqual(first, second)
        self.assertEqual(first["manifest_sha256"], digest)
        self.assertEqual(first["configuration"]["seed"], 42)
        self.assertEqual(first["configuration"]["n_splits"], 3)
        self.assertEqual(first["group_count"], 12)
        self.assertEqual(len(first["study_assignments"]), len(self.manifest["records"]))
        assignments = {item["study_id"]: item["fold"] for item in first["study_assignments"]}
        groups = {}
        for record in self.manifest["records"]:
            groups.setdefault(record["normalized_report_sha256"], set()).add(assignments[record["study_id"]])
        self.assertTrue(all(len(folds) == 1 for folds in groups.values()))
        self.assertEqual(first["audit"]["group_overlap_count"], 0)
        for fold_audit in first["audit"]["folds"]:
            for target in LABEL_COLUMNS:
                self.assertGreater(fold_audit["train"]["positive_counts"][target], 0)
                self.assertGreater(fold_audit["train"]["negative_counts"][target], 0)
                self.assertGreater(fold_audit["validation"]["positive_counts"][target], 0)
                self.assertGreater(fold_audit["validation"]["negative_counts"][target], 0)

    def test_group_or_labels_keep_discordant_studies_together(self):
        from rsna_knee.validation import cohort

        with patch.object(cohort, "iterative_multilabel_folds",
                          wraps=cohort.iterative_multilabel_folds) as allocator:
            artifact = build_grouped_training_folds(
                self.manifest, manifest_sha256=self._manifest_sha(self.manifest),
            )
        group_targets = allocator.call_args.args[1]
        self.assertEqual(group_targets[0], [1] * len(LABEL_COLUMNS))
        self.assertEqual(group_targets[4], [0] * len(LABEL_COLUMNS))
        assignments = {item["study_id"]: item["fold"] for item in artifact["study_assignments"]}
        self.assertEqual(assignments["study-0-a"], assignments["study-0-b"])
        self.assertEqual(artifact["group_assignments"][0]["study_ids"],
                         ["study-0-a", "study-0-b"])

    def test_fails_closed_when_rare_target_cannot_have_both_classes_in_every_fold(self):
        sparse = self._manifest()
        for record in sparse["records"]:
            record["labels"][LABEL_COLUMNS[0]] = 0
        sparse["records"][0]["labels"][LABEL_COLUMNS[0]] = 1
        sparse["records"][1]["labels"][LABEL_COLUMNS[0]] = 1
        with self.assertRaises(ValueError):
            build_grouped_training_folds(sparse, manifest_sha256=self._manifest_sha(sparse))

    def test_rejects_duplicate_ids_invalid_binary_labels_bad_hashes_and_unknown_provenance(self):
        cases = []
        duplicate = self._manifest()
        duplicate["records"][1]["study_id"] = duplicate["records"][0]["study_id"]
        cases.append(duplicate)
        invalid_label = self._manifest()
        invalid_label["records"][0]["labels"][LABEL_COLUMNS[0]] = True
        cases.append(invalid_label)
        invalid_group = self._manifest()
        invalid_group["records"][0]["normalized_report_sha256"] = "not-a-hash"
        cases.append(invalid_group)
        invalid_prompt = self._manifest()
        invalid_prompt["records"][0]["prompt_version"] = "unknown-prompt"
        cases.append(invalid_prompt)
        invalid_schema = self._manifest()
        invalid_schema["schema_version"] = "other"
        cases.append(invalid_schema)
        for manifest in cases:
            with self.subTest(manifest=manifest["schema_version"]):
                with self.assertRaises(ValueError):
                    build_grouped_training_folds(manifest, manifest_sha256=self._manifest_sha(manifest))

    def test_private_writer_refuses_outside_path_and_existing_output(self):
        from rsna_knee.validation.cohort import write_private_folds

        artifact = build_grouped_training_folds(self.manifest, manifest_sha256=self._manifest_sha(self.manifest))
        with self.assertRaises(ValueError):
            write_private_folds(artifact, output_path=self.root / "outside.json", project_root=self.root)
        output = self.root / "data" / "private" / "folds.json"
        write_private_folds(artifact, output_path=output, project_root=self.root)
        contents = output.read_bytes()
        with self.assertRaises(FileExistsError):
            write_private_folds(artifact, output_path=output, project_root=self.root)
        self.assertEqual(output.read_bytes(), contents)

    def test_cli_prints_aggregates_only_and_writes_private_assignments(self):
        manifest_path = self.root / "data" / "private" / "cohort.json"
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        output = self.root / "data" / "private" / "folds.json"
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("scripts.prepare_training_folds.REPOSITORY_ROOT", self.root):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                status = prepare_main(["--manifest", str(manifest_path), "--output", str(output)])
        self.assertEqual(status, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertNotIn("study-", stdout.getvalue())
        self.assertNotIn(str(output), stdout.getvalue())
        saved = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(len(saved["study_assignments"]), len(self.manifest["records"]))


if __name__ == "__main__":
    unittest.main()
