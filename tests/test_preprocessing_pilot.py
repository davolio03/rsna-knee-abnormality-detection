"""Synthetic coverage for the bounded private preprocessing pilot harness."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("optional NumPy dependency is not installed") from exc

from rsna_knee.data.dicom_preprocessing import DicomPreprocessedSeries, DicomPreprocessingError
from rsna_knee.data.preprocessing_pilot import run_preprocessing_pilot


def _manifest(studies: int = 7, series_per_study: int = 6):
    from rsna_knee.data.cohort import REPORT_NORMALIZATION_VERSION
    from rsna_knee.labels.subscription import LABEL_COLUMNS

    versions = {"legacy": 0, "c2-literal-evidence-v1": studies,
                "c2-literal-evidence-v2": 0, "c2-literal-evidence-v3": 0}
    return {
        "schema_version": "baseline-training-cohort-v1",
        "configuration": {
            "report_group_normalization": REPORT_NORMALIZATION_VERSION,
            "cohort_size": studies,
        },
        "source_sha256": {key: "a" * 64 for key in (
            "train_csv", "labels_jsonl", "series_csv", "gold_exclusions", "prompt_journal"
        )},
        "prompt_provenance_counts": versions,
        "records": [
            {
                "study_id": f"study-private-{study}",
                "labels": {name: 0 for name in LABEL_COLUMNS},
                "prompt_version": "c2-literal-evidence-v1",
                "normalized_report_sha256": "b" * 64,
                "series": [
                    {"series_id": f"series-private-{study}-{series}",
                     "fluid_sensitive": 0, "fat_suppression": 0,
                     "anatomical_plane": "unknown"}
                    for series in range(series_per_study)
                ],
            }
            for study in range(studies)
        ],
    }


def _result(value: float = 0.25, size: int = 8):
    images = np.full((2, 1, size, size), value, dtype=np.float32)
    mask = np.ones(images.shape, dtype=np.bool_)
    return DicomPreprocessedSeries(
        images,
        mask,
        {
            "slice_count_scanned": 17,
            "slice_count_selected": 2,
            "pp01": {
                "config": {"max_slices": 8},
                "transforms": [
                    {"native_pixel_spacing": [0.7, 0.8]},
                    {"native_pixel_spacing": [0.7, 0.8]},
                ],
            },
        },
    )


class PreprocessingPilotTests(unittest.TestCase):
    def test_limits_to_first_five_studies_all_series_and_eight_slices(self):
        calls = []
        manifest = _manifest()
        manifest["records"][5] = None  # The harness must not inspect any later cohort record.

        def process(path, *, max_slices, output_size):
            calls.append((str(path), max_slices, output_size))
            return _result()

        report = run_preprocessing_pilot(
            manifest, Path("/private/images"), series_preprocessor=process, output_size=8
        )

        self.assertEqual(report["studies_requested"], 5)
        self.assertEqual(report["series_requested"], 30)
        self.assertEqual(report["series_succeeded"], 30)
        self.assertEqual(len(calls), 30)
        self.assertTrue(all(call[1:] == (8, 8) for call in calls))
        self.assertEqual([entry["series_index"] for entry in report["series"]], list(range(1, 31)))
        self.assertTrue(all("reason_code" not in entry for entry in report["series"]))
        self.assertEqual(report["series"][0]["native_pixel_spacing_by_slice"], [[0.7, 0.8]] * 2)

    def test_failures_are_aggregated_without_exception_or_identifier_text(self):
        def process(path, *, max_slices, output_size):
            if "series-private-0-0" in str(path):
                raise DicomPreprocessingError("pixel_decode_failed", "secret file path")
            return _result()

        report = run_preprocessing_pilot(
            _manifest(studies=5), Path("/private/images"),
            series_preprocessor=process, output_size=8,
        )
        rendered = json.dumps(report)
        self.assertEqual(report["series"][0]["status"], "failed")
        self.assertEqual(report["series"][0]["reason_code"], "pixel_decode_failed")
        self.assertNotIn("study-private", rendered)
        self.assertNotIn("series-private", rendered)
        self.assertNotIn("secret file path", rendered)
        self.assertEqual(report["status_counts"], {"failed": 1, "ok": 29})

    def test_invalid_tensors_fail_closed_as_safe_series_status(self):
        def process(path, *, max_slices, output_size):
            output = _result()
            broken = output.images.copy()
            broken[0, 0, 0, 0] = np.nan
            return DicomPreprocessedSeries(broken, output.valid_mask, output.metadata)

        report = run_preprocessing_pilot(
            _manifest(studies=5), Path("/private/images"),
            series_preprocessor=process, output_size=8,
        )
        self.assertEqual(report["series"][0]["status"], "failed")
        self.assertEqual(report["series"][0]["reason_code"], "invalid_tensor")

    def test_writes_id_free_panel_and_json_report_for_every_successful_series(self):
        directory = Path(__file__).parent / "preprocessing-pilot-test-output"
        directory.mkdir(exist_ok=True)
        try:
            report = run_preprocessing_pilot(
                _manifest(studies=5), Path("/private/images"),
                series_preprocessor=lambda path, **kwargs: _result(),
                output_size=8, output_dir=directory,
            )
            paths = sorted(directory.iterdir())
            self.assertEqual([path.name for path in paths], [
                "preprocessing-pilot-montage.png", "preprocessing-pilot-report.json"
            ])
            self.assertEqual(report["panel_series_count"], 30)
            saved = paths[1].read_text(encoding="utf-8")
            self.assertNotIn("study-private", saved)
            self.assertNotIn("series-private", saved)
        finally:
            for path in directory.iterdir():
                path.unlink()
            directory.rmdir()

    def test_montage_keeps_failure_placeholders_and_late_success_visible(self):
        directory = Path(__file__).parent / "preprocessing-pilot-late-success-output"
        directory.mkdir(exist_ok=True)

        def process(path, *, max_slices, output_size):
            if "series-private-4-5" not in str(path):
                raise DicomPreprocessingError("pixel_decode_failed", "private path detail")
            return _result(value=0.25, size=output_size)

        try:
            report = run_preprocessing_pilot(
                _manifest(studies=5), Path("/private/images"),
                series_preprocessor=process, output_size=128, output_dir=directory,
            )
            from PIL import Image

            montage = Image.open(directory / "preprocessing-pilot-montage.png").convert("RGB")
            self.assertEqual(report["series_succeeded"], 1)
            self.assertEqual(report["panel_series_count"], 30)
            self.assertEqual(report["panel_failure_placeholders"], 29)
            self.assertEqual(montage.size, (1536, 770))
            # Series 30 is at the lower-right; its image and mask must remain visible.
            self.assertEqual(montage.getpixel((1344, 706)), (64, 64, 64))
            self.assertEqual(montage.getpixel((1472, 706)), (255, 255, 255))
            # Failed series 1 keeps a visible non-black placeholder tile.
            self.assertGreater(montage.getpixel((5, 40))[0], 0)
            self.assertNotIn("study-private", json.dumps(report))
            self.assertNotIn("private path detail", json.dumps(report))
        finally:
            for path in directory.iterdir():
                path.unlink()
            directory.rmdir()


if __name__ == "__main__":
    unittest.main()
