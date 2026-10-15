import hashlib
import json
import os
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsna_knee.data.dicom_benchmark import (
    DicomBenchmarkError,
    run_dicom_benchmark,
)
from rsna_knee.labels.subscription import LABEL_COLUMNS


class FakePixels:
    def __init__(self, shape=(2, 3), nbytes=12):
        self.shape = shape
        self.nbytes = nbytes
        self.ndim = len(shape)
        self.size = 1
        for value in shape:
            self.size *= value


class FakeDataset:
    def __init__(self, *, study="study-1", series="series-a", position=(0, 0, 0),
                 orientation=(1, 0, 0, 0, 1, 0), spacing=(0.5, 0.5), rows=2,
                 columns=3, samples=1, photo="MONOCHROME2", frames=1,
                 pixel_array=None, compressed=False):
        self.StudyInstanceUID = study
        self.SeriesInstanceUID = series
        self.ImagePositionPatient = position
        self.ImageOrientationPatient = orientation
        self.PixelSpacing = spacing
        self.Rows = rows
        self.Columns = columns
        self.SamplesPerPixel = samples
        self.PhotometricInterpretation = photo
        self.NumberOfFrames = frames
        self.pixel_array = pixel_array or FakePixels()
        self.file_meta = SimpleNamespace(
            TransferSyntaxUID=SimpleNamespace(is_compressed=compressed),
        )


class DicomBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.image_root = self.root / "images"
        self.image_root.mkdir()
        self.manifest = self._manifest()
        self.header_data = {}
        self.pixel_data = {}
        self.pixel_reads = []
        self._dependencies()

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def _manifest():
        records = []
        for study, series_ids in (("study-1", ["series-b", "series-a"]),
                                  ("study-2", ["series-c"])):
            records.append({
                "study_id": study,
                "labels": {target: 0 for target in LABEL_COLUMNS},
                "series": [{"series_id": series_id, "fluid_sensitive": 1,
                            "fat_suppression": 1, "anatomical_plane": "Sagittal"}
                           for series_id in series_ids],
                "prompt_version": "c2-literal-evidence-v1",
                "normalized_report_sha256": hashlib.sha256(study.encode()).hexdigest(),
            })
        return {
            "schema_version": "baseline-training-cohort-v1",
            "configuration": {"cohort_size": 2,
                              "report_group_normalization": "python-casefold-collapse-whitespace-retain-punctuation-v1"},
            "source_sha256": {name: "a" * 64 for name in (
                "train_csv", "labels_jsonl", "series_csv", "gold_exclusions", "prompt_journal",
            )},
            "prompt_provenance_counts": {"legacy": 0, "c2-literal-evidence-v1": 2,
                                          "c2-literal-evidence-v2": 0,
                                          "c2-literal-evidence-v3": 0},
            "records": records,
        }

    def _dependencies(self):
        def dcmread(path, *, stop_before_pixels=False, specific_tags=None):
            key = Path(path).name
            if stop_before_pixels:
                return self.header_data[key]
            self.pixel_reads.append(key)
            return self.pixel_data[key]

        fake_pydicom = SimpleNamespace(dcmread=dcmread)
        fake_numpy = SimpleNamespace(asarray=lambda array: array)
        self.dependencies = (fake_pydicom, fake_numpy)
        self.dependency_patch = patch(
            "rsna_knee.data.dicom_benchmark._load_dependencies",
            return_value=self.dependencies,
        )
        self.dependency_patch.start()
        self.addCleanup(self.dependency_patch.stop)

    def _series_dir(self, study="study-1", series="series-a"):
        path = self.image_root / study / series
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _add_slice(self, name, *, study="study-1", series="series-a", z=0,
                   file_size=4, dataset=None):
        path = self._series_dir(study, series) / name
        path.write_bytes(b"x" * file_size)
        self.header_data[name] = dataset or FakeDataset(
            study=study, series=series, position=(0, 0, z),
        )
        self.pixel_data[name] = dataset or FakeDataset(study=study, series=series)
        return path

    def test_selection_is_bounded_and_slices_sort_by_geometry_not_filename_or_instance(self):
        # Filenames and InstanceNumber are deliberately unrelated to slice position.
        self._add_slice("zeta.dcm", z=2, dataset=FakeDataset(position=(0, 0, 2)))
        self._add_slice("alpha.dcm", z=0, dataset=FakeDataset(position=(0, 0, 0)))
        self._add_slice("middle.dcm", z=1, dataset=FakeDataset(position=(0, 0, 1)))
        for name, z in (("b2.dcm", 2), ("b0.dcm", 0), ("b1.dcm", 1)):
            self._add_slice(name, series="series-b", z=z,
                            dataset=FakeDataset(series="series-b", position=(0, 0, z)))
        result = run_dicom_benchmark(
            manifest=self.manifest, image_root=self.image_root,
            max_studies=1, max_slices_per_series=2,
        )
        self.assertEqual(result["studies_requested"], 1)
        self.assertEqual(result["series_requested"], 2)
        self.assertEqual(result["header_files_scanned"], 6)
        self.assertEqual(result["slices_decoded"], 4)
        self.assertEqual(result["decoded_input_bytes"], 16)
        self.assertEqual(result["decoded_pixel_bytes"], 48)
        self.assertEqual(result["max_pixel_shape"], [2, 3])
        self.assertEqual(self.pixel_reads, ["alpha.dcm", "middle.dcm", "b0.dcm", "b1.dcm"])
        self.assertNotIn("study-1", json.dumps(result))
        self.assertNotIn("series-a", json.dumps(result))

    def test_rejects_missing_or_inconsistent_geometry_without_instance_number_fallback(self):
        first = FakeDataset(position=(0, 0, 0))
        missing = FakeDataset(position=(0, 0, 1))
        del missing.ImagePositionPatient
        self._add_slice("one.dcm", z=0, dataset=first)
        self._add_slice("two.dcm", z=1, dataset=missing)
        with self.assertRaises(DicomBenchmarkError) as raised:
            run_dicom_benchmark(self.manifest, self.image_root, max_studies=1)
        self.assertEqual(raised.exception.reason_code, "missing_geometry")

    def test_rejects_inconsistent_orientation_and_duplicate_projected_positions(self):
        self._add_slice("one.dcm", z=0, dataset=FakeDataset(position=(0, 0, 0)))
        self._add_slice("two.dcm", z=1, dataset=FakeDataset(
            position=(0, 0, 1), orientation=(0, 1, 0, -1, 0, 0),
        ))
        with self.assertRaises(DicomBenchmarkError) as raised:
            run_dicom_benchmark(self.manifest, self.image_root, max_studies=1)
        self.assertEqual(raised.exception.reason_code, "inconsistent_orientation")

        self.header_data.clear()
        self.pixel_data.clear()
        self._add_slice("one.dcm", z=0, dataset=FakeDataset(position=(0, 0, 0)))
        self._add_slice("two.dcm", z=1, dataset=FakeDataset(position=(0, 0, 0)))
        with self.assertRaises(DicomBenchmarkError) as raised:
            run_dicom_benchmark(self.manifest, self.image_root, max_studies=1)
        self.assertEqual(raised.exception.reason_code, "duplicate_slice_position")

    def test_rejects_identity_mismatch_and_unsupported_multiframe_data(self):
        self._add_slice("wrong-study.dcm", dataset=FakeDataset(study="other-study"))
        with self.assertRaises(DicomBenchmarkError) as raised:
            run_dicom_benchmark(self.manifest, self.image_root, max_studies=1)
        self.assertEqual(raised.exception.reason_code, "study_identity_mismatch")

        self.header_data.clear()
        self.pixel_data.clear()
        self._add_slice("multiframe.dcm", dataset=FakeDataset(frames=2))
        with self.assertRaises(DicomBenchmarkError) as raised:
            run_dicom_benchmark(self.manifest, self.image_root, max_studies=1)
        self.assertEqual(raised.exception.reason_code, "unsupported_frame_count")

    def test_decode_failures_are_categorical_and_optional_compression_failures_are_counted(self):
        secret = "PRIVATE_PATH_REPORT_CANARY"
        class BrokenDataset(FakeDataset):
            def __init__(self):
                self.StudyInstanceUID = "study-1"
                self.SeriesInstanceUID = "series-a"
                self.ImagePositionPatient = (0, 0, 0)
                self.ImageOrientationPatient = (1, 0, 0, 0, 1, 0)
                self.PixelSpacing = (0.5, 0.5)
                self.Rows = self.Columns = 2
                self.SamplesPerPixel = 1
                self.PhotometricInterpretation = "MONOCHROME2"
                self.NumberOfFrames = 1
                self.file_meta = SimpleNamespace(
                    TransferSyntaxUID=SimpleNamespace(is_compressed=True),
                )

            @property
            def pixel_array(self):
                raise RuntimeError(secret)

        broken = BrokenDataset()
        self._add_slice("compressed.dcm", dataset=broken)
        self._add_slice("healthy.dcm", series="series-b", dataset=FakeDataset(series="series-b"))
        result = run_dicom_benchmark(self.manifest, self.image_root, max_studies=1)
        encoded = json.dumps(result)
        self.assertNotIn(secret, encoded)
        self.assertEqual(result["decode_failures"], {"compressed_pixel_decode_failed": 1})
        self.assertEqual(result["slices_decoded"], 1)

    def test_rejects_path_traversal_and_symlink_escape(self):
        unsafe = self._manifest()
        unsafe["records"][0]["series"][0]["series_id"] = "../escape"
        with self.assertRaises(DicomBenchmarkError) as raised:
            run_dicom_benchmark(unsafe, self.image_root, max_studies=1)
        self.assertEqual(raised.exception.reason_code, "unsafe_path_component")

        self._series_dir()
        target = self.root / "outside"
        target.mkdir()
        series_dir = self.image_root / "study-1" / "series-a"
        with patch("rsna_knee.data.dicom_benchmark._path_is_symlink",
                   side_effect=lambda path: Path(path) == series_dir):
            with self.assertRaises(DicomBenchmarkError) as raised:
                run_dicom_benchmark(self.manifest, self.image_root, max_studies=1)
        self.assertEqual(raised.exception.reason_code, "symlink_path")

    def test_enforces_byte_shape_and_time_budgets(self):
        self._add_slice("large.dcm", file_size=10)
        with self.assertRaises(DicomBenchmarkError) as raised:
            run_dicom_benchmark(self.manifest, self.image_root, max_studies=1,
                                max_file_bytes=5)
        self.assertEqual(raised.exception.reason_code, "file_size_limit")

        self.header_data.clear()
        self.pixel_data.clear()
        self._add_slice("large-shape.dcm", dataset=FakeDataset(rows=4, columns=4))
        with self.assertRaises(DicomBenchmarkError) as raised:
            run_dicom_benchmark(self.manifest, self.image_root, max_studies=1,
                                max_pixels_per_slice=8)
        self.assertEqual(raised.exception.reason_code, "pixel_shape_limit")

        with patch("rsna_knee.data.dicom_benchmark.time.monotonic", side_effect=[0.0, 2.0]):
            with self.assertRaises(DicomBenchmarkError) as raised:
                run_dicom_benchmark(self.manifest, self.image_root, max_studies=1,
                                    max_seconds=1.0)
        self.assertEqual(raised.exception.reason_code, "time_limit")

    def test_missing_runtime_dependencies_fail_with_a_clear_prerequisite_code(self):
        with patch("rsna_knee.data.dicom_benchmark._load_dependencies",
                   side_effect=DicomBenchmarkError("missing_dependency", "Required dependency is missing.")):
            with self.assertRaises(DicomBenchmarkError) as raised:
                run_dicom_benchmark(self.manifest, self.image_root, max_studies=1)
        self.assertEqual(raised.exception.reason_code, "missing_dependency")

    def test_series_enumeration_oserror_is_sanitized_and_returns_no_partial_result(self):
        self._add_slice("one.dcm")
        secret = "PRIVATE_CANARY"
        series_dir = self.image_root / "study-1" / "series-a"
        original_iterdir = Path.iterdir

        def fail_selected_directory(path):
            if path == series_dir:
                raise PermissionError(secret)
            return original_iterdir(path)

        with patch.object(Path, "iterdir", fail_selected_directory):
            with self.assertRaises(DicomBenchmarkError) as raised:
                run_dicom_benchmark(self.manifest, self.image_root, max_studies=1)
        self.assertEqual(raised.exception.reason_code, "series_scan_failed")
        self.assertNotIn(secret, str(raised.exception))

    def test_decode_file_stat_oserror_is_sanitized_and_returns_no_partial_result(self):
        path = self._add_slice("one.dcm")
        original_stat = Path.stat
        calls = 0
        secret = "PRIVATE_CANARY"

        def fail_on_decode_stat(candidate, *args, **kwargs):
            nonlocal calls
            if candidate == path:
                calls += 1
                if calls == 2:
                    raise PermissionError(secret)
            return original_stat(candidate, *args, **kwargs)

        with patch.object(Path, "stat", fail_on_decode_stat):
            with self.assertRaises(DicomBenchmarkError) as raised:
                run_dicom_benchmark(self.manifest, self.image_root, max_studies=1)
        self.assertEqual(raised.exception.reason_code, "file_read_failed")
        self.assertNotIn(secret, str(raised.exception))

    def test_root_and_series_directory_oserrors_are_sanitized(self):
        self._add_slice("one.dcm")
        secret = "PRIVATE_CANARY"
        original_is_dir = Path.is_dir

        def fail_selected_directory(path):
            if path in {self.image_root, self.image_root / "study-1" / "series-a"}:
                raise PermissionError(secret)
            return original_is_dir(path)

        with patch.object(Path, "is_dir", fail_selected_directory):
            with self.assertRaises(DicomBenchmarkError) as raised:
                run_dicom_benchmark(self.manifest, self.image_root, max_studies=1)
        self.assertEqual(raised.exception.reason_code, "image_root_unavailable")
        self.assertNotIn(secret, str(raised.exception))

        def fail_series_only(path):
            if path == self.image_root / "study-1" / "series-a":
                raise PermissionError(secret)
            return original_is_dir(path)

        with patch.object(Path, "is_dir", fail_series_only):
            with self.assertRaises(DicomBenchmarkError) as raised:
                run_dicom_benchmark(self.manifest, self.image_root, max_studies=1)
        self.assertEqual(raised.exception.reason_code, "series_directory_unavailable")
        self.assertNotIn(secret, str(raised.exception))

    def test_symlink_check_oserrors_are_sanitized_for_study_and_file_paths(self):
        file_path = self._add_slice("one.dcm")
        secret = "PRIVATE_CANARY"
        original_is_symlink = Path.is_symlink
        study_path = self.image_root / "study-1"

        def fail_study_check(path):
            if path == study_path:
                raise PermissionError(secret)
            return original_is_symlink(path)

        with patch.object(Path, "is_symlink", fail_study_check):
            with self.assertRaises(DicomBenchmarkError) as raised:
                run_dicom_benchmark(self.manifest, self.image_root, max_studies=1)
        self.assertEqual(raised.exception.reason_code, "filesystem_check_failed")
        self.assertNotIn(secret, str(raised.exception))

        def fail_file_check(path):
            if path == file_path:
                raise PermissionError(secret)
            return original_is_symlink(path)

        with patch.object(Path, "is_symlink", fail_file_check):
            with self.assertRaises(DicomBenchmarkError) as raised:
                run_dicom_benchmark(self.manifest, self.image_root, max_studies=1)
        self.assertEqual(raised.exception.reason_code, "filesystem_check_failed")
        self.assertNotIn(secret, str(raised.exception))

    def test_notebook_configuration_probe_oserror_is_sanitized(self):
        source_root = Path(__file__).resolve().parents[1]
        manifest = self.root / "manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        image_root = self.root / "notebook-images"
        image_root.mkdir()
        secret = "PRIVATE_CANARY"
        original_is_dir = Path.is_dir
        source_path = source_root / "src"

        def fail_source_probe(path):
            if path == source_path:
                raise PermissionError(secret)
            return original_is_dir(path)

        env = {
            "SOURCE_ROOT": str(source_root),
            "PRIVATE_MANIFEST": str(manifest),
            "COMPETITION_IMAGE_ROOT": str(image_root),
        }
        notebook = source_root / "notebooks" / "baseline_dicom_benchmark.py"
        with patch.dict(os.environ, env), patch.object(Path, "is_dir", fail_source_probe):
            with self.assertRaises(RuntimeError) as raised:
                runpy.run_path(str(notebook))
        self.assertNotIn(secret, str(raised.exception))


if __name__ == "__main__":
    unittest.main()
