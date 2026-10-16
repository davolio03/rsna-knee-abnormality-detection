import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import numpy as np
    import PIL  # noqa: F401
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("optional NumPy/Pillow image dependencies are not installed") from exc

from rsna_knee.data.dicom_preprocessing import (
    DicomPreprocessingError,
    preprocess_dicom_series,
)


def dataset(z, pixels=None, **overrides):
    values = {
        "ImageOrientationPatient": [1, 0, 0, 0, 1, 0],
        "ImagePositionPatient": [0, 0, z],
        "PixelSpacing": [0.7, 0.8],
        "Rows": 2,
        "Columns": 3,
        "SamplesPerPixel": 1,
        "PhotometricInterpretation": "MONOCHROME2",
        "PixelData": b"synthetic",
    }
    values.update(overrides)
    obj = SimpleNamespace(**values)
    if pixels is not None:
        obj.pixel_array = pixels
    return obj


class PixelAccessGuard:
    def __init__(self, source):
        self.__dict__.update(vars(source))

    @property
    def pixel_array(self):
        raise AssertionError("pixel data must not be accessed after metadata mismatch")


class FakeDicomReader:
    def __init__(self, headers, decoded):
        self.headers = headers
        self.decoded = decoded
        self.calls = []

    def dcmread(self, path, **kwargs):
        self.calls.append((Path(path).name, kwargs))
        if kwargs.get("stop_before_pixels"):
            return self.headers[Path(path).name]
        return self.decoded[Path(path).name]


class DicomPreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.files = []
        self.headers = {}
        self.decoded = {}
        for index, z in enumerate((3, 1, 2)):
            name = f"slice-{index}.dcm"
            path = self.root / name
            path.write_bytes(b"x")
            self.files.append(path)
            self.headers[name] = dataset(z)
            self.decoded[name] = dataset(z, np.full((2, 3), z, dtype=np.uint16))
        self.reader = FakeDicomReader(self.headers, self.decoded)

    def run_adapter(self, **kwargs):
        return preprocess_dicom_series(self.root, dcmread=self.reader.dcmread, output_size=4, **kwargs)

    def test_orders_by_patient_space_and_passes_selected_pixels_to_pp01(self):
        result = self.run_adapter()
        self.assertEqual(result.metadata["selected_positions_mm"], [1.0, 2.0, 3.0])
        self.assertEqual(result.metadata["slice_count_scanned"], 3)
        self.assertEqual(result.metadata["slice_count_selected"], 3)
        self.assertEqual(result.images.shape, (3, 1, 4, 4))
        self.assertEqual(result.metadata["preprocessing_version"], "pp02-v1")
        self.assertEqual(result.metadata["pp01"]["version"], "pp01-v1")

    def test_uniform_selection_happens_before_pixel_decode(self):
        for path in self.root.glob("slice-*.dcm"):
            path.unlink()
            self.headers.pop(path.name)
            self.decoded.pop(path.name)
        for i in range(9):
            name = f"s-{i:02}.dcm"
            (self.root / name).write_bytes(b"x")
            self.headers[name] = dataset(i)
            self.decoded[name] = dataset(i, np.full((2, 3), i, dtype=np.uint16))
        result = self.run_adapter()
        self.assertEqual(result.metadata["slice_count_scanned"], 9)
        self.assertEqual(result.metadata["slice_count_selected"], 8)
        decoded_names = [name for name, kw in self.reader.calls if not kw.get("stop_before_pixels")]
        self.assertEqual(decoded_names, [f"s-{i:02}.dcm" for i in range(7)] + ["s-08.dcm"])

    def test_limit_one_selects_first_geometric_slice(self):
        result = self.run_adapter(max_slices=1)
        self.assertEqual(result.metadata["selected_positions_mm"], [1.0])
        self.assertEqual(result.images.shape[0], 1)

    def test_limit_cannot_exceed_eight(self):
        with self.assertRaises(DicomPreprocessingError) as caught:
            self.run_adapter(max_slices=9)
        self.assertEqual(caught.exception.reason_code, "invalid_limit")

    def test_dicom_decimal_strings_become_native_pp01_intensity_metadata(self):
        for name in self.headers:
            values = {"RescaleSlope": "2.5", "RescaleIntercept": "-3", "PixelPaddingValue": "0"}
            self.headers[name] = dataset(
                self.headers[name].ImagePositionPatient[2], **values
            )
            self.decoded[name] = dataset(
                self.decoded[name].ImagePositionPatient[2],
                np.arange(6, dtype=np.uint16).reshape(2, 3),
                **values,
            )
        result = preprocess_dicom_series(
            self.root,
            dcmread=self.reader.dcmread,
            output_size=4,
            lower_percentile=0,
            upper_percentile=100,
        )
        self.assertEqual(result.metadata["pp01"]["normalization"]["valid_min"], -0.5)
        self.assertEqual(result.metadata["pp01"]["normalization"]["valid_max"], 9.5)

    def test_unsafe_integer_padding_metadata_is_rejected_before_float_conversion(self):
        too_large = 2**53 + 1
        for name in self.headers:
            position = self.headers[name].ImagePositionPatient[2]
            self.headers[name] = dataset(position, PixelPaddingValue=too_large)
            self.decoded[name] = dataset(
                position,
                np.full((2, 3), 2**53, dtype=np.uint64),
                PixelPaddingValue=too_large,
            )
        with self.assertRaises(DicomPreprocessingError) as caught:
            self.run_adapter()
        self.assertEqual(caught.exception.reason_code, "invalid_intensity_metadata")

    def test_decoded_intensity_metadata_must_match_scanned_header(self):
        self.headers["slice-1.dcm"] = dataset(
            1, RescaleSlope=1, RescaleIntercept=0, PixelPaddingValue=0
        )
        self.decoded["slice-1.dcm"] = dataset(
            1,
            np.ones((2, 3)),
            RescaleSlope=2,
            RescaleIntercept=10,
            PixelPaddingValue=0,
        )
        self.decoded["slice-1.dcm"] = PixelAccessGuard(self.decoded["slice-1.dcm"])
        with self.assertRaises(DicomPreprocessingError) as caught:
            self.run_adapter(max_slices=1)
        self.assertEqual(caught.exception.reason_code, "decoded_metadata_mismatch")

    def test_duplicate_projected_positions_fail(self):
        self.headers["slice-0.dcm"] = dataset(1)
        with self.assertRaises(DicomPreprocessingError) as caught:
            self.run_adapter()
        self.assertEqual(caught.exception.reason_code, "duplicate_slice_position")

    def test_inconsistent_geometry_fails(self):
        self.headers["slice-0.dcm"] = dataset(3, PixelSpacing=[0.9, 0.8])
        with self.assertRaises(DicomPreprocessingError) as caught:
            self.run_adapter()
        self.assertEqual(caught.exception.reason_code, "inconsistent_spacing")

    def test_missing_required_geometry_fails_closed(self):
        self.headers["slice-0.dcm"] = dataset(3, PixelSpacing=None)
        with self.assertRaises(DicomPreprocessingError) as caught:
            self.run_adapter()
        self.assertEqual(caught.exception.reason_code, "missing_geometry")

    def test_missing_pixel_layout_metadata_fails_closed(self):
        self.headers["slice-0.dcm"] = dataset(3, SamplesPerPixel=None)
        with self.assertRaises(DicomPreprocessingError):
            self.run_adapter()

    def test_rgb_and_multiframe_are_rejected(self):
        for overrides, expected in (({"SamplesPerPixel": 3, "PhotometricInterpretation": "RGB"}, "unsupported_pixel_layout"),
                                    ({"NumberOfFrames": "2"}, "unsupported_frame_count")):
            self.headers["slice-0.dcm"] = dataset(3, **overrides)
            with self.subTest(expected=expected), self.assertRaises(DicomPreprocessingError) as caught:
                self.run_adapter()
            self.assertEqual(caught.exception.reason_code, expected)
            self.headers["slice-0.dcm"] = dataset(3)

    def test_missing_or_incomplete_rescale_and_padding_pairs_fail(self):
        for overrides in ({"RescaleSlope": 2}, {"PixelPaddingRangeLimit": 0}):
            self.headers["slice-0.dcm"] = dataset(3, **overrides)
            self.decoded["slice-0.dcm"] = dataset(3, np.ones((2, 3)), **overrides)
            with self.subTest(overrides=overrides), self.assertRaises(DicomPreprocessingError):
                self.run_adapter()
            self.headers["slice-0.dcm"] = dataset(3)
            self.decoded["slice-0.dcm"] = dataset(3, np.ones((2, 3)))

    def test_lut_is_rejected(self):
        self.headers["slice-0.dcm"] = dataset(3, ModalityLUTSequence=[object()])
        with self.assertRaises(DicomPreprocessingError) as caught:
            self.run_adapter()
        self.assertEqual(caught.exception.reason_code, "unsupported_modality_lut")

    def test_decoded_geometry_must_match_header(self):
        self.decoded["slice-1.dcm"] = dataset(1, np.ones((3, 3)))
        with self.assertRaises(DicomPreprocessingError) as caught:
            self.run_adapter()
        self.assertEqual(caught.exception.reason_code, "decoded_metadata_mismatch")

    def test_decoder_error_text_and_paths_are_not_exposed(self):
        def broken(path, **kwargs):
            raise RuntimeError(f"secret {path}")
        with self.assertRaises(DicomPreprocessingError) as caught:
            preprocess_dicom_series(self.root, dcmread=broken)
        self.assertNotIn(str(self.root), str(caught.exception))
        self.assertNotIn("secret", str(caught.exception))

    def test_non_dicom_and_empty_series_fail_safely(self):
        (self.root / "notes.txt").write_text("x")
        result = self.run_adapter()
        self.assertEqual(result.metadata["slice_count_scanned"], 3)
        for path in self.root.glob("*.dcm"):
            path.unlink()
        with self.assertRaises(DicomPreprocessingError) as caught:
            self.run_adapter()
        self.assertEqual(caught.exception.reason_code, "series_files_missing")

    def test_public_report_contains_no_path_or_identifiers(self):
        result = self.run_adapter()
        self.assertNotIn(str(self.root), repr(result.metadata))
        self.assertNotIn("slice-", repr(result.metadata))
        self.assertNotIn("StudyInstanceUID", repr(result.metadata))


if __name__ == "__main__":
    unittest.main()
