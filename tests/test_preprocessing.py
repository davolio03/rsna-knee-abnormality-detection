"""Synthetic contract tests for the local decoded-array preprocessing prototype."""

import json
import unittest

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("optional NumPy dependency is not installed") from exc

from rsna_knee.data.preprocessing import (
    DecodedSlice,
    preprocess_series,
    uniformly_spaced_indices,
)


def slice_(pixels, **metadata):
    return DecodedSlice(np.asarray(pixels), **metadata)


class UniformSamplingTests(unittest.TestCase):
    def test_short_series_returns_every_index(self):
        self.assertEqual(uniformly_spaced_indices(3), (0, 1, 2))

    def test_sampling_is_unique_sorted_and_keeps_endpoints(self):
        result = uniformly_spaced_indices(23, limit=8)
        self.assertEqual(len(result), 8)
        self.assertEqual(tuple(sorted(set(result))), result)
        self.assertEqual((result[0], result[-1]), (0, 22))
        self.assertEqual(result, uniformly_spaced_indices(23, limit=8))

    def test_invalid_sampling_parameters_fail(self):
        for length, limit in ((0, 8), (-1, 8), (4, 0), (4, -1)):
            with self.subTest(length=length, limit=limit):
                with self.assertRaises(ValueError):
                    uniformly_spaced_indices(length, limit)


class SeriesPreprocessingTests(unittest.TestCase):
    def test_affine_rescale_and_padding_are_applied_in_stored_value_space(self):
        image = np.array([[0, 1, -1], [2, 3, 4]], dtype=np.int16)
        original = image.copy()
        result = preprocess_series(
            [slice_(image, photometric_interpretation="MONOCHROME2", slope=2, intercept=-10,
                    pixel_padding_value=0, pixel_padding_range_limit=2, pixel_spacing=(0.5, 0.7))],
            output_size=6,
        )
        self.assertFalse(result.valid_mask[0, 0, 0, 0])
        self.assertFalse(result.valid_mask[0, 0, 0, -1])
        self.assertTrue(result.valid_mask[0, 0, 3, 3])
        self.assertTrue(np.array_equal(image, original))
        self.assertEqual(result.metadata["normalization"]["scope"], "series")
        self.assertEqual(result.metadata["transforms"][0]["native_pixel_spacing"], [0.5, 0.7])

    def test_zero_is_valid_without_explicit_padding_metadata(self):
        result = preprocess_series(
            [slice_([[0, 1], [2, 3]], photometric_interpretation="MONOCHROME2")],
            output_size=4,
        )
        self.assertTrue(result.valid_mask.all())

    def test_mono1_inverts_only_valid_pixels(self):
        mono2 = preprocess_series(
            [slice_([[0, 1], [2, 3]], photometric_interpretation="MONOCHROME2")], output_size=4
        )
        mono1 = preprocess_series(
            [slice_([[0, 1], [2, 3]], photometric_interpretation="MONOCHROME1")], output_size=4
        )
        self.assertTrue(np.allclose(mono1.images, 1.0 - mono2.images))
        padded = preprocess_series(
            [slice_([[0, 1], [2, 3]], photometric_interpretation="MONOCHROME1", pixel_padding_value=0)],
            output_size=4,
        )
        self.assertTrue(np.all(padded.images[~padded.valid_mask] == 0))

    def test_negative_slope_and_paired_affine_metadata(self):
        result = preprocess_series(
            [slice_([[0, 1], [2, 3]], photometric_interpretation="MONOCHROME2", slope=-2, intercept=3)],
            output_size=4,
        )
        self.assertEqual(result.metadata["normalization"]["valid_min"], -3.0)
        with self.assertRaises(ValueError):
            preprocess_series([slice_([[1]], photometric_interpretation="MONOCHROME2", slope=2)])
        with self.assertRaises(ValueError):
            preprocess_series([slice_([[1]], photometric_interpretation="MONOCHROME2", intercept=3)])

    def test_padding_range_is_inclusive_and_order_independent(self):
        low_first = preprocess_series(
            [slice_([[0, 1, 2, 3]], photometric_interpretation="MONOCHROME2",
                    pixel_padding_value=1, pixel_padding_range_limit=2)], output_size=4
        )
        high_first = preprocess_series(
            [slice_([[0, 1, 2, 3]], photometric_interpretation="MONOCHROME2",
                    pixel_padding_value=2, pixel_padding_range_limit=1)], output_size=4
        )
        self.assertEqual(low_first.valid_mask.tolist(), high_first.valid_mask.tolist())
        self.assertEqual(low_first.valid_mask[0, 0, 1].tolist(), [True, False, False, True])

    def test_per_series_not_per_slice_normalization(self):
        slices = [
            slice_([[0, 1], [2, 3]], photometric_interpretation="MONOCHROME2"),
            slice_([[100, 101], [102, 103]], photometric_interpretation="MONOCHROME2"),
        ]
        result = preprocess_series(slices, output_size=4)
        self.assertEqual(result.metadata["normalization"]["scope"], "series")
        self.assertGreater(float(result.images[1].mean()), float(result.images[0].mean()))
        self.assertAlmostEqual(result.metadata["normalization"]["valid_min"], 0.0)
        self.assertAlmostEqual(result.metadata["normalization"]["valid_max"], 103.0)

    def test_percentile_degenerate_nonconstant_falls_back_to_minmax(self):
        result = preprocess_series(
            [slice_([[0, 0, 0], [0, 10, 0]], photometric_interpretation="MONOCHROME2")],
            output_size=3, lower_percentile=40, upper_percentile=60,
        )
        self.assertTrue(result.metadata["normalization"]["used_min_max_fallback"])
        self.assertFalse(result.metadata["normalization"]["is_constant"])
        self.assertEqual(float(result.images.max()), 1.0)

    def test_constant_valid_input_outputs_zeros_and_is_flagged(self):
        result = preprocess_series(
            [slice_([[7, 7], [7, 7]], photometric_interpretation="MONOCHROME2")], output_size=4
        )
        self.assertTrue(result.metadata["normalization"]["is_constant"])
        self.assertFalse(result.metadata["normalization"]["used_min_max_fallback"])
        self.assertTrue(np.all(result.images == 0))
        mono1_constant = preprocess_series(
            [slice_([[7, 7], [7, 7]], photometric_interpretation="MONOCHROME1")], output_size=4
        )
        self.assertTrue(np.all(mono1_constant.images == 0))

    def test_invalid_or_empty_pixel_data_fails_closed(self):
        for pixels in (np.array([[np.nan, 1.0]]), np.array([[np.inf, 1.0]])):
            with self.subTest(pixels=pixels):
                with self.assertRaises(ValueError):
                    preprocess_series([slice_(pixels, photometric_interpretation="MONOCHROME2")])
        with self.assertRaises(ValueError):
            preprocess_series([slice_([[0, 0]], photometric_interpretation="MONOCHROME2",
                                      pixel_padding_value=0)])

    def test_complex_pixels_are_rejected_instead_of_truncated(self):
        with self.assertRaisesRegex(ValueError, "real"):
            preprocess_series(
                [slice_(np.array([[1 + 2j, 3 + 0j]]), photometric_interpretation="MONOCHROME2")]
            )

    def test_extreme_finite_float_range_fails_closed_before_overflow(self):
        with np.errstate(over="ignore", invalid="ignore"):
            with self.assertRaisesRegex(ValueError, "range|finite|representable"):
                preprocess_series(
                    [slice_(np.array([[-1e308, 1e308]], dtype=np.float64),
                            photometric_interpretation="MONOCHROME2")]
                )

    def test_differently_shaped_python_lists_are_rejected_within_series(self):
        slices = [
            slice_([[1, 2]], photometric_interpretation="MONOCHROME2"),
            slice_([[1, 2, 3]], photometric_interpretation="MONOCHROME2"),
        ]
        with self.assertRaisesRegex(ValueError, "shape"):
            preprocess_series(slices)

    def test_nonfinite_padding_metadata_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "padding"):
            preprocess_series(
                [slice_([[1, 2]], photometric_interpretation="MONOCHROME2",
                        pixel_padding_value=float("nan"))]
            )

    def test_boolean_rescale_metadata_is_rejected(self):
        for key in ("slope", "intercept"):
            metadata = {"slope": 1.0, "intercept": 0.0}
            metadata[key] = True
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                preprocess_series([slice_([[1, 2]], photometric_interpretation="MONOCHROME2", **metadata)])

    def test_numeric_string_spacing_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "spacing"):
            preprocess_series(
                [slice_([[1, 2]], photometric_interpretation="MONOCHROME2",
                        pixel_spacing=("0.5", 0.5))]
            )

    def test_bilinear_resize_does_not_attenuate_valid_pixels_next_to_padding(self):
        result = preprocess_series(
            [slice_([[10, 0, 20, 30]], photometric_interpretation="MONOCHROME2",
                    pixel_padding_value=0)],
            output_size=8, lower_percentile=0, upper_percentile=100,
        )
        self.assertTrue(result.valid_mask[0, 0, 3, 4])
        self.assertAlmostEqual(float(result.images[0, 0, 3, 4]), 0.5, places=2)

    def test_nonuniform_input_spacing_is_preserved_without_physical_resampling(self):
        result = preprocess_series(
            [slice_([[1, 2], [3, 4]], photometric_interpretation="MONOCHROME2",
                    pixel_spacing=(0.4, 0.8))], output_size=8
        )
        transform = result.metadata["transforms"][0]
        self.assertEqual(transform["native_pixel_spacing"], [0.4, 0.8])
        self.assertEqual(transform["resized_shape"], [8, 8])

    def test_canvas_letterboxes_without_cropping_and_returns_expected_contract(self):
        result = preprocess_series(
            [slice_(np.arange(12).reshape(3, 4), photometric_interpretation="MONOCHROME2")],
            output_size=8,
        )
        self.assertEqual(result.images.shape, (1, 1, 8, 8))
        self.assertEqual(result.images.dtype, np.float32)
        self.assertTrue(np.isfinite(result.images).all())
        self.assertGreaterEqual(float(result.images.min()), 0.0)
        self.assertLessEqual(float(result.images.max()), 1.0)
        transform = result.metadata["transforms"][0]
        self.assertEqual(transform["native_shape"], [3, 4])
        self.assertEqual(transform["resized_shape"], [6, 8])
        self.assertEqual(transform["padding_top"], 1)
        self.assertEqual(transform["padding_bottom"], 1)
        self.assertEqual(result.valid_mask.dtype, np.bool_)
        self.assertEqual(result.valid_mask.shape, (1, 1, 8, 8))
        self.assertFalse(result.valid_mask[:, :, 0, :].any())

    def test_output_selection_caps_slices_at_limit_and_metadata_has_no_arrays(self):
        slices = [slice_([[i, i + 1], [i + 2, i + 3]], photometric_interpretation="MONOCHROME2")
                  for i in range(10)]
        result = preprocess_series(slices, max_slices=4, output_size=4)
        self.assertEqual(result.images.shape[0], 4)
        self.assertEqual(result.metadata["selected_indices"], [0, 3, 6, 9])
        self.assertEqual(result.metadata["version"], "pp01-v1")
        serialized = str(result.metadata).lower()
        for forbidden in ("identifier", "report", "pixels", "array"):
            self.assertNotIn(forbidden, serialized)
        self.assertIsInstance(result.metadata, dict)
        self.assertEqual(result.metadata["normalization"]["method"], "percentile_1_99_linear")
        json.dumps(result.metadata)

    def test_rejects_incompatible_series_metadata_and_unsupported_luts(self):
        base = slice_([[1, 2]], photometric_interpretation="MONOCHROME2", pixel_spacing=(0.5, 0.5))
        with self.assertRaises(ValueError):
            preprocess_series([base, slice_([[1, 2]], photometric_interpretation="MONOCHROME1",
                                             pixel_spacing=(0.5, 0.5))])
        with self.assertRaises(ValueError):
            preprocess_series([base, slice_([[1, 2, 3]], photometric_interpretation="MONOCHROME2",
                                             pixel_spacing=(0.5, 0.5))])
        with self.assertRaises(ValueError):
            preprocess_series([slice_([[1]], photometric_interpretation="MONOCHROME2",
                                      modality_lut_sequence_present=True)])

    def test_direct_array_spacing_contract_is_still_exact(self):
        base = slice_([[1, 2]], photometric_interpretation="MONOCHROME2", pixel_spacing=(0.5, 0.5))
        near = slice_([[1, 2]], photometric_interpretation="MONOCHROME2", pixel_spacing=(0.500001, 0.5))
        with self.assertRaisesRegex(ValueError, "pixel spacing must be consistent"):
            preprocess_series([base, near])

    def test_invalid_parameters_and_spacing_are_rejected(self):
        image = slice_([[1, 2]], photometric_interpretation="MONOCHROME2")
        for kwargs in ({"output_size": 0}, {"max_slices": 0}, {"lower_percentile": -1},
                       {"upper_percentile": 101}, {"lower_percentile": 90, "upper_percentile": 10}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                preprocess_series([image], **kwargs)
        with self.assertRaises(ValueError):
            preprocess_series([slice_([[1]], photometric_interpretation="MONOCHROME2",
                                      pixel_spacing=(0, 1))])
        with self.assertRaises(ValueError):
            preprocess_series([slice_([[1]], photometric_interpretation="MONOCHROME2", slope=0, intercept=0)])
        with self.assertRaises(ValueError):
            preprocess_series([slice_([[1]], photometric_interpretation="MONOCHROME2", slope=float("nan"), intercept=1)])
        with self.assertRaisesRegex(ValueError, "percentile"):
            preprocess_series([image], lower_percentile=False)

    def test_integer_pixels_and_padding_tags_must_fit_exact_float64_integer_domain(self):
        raw = np.array([[2**63, 2**63 + 1, 2**63 + 2]], dtype=np.uint64)
        with self.assertRaisesRegex(ValueError, "exact|representable|float64"):
            preprocess_series(
                [slice_(raw, photometric_interpretation="MONOCHROME2", pixel_padding_value=2**63)]
            )
        with self.assertRaisesRegex(ValueError, "exact|representable|float64"):
            preprocess_series(
                [slice_([[1, 2]], photometric_interpretation="MONOCHROME2",
                        pixel_padding_value=2**53 + 1)]
            )


if __name__ == "__main__":
    unittest.main()
