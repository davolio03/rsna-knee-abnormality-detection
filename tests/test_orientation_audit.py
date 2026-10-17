import unittest
from pathlib import Path
from types import SimpleNamespace

from rsna_knee.data.orientation_audit import (
    _angle_degrees,
    audit_orientation_headers,
    extract_manifest_series,
    scan_first_inconsistent_series,
)


def rotated(degrees):
    import math

    angle = math.radians(degrees)
    return (math.cos(angle), math.sin(angle), 0.0,
            -math.sin(angle), math.cos(angle), 0.0)


class OrientationAuditTests(unittest.TestCase):
    def test_manifest_extraction_preserves_training_order_without_decoding_labels(self):
        raw = b'''{"records":[
          {"study_id":"study-b","labels":{"secret":"DO_NOT_READ"},
           "series":[{"series_id":"series-2","plane":"x"},{"series_id":"series-1"}]},
          {"study_id":"study-a","labels":{"nested":["private"]},
           "series":[{"series_id":"series-3"}]}
        ],"ignored":{"labels":{"secret":"also-not-read"}}}'''
        parsed = extract_manifest_series(raw)
        self.assertEqual(parsed, [
            ("study-b", ("series-2", "series-1")),
            ("study-a", ("series-3",)),
        ])
        self.assertNotIn("DO_NOT_READ", repr(parsed))
        self.assertNotIn("private", repr(parsed))

    def test_manifest_extractor_rejects_invalid_selected_structure_safely(self):
        with self.assertRaisesRegex(ValueError, "^manifest_invalid$"):
            extract_manifest_series(b'{"records":[{"study_id":"x","series":[]}]}')

    def test_reports_component_delta_and_normalized_small_angles(self):
        result = audit_orientation_headers([
            (1, 0, 0, 0, 1, 0),
            rotated(0.0005),
        ])
        self.assertEqual(result["header_count"], 2)
        self.assertEqual(result["component_count_above_threshold"], 0)
        self.assertAlmostEqual(result["max_row_angle_degrees"], 0.0005, places=7)
        self.assertAlmostEqual(result["max_column_angle_degrees"], 0.0005, places=7)
        self.assertAlmostEqual(result["max_normal_angle_degrees"], 0.0, places=8)
        self.assertLess(result["max_component_delta"], 1e-5)

    def test_counts_over_threshold_components_and_large_plane_change(self):
        result = audit_orientation_headers([
            (1, 0, 0, 0, 1, 0),
            (0, 1, 0, -1, 0, 0),
        ])
        self.assertEqual(result["component_count_above_threshold"], 4)
        self.assertEqual(result["header_count_above_threshold"], 1)
        self.assertAlmostEqual(result["max_row_angle_degrees"], 90.0)
        self.assertAlmostEqual(result["max_column_angle_degrees"], 90.0)
        self.assertAlmostEqual(result["max_normal_angle_degrees"], 0.0)

    def test_counts_invalid_nonfinite_missing_and_zero_norm_headers(self):
        result = audit_orientation_headers([
            (1, 0, 0, 0, 1, 0),
            None,
            (1, 0, 0, 0, 1),
            (float("nan"), 0, 0, 0, 1, 0),
            (0, 0, 0, 0, 1, 0),
        ])
        self.assertEqual(result["header_count"], 5)
        self.assertEqual(result["valid_orientation_count"], 1)
        self.assertEqual(result["invalid_orientation_count"], 4)
        self.assertEqual(result["max_component_delta"], 0.0)

    def test_angle_math_preserves_tiny_angles_and_handles_identical_and_opposite_vectors(self):
        import math

        tiny_degrees = 1e-7
        tiny = (math.cos(math.radians(tiny_degrees)),
                math.sin(math.radians(tiny_degrees)), 0.0)
        self.assertAlmostEqual(_angle_degrees((1.0, 0.0, 0.0), tiny), tiny_degrees, places=10)
        self.assertEqual(_angle_degrees((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)), 0.0)
        self.assertEqual(_angle_degrees((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)), 180.0)

    def test_nonunit_vectors_are_normalized_only_for_angle_measurement(self):
        result = audit_orientation_headers([
            (1, 0, 0, 0, 1, 0),
            (2, 0, 0, 0, 3, 0),
        ])
        self.assertEqual(result["component_count_above_threshold"], 2)
        self.assertEqual(result["max_row_angle_degrees"], 0.0)
        self.assertEqual(result["max_column_angle_degrees"], 0.0)
        self.assertEqual(result["max_normal_angle_degrees"], 0.0)

    def test_large_finite_components_do_not_overflow_angle_math(self):
        result = audit_orientation_headers([
            (1e300, 0, 0, 0, 1e300, 0),
            (2e300, 0, 0, 0, 3e300, 0),
        ])
        self.assertEqual(result["valid_orientation_count"], 2)
        self.assertEqual(result["max_row_angle_degrees"], 0.0)
        self.assertEqual(result["max_column_angle_degrees"], 0.0)
        self.assertEqual(result["max_normal_angle_degrees"], 0.0)

    def test_threshold_cannot_be_overridden(self):
        for threshold in (0.0, 1e-6, 1e-4, True, float("nan")):
            with self.subTest(threshold=threshold):
                with self.assertRaisesRegex(ValueError, "fixed by the audited preprocessor"):
                    audit_orientation_headers([(1, 0, 0, 0, 1, 0)], threshold=threshold)

    def test_scans_full_first_inconsistent_series_then_stops_before_next(self):
        groups = [
            [Path("a.dcm"), Path("b.dcm"), Path("c.dcm")],
            [Path("d.dcm")],
        ]
        headers = {
            "a.dcm": (1, 0, 0, 0, 1, 0),
            "b.dcm": rotated(2.0),
            "c.dcm": rotated(3.0),
            "d.dcm": (1, 0, 0, 0, 1, 0),
        }
        calls = []

        def reader(path, **kwargs):
            calls.append((Path(path).name, kwargs))
            return SimpleNamespace(ImageOrientationPatient=headers[Path(path).name])

        result = scan_first_inconsistent_series(groups, reader)
        self.assertEqual([name for name, _ in calls], ["a.dcm", "b.dcm", "c.dcm"])
        for _, kwargs in calls:
            self.assertTrue(kwargs["stop_before_pixels"])
            self.assertEqual(kwargs["specific_tags"], ["ImageOrientationPatient"])
        self.assertEqual(result["series_ordinal"], 1)
        self.assertEqual(result["headers_read_in_series"], 3)
        self.assertTrue(result["inconsistent_orientation"])
        self.assertNotIn("a.dcm", repr(result))

    def test_reader_failures_are_redacted_and_counted(self):
        calls = []

        def reader(path, **kwargs):
            calls.append(Path(path).name)
            if Path(path).name == "private.dcm":
                raise RuntimeError("patient-id=/private/path/sentinel")
            return SimpleNamespace(ImageOrientationPatient=(1, 0, 0, 0, 1, 0))

        result = scan_first_inconsistent_series(
            [[Path("private.dcm"), Path("safe.dcm")]], reader,
        )
        self.assertEqual(result["header_read_failure_count"], 1)
        self.assertEqual(result["headers_read_in_series"], 2)
        rendered = repr(result)
        self.assertNotIn("private.dcm", rendered)
        self.assertNotIn("patient-id", rendered)
        self.assertNotIn("/private/path", rendered)

    def test_invalid_first_header_does_not_claim_preprocessor_orientation_failure(self):
        calls = []
        orientations = {
            "invalid.dcm": None,
            "baseline.dcm": (1, 0, 0, 0, 1, 0),
            "rotated.dcm": rotated(3.0),
            "next-a.dcm": (1, 0, 0, 0, 1, 0),
            "next-b.dcm": rotated(3.0),
        }

        def reader(path, **kwargs):
            name = Path(path).name
            calls.append(name)
            return SimpleNamespace(ImageOrientationPatient=orientations[name])

        result = scan_first_inconsistent_series([
            [Path("invalid.dcm"), Path("baseline.dcm"), Path("rotated.dcm")],
            [Path("next-a.dcm"), Path("next-b.dcm")],
        ], reader)
        self.assertEqual(result["series_ordinal"], 2)
        self.assertEqual(calls, ["invalid.dcm", "baseline.dcm", "rotated.dcm", "next-a.dcm", "next-b.dcm"])

    def test_header_cap_reports_partial_without_scanning_next_header(self):
        calls = []

        def reader(path, **kwargs):
            calls.append(Path(path).name)
            return SimpleNamespace(ImageOrientationPatient=(1, 0, 0, 0, 1, 0))

        result = scan_first_inconsistent_series(
            [[Path("a.dcm"), Path("b.dcm")], [Path("c.dcm")]],
            reader,
            max_headers=2,
        )
        self.assertEqual(calls, ["a.dcm", "b.dcm"])
        self.assertTrue(result["partial"])
        self.assertEqual(result["partial_reason"], "header_limit_reached")

    def test_disabled_or_invalid_provenance_prevents_reader_calls(self):
        calls = []

        def reader(path, **kwargs):
            calls.append(path)
            raise AssertionError("reader must not run")

        disabled = scan_first_inconsistent_series(
            [[Path("private.dcm")]], reader, enabled=False, provenance_valid=True,
        )
        invalid = scan_first_inconsistent_series(
            [[Path("private.dcm")]], reader, enabled=True, provenance_valid=False,
        )
        self.assertEqual(disabled["status"], "disabled")
        self.assertEqual(invalid["status"], "blocked")
        self.assertEqual(invalid["blocked_reason"], "provenance_invalid")
        self.assertEqual(calls, [])

    def test_empty_series_and_empty_series_list_fail_closed(self):
        def reader(path, **kwargs):
            raise AssertionError("reader must not run for empty series")

        for groups in ([], [[]], [[], [Path("unused.dcm")]]):
            with self.subTest(groups=groups):
                with self.assertRaisesRegex(ValueError, "^series_files_missing$"):
                    scan_first_inconsistent_series(groups, reader)


if __name__ == "__main__":
    unittest.main()
