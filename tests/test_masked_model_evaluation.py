"""Synthetic soft-label comparison contracts; no medical images."""
import copy
import math
import unittest

from rsna_knee.validation.masked_baseline import compare_constant_baseline
from rsna_knee.labels.full_cohort_prefilter import TARGET_COLUMNS


def row(key, value, observed=True):
    return {"study_id": key, "soft_targets": dict.fromkeys(TARGET_COLUMNS, value),
            "observed_mask": dict.fromkeys(TARGET_COLUMNS, observed)}


def report(value=.3, count=2):
    return {"epochs": [{"validation": {"observed_cell_count": count * 12,
            "masked_bce": value, "brier": value / 2,
            "per_target": [{"target": name, "observed_count": count,
                            "masked_bce": value, "brier": value / 2}
                           for name in TARGET_COLUMNS]}}]}


class ConstantBaselineTests(unittest.TestCase):
    def test_fits_training_soft_mean_only_and_evaluates_same_masks(self):
        records = [row("t1", .2), row("t2", .4), row("v1", .9), row("v2", .8)]
        result = compare_constant_baseline(records, {"t1": 1, "t2": 2, "v1": 0, "v2": 0}, report())
        target = result["per_target"][0]
        self.assertAlmostEqual(target["constant_probability"], .3)
        expected = -(.85 * math.log(.3) + .15 * math.log(.7))
        self.assertAlmostEqual(target["constant_bce"], expected)
        self.assertAlmostEqual(result["constant"]["masked_bce"], expected)
        self.assertAlmostEqual(result["constant"]["brier"], ((.9-.3)**2+(.8-.3)**2)/2)
        self.assertEqual(result["bce_targets_better_than_constant"], 12)
        self.assertNotIn("study_id", str(result))

    def test_unknown_cells_never_change_training_mean_or_validation_counts(self):
        records = [row("t", .25), row("tu", .5, False), row("v", .8), row("vu", .5, False)]
        result = compare_constant_baseline(records, {"t": 1, "tu": 1, "v": 0, "vu": 0}, report(count=1))
        self.assertEqual(result["constant"]["observed_cell_count"], 12)
        self.assertEqual(result["per_target"][0]["constant_probability"], .25)

    def test_endpoint_constants_are_stable_and_mismatches_fail(self):
        records = [row("t", 0.), row("v", 1.)]
        result = compare_constant_baseline(records, {"t": 1, "v": 0}, report(count=1))
        self.assertTrue(math.isfinite(result["constant"]["masked_bce"]))
        with self.assertRaises(ValueError):
            compare_constant_baseline(records, {"t": 1, "v": 0}, report(count=2))
        bad = copy.deepcopy(records); bad[0]["soft_targets"][TARGET_COLUMNS[0]] = float("nan")
        with self.assertRaises(ValueError): compare_constant_baseline(bad, {"t": 1, "v": 0}, report(count=1))

    def test_report_target_order_and_aggregate_are_checked(self):
        records = [row("t", .2), row("v", .4)]
        reversed_report=report(count=1);reversed_report["epochs"][-1]["validation"]["per_target"].reverse()
        wrong_aggregate=report(count=1);wrong_aggregate["epochs"][-1]["validation"]["masked_bce"]=.8
        for bad in (reversed_report,wrong_aggregate):
            with self.assertRaises(ValueError): compare_constant_baseline(records, {"t": 1, "v": 0}, bad)
