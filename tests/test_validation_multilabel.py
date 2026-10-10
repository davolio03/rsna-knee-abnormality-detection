"""Tests for deterministic multilabel fold allocation."""

import unittest

from rsna_knee.validation.multilabel import iterative_multilabel_folds


class IterativeMultilabelFoldsTests(unittest.TestCase):
    def test_folds_are_deterministic_and_partition_studies_once(self):
        study_keys = [f"synthetic-study-{index}" for index in range(12)]
        targets = [
            [index % 2, (index // 2) % 2, (index // 3) % 2]
            for index in range(12)
        ]

        first = iterative_multilabel_folds(study_keys, targets, n_splits=3, seed=17)
        second = iterative_multilabel_folds(study_keys, targets, n_splits=3, seed=17)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual(sorted(key for fold in first for key in fold), sorted(study_keys))
        self.assertTrue(all(len(fold) == 4 for fold in first))

    def test_balances_independent_labels_when_each_has_feasible_support(self):
        study_keys = [f"study-{index}" for index in range(12)]
        targets = [[index % 2, (index // 2) % 2] for index in range(12)]

        folds = iterative_multilabel_folds(study_keys, targets, n_splits=3, seed=5)
        key_to_target = dict(zip(study_keys, targets))

        for label_index in range(2):
            positive_counts = [
                sum(key_to_target[key][label_index] for key in fold)
                for fold in folds
            ]
            self.assertLessEqual(max(positive_counts) - min(positive_counts), 1)

    def test_rejects_duplicate_study_keys(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            iterative_multilabel_folds(
                ["study-a", "study-a"], [[1], [0]], n_splits=2
            )

    def test_rejects_mismatched_or_ragged_target_rows(self):
        with self.assertRaisesRegex(ValueError, "same number"):
            iterative_multilabel_folds(["a", "b"], [[1]], n_splits=2)

        with self.assertRaisesRegex(ValueError, "same width"):
            iterative_multilabel_folds(["a", "b"], [[1, 0], [1]], n_splits=2)

    def test_rejects_non_binary_targets(self):
        with self.assertRaisesRegex(ValueError, "binary"):
            iterative_multilabel_folds(["a", "b"], [[1], [2]], n_splits=2)

    def test_requires_enough_studies_for_requested_folds(self):
        with self.assertRaisesRegex(ValueError, "at least one study per fold"):
            iterative_multilabel_folds(["only-one"], [[1]], n_splits=2)


if __name__ == "__main__":
    unittest.main()
