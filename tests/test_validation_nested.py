"""Synthetic tests for nested study-level multilabel validation."""

import copy
import random
import unittest

from rsna_knee.validation.multilabel import (
    iterative_multilabel_folds,
    nested_multilabel_splits,
)


class NestedMultilabelSplitsTests(unittest.TestCase):
    def setUp(self):
        self.keys = [f"synthetic-{index}" for index in range(18)]
        self.targets = [[index % 2, (index // 2) % 2] for index in range(18)]

    def test_default_three_by_three_is_exhaustive_and_isolated(self):
        splits = nested_multilabel_splits(self.keys, self.targets)
        self.assertEqual(len(splits), 3)
        self.assertEqual(
            sorted(key for split in splits for key in split.test), sorted(self.keys)
        )
        for outer in splits:
            with self.subTest(test=outer.test):
                self.assertTrue(outer.train and outer.test)
                self.assertFalse(set(outer.train) & set(outer.test))
                self.assertEqual(set(outer.train) | set(outer.test), set(self.keys))
                self.assertEqual(len(outer.inner_splits), 3)
                self.assertEqual(
                    sorted(key for inner in outer.inner_splits for key in inner.validation),
                    sorted(outer.train),
                )
                for inner in outer.inner_splits:
                    self.assertTrue(inner.train and inner.validation)
                    self.assertFalse(set(inner.train) & set(inner.validation))
                    self.assertEqual(
                        set(inner.train) | set(inner.validation), set(outer.train)
                    )
                    self.assertFalse(
                        set(outer.test) & (set(inner.train) | set(inner.validation))
                    )

    def test_reuses_allocator_on_aligned_outer_training_rows(self):
        splits = nested_multilabel_splits(self.keys, self.targets, seed=17)
        self.assertEqual(
            tuple(split.test for split in splits),
            iterative_multilabel_folds(self.keys, self.targets, seed=17),
        )
        row_by_key = dict(zip(self.keys, self.targets))
        for index, outer in enumerate(splits):
            expected = iterative_multilabel_folds(
                outer.train, [row_by_key[key] for key in outer.train], seed=18 + index
            )
            self.assertEqual(tuple(inner.validation for inner in outer.inner_splits), expected)

    def test_seed_is_repeatable_without_mutating_input_or_global_rng(self):
        original = copy.deepcopy((self.keys, self.targets))
        rng_state = random.getstate()
        for seed in (0, 42, -7):
            self.assertEqual(
                nested_multilabel_splits(self.keys, self.targets, seed=seed),
                nested_multilabel_splits(self.keys, self.targets, seed=seed),
            )
        self.assertEqual((self.keys, self.targets), original)
        self.assertEqual(random.getstate(), rng_state)

    def test_custom_fold_counts_and_all_negative_targets(self):
        splits = nested_multilabel_splits(
            self.keys, [[0, 0] for _ in self.keys], outer_n_splits=2, inner_n_splits=4
        )
        self.assertEqual(len(splits), 2)
        self.assertTrue(all(len(outer.inner_splits) == 4 for outer in splits))

    def test_preserves_original_order_and_non_string_keys(self):
        keys = [(index, "synthetic") for index in range(18)]
        splits = nested_multilabel_splits(keys, self.targets)
        position = {key: index for index, key in enumerate(keys)}
        for outer in splits:
            partitions = [outer.train, outer.test]
            for inner in outer.inner_splits:
                partitions.extend([inner.train, inner.validation])
            for partition in partitions:
                self.assertEqual(list(partition), sorted(partition, key=position.get))

    def test_rejects_insufficient_outer_training_population(self):
        with self.assertRaisesRegex(ValueError, "outer fold.*inner_n_splits"):
            nested_multilabel_splits(["a", "b", "c"], [[0], [0], [0]])

    def test_rejects_invalid_inner_fold_counts(self):
        for value in (True, 1, 0, -1, 3.0, "3", None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "inner_n_splits.*integer"):
                    nested_multilabel_splits(self.keys, self.targets, inner_n_splits=value)

    def test_retains_allocator_input_validation(self):
        cases = [
            (["a", "a", "b"], [[0], [0], [0]], "unique"),
            (self.keys, self.targets[:-1], "same number"),
            (self.keys, [[2] for _ in self.keys], "binary"),
        ]
        for keys, targets, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    nested_multilabel_splits(keys, targets)


if __name__ == "__main__":
    unittest.main()
