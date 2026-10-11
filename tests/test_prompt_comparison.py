"""Synthetic tests for the fail-closed prompt comparison boundary."""

import unittest

from rsna_knee.labels.prompt_comparison import (
    CANDIDATE_IDS,
    run_nested_prompt_comparison,
    select_candidate,
)
from rsna_knee.validation.multilabel import nested_multilabel_splits


class PromptComparisonTests(unittest.TestCase):
    def setUp(self):
        self.keys = tuple(f"synthetic-{index}" for index in range(18))
        targets = [[index % 2, (index // 2) % 2] for index in range(len(self.keys))]
        self.outer = nested_multilabel_splits(self.keys, targets)[0]
        self.schema = {"label": bool, "score": float}

    def test_only_predeclared_candidate_ids_are_selectable(self):
        self.assertEqual(CANDIDATE_IDS, ("candidate_1", "candidate_2", "candidate_3"))
        with self.assertRaisesRegex(ValueError, "candidate"):
            select_candidate({"unknown": 0.9})

    def test_candidate_selection_uses_inner_scores_and_fixed_tie_order(self):
        self.assertEqual(
            select_candidate(
                {"candidate_3": 0.8, "candidate_2": 0.8, "candidate_1": 0.8}
            ),
            "candidate_1",
        )
        self.assertEqual(
            select_candidate(
                {"candidate_1": 0.6, "candidate_2": 0.7, "candidate_3": 0.8}
            ),
            "candidate_3",
        )

    def test_nested_boundary_hides_outer_test_from_inner_selection(self):
        outer_test = set(self.outer.test)
        observed = []
        evaluated = []

        def payload_factory(train_keys, evaluation_keys):
            observed.append((tuple(train_keys), tuple(evaluation_keys)))
            return {"synthetic": True, "keys": tuple(evaluation_keys)}

        def evaluator(candidate_id, payload, output_schema):
            self.assertIn(candidate_id, CANDIDATE_IDS)
            self.assertEqual(output_schema, self.schema)
            self.assertEqual(set(payload), {"synthetic", "keys"})
            evaluated.append(candidate_id)
            return {"label": True, "score": CANDIDATE_IDS.index(candidate_id) / 10}

        def scorer(output, evaluation_keys):
            if set(evaluation_keys) == outer_test:
                # Outer scoring happens after inner selection and cannot feed back.
                self.assertEqual(evaluated[-1], "candidate_3")
            else:
                self.assertFalse(outer_test.intersection(evaluation_keys))
            return output["score"]

        result = run_nested_prompt_comparison(
            self.outer,
            payload_factory=payload_factory,
            evaluator=evaluator,
            scorer=scorer,
            output_schema=self.schema,
        )

        self.assertEqual(result.selected_candidate_id, "candidate_3")
        self.assertEqual(result.outer_score, 0.2)
        self.assertEqual(len(observed), len(CANDIDATE_IDS) * len(self.outer.inner_splits) + 1)
        for train_keys, evaluation_keys in observed[:-1]:
            self.assertFalse(outer_test.intersection(train_keys))
            self.assertFalse(outer_test.intersection(evaluation_keys))
        self.assertEqual(set(observed[-1][1]), outer_test)

    def test_missing_callbacks_payload_or_schema_fail_closed(self):
        for callback_name in ("payload_factory", "evaluator", "scorer"):
            callbacks = {
                "payload_factory": lambda *_: {"synthetic": True},
                "evaluator": lambda *_: {"label": True, "score": 0.0},
                "scorer": lambda *_: 0.0,
            }
            callbacks[callback_name] = None
            with self.subTest(callback=callback_name):
                with self.assertRaisesRegex(ValueError, "callback"):
                    run_nested_prompt_comparison(
                        self.outer,
                        payload_factory=callbacks["payload_factory"],
                        evaluator=callbacks["evaluator"],
                        scorer=callbacks["scorer"],
                        output_schema=self.schema,
                    )
        with self.assertRaisesRegex(ValueError, "schema"):
            run_nested_prompt_comparison(self.outer, payload_factory=lambda *_: {},
                                         evaluator=lambda *_: {}, scorer=lambda *_: 0.0,
                                         output_schema=None)
        with self.assertRaisesRegex(ValueError, "payload"):
            run_nested_prompt_comparison(self.outer, payload_factory=lambda *_: None,
                                         evaluator=lambda *_: {}, scorer=lambda *_: 0.0,
                                         output_schema=self.schema)

    def test_callback_failure_does_not_echo_payload_details(self):
        with self.assertRaises(ValueError) as raised:
            run_nested_prompt_comparison(
                self.outer,
                payload_factory=lambda *_: {"synthetic": True},
                evaluator=lambda *_: (_ for _ in ()).throw(
                    RuntimeError("synthetic provider payload")
                ),
                scorer=lambda *_: 0.0,
                output_schema=self.schema,
            )
        self.assertNotIn("synthetic provider payload", str(raised.exception))

    def test_invalid_candidate_output_schema_and_outer_fold_overlap_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "schema"):
            run_nested_prompt_comparison(
                self.outer,
                payload_factory=lambda *_: {"synthetic": True},
                evaluator=lambda *_: {"unexpected": 1},
                scorer=lambda *_: 0.0,
                output_schema=self.schema,
            )

        leaked = type(self.outer)(
            train=self.outer.train,
            test=self.outer.test,
            inner_splits=(
                type(self.outer.inner_splits[0])(
                    train=self.outer.inner_splits[0].train,
                    validation=self.outer.test,
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "outer"):
            run_nested_prompt_comparison(
                leaked,
                payload_factory=lambda *_: {"synthetic": True},
                evaluator=lambda *_: {"label": True, "score": 0.0},
                scorer=lambda *_: 0.0,
                output_schema=self.schema,
            )

    def test_all_splits_are_prevalidated_before_any_callback(self):
        malformed_later = type(self.outer)(
            train=self.outer.train,
            test=self.outer.test,
            inner_splits=(
                self.outer.inner_splits[0],
                type(self.outer.inner_splits[1])(
                    train=self.outer.inner_splits[1].train,
                    validation=self.outer.test,
                ),
            ),
        )
        calls = []

        def callback(*args):
            calls.append(args)
            return {"label": True, "score": 0.0}

        with self.assertRaisesRegex(ValueError, "inner split"):
            run_nested_prompt_comparison(
                malformed_later,
                payload_factory=lambda *_: calls.append("payload") or {"synthetic": True},
                evaluator=callback,
                scorer=lambda *_: calls.append("score") or 0.0,
                output_schema=self.schema,
            )
        self.assertEqual(calls, [])

    def test_duplicate_study_keys_are_rejected_before_callbacks(self):
        duplicate_outer = type(self.outer)(
            train=self.outer.train + (self.outer.train[0],),
            test=self.outer.test,
            inner_splits=self.outer.inner_splits,
        )
        duplicate_inner = type(self.outer)(
            train=self.outer.train,
            test=self.outer.test,
            inner_splits=(
                type(self.outer.inner_splits[0])(
                    train=self.outer.inner_splits[0].train
                    + (self.outer.inner_splits[0].train[0],),
                    validation=self.outer.inner_splits[0].validation,
                ),
                *self.outer.inner_splits[1:],
            ),
        )
        for split in (duplicate_outer, duplicate_inner):
            calls = []
            with self.subTest(split=split):
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    run_nested_prompt_comparison(
                        split,
                        payload_factory=lambda *_: calls.append("payload")
                        or {"synthetic": True},
                        evaluator=lambda *_: calls.append("evaluate")
                        or {"label": True, "score": 0.0},
                        scorer=lambda *_: calls.append("score") or 0.0,
                        output_schema=self.schema,
                    )
            self.assertEqual(calls, [])

    def test_outer_outcome_changes_do_not_change_selected_candidate(self):
        outer_test = set(self.outer.test)

        def run(outer_outcome):
            def payload_factory(train_keys, evaluation_keys):
                is_outer = set(evaluation_keys) == outer_test
                return {
                    "synthetic": True,
                    "outer": is_outer,
                    "outcome": outer_outcome if is_outer else None,
                }

            def evaluator(candidate_id, payload, _schema):
                score = payload["outcome"] if payload["outer"] else (
                    CANDIDATE_IDS.index(candidate_id) / 10
                )
                return {"label": True, "score": float(score)}

            return run_nested_prompt_comparison(
                self.outer,
                payload_factory=payload_factory,
                evaluator=evaluator,
                scorer=lambda output, _keys: output["score"],
                output_schema=self.schema,
            )

        first = run(0.0)
        second = run(1.0)
        self.assertEqual(first.selected_candidate_id, "candidate_3")
        self.assertEqual(second.selected_candidate_id, first.selected_candidate_id)
        self.assertNotEqual(first.outer_score, second.outer_score)


if __name__ == "__main__":
    unittest.main()
