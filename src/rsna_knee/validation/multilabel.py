"""Deterministic iterative stratification for multilabel study keys."""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from numbers import Integral
from typing import TypeVar


StudyKey = TypeVar("StudyKey")


def iterative_multilabel_folds(
    study_keys: Sequence[StudyKey],
    targets: Sequence[Sequence[int]],
    n_splits: int = 3,
    seed: int = 42,
) -> tuple[tuple[StudyKey, ...], ...]:
    """Allocate studies to folds with deterministic iterative multilabel stratification.

    The algorithm prioritizes the least frequent remaining positive label, then
    assigns studies carrying it to folds with the greatest remaining need for
    that label. Ties use the remaining fold capacity and a local seeded random
    generator. Input order is never mutated and each study appears in exactly
    one returned fold. Rare labels need not occur in every fold when their
    support is smaller than ``n_splits``.

    Args:
        study_keys: Unique, hashable keys corresponding one-to-one with rows in
            ``targets``.
        targets: Rectangular binary matrix, one row per study and one column
            per multilabel target.
        n_splits: Number of folds; must be at least two and no greater than
            the number of studies.
        seed: Integer seed controlling only tie-breaking.

    Returns:
        A tuple of folds, each containing the original study keys.

    Raises:
        ValueError: If keys or target rows are invalid, or there are not enough
            studies to populate every fold.
    """
    keys = _materialize(study_keys, "study_keys")
    rows = _materialize(targets, "targets")

    if not isinstance(n_splits, Integral) or isinstance(n_splits, bool):
        raise ValueError("n_splits must be an integer of at least two")
    n_splits = int(n_splits)
    if n_splits < 2:
        raise ValueError("n_splits must be an integer of at least two")
    if not isinstance(seed, Integral) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    seed = int(seed)

    if len(keys) != len(rows):
        raise ValueError("study_keys and targets must contain the same number of rows")
    if len(keys) < n_splits:
        raise ValueError("need at least one study per fold")

    seen = set()
    for key in keys:
        try:
            if key in seen:
                raise ValueError("study_keys must be unique")
            seen.add(key)
        except TypeError as exc:
            raise ValueError("study_keys must contain hashable values") from exc

    matrix = []
    width = None
    for row in rows:
        values = _materialize(row, "target row")
        if width is None:
            width = len(values)
            if width == 0:
                raise ValueError("targets must contain at least one column")
        elif len(values) != width:
            raise ValueError("target rows must have the same width")
        if any(value not in (0, 1, False, True) for value in values):
            raise ValueError("targets must contain only binary values 0 and 1")
        matrix.append(tuple(int(value) for value in values))

    rng = random.Random(seed)
    fold_capacity = [len(keys) / n_splits] * n_splits
    label_capacity = [
        [sum(row[label] for row in matrix) / n_splits for label in range(width)]
        for _ in range(n_splits)
    ]
    remaining_by_label = [
        [index for index, row in enumerate(matrix) if row[label]]
        for label in range(width)
    ]
    unassigned = set(range(len(keys)))
    assigned = [[] for _ in range(n_splits)]

    while any(remaining_by_label):
        active_labels = [
            label for label, indices in enumerate(remaining_by_label) if indices
        ]
        rarest_count = min(len(remaining_by_label[label]) for label in active_labels)
        rarest_labels = [
            label
            for label in active_labels
            if len(remaining_by_label[label]) == rarest_count
        ]
        label = rng.choice(rarest_labels)
        candidates = remaining_by_label[label][:]
        rng.shuffle(candidates)

        for index in candidates:
            if index not in unassigned:
                continue
            fold = _choose_fold(label_capacity, fold_capacity, label, rng)
            assigned[fold].append(index)
            unassigned.remove(index)
            fold_capacity[fold] -= 1
            for positive_label, value in enumerate(matrix[index]):
                if value:
                    label_capacity[fold][positive_label] -= 1
                    remaining_by_label[positive_label].remove(index)

    for index in sorted(unassigned):
        fold = _choose_fold(label_capacity, fold_capacity, None, rng)
        assigned[fold].append(index)
        fold_capacity[fold] -= 1

    return tuple(
        tuple(keys[index] for index in range(len(keys)) if index in fold_indices)
        for fold_indices in assigned
    )


def _choose_fold(
    label_capacity: list[list[float]],
    fold_capacity: list[float],
    label: int | None,
    rng: random.Random,
) -> int:
    if label is None:
        best_need = max(fold_capacity)
    else:
        best_need = max(capacity[label] for capacity in label_capacity)
    if label is None:
        candidates = [
            fold
            for fold, capacity in enumerate(fold_capacity)
            if capacity == best_need
        ]
    else:
        candidates = [
            fold
            for fold, capacity in enumerate(label_capacity)
            if capacity[label] == best_need
        ]
    if label is not None and len(candidates) > 1:
        best_size = max(fold_capacity[fold] for fold in candidates)
        candidates = [fold for fold in candidates if fold_capacity[fold] == best_size]
    return rng.choice(candidates)


def _materialize(values: Iterable[StudyKey], name: str) -> list[StudyKey]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be an iterable of values, not a string")
    try:
        return list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be iterable") from exc
