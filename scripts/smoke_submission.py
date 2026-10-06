"""Generate and validate a tiny, synthetic-only submission contract fixture."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Sequence


HEADER = (
    "StudyInstanceUID",
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)
SYNTHETIC_IDS = (
    "synthetic-study-001",
    "synthetic-study-002",
    "synthetic-study-003",
)
SYNTHETIC_SCORE = "0.5"
DEFAULT_OUTPUT = Path("outputs/phase0-smoke/submission.csv")


class SubmissionValidationError(ValueError):
    """Raised when a CSV does not match the synthetic submission contract."""


def validate_submission(path: Path, expected_ids: Sequence[str] = SYNTHETIC_IDS) -> None:
    """Validate exact header, row shape, identifiers, and finite probabilities."""
    expected = tuple(expected_ids)
    if not expected or any(not study_id.strip() for study_id in expected):
        raise SubmissionValidationError("Expected fixture identifiers must be nonblank.")
    if len(set(expected)) != len(expected):
        raise SubmissionValidationError("Expected fixture identifiers must be unique.")

    try:
        with path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file, strict=True)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise SubmissionValidationError("CSV is empty; the submission header is missing.") from exc
            if tuple(header) != HEADER:
                raise SubmissionValidationError("CSV header does not exactly match the required ordered header.")

            observed_ids: list[str] = []
            for line_number, row in enumerate(reader, start=2):
                if len(row) != len(HEADER):
                    raise SubmissionValidationError(
                        f"Row {line_number} has {len(row)} columns; expected {len(HEADER)}."
                    )
                study_id = row[0]
                if not study_id.strip():
                    raise SubmissionValidationError(f"Row {line_number} has a blank study identifier.")
                observed_ids.append(study_id)
                for column, raw_value in zip(HEADER[1:], row[1:]):
                    try:
                        value = float(raw_value)
                    except ValueError as exc:
                        raise SubmissionValidationError(
                            f"Row {line_number}, {column!r} is not numeric: {raw_value!r}."
                        ) from exc
                    if not math.isfinite(value):
                        raise SubmissionValidationError(f"Row {line_number}, {column!r} must be finite.")
                    if not 0.0 <= value <= 1.0:
                        raise SubmissionValidationError(f"Row {line_number}, {column!r} is outside [0, 1].")
    except csv.Error as exc:
        raise SubmissionValidationError(f"Malformed CSV: {exc}.") from exc

    if not observed_ids:
        raise SubmissionValidationError("CSV contains a header but no study rows.")
    if len(observed_ids) != len(set(observed_ids)):
        raise SubmissionValidationError("CSV contains duplicate study identifiers.")
    if set(observed_ids) != set(expected) or len(observed_ids) != len(expected):
        missing = sorted(set(expected) - set(observed_ids))
        extra = sorted(set(observed_ids) - set(expected))
        raise SubmissionValidationError(f"Study identifiers do not match fixture; missing={missing}, extra={extra}.")


def generate_submission(path: Path) -> None:
    """Create the fixed synthetic fixture, refusing to replace existing files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(HEADER)
        for study_id in SYNTHETIC_IDS:
            writer.writerow((study_id, *([SYNTHETIC_SCORE] * (len(HEADER) - 1))))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate and validate a synthetic local CSV only; no competition data or network access."
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help=f"output path (default: {DEFAULT_OUTPUT})"
    )
    args = parser.parse_args(argv)
    try:
        generate_submission(args.output)
        validate_submission(args.output)
    except (OSError, SubmissionValidationError) as exc:
        print(f"Synthetic local smoke test failed: {exc}", file=sys.stderr)
        return 1

    print(f"Synthetic local smoke test passed: {args.output}")
    print(f"Validated {len(SYNTHETIC_IDS)} fictitious study identifiers and {len(HEADER) - 1} scores per row.")
    print("This does not validate competition data, model quality, or Kaggle runtime/submission behavior.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
