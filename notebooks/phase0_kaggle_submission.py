"""Create and validate the Phase 0 Kaggle constant-prediction smoke submission.

Only test.csv and the optional sample_submission.csv are read. No identifier values
are printed. The candidate roots are deliberately fixed rather than recursively
walking Kaggle's input mount.
"""

import csv
import math
from pathlib import Path

TARGETS = [
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
]
HEADER = ["StudyInstanceUID", *TARGETS]
INPUT_ROOTS = (
    Path("/kaggle/input/rsna-knee-abnormality-detection"),
    Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
)


def locate_test_csv(roots=INPUT_ROOTS):
    """Find test.csv by checking only the explicitly permitted candidate paths."""
    for root in roots:
        test_path = root / "test.csv"
        if test_path.is_file():
            return root, test_path
    raise FileNotFoundError("test.csv not found at an allowed competition input path")


def read_test_ids(test_path):
    with test_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or "StudyInstanceUID" not in reader.fieldnames:
            raise ValueError("test.csv is missing the StudyInstanceUID column")
        ids = []
        for row in reader:
            study_id = row["StudyInstanceUID"]
            if study_id is None or not study_id.strip():
                raise ValueError("test.csv contains a blank StudyInstanceUID")
            ids.append(study_id)
    if not ids:
        raise ValueError("test.csv contains no studies")
    if len(ids) != len(set(ids)):
        raise ValueError("test.csv contains duplicate StudyInstanceUID values")
    return ids


def validate_sample_header(sample_path):
    if sample_path.is_file():
        with sample_path.open("r", encoding="utf-8-sig", newline="") as source:
            header = next(csv.reader(source), None)
        if header != HEADER:
            raise ValueError("sample_submission.csv header does not match the contract")


def write_and_validate_submission(test_path, output_path, sample_path=None):
    if sample_path is not None:
        validate_sample_header(sample_path)
    ids = read_test_ids(test_path)
    output_path = Path(output_path)
    with output_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination, lineterminator="\n")
        writer.writerow(HEADER)
        writer.writerows([study_id, *([0.5] * len(TARGETS))] for study_id in ids)

    with output_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.reader(source)
        if next(reader, None) != HEADER:
            raise ValueError("submission.csv header validation failed")
        observed_ids = []
        row_count = 0
        for row in reader:
            if len(row) != len(HEADER):
                raise ValueError("submission.csv contains a row with an invalid width")
            study_id = row[0]
            if not study_id.strip():
                raise ValueError("submission.csv contains a blank StudyInstanceUID")
            scores = [float(value) for value in row[1:]]
            if not all(math.isfinite(score) and 0.0 <= score <= 1.0 for score in scores):
                raise ValueError("submission.csv contains an invalid probability")
            if any(score != 0.5 for score in scores):
                raise ValueError("submission.csv contains a non-constant prediction")
            observed_ids.append(study_id)
            row_count += 1
    if observed_ids != ids:
        raise ValueError("submission.csv identifiers/order do not match test.csv")
    if row_count != len(ids):
        raise ValueError("submission.csv row count does not match test.csv")
    return row_count


def main():
    root, test_path = locate_test_csv()
    sample_path = root / "sample_submission.csv"
    count = write_and_validate_submission(
        test_path,
        Path("/kaggle/working/submission.csv"),
        sample_path=sample_path,
    )
    print(f"Validation passed: {count} test rows; exact header; finite 0.5 scores.")
    print("Created and read back /kaggle/working/submission.csv.")


if __name__ == "__main__":
    main()
