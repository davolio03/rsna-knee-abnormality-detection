import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from smoke_submission import (
    HEADER,
    SYNTHETIC_IDS,
    SubmissionValidationError,
    generate_submission,
    validate_submission,
)


def write_csv(path: Path, header=HEADER, rows=()):
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(rows)


def fixture_rows(ids=SYNTHETIC_IDS, score="0.5"):
    return [(study_id, *([score] * (len(HEADER) - 1))) for study_id in ids]


class SmokeSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = Path(self.temp_dir.name) / "submission.csv"

    def test_generator_writes_and_validates_exact_synthetic_fixture(self):
        generate_submission(self.path)
        validate_submission(self.path)
        with self.path.open(newline="", encoding="utf-8") as csv_file:
            rows = list(csv.reader(csv_file))
        self.assertEqual(tuple(rows[0]), HEADER)
        self.assertEqual([row[0] for row in rows[1:]], list(SYNTHETIC_IDS))
        self.assertTrue(all(len(row) == len(HEADER) for row in rows[1:]))
        self.assertTrue(all(value == "0.5" for row in rows[1:] for value in row[1:]))

    def test_generator_refuses_to_overwrite_existing_file(self):
        self.path.write_text("preserve me", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            generate_submission(self.path)
        self.assertEqual(self.path.read_text(encoding="utf-8"), "preserve me")

    def test_probability_boundaries_zero_and_one_are_accepted(self):
        write_csv(self.path, rows=fixture_rows(score="0"))
        validate_submission(self.path)
        write_csv(self.path, rows=fixture_rows(score="1"))
        validate_submission(self.path)

    def test_wrong_reordered_or_duplicate_header_is_rejected(self):
        headers = (("wrong", *HEADER[1:]), tuple(reversed(HEADER)), (*HEADER[:-1], HEADER[-2]))
        for header in headers:
            with self.subTest(header=header):
                write_csv(self.path, header=header, rows=fixture_rows())
                with self.assertRaisesRegex(SubmissionValidationError, "header"):
                    validate_submission(self.path)

    def test_empty_and_header_only_csv_are_rejected(self):
        self.path.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(SubmissionValidationError, "empty"):
            validate_submission(self.path)
        write_csv(self.path)
        with self.assertRaisesRegex(SubmissionValidationError, "no study rows"):
            validate_submission(self.path)

    def test_missing_extra_duplicate_and_blank_identifiers_are_rejected(self):
        cases = (
            fixture_rows(SYNTHETIC_IDS[:-1]),
            fixture_rows((*SYNTHETIC_IDS, "unexpected-study")),
            fixture_rows((SYNTHETIC_IDS[0], SYNTHETIC_IDS[0], SYNTHETIC_IDS[2])),
            fixture_rows(("", SYNTHETIC_IDS[1], SYNTHETIC_IDS[2])),
        )
        for rows in cases:
            with self.subTest(ids=[row[0] for row in rows]):
                write_csv(self.path, rows=rows)
                with self.assertRaises(SubmissionValidationError):
                    validate_submission(self.path)

    def test_whitespace_padded_identifiers_are_not_normalized(self):
        for study_id in (f" {SYNTHETIC_IDS[0]}", f"{SYNTHETIC_IDS[0]} "):
            ids = (study_id, SYNTHETIC_IDS[1], SYNTHETIC_IDS[2])
            with self.subTest(study_id=study_id):
                write_csv(self.path, rows=fixture_rows(ids))
                with self.assertRaisesRegex(SubmissionValidationError, "identifiers do not match"):
                    validate_submission(self.path)

    def test_wrong_width_and_malformed_csv_are_rejected(self):
        write_csv(self.path, rows=[fixture_rows()[0][:-1]])
        with self.assertRaisesRegex(SubmissionValidationError, "columns"):
            validate_submission(self.path)
        self.path.write_text(",".join(HEADER) + '\n"unterminated', encoding="utf-8")
        with self.assertRaisesRegex(SubmissionValidationError, "Malformed CSV"):
            validate_submission(self.path)

    def test_nonnumeric_nonfinite_and_out_of_range_values_are_rejected(self):
        for value in ("not-a-number", "nan", "inf", "-inf", "-0.01", "1.01"):
            with self.subTest(value=value):
                write_csv(self.path, rows=fixture_rows(score=value))
                with self.assertRaises(SubmissionValidationError):
                    validate_submission(self.path)


if __name__ == "__main__":
    unittest.main()
