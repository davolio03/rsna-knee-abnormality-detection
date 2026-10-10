"""Run a bounded, resumable local Codex CLI report-label pilot."""

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rsna_knee.labels.subscription import LabelingError, run_pilot


def _read_rows(path: Path):
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise LabelingError("Training CSV has no header.")
            return list(reader)
    except (OSError, UnicodeError, csv.Error):
        raise LabelingError("Training CSV could not be read.") from None


def _read_gold_ids(path: Path) -> set[str]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["StudyInstanceUID"]:
                raise LabelingError("Gold exclusion CSV must have only StudyInstanceUID.")
            ids = [row["StudyInstanceUID"] for row in reader]
    except (OSError, UnicodeError, csv.Error, KeyError):
        raise LabelingError("Gold exclusion CSV could not be read.") from None
    if not ids or any(not value.strip() for value in ids):
        raise LabelingError("Gold exclusion CSV is empty or invalid.")
    return set(ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--gold-exclusions", type=Path, required=True)
    parser.add_argument("--max-records", type=int, required=True,
                        help="Explicit pilot limit (1–10).")
    parser.add_argument("--subscription-oauth-confirmed", action="store_true", required=True,
                        help="Assert Codex CLI is authenticated with ChatGPT subscription OAuth.")
    parser.add_argument("--output", type=Path,
                        default=Path("data/outputs/codex-report-labels.jsonl"))
    args = parser.parse_args()
    try:
        total = run_pilot(
            _read_rows(args.train_csv), gold_ids=_read_gold_ids(args.gold_exclusions),
            max_records=args.max_records, output_path=args.output,
            subscription_oauth_confirmed=args.subscription_oauth_confirmed,
        )
    except LabelingError as exc:
        parser.error(str(exc))
    # Do not print IDs, reports, predictions, CLI output, or error payloads.
    print(f"Local pilot output records: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
