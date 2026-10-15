"""Build a validated, report-free 500-study training manifest locally."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from rsna_knee.data.cohort import build_training_cohort, write_private_manifest  # noqa: E402
from rsna_knee.labels.subscription import LabelingError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=REPOSITORY_ROOT / "data" / "train.csv")
    parser.add_argument("--labels", type=Path,
                        default=REPOSITORY_ROOT / "data" / "outputs" / "siwc-c2-cohort-run01.jsonl")
    parser.add_argument("--series-csv", type=Path,
                        default=REPOSITORY_ROOT / "data" / "train_series.csv")
    parser.add_argument("--gold-exclusions", type=Path,
                        default=REPOSITORY_ROOT / "data" / "private" / "gold58_study_ids.csv")
    parser.add_argument("--prompt-journal", type=Path,
                        default=REPOSITORY_ROOT / "data" / "outputs" /
                        "siwc-c2-cohort-run01-diagnostics.jsonl")
    parser.add_argument("--output", type=Path,
                        default=REPOSITORY_ROOT / "data" / "private" /
                        "500-study-training-cohort.json")
    parser.add_argument("--cohort-size", type=int, default=500)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = build_training_cohort(
            train_csv=args.train_csv,
            labels_jsonl=args.labels,
            series_csv=args.series_csv,
            gold_exclusions=args.gold_exclusions,
            prompt_journal=args.prompt_journal,
            cohort_size=args.cohort_size,
        )
        write_private_manifest(manifest, output_path=args.output, project_root=REPOSITORY_ROOT)
    except FileExistsError:
        parser.error("Private manifest already exists; choose a new output path.")
    except (LabelingError, OSError, ValueError):
        parser.error("Manifest preparation failed; details suppressed for privacy.")
    counts = manifest["prompt_provenance_counts"]
    series_count = sum(len(record["series"]) for record in manifest["records"])
    print(f"Private manifest created: {len(manifest['records'])} studies; {series_count} series.")
    print("Prompt provenance totals: " + ", ".join(
        f"{name}={count}" for name, count in counts.items()
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
