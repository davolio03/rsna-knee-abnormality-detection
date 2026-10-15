"""Create deterministic three-fold report-group assignments locally."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from rsna_knee.validation.cohort import load_manifest_and_build_folds, write_private_folds  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=REPOSITORY_ROOT / "data" / "private" /
                        "500-study-training-cohort.json")
    parser.add_argument("--output", type=Path,
                        default=REPOSITORY_ROOT / "data" / "private" /
                        "500-study-training-folds.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        artifact = load_manifest_and_build_folds(args.manifest)
        write_private_folds(artifact, output_path=args.output, project_root=REPOSITORY_ROOT)
    except FileExistsError:
        parser.error("Private fold artifact already exists; choose a new output path.")
    except (OSError, ValueError):
        parser.error("Fold preparation failed; details suppressed for privacy.")

    study_count = len(artifact["study_assignments"])
    print(f"Prepared 3 grouped folds for {study_count} studies across {artifact['group_count']} report groups.")
    print("All 12 targets retain both classes in training and validation for each fold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
