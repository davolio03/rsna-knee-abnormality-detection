"""Validate the checked-in workflow source inventory without executing it."""

import argparse
import csv
import subprocess
import sys
from collections import Counter
from pathlib import Path, PurePosixPath


INVENTORY_COLUMNS = (
    "path",
    "role",
    "lifecycle",
    "purpose",
    "known_consumers",
    "evidence",
    "compatibility_notes",
)
ROLES = {"current", "shared", "historical", "generated"}
LIFECYCLES = {"current", "historical", "optional", "research-scoped"}
SOURCE_SUFFIXES = {".py", ".ipynb"}


class InventoryError(ValueError):
    """Raised when catalog coverage or a public reference is invalid."""


def _canonical_repo_path(value, field, row, root):
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or path.as_posix() != value
    ):
        raise InventoryError(f"row {row}: {field} path is not canonical")
    resolved = root.joinpath(*path.parts)
    try:
        resolved.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise InventoryError(f"row {row}: {field} path does not exist inside repository") from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise InventoryError(f"row {row}: {field} path must be a regular non-symlink file")
    return path.as_posix()


def _source_paths_from_git(root):
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "scripts", "notebooks"],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise InventoryError("git ls-files failed; source coverage was not checked") from exc
    try:
        paths = result.stdout.decode("utf-8").split("\0")
    except UnicodeDecodeError as exc:
        raise InventoryError("git ls-files returned non-UTF-8 paths") from exc
    return sorted(path for path in paths if path and Path(path).suffix in SOURCE_SUFFIXES)


def _references(value, field, row, root):
    refs = []
    for item in value.split(";"):
        reference = item.strip()
        if not reference:
            continue
        path = reference.split("#", 1)[0].strip()
        refs.append(_canonical_repo_path(path, field, row, root))
    return refs


def validate_inventory(root, inventory_path, source_paths=None):
    """Validate exact source coverage and repository-local catalog references."""
    root = Path(root).resolve()
    inventory_path = Path(inventory_path)
    try:
        with inventory_path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(INVENTORY_COLUMNS):
                raise InventoryError("inventory columns do not match the required schema")
            rows = list(reader)
    except (OSError, csv.Error, UnicodeError) as exc:
        raise InventoryError("inventory CSV could not be read") from exc

    catalog_paths = []
    roles = Counter()
    lifecycles = Counter()
    for number, row in enumerate(rows, start=2):
        if None in row or any(row.get(column) is None for column in INVENTORY_COLUMNS):
            raise InventoryError(f"row {number}: missing or extra CSV fields")
        path = _canonical_repo_path(row["path"], "source", number, root)
        if not path.startswith(("scripts/", "notebooks/")) or Path(path).suffix not in SOURCE_SUFFIXES:
            raise InventoryError(f"row {number}: source is outside scripts/notebooks Python or notebook scope")
        if not row["purpose"].strip() or not row["evidence"].strip() or not row["compatibility_notes"].strip():
            raise InventoryError(f"row {number}: purpose, evidence, and compatibility_notes must be nonblank")
        if row["role"] not in ROLES:
            raise InventoryError(f"row {number}: invalid role")
        if row["lifecycle"] not in LIFECYCLES:
            raise InventoryError(f"row {number}: invalid lifecycle")
        evidence = _references(row["evidence"], "evidence", number, root)
        if not evidence:
            raise InventoryError(f"row {number}: evidence must contain at least one repository path")
        consumers = _references(row["known_consumers"], "consumer", number, root)
        for consumer in consumers:
            if not consumer.startswith(("src/", "scripts/", "notebooks/")) or Path(consumer).suffix != ".py":
                raise InventoryError(f"row {number}: consumer must be a runtime Python source")
        catalog_paths.append(path)
        roles[row["role"]] += 1
        lifecycles[row["lifecycle"]] += 1

    duplicates = sorted(path for path, count in Counter(catalog_paths).items() if count > 1)
    if duplicates:
        raise InventoryError("duplicate source rows: " + ", ".join(duplicates))

    actual_paths = sorted(source_paths if source_paths is not None else _source_paths_from_git(root))
    missing = sorted(set(actual_paths) - set(catalog_paths))
    extra = sorted(set(catalog_paths) - set(actual_paths))
    if missing:
        raise InventoryError("missing source rows: " + ", ".join(missing))
    if extra:
        raise InventoryError("extra source rows: " + ", ".join(extra))
    return {"sources": len(catalog_paths), "roles": dict(roles), "lifecycles": dict(lifecycles)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate the catalog (the default)")
    args = parser.parse_args(argv)
    del args
    root = Path(__file__).resolve().parents[1]
    inventory = root / "docs" / "WORKFLOW_INVENTORY.csv"
    try:
        summary = validate_inventory(root, inventory)
    except InventoryError as exc:
        print(f"Workflow inventory check failed: {exc}", file=sys.stderr)
        return 1
    roles = ", ".join(f"{name}={count}" for name, count in sorted(summary["roles"].items()))
    print(f"Workflow inventory OK: {summary['sources']} sources ({roles})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
