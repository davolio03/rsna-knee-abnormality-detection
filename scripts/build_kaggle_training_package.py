"""Build a fresh, import-closed, allowlisted private Kaggle training package."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import re
import stat
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

EXPECTED_MANIFEST_SHA256 = "e88c61c22a463737f16951b7c76db03834034aecd1b31dc61b4ce495080b2be2"
EXPECTED_FOLDS_SHA256 = "b82b376b0086201e3364b11a0203b62890e628442075a52ea4a9e09ee0e778ac"
ENTRY_MODULES = ("rsna_knee.training.baseline", "rsna_knee.data.training_dataset", "rsna_knee.models.baseline")
# Audited source allowlist. The two deferred runner imports are needed for DICOM
# decoding and fold-mode metrics; unrelated labeler/provider modules are excluded.
SOURCE_MODULE_ALLOWLIST = frozenset({
    "rsna_knee.training.baseline", "rsna_knee.data.training_dataset",
    "rsna_knee.models.baseline", "rsna_knee.validation.metrics",
    "rsna_knee.data.dicom_preprocessing", "rsna_knee.data.preprocessing",
    "rsna_knee.data.dicom_benchmark", "rsna_knee.validation.cohort",
    "rsna_knee.validation.multilabel", "rsna_knee.data.cohort",
    "rsna_knee.labels.subscription", "rsna_knee.labels.siwc_responses",
    "rsna_knee.labels.siwc_auth", "rsna_knee.labels.prompt_comparison",
})
PACKAGE_PREFIX = "rsna_knee"


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _module_path(source_root: Path, module: str) -> Path | None:
    if not module.startswith(PACKAGE_PREFIX + ".") and module != PACKAGE_PREFIX:
        return None
    base = source_root.joinpath(*module.split("."))
    module_file = base.with_suffix(".py")
    init_file = base / "__init__.py"
    if module_file.is_file():
        return module_file
    if init_file.is_file():
        return init_file
    return None


def _local_import_closure(source_root: Path) -> set[Path]:
    pending = [*ENTRY_MODULES, "rsna_knee.validation.metrics"]
    found: set[Path] = set()
    while pending:
        module = pending.pop()
        if module not in SOURCE_MODULE_ALLOWLIST:
            continue
        path = _module_path(source_root, module)
        if path is None or path in found:
            continue
        found.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        package = module.rpartition(".")[0] if path.name != "__init__.py" else module
        class EagerImportVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                return None  # Function-local imports are not required to import the training runtime.

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Import(self, node):
                local = [alias.name for alias in node.names if alias.name.startswith(PACKAGE_PREFIX)]
                if any(name not in SOURCE_MODULE_ALLOWLIST for name in local):
                    raise ValueError("training source has an unapproved local import")
                pending.extend(local)

            def visit_ImportFrom(self, node):
                base = node.module or ""
                if node.level:
                    parts = package.split(".")
                    base = ".".join(parts[: len(parts) - node.level + 1] + ([base] if base else []))
                if base.startswith(PACKAGE_PREFIX):
                    if base not in SOURCE_MODULE_ALLOWLIST:
                        raise ValueError(f"training source imports unapproved local module: {module} -> {base}")
                    pending.append(base)
                    for alias in node.names:
                        candidate = f"{base}.{alias.name}"
                        if _module_path(source_root, candidate) is not None:
                            if candidate not in SOURCE_MODULE_ALLOWLIST:
                                raise ValueError(f"training source imports unapproved local submodule: {module} -> {candidate}")
                            pending.append(candidate)

        EagerImportVisitor().visit(tree)
    # Every package directory is represented explicitly; this works for zip import.
    for path in tuple(found):
        parent = path.parent
        while parent != source_root.parent and parent.is_relative_to(source_root):
            init_file = parent / "__init__.py"
            if init_file.is_file():
                found.add(init_file)
            parent = parent.parent
    if any(not path.is_relative_to(source_root) for path in found):
        raise ValueError("package source escaped the allowlisted root")
    return found


def build_package(
    repository_root: Path,
    archive_path: Path,
    inventory_path: Path,
    *,
    manifest_path: Path,
    folds_path: Path,
    notebook_path: Path,
    config_path: Path,
) -> dict[str, object]:
    """Write a new ZIP and hash inventory; never overwrite existing artifacts."""
    repository = Path(repository_root).resolve(strict=True)
    archive_path = Path(archive_path)
    inventory_path = Path(inventory_path)
    if archive_path.exists() or archive_path.is_symlink() or inventory_path.exists() or inventory_path.is_symlink():
        raise ValueError("package outputs must be new")
    if archive_path.absolute() == inventory_path.absolute():
        raise ValueError("package paths must be distinct")

    inputs = {
        "data/private/500-study-training-cohort.json": (Path(manifest_path), EXPECTED_MANIFEST_SHA256),
        "data/private/500-study-training-folds.json": (Path(folds_path), EXPECTED_FOLDS_SHA256),
        "notebooks/kaggle_eight_study_overfit.ipynb": (Path(notebook_path), None),
        "configs/kaggle_eight_study_overfit.json": (Path(config_path), None),
    }
    members: dict[str, bytes] = {}
    source_root = repository / "src"
    for path in _local_import_closure(source_root):
        relative = path.relative_to(repository).as_posix()
        if path.suffix != ".py" or "__pycache__" in path.parts:
            raise ValueError("source allowlist rejected a non-source file")
        members[relative] = path.read_bytes()
    for archive_name, (path, expected_hash) in inputs.items():
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(repository):
            raise ValueError("package input escaped repository")
        raw = resolved.read_bytes()
        if expected_hash and _sha256(raw) != expected_hash:
            raise ValueError("private source artifact hash mismatch")
        if archive_name.endswith(".ipynb"):
            notebook = json.loads(raw.decode("utf-8"))
            if any(cell.get("outputs", []) for cell in notebook.get("cells", [])):
                raise ValueError("notebook must not contain saved outputs")
        members[archive_name] = raw

    # Enforce path and content policy before writing any package bytes.
    names = tuple(sorted(members))
    if any(name.startswith("/") or ".." in Path(name).parts for name in names):
        raise ValueError("package member path is unsafe")
    if any(name.endswith((".pyc", ".pyo", ".pt", ".pth", ".ckpt", ".dcm", ".dicom"))
           or "__pycache__" in Path(name).parts for name in names):
        raise ValueError("package member type is not allowed")
    source_names = {name for name in names if name.startswith("src/")}
    if not source_names or "src/rsna_knee/__init__.py" not in source_names:
        raise ValueError("package is not import closed")

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_members = {name: _sha256(members[name]) for name in names}
    try:
        with ZipFile(archive_path, "x", compression=ZIP_DEFLATED, compresslevel=9) as archive:
            for name in names:
                info = ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
                info.compress_type = ZIP_DEFLATED
                info.external_attr = (stat.S_IFREG | 0o600) << 16
                archive.writestr(info, members[name])
        package_bytes = archive_path.read_bytes()
        inventory = {
            "schema_version": "kaggle-training-package-inventory-v1",
            "package_sha256": _sha256(package_bytes),
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "folds_sha256": EXPECTED_FOLDS_SHA256,
            "members": inventory_members,
            "exclusions": ["bytecode", "reports", "evidence", "images", "checkpoints", "outputs", "credentials"],
        }
        inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception:
        # Preserve an uncertain output; do not silently overwrite/retry it.
        raise
    return inventory


def build_first_fold_package(
    repository_root: Path,
    output_dir: Path,
    *,
    manifest_path: Path,
    folds_path: Path,
    config_path: Path,
    notebook_path: Path,
) -> dict[str, object]:
    """Create a fresh first-fold package and an independently pinned notebook.

    The notebook is deliberately outside the ZIP inventory: it embeds the inventory
    bytes, so including it in that inventory would create a self-referential hash.
    """
    repository = Path(repository_root).resolve(strict=True)
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("first-fold package output directory must be new")

    inputs = {
        "data/private/500-study-training-cohort.json": (Path(manifest_path), EXPECTED_MANIFEST_SHA256),
        "data/private/500-study-training-folds.json": (Path(folds_path), EXPECTED_FOLDS_SHA256),
    }
    config_raw = Path(config_path).resolve(strict=True).read_bytes()
    if not Path(config_path).resolve(strict=True).is_relative_to(repository):
        raise ValueError("package input escaped repository")
    config_template = json.loads(config_raw.decode("utf-8"))
    if not isinstance(config_template, dict):
        raise ValueError("first-fold config must be an object")
    members: dict[str, bytes] = {}
    for path in _local_import_closure(repository / "src"):
        relative = path.relative_to(repository).as_posix()
        if path.suffix != ".py" or "__pycache__" in path.parts:
            raise ValueError("source allowlist rejected a non-source file")
        members[relative] = path.read_bytes()
    for archive_name, (path, expected_hash) in inputs.items():
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(repository):
            raise ValueError("package input escaped repository")
        raw = resolved.read_bytes()
        if expected_hash and _sha256(raw) != expected_hash:
            raise ValueError("private source artifact hash mismatch")
        members[archive_name] = raw

    names = tuple(sorted(members))
    if any(name.startswith("/") or ".." in Path(name).parts for name in names):
        raise ValueError("package member path is unsafe")
    if any(name.endswith((".pyc", ".pyo", ".pt", ".pth", ".ckpt", ".dcm", ".dicom"))
           or "__pycache__" in Path(name).parts for name in names):
        raise ValueError("package member type is not allowed")
    source_names = {name for name in names if name.startswith("src/")}
    if not source_names or "src/rsna_knee/__init__.py" not in source_names:
        raise ValueError("package is not import closed")

    package_dir = output_dir / "package"
    archive_path = output_dir / "kaggle-first-fold-diagnostic.zip"
    inventory_path = output_dir / "inventory.json"
    prepared_notebook_path = output_dir / "kaggle_first_fold_baseline_run02.ipynb"
    prepared_config_path = output_dir / "kaggle_first_fold_baseline_run02.json"
    output_dir.mkdir(parents=True, exist_ok=False)
    package_dir.mkdir()
    inventory_members = {name: _sha256(members[name]) for name in names}
    try:
        for name in names:
            target = package_dir.joinpath(*Path(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(members[name])
        with ZipFile(archive_path, "x", compression=ZIP_DEFLATED, compresslevel=9) as archive:
            for name in names:
                info = ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
                info.compress_type = ZIP_DEFLATED
                info.external_attr = (stat.S_IFREG | 0o600) << 16
                archive.writestr(info, members[name])
        package_bytes = archive_path.read_bytes()
        inventory = {
            "schema_version": "kaggle-training-package-inventory-v1",
            "package_sha256": _sha256(package_bytes),
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "folds_sha256": EXPECTED_FOLDS_SHA256,
            "members": inventory_members,
            "exclusions": ["bytecode", "reports", "evidence", "images", "checkpoints", "outputs", "credentials"],
        }
        inventory_bytes = (json.dumps(inventory, indent=2, sort_keys=True) + "\n").encode("utf-8")
        inventory_path.write_bytes(inventory_bytes)

        config = dict(config_template)
        config.update({
            "output_dir": "/kaggle/working/kaggle-first-fold-baseline/run02",
            "expected_member_count": len(names),
            "expected_inventory_sha256": _sha256(inventory_bytes),
            "expected_archive_provenance": inventory["package_sha256"],
        })
        prepared_config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

        notebook_raw = Path(notebook_path).resolve(strict=True).read_bytes()
        if not Path(notebook_path).resolve(strict=True).is_relative_to(repository):
            raise ValueError("notebook input escaped repository")
        notebook = json.loads(notebook_raw.decode("utf-8"))
        if any(cell.get("outputs", []) for cell in notebook.get("cells", [])):
            raise ValueError("notebook must not contain saved outputs")
        code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
        if len(code_cells) != 1:
            raise ValueError("first-fold notebook code-cell contract changed")
        code = "".join(code_cells[0].get("source", []))
        replacements = {
            r'"output_dir": "[^"]+"': '"output_dir": "/kaggle/working/kaggle-first-fold-baseline/run02"',
            r'"expected_member_count": \d+': f'"expected_member_count": {len(names)}',
            r'"expected_inventory_sha256": "[0-9a-f]{64}"': f'"expected_inventory_sha256": "{_sha256(inventory_bytes)}"',
            r'"expected_archive_provenance": "[0-9a-f]{64}"': f'"expected_archive_provenance": "{inventory["package_sha256"]}"',
            r'EXPECTED_INVENTORY_B64 = "[^"]+"': f'EXPECTED_INVENTORY_B64 = "{base64.b64encode(inventory_bytes).decode("ascii")}"',
        }
        for pattern, replacement in replacements.items():
            code, count = re.subn(pattern, lambda _match, value=replacement: value, code, count=1)
            if count != 1:
                raise ValueError("first-fold notebook pin contract changed")
        if "RUN_TRAINING = False" not in code:
            raise ValueError("first-fold notebook must remain disabled by default")
        code_cells[0]["source"] = code.splitlines(keepends=True)
        prepared_notebook_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    except Exception:
        # Keep an uncertain output intact for inspection; never silently replace it.
        raise
    return {
        **inventory,
        "inventory_sha256": _sha256(inventory_bytes),
        "archive_path": str(archive_path),
        "package_dir": str(package_dir),
        "inventory_path": str(inventory_path),
        "notebook_path": str(prepared_notebook_path),
        "config_path": str(prepared_config_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    default_output = root / "data/private/kaggle-eight-study-overfit"
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--first-fold-output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.first_fold_output_dir is not None:
        result = build_first_fold_package(
            root,
            args.first_fold_output_dir,
            manifest_path=root / "data/private/500-study-training-cohort.json",
            folds_path=root / "data/private/500-study-training-folds.json",
            config_path=root / "configs/kaggle_first_fold_baseline.json",
            notebook_path=root / "notebooks/kaggle_first_fold_baseline.ipynb",
        )
        print(json.dumps({key: result[key] for key in (
            "package_sha256", "inventory_sha256", "manifest_sha256", "folds_sha256",
            "members", "archive_path", "package_dir", "inventory_path", "notebook_path",
            "config_path",
        )}, indent=2))
        return 0
    result = build_package(
        root,
        args.output_dir / "kaggle-eight-study-overfit.zip",
        args.output_dir / "inventory.json",
        manifest_path=root / "data/private/500-study-training-cohort.json",
        folds_path=root / "data/private/500-study-training-folds.json",
        notebook_path=root / "notebooks/kaggle_eight_study_overfit.ipynb",
        config_path=root / "configs/kaggle_eight_study_overfit.json",
    )
    print(json.dumps({key: result[key] for key in ("package_sha256", "manifest_sha256", "folds_sha256", "members")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
