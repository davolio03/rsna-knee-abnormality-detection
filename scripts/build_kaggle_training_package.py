"""Build a fresh, import-closed, allowlisted private Kaggle training package."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import re
import shutil
import stat
from tempfile import TemporaryDirectory
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


def _validate_resnet_fold(fold: int) -> None:
    if type(fold) is not int or fold not in (0, 1, 2):
        raise ValueError("ResNet-18 fold must be one of 0, 1, or 2")


def _resnet_fold_protocol(manifest_raw: bytes, folds_raw: bytes, fold: int) -> dict[str, int]:
    _validate_resnet_fold(fold)
    if _sha256(manifest_raw) != EXPECTED_MANIFEST_SHA256 or _sha256(folds_raw) != EXPECTED_FOLDS_SHA256:
        raise ValueError("private source artifact hash mismatch")
    manifest = json.loads(manifest_raw.decode("utf-8"))
    folds = json.loads(folds_raw.decode("utf-8"))
    records = manifest.get("records")
    assignments = folds.get("study_assignments")
    groups = folds.get("group_assignments")
    if (manifest.get("schema_version") != "baseline-training-cohort-v1"
            or manifest.get("configuration", {}).get("cohort_size") != 500
            or manifest.get("configuration", {}).get("gold_exclusion_count") != 58
            or folds.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256
            or not isinstance(records, list) or len(records) != 500
            or not isinstance(assignments, list) or len(assignments) != 500
            or not isinstance(groups, list) or not groups):
        raise ValueError("pinned ResNet-18 cohort or fold schema is invalid")

    record_ids = [record.get("study_id") for record in records if isinstance(record, dict)]
    assignment_by_study: dict[str, int] = {}
    for assignment in assignments:
        if (not isinstance(assignment, dict)
                or not isinstance(assignment.get("study_id"), str)
                or type(assignment.get("fold")) is not int
                or assignment["fold"] not in (0, 1, 2)
                or assignment["study_id"] in assignment_by_study):
            raise ValueError("pinned ResNet-18 study assignments are invalid")
        assignment_by_study[assignment["study_id"]] = assignment["fold"]
    if len(set(record_ids)) != 500 or set(record_ids) != set(assignment_by_study):
        raise ValueError("pinned ResNet-18 study assignments do not match the cohort")

    grouped_studies: set[str] = set()
    for group in groups:
        if (not isinstance(group, dict) or not isinstance(group.get("normalized_report_sha256"), str)
                or type(group.get("fold")) is not int or group["fold"] not in (0, 1, 2)
                or not isinstance(group.get("study_ids"), list) or not group["study_ids"]):
            raise ValueError("pinned ResNet-18 report groups are invalid")
        for study_id in group["study_ids"]:
            if (study_id in grouped_studies or assignment_by_study.get(study_id) != group["fold"]):
                raise ValueError("pinned ResNet-18 report groups cross folds or duplicate studies")
            grouped_studies.add(study_id)
    if grouped_studies != set(record_ids):
        raise ValueError("pinned ResNet-18 report groups do not cover the cohort")

    train_ids = {study_id for study_id, assigned_fold in assignment_by_study.items() if assigned_fold != fold}
    validation_ids = {study_id for study_id, assigned_fold in assignment_by_study.items() if assigned_fold == fold}
    if not train_ids or not validation_ids or train_ids & validation_ids:
        raise ValueError("pinned ResNet-18 partition is invalid")
    labels_by_study = {record["study_id"]: record.get("labels") for record in records}
    target_names = list(manifest.get("records", [{}])[0].get("labels", {}))
    for partition in (train_ids, validation_ids):
        for target in target_names:
            values = [labels_by_study[study_id].get(target) for study_id in partition]
            if any(type(value) is not int or value not in (0, 1) for value in values):
                raise ValueError("pinned ResNet-18 labels are invalid")
            if not any(values) or all(values):
                raise ValueError("pinned ResNet-18 partition lacks both target classes")

    audit_rows = folds.get("audit", {}).get("folds")
    audit = next((row for row in audit_rows or [] if row.get("fold") == fold), None)
    if (folds.get("audit", {}).get("group_overlap_count") != 0 or audit is None
            or audit.get("group_overlap_count") != 0
            or audit.get("train_study_count") != len(train_ids)
            or audit.get("validation_study_count") != len(validation_ids)):
        raise ValueError("pinned ResNet-18 fold audit does not match assignments")
    return {
        "fold": fold,
        "training_count": len(train_ids),
        "validation_count": len(validation_ids),
        "steps": len(train_ids) * 5,
    }


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


def build_resnet18_comparison_package(
    repository_root: Path,
    output_dir: Path,
    *,
    manifest_path: Path,
    folds_path: Path,
    weights_path: Path,
    expected_weights_sha256: str,
    expected_weights_size: int,
    fold: int = 0,
) -> dict[str, object]:
    """Prepare two disabled, identical-protocol ResNet-18 arm notebooks and upload files."""
    repository = Path(repository_root).resolve(strict=True)
    output_dir = Path(output_dir)
    _validate_resnet_fold(fold)
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("ResNet-18 comparison output directory must be new")
    if (type(expected_weights_sha256) is not str
            or expected_weights_sha256 != "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
            or type(expected_weights_size) is not int or expected_weights_size != 46830571):
        raise ValueError("official ImageNet V1 weight pin is invalid")
    weights = Path(weights_path).resolve(strict=True)
    if not weights.is_relative_to(repository) or weights.is_symlink() or not weights.is_file():
        raise ValueError("weight source escaped the private repository boundary")
    weight_digest = hashlib.sha256()
    weight_size = 0
    with weights.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            weight_size += len(block)
            weight_digest.update(block)
    if weight_size != expected_weights_size or weight_digest.hexdigest() != expected_weights_sha256:
        raise ValueError("official ImageNet V1 weight hash or size mismatch")

    inputs = {
        "data/private/500-study-training-cohort.json": (Path(manifest_path), EXPECTED_MANIFEST_SHA256),
        "data/private/500-study-training-folds.json": (Path(folds_path), EXPECTED_FOLDS_SHA256),
    }
    members: dict[str, bytes] = {}
    for source in _local_import_closure(repository / "src"):
        relative = source.relative_to(repository).as_posix()
        members[relative] = source.read_bytes()
    adapter_relative = "src/rsna_knee/models/resnet18.py"
    adapter_path = repository / adapter_relative
    if not adapter_path.is_file():
        raise ValueError("ResNet-18 study model source is unavailable")
    members[adapter_relative] = adapter_path.read_bytes()
    for archive_name, (source, expected_hash) in inputs.items():
        resolved = source.resolve(strict=True)
        if not resolved.is_relative_to(repository):
            raise ValueError("private package input escaped repository")
        raw = resolved.read_bytes()
        if _sha256(raw) != expected_hash:
            raise ValueError("private source artifact hash mismatch")
        members[archive_name] = raw

    fold_protocol = _resnet_fold_protocol(
        members["data/private/500-study-training-cohort.json"],
        members["data/private/500-study-training-folds.json"], fold,
    )

    names = tuple(sorted(members))
    if (any(name.startswith("/") or ".." in Path(name).parts for name in names)
            or any(name.endswith((".pyc", ".pyo", ".pt", ".pth", ".ckpt", ".dcm", ".dicom"))
                   or "__pycache__" in Path(name).parts for name in names)):
        raise ValueError("comparison package contains an unsafe member")
    hashes = {name: _sha256(members[name]) for name in names}
    package_dir = output_dir / "package"
    archive_path = output_dir / "kaggle-resnet18-comparison.zip"
    inventory_path = output_dir / "inventory.json"
    upload_dir = output_dir / "upload"
    output_dir.mkdir(parents=True, exist_ok=False)
    package_dir.mkdir()
    upload_dir.mkdir()
    upload_package = upload_dir / "package"
    upload_package.mkdir()
    for name in names:
        for root in (package_dir, upload_package):
            destination = root.joinpath(*Path(name).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(members[name])
    with ZipFile(archive_path, "x", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name in names:
            info = ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            archive.writestr(info, members[name])
    archive_bytes = archive_path.read_bytes()
    inventory = {
        "schema_version": "kaggle-resnet18-comparison-inventory-v1",
        "package_sha256": _sha256(archive_bytes),
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "folds_sha256": EXPECTED_FOLDS_SHA256,
        "members": hashes,
        "weights": {
            "relative_path": "weights/resnet18-f37072fd.pth",
            "sha256": expected_weights_sha256,
            "size_bytes": expected_weights_size,
        },
        "excluded": ["competition_images", "reports", "predictions", "checkpoints", "credentials", "notebook_outputs"],
    }
    inventory_bytes = (json.dumps(inventory, indent=2, sort_keys=True) + "\n").encode("utf-8")
    inventory_path.write_bytes(inventory_bytes)
    inventory_sha256 = _sha256(inventory_bytes)
    (upload_dir / "inventory.json").write_bytes(inventory_bytes)
    (upload_dir / "kaggle-resnet18-comparison.zip.b64").write_bytes(base64.b64encode(archive_bytes))
    upload_weights = upload_dir / inventory["weights"]["relative_path"]
    upload_weights.parent.mkdir(parents=True)
    upload_weights.write_bytes(weights.read_bytes())

    source_pins = {
        "baseline": hashes["src/rsna_knee/training/baseline.py"],
        "resnet18": hashes[adapter_relative],
    }
    config_paths: dict[str, str] = {}
    notebook_paths: dict[str, str] = {}
    inventory_b64 = base64.b64encode(inventory_bytes).decode("ascii")
    run_suffix = "run01" if fold == 0 else f"fold{fold}-run01"
    for arm, model_kind in (("random", "resnet18_random"), ("imagenet1k_v1", "resnet18_imagenet1k_v1")):
        config_path = output_dir / f"kaggle_resnet18_{arm}.json"
        output_name = (
            f"/kaggle/working/resnet18-comparison/{run_suffix}-imagenet1k-v1"
            if fold == 0 and arm == "imagenet1k_v1"
            else f"/kaggle/working/resnet18-comparison/{run_suffix}-{arm}"
        )
        config: dict[str, object] = {
            "mode": "fold", "manifest_path": "resolved-from-private-package",
            "folds_path": "resolved-from-private-package", "image_root": "/kaggle/input/competitions/rsna-knee-abnormality-detection/train_series",
            "output_dir": output_name, "fold": fold, "seed": 42, "model_seed": 42,
            "epochs": 5, "max_steps": None, "overfit_limit": None,
            "output_size": 256, "max_slices": 8, "device": "cuda",
            "learning_rate": 0.001, "feature_width": 64, "model_kind": model_kind,
            "slice_microbatch": 2, "expected_member_count": len(names),
            "expected_inventory_sha256": inventory_sha256,
            "expected_archive_provenance": inventory["package_sha256"],
            "expected_manifest_sha256": inventory["manifest_sha256"],
            "expected_folds_sha256": inventory["folds_sha256"],
            "expected_source_sha256": source_pins["baseline"],
            "expected_resnet18_source_sha256": source_pins["resnet18"],
            "expected_training_count": fold_protocol["training_count"],
            "expected_validation_count": fold_protocol["validation_count"],
            "expected_steps": fold_protocol["steps"],
            "private_input_root": "/kaggle/input/datasets/davidsulleiro/rsna-knee-five-study-reader-private",
        }
        if model_kind == "resnet18_imagenet1k_v1":
            config.update({
                "weights_path": "resolved-from-private-input/weights/resnet18-f37072fd.pth",
                "weights_sha256": expected_weights_sha256,
                "weights_size": expected_weights_size,
            })
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        config_bytes = config_path.read_bytes()
        (upload_dir / config_path.name).write_bytes(config_bytes)
        config_paths[arm] = str(config_path)

        code = f'''RUN_TRAINING = False
if not RUN_TRAINING:
    print({{"training": "disabled", "arm": "{arm}", "protocol": "fold {fold}; {fold_protocol['training_count']} train; {fold_protocol['validation_count']} validation; 5 epochs; {fold_protocol['steps']} steps"}})
else:
    import base64, hashlib, io, json, sys, time, zipfile
    from pathlib import Path

    def sha256(raw):
        return hashlib.sha256(raw).hexdigest()

    def resolve_input_root(candidates, expected_sha256):
        for candidate in candidates:
            try:
                raw = (candidate / 'inventory.json').read_bytes()
            except OSError:
                continue
            if sha256(raw) == expected_sha256:
                return candidate
        raise RuntimeError('input_mount_unavailable')

    EXPECTED_INVENTORY_SHA256 = '{inventory_sha256}'
    INPUT_CANDIDATES = (
        Path('/kaggle/input/datasets/davidsulleiro/rsna-knee-five-study-reader-private'),
        Path('/kaggle/input/rsna-knee-five-study-reader-private'),
    )
    INPUT_ROOT = resolve_input_root(INPUT_CANDIDATES, EXPECTED_INVENTORY_SHA256)
    INVENTORY_BYTES = (INPUT_ROOT / 'inventory.json').read_bytes()
    if INVENTORY_BYTES != base64.b64decode('{inventory_b64}', validate=True):
        raise RuntimeError('inventory_content_mismatch')
    INVENTORY = json.loads(INVENTORY_BYTES)
    if INVENTORY['schema_version'] != 'kaggle-resnet18-comparison-inventory-v1':
        raise RuntimeError('inventory_schema_mismatch')
    ARCHIVE_BYTES = base64.b64decode((INPUT_ROOT / 'kaggle-resnet18-comparison.zip.b64').read_bytes(), validate=True)
    if sha256(ARCHIVE_BYTES) != INVENTORY['package_sha256'] or INVENTORY['package_sha256'] != '{inventory['package_sha256']}':
        raise RuntimeError('package_hash_mismatch')
    PACKAGE_ROOT = INPUT_ROOT / 'package'
    with zipfile.ZipFile(io.BytesIO(ARCHIVE_BYTES)) as archive:
        if set(archive.namelist()) != set(INVENTORY['members']):
            raise RuntimeError('package_member_mismatch')
        for name, expected_hash in INVENTORY['members'].items():
            raw = archive.read(name)
            if sha256(raw) != expected_hash or sha256((PACKAGE_ROOT / name).read_bytes()) != expected_hash:
                raise RuntimeError('package_member_hash_mismatch')
    if (INVENTORY['manifest_sha256'] != '{inventory['manifest_sha256']}'
            or INVENTORY['folds_sha256'] != '{inventory['folds_sha256']}'
            or INVENTORY['members']['src/rsna_knee/training/baseline.py'] != '{source_pins['baseline']}'
            or INVENTORY['members']['{adapter_relative}'] != '{source_pins['resnet18']}'):
        raise RuntimeError('source_or_fold_pin_mismatch')
    sys.path.insert(0, str(PACKAGE_ROOT / 'src'))
    import torch
    import torchvision
    from torchvision.models import resnet18
    probe = resnet18(weights=None)
    if probe.conv1.in_channels != 3 or probe.fc.out_features != 1000:
        raise RuntimeError('torchvision_resnet18_incompatible')
    del probe
    CONFIG = json.loads((INPUT_ROOT / '{config_path.name}').read_text(encoding='utf-8'))
    if (CONFIG['expected_inventory_sha256'] != EXPECTED_INVENTORY_SHA256
            or CONFIG['expected_member_count'] != len(INVENTORY['members'])
            or CONFIG['expected_archive_provenance'] != INVENTORY['package_sha256']
            or CONFIG['expected_manifest_sha256'] != INVENTORY['manifest_sha256']
            or CONFIG['expected_folds_sha256'] != INVENTORY['folds_sha256']
            or CONFIG['expected_source_sha256'] != INVENTORY['members']['src/rsna_knee/training/baseline.py']
            or CONFIG['expected_resnet18_source_sha256'] != INVENTORY['members']['{adapter_relative}']):
        raise RuntimeError('config_inventory_pin_mismatch')
    print({{'event': 'resnet18_run_start', 'model_kind': CONFIG['model_kind'],
        'torch_version': torch.__version__, 'torchvision_version': torchvision.__version__,
        'device': CONFIG['device'], 'training_study_count': CONFIG['expected_training_count'],
        'validation_study_count': CONFIG['expected_validation_count'],
        'epochs': CONFIG['epochs'], 'planned_steps': CONFIG['expected_steps'],
        'inventory_sha256': EXPECTED_INVENTORY_SHA256,
        'archive_sha256': INVENTORY['package_sha256'],
        'weights_sha256': INVENTORY['weights']['sha256'] if CONFIG['model_kind'] == 'resnet18_imagenet1k_v1' else None}})
    CONFIG.update({{
        'manifest_path': str(PACKAGE_ROOT / 'data/private/500-study-training-cohort.json'),
        'folds_path': str(PACKAGE_ROOT / 'data/private/500-study-training-folds.json'),
        'image_root': '/kaggle/input/competitions/rsna-knee-abnormality-detection/train_series',
    }})
    if CONFIG['model_kind'] == 'resnet18_imagenet1k_v1':
        weights_path = INPUT_ROOT / INVENTORY['weights']['relative_path']
        raw = weights_path.read_bytes()
        if len(raw) != INVENTORY['weights']['size_bytes'] or sha256(raw) != INVENTORY['weights']['sha256']:
            raise RuntimeError('offline_weights_pin_mismatch')
        if CONFIG['weights_sha256'] != INVENTORY['weights']['sha256'] or CONFIG['weights_size'] != INVENTORY['weights']['size_bytes']:
            raise RuntimeError('configured_weights_pin_mismatch')
        CONFIG['weights_path'] = str(weights_path)
    elif CONFIG['model_kind'] != 'resnet18_random':
        raise RuntimeError('model_kind_mismatch')
    if (CONFIG['model_kind'] != '{model_kind}' or CONFIG['fold'] != {fold} or CONFIG['seed'] != 42
            or CONFIG['epochs'] != 5 or CONFIG['expected_training_count'] != {fold_protocol['training_count']}
            or CONFIG['expected_validation_count'] != {fold_protocol['validation_count']} or CONFIG['expected_steps'] != {fold_protocol['steps']}
            or CONFIG['output_dir'] != '{output_name}'):
        raise RuntimeError('training_protocol_mismatch')
    OUTPUT_DIR = Path(CONFIG['output_dir'])
    if OUTPUT_DIR.exists():
        raise RuntimeError('private_output_must_be_new')
    from rsna_knee.training.baseline import TrainingConfig, run_training
    torch.set_num_threads(1)
    started = time.perf_counter()
    REPORT = run_training(TrainingConfig.from_mapping(CONFIG), execute=True)
    print({{'training_status': REPORT.get('status'), 'model_kind': REPORT.get('model_kind'),
        'training_study_count': REPORT.get('training_study_count'),
        'validation_study_count': REPORT.get('validation_study_count'),
        'steps_completed': REPORT.get('steps_completed'), 'macro_auc': REPORT.get('macro_auc'),
        'per_target_auc': REPORT.get('per_target_auc'), 'training_loss_by_epoch': REPORT.get('training_loss_by_epoch'),
        'model_initialization': REPORT.get('model_initialization'),
        'runtime': REPORT.get('runtime'), 'notebook_elapsed_seconds': time.perf_counter() - started}})
'''
        notebook_path = output_dir / f"kaggle_resnet18_{arm}.ipynb"
        notebook = {
            "cells": [
                {"cell_type": "markdown", "metadata": {}, "source": [
                    f"# Matched ResNet-18 {arm} training arm\n", "\n",
                    "Prepared only; execution is disabled by default. The run uses the unchanged fold-0 cohort and five-epoch protocol. It is one initialization arm of a two-run comparison, not a generalization or clinical claim.\n",
                ]},
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(keepends=True)},
            ],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5,
        }
        notebook_path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
        (upload_dir / notebook_path.name).write_bytes(notebook_path.read_bytes())
        notebook_paths[arm] = str(notebook_path)

    return {
        **inventory, "inventory_sha256": inventory_sha256,
        "archive_path": str(archive_path), "package_dir": str(package_dir),
        "inventory_path": str(inventory_path), "transport_dir": str(upload_dir),
        "transport_weight_path": str(upload_weights),
        "config_paths": config_paths, "notebook_paths": notebook_paths,
        "members": hashes,
    }


def build_resnet18_frozen_control_package(
    repository_root: Path,
    output_dir: Path,
    *,
    manifest_path: Path,
    folds_path: Path,
    weights_path: Path,
    expected_weights_sha256: str,
    expected_weights_size: int,
    fold: int = 0,
) -> dict[str, object]:
    """Prepare a separate disabled, one-arm ImageNet V1 frozen-backbone package."""
    output_dir = Path(output_dir)
    _validate_resnet_fold(fold)
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("frozen ResNet-18 output directory must be new")
    with TemporaryDirectory(prefix="rsna-resnet18-frozen-base-") as temporary:
        base = build_resnet18_comparison_package(
            repository_root, Path(temporary) / "comparison",
            manifest_path=manifest_path, folds_path=folds_path, weights_path=weights_path,
            expected_weights_sha256=expected_weights_sha256,
            expected_weights_size=expected_weights_size,
            fold=fold,
        )
        base_inventory = json.loads(Path(base["inventory_path"]).read_text(encoding="utf-8"))
        base_transport = Path(base["transport_dir"])
        source_config = json.loads(Path(base["config_paths"]["imagenet1k_v1"]).read_text(encoding="utf-8"))
        source_config.update({
            "freeze_backbone": True,
            "output_dir": ("/kaggle/working/resnet18-frozen-backbone/run01" if fold == 0
                            else f"/kaggle/working/resnet18-frozen-backbone/fold{fold}-run01"),
        })
        config_name = "kaggle_resnet18_frozen_backbone.json"
        notebook_name = "kaggle_resnet18_frozen_backbone.ipynb"
        output_dir.mkdir(parents=True, exist_ok=False)
        package_dir = output_dir / "package"
        shutil.copytree(Path(base["package_dir"]), package_dir)
        archive_path = output_dir / "kaggle-resnet18-frozen-backbone.zip"
        shutil.copyfile(base["archive_path"], archive_path)
        inventory = dict(base_inventory)
        inventory["schema_version"] = "kaggle-resnet18-frozen-control-inventory-v1"
        inventory["freeze_backbone"] = True
        inventory_bytes = (json.dumps(inventory, indent=2, sort_keys=True) + "\n").encode("utf-8")
        source_config["expected_inventory_sha256"] = _sha256(inventory_bytes)
        inventory_path = output_dir / "inventory.json"
        inventory_path.write_bytes(inventory_bytes)

        transport_dir = output_dir / "upload"
        transport_dir.mkdir()
        shutil.copytree(package_dir, transport_dir / "package")
        (transport_dir / "inventory.json").write_bytes(inventory_bytes)
        (transport_dir / "kaggle-resnet18-comparison.zip.b64").write_bytes(
            (base_transport / "kaggle-resnet18-comparison.zip.b64").read_bytes()
        )
        weights_name = inventory["weights"]["relative_path"]
        weights_target = transport_dir / weights_name
        weights_target.parent.mkdir(parents=True)
        shutil.copyfile(base_transport / weights_name, weights_target)

        config_path = output_dir / config_name
        config_path.write_text(json.dumps(source_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (transport_dir / config_name).write_bytes(config_path.read_bytes())
        source_notebook = json.loads(Path(base["notebook_paths"]["imagenet1k_v1"]).read_text(encoding="utf-8"))
        markdown_cells = [cell for cell in source_notebook["cells"] if cell.get("cell_type") == "markdown"]
        if len(markdown_cells) != 1:
            raise ValueError("ResNet-18 notebook markdown-cell contract changed")
        markdown_cells[0]["source"] = [
            f"# Frozen ImageNet V1 head-only control — fold {fold}\n",
            "\n",
            f"Prepared only; execution is disabled by default. This single frozen-backbone control uses the existing fold {fold} assignment and the five-epoch protocol. It is not a two-arm comparison, external test, generalization, or clinical claim.\n",
        ]
        code_cells = [cell for cell in source_notebook["cells"] if cell.get("cell_type") == "code"]
        if len(code_cells) != 1:
            raise ValueError("ResNet-18 notebook code-cell contract changed")
        code = "".join(code_cells[0]["source"])
        old_inventory_b64 = base64.b64encode(Path(base["inventory_path"]).read_bytes()).decode("ascii")
        new_inventory_b64 = base64.b64encode(inventory_bytes).decode("ascii")
        code = code.replace(old_inventory_b64, new_inventory_b64)
        code = code.replace(base["inventory_sha256"], _sha256(inventory_bytes))
        code = code.replace("kaggle-resnet18-comparison-inventory-v1", "kaggle-resnet18-frozen-control-inventory-v1")
        protocol = (f"fold {fold}; {source_config['expected_training_count']} train; "
                    f"{source_config['expected_validation_count']} validation; 5 epochs; "
                    f"{source_config['expected_steps']} steps")
        code = code.replace(
            f'"arm": "imagenet1k_v1", "protocol": "{protocol}"',
            f'"arm": "imagenet1k_v1_frozen_backbone", "protocol": "{protocol}; frozen backbone"',
        )
        code = code.replace("'kaggle_resnet18_imagenet1k_v1.json'", repr(config_name))
        source_output = (
            "/kaggle/working/resnet18-comparison/run01-imagenet1k-v1" if fold == 0
            else f"/kaggle/working/resnet18-comparison/fold{fold}-run01-imagenet1k_v1"
        )
        code = code.replace(
            f"CONFIG['output_dir'] != '{source_output}'",
            f"CONFIG['output_dir'] != '{source_config['output_dir']}'\n            or CONFIG.get('freeze_backbone') is not True",
        )
        code_cells[0]["source"] = code.splitlines(keepends=True)
        notebook_path = output_dir / notebook_name
        notebook_path.write_text(json.dumps(source_notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        (transport_dir / notebook_name).write_bytes(notebook_path.read_bytes())

    return {
        "package_sha256": _sha256(archive_path.read_bytes()),
        "inventory_sha256": _sha256(inventory_bytes),
        "manifest_sha256": inventory["manifest_sha256"],
        "folds_sha256": inventory["folds_sha256"],
        "weights_sha256": inventory["weights"]["sha256"],
        "weights_size": inventory["weights"]["size_bytes"],
        "member_count": len(inventory["members"]),
        "upload_file_count": sum(1 for path in transport_dir.rglob("*") if path.is_file()),
        "archive_path": str(archive_path), "package_dir": str(package_dir),
        "inventory_path": str(inventory_path), "transport_dir": str(transport_dir),
        "config_path": str(config_path), "notebook_path": str(notebook_path),
        "members": inventory["members"],
    }


def build_overfit_sanity_package(
    repository_root: Path,
    output_dir: Path,
    *,
    manifest_path: Path,
    folds_path: Path,
    config_path: Path,
    notebook_path: Path,
) -> dict[str, object]:
    """Prepare a fresh, disabled eight-study / 240-step diagnostic bundle."""
    repository = Path(repository_root).resolve(strict=True)
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("overfit sanity package output directory must be new")

    config_resolved = Path(config_path).resolve(strict=True)
    notebook_resolved = Path(notebook_path).resolve(strict=True)
    if not config_resolved.is_relative_to(repository) or not notebook_resolved.is_relative_to(repository):
        raise ValueError("package input escaped repository")
    config_template = json.loads(config_resolved.read_text(encoding="utf-8"))
    if not isinstance(config_template, dict):
        raise ValueError("overfit sanity config must be an object")
    required_template = {
        "mode": "overfit", "fold": 0, "seed": 42, "overfit_limit": 8,
        "output_size": 256, "max_slices": 8, "device": "cuda",
        "learning_rate": 0.001, "feature_width": 64,
    }
    if any(config_template.get(key) != value for key, value in required_template.items()):
        raise ValueError("overfit sanity config template changed")

    inputs = {
        "data/private/500-study-training-cohort.json": (Path(manifest_path), EXPECTED_MANIFEST_SHA256),
        "data/private/500-study-training-folds.json": (Path(folds_path), EXPECTED_FOLDS_SHA256),
    }
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
        if _sha256(raw) != expected_hash:
            raise ValueError("private source artifact hash mismatch")
        members[archive_name] = raw

    names = tuple(sorted(members))
    if any(name.startswith("/") or ".." in Path(name).parts for name in names):
        raise ValueError("package member path is unsafe")
    if any(name.endswith((".pyc", ".pyo", ".pt", ".pth", ".ckpt", ".dcm", ".dicom"))
           or "__pycache__" in Path(name).parts for name in names):
        raise ValueError("package member type is not allowed")
    if "src/rsna_knee/__init__.py" not in members:
        raise ValueError("package is not import closed")

    output_dir.mkdir(parents=True, exist_ok=False)
    package_dir = output_dir / "package"
    package_dir.mkdir()
    archive_path = output_dir / "kaggle-training-learning-diagnostics-sanity.zip"
    transport_dir = output_dir / "upload"
    transport_package_dir = transport_dir / "package"
    transport_inventory_path = transport_dir / "inventory.json"
    transport_config_path = transport_dir / "kaggle_training_learning_diagnostics_sanity.json"
    transport_archive_path = transport_dir / "kaggle-training-learning-diagnostics-sanity.zip.b64"
    inventory_path = output_dir / "inventory.json"
    prepared_config_path = output_dir / "kaggle_training_learning_diagnostics_sanity.json"
    prepared_notebook_path = output_dir / "kaggle_training_learning_diagnostics_sanity.ipynb"
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
        inventory = {
            "schema_version": "kaggle-training-package-inventory-v1",
            "package_sha256": _sha256(archive_path.read_bytes()),
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "folds_sha256": EXPECTED_FOLDS_SHA256,
            "members": inventory_members,
            "exclusions": ["bytecode", "reports", "evidence", "images", "checkpoints", "outputs", "credentials"],
        }
        inventory_bytes = (json.dumps(inventory, indent=2, sort_keys=True) + "\n").encode("utf-8")
        inventory_path.write_bytes(inventory_bytes)

        prepared_config = dict(config_template)
        prepared_config.update({
            "manifest_path": "resolved-from-private-input",
            "folds_path": "resolved-from-private-input",
            "output_dir": "/kaggle/working/training-learning-diagnostics/sanity-run01",
            "epochs": 30,
            "max_steps": 240,
            "expected_member_count": len(names),
            "expected_manifest_sha256": inventory["manifest_sha256"],
            "expected_folds_sha256": inventory["folds_sha256"],
            "expected_source_sha256": inventory_members["src/rsna_knee/training/baseline.py"],
            "expected_inventory_sha256": _sha256(inventory_bytes),
            "expected_archive_provenance": inventory["package_sha256"],
            "expected_training_count": 8,
            "expected_steps": 240,
            "diagnostic_target_bce": 0.10,
        })
        prepared_config_path.write_text(json.dumps(prepared_config, indent=2) + "\n", encoding="utf-8")

        # Kaggle recursively extracts uploaded ZIP files, so transport the pinned
        # archive as strict ASCII base64 while retaining the package tree itself.
        transport_dir.mkdir()
        for name in names:
            target = transport_package_dir.joinpath(*Path(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(members[name])
        transport_inventory_path.write_bytes(inventory_bytes)
        transport_config_path.write_bytes(prepared_config_path.read_bytes())
        transport_archive_path.write_bytes(base64.b64encode(archive_path.read_bytes()))

        notebook = json.loads(notebook_resolved.read_text(encoding="utf-8"))
        if any(cell.get("outputs", []) for cell in notebook.get("cells", [])):
            raise ValueError("notebook must not contain saved outputs")
        markdown = {
            "cell_type": "markdown", "metadata": {},
            "source": [
                "# Eight-study training-subset sanity diagnostic\n",
                "\n",
                "Prepared only; execution is disabled by default. The predeclared BCE threshold is a diagnostic target, not a guarantee. Results describe the same eight training studies and are not validation or generalization evidence.\n",
            ],
        }
        code = '''import hashlib
import base64
import io
import json
import sys
import time
from pathlib import Path

RUN_TRAINING = False
INPUT_ROOT = Path('/kaggle/input/rsna-knee-five-study-reader-private')
IMAGE_ROOT = Path('/kaggle/input/competitions/rsna-knee-abnormality-detection/train_series')
OUTPUT_DIR = Path('/kaggle/working/training-learning-diagnostics/sanity-run01')
PACKAGE_ROOT = INPUT_ROOT / 'package'
ARCHIVE_B64_PATH = INPUT_ROOT / 'kaggle-training-learning-diagnostics-sanity.zip.b64'
INVENTORY_PATH = INPUT_ROOT / 'inventory.json'
CONFIG_PATH = INPUT_ROOT / 'kaggle_training_learning_diagnostics_sanity.json'
EXPECTED_INVENTORY_SHA256 = "__INVENTORY_SHA256__"
EXPECTED_ARCHIVE_SHA256 = "__ARCHIVE_SHA256__"
EXPECTED_SOURCE_SHA256 = "__SOURCE_SHA256__"
EXPECTED_INVENTORY_B64 = "__INVENTORY_B64__"

def sha256(raw):
    return hashlib.sha256(raw).hexdigest()

if not RUN_TRAINING:
    print({'training': 'disabled', 'protocol': '8 studies; fold 0; seed 42; 30 epochs; maximum 240 steps'})

inventory_bytes = INVENTORY_PATH.read_bytes()
if sha256(inventory_bytes) != EXPECTED_INVENTORY_SHA256:
    raise RuntimeError('inventory_hash_mismatch')
if inventory_bytes.decode('utf-8') != __import__('base64').b64decode(EXPECTED_INVENTORY_B64).decode('utf-8'):
    raise RuntimeError('inventory_content_mismatch')
inventory = json.loads(inventory_bytes)
archive_bytes = base64.b64decode(ARCHIVE_B64_PATH.read_bytes().decode('ascii'), validate=True)
if sha256(archive_bytes) != EXPECTED_ARCHIVE_SHA256 or inventory['package_sha256'] != EXPECTED_ARCHIVE_SHA256:
    raise RuntimeError('package_hash_mismatch')
import zipfile
with zipfile.ZipFile(io.BytesIO(archive_bytes)) as package:
    if set(package.namelist()) != set(inventory['members']):
        raise RuntimeError('package_member_mismatch')
    for name, expected_hash in inventory['members'].items():
        if sha256(package.read(name)) != expected_hash:
            raise RuntimeError('package_member_hash_mismatch')
        if sha256((PACKAGE_ROOT / name).read_bytes()) != expected_hash:
            raise RuntimeError('package_tree_hash_mismatch')
if inventory['manifest_sha256'] != "__MANIFEST_SHA256__" or inventory['folds_sha256'] != "__FOLDS_SHA256__":
    raise RuntimeError('private_input_hash_mismatch')
if inventory['members']['src/rsna_knee/training/baseline.py'] != EXPECTED_SOURCE_SHA256:
    raise RuntimeError('training_source_hash_mismatch')
sys.path.insert(0, str(PACKAGE_ROOT / 'src'))
from rsna_knee.training.baseline import TrainingConfig, run_training
config_value = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
config_value.update({
    'manifest_path': str(PACKAGE_ROOT / 'data/private/500-study-training-cohort.json'),
    'folds_path': str(PACKAGE_ROOT / 'data/private/500-study-training-folds.json'),
    'image_root': str(IMAGE_ROOT),
    'output_dir': str(OUTPUT_DIR),
})
config = TrainingConfig.from_mapping(config_value)
if (config.mode, config.fold, config.seed, config.epochs, config.max_steps, config.overfit_limit,
        config.output_size, config.max_slices, config.feature_width, config.learning_rate) != (
        'overfit', 0, 42, 30, 240, 8, 256, 8, 64, 0.001):
    raise RuntimeError('training_protocol_mismatch')
if RUN_TRAINING:
    if OUTPUT_DIR.exists():
        raise RuntimeError('private_output_must_be_new')
    started = time.perf_counter()
    report = run_training(config, execute=True)
    print({'training': {
        'status': report.get('status'), 'training_study_count': report.get('training_study_count'),
        'steps_completed': report.get('steps_completed'), 'train_loss_start': report.get('train_loss_start'),
        'train_loss_end': report.get('train_loss_end'),
        'final_subset_diagnostics': report.get('final_subset_diagnostics'),
        'training_loss_by_epoch': report.get('training_loss_by_epoch'),
        'notebook_elapsed_seconds': time.perf_counter() - started,
    }})
'''
        replacements = {
            "__INVENTORY_SHA256__": _sha256(inventory_bytes),
            "__ARCHIVE_SHA256__": inventory["package_sha256"],
            "__SOURCE_SHA256__": inventory_members["src/rsna_knee/training/baseline.py"],
            "__INVENTORY_B64__": base64.b64encode(inventory_bytes).decode("ascii"),
            "__MANIFEST_SHA256__": inventory["manifest_sha256"],
            "__FOLDS_SHA256__": inventory["folds_sha256"],
        }
        for marker, value in replacements.items():
            if marker not in code:
                raise ValueError("overfit sanity notebook pin contract changed")
            code = code.replace(marker, value)
        code_cell = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                     "source": code.splitlines(keepends=True)}
        notebook["cells"] = [markdown, code_cell]
        prepared_notebook_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    except Exception:
        # Preserve uncertain output for inspection; never silently replace it.
        raise

    return {
        **inventory,
        "inventory_sha256": _sha256(inventory_bytes),
        "archive_path": str(archive_path),
        "package_dir": str(package_dir),
        "transport_dir": str(transport_dir),
        "transport_archive_path": str(transport_archive_path),
        "inventory_path": str(inventory_path),
        "notebook_path": str(prepared_notebook_path),
        "config_path": str(prepared_config_path),
    }


def build_learning_path_diagnostic_package(
    repository_root: Path,
    output_dir: Path,
    *,
    manifest_path: Path,
    folds_path: Path,
    config_path: Path,
    notebook_path: Path,
) -> dict[str, object]:
    """Prepare a new disabled-default first-eight-study learning-path bundle."""
    repository = Path(repository_root).resolve(strict=True)
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("learning-path diagnostic output directory must be new")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="learning-path-package-", dir=output_dir.parent) as temp:
        base = build_overfit_sanity_package(
            repository, Path(temp) / "base",
            manifest_path=manifest_path, folds_path=folds_path,
            config_path=config_path, notebook_path=notebook_path,
        )
        base_exclusions = json.loads(Path(base["inventory_path"]).read_text(encoding="utf-8"))["exclusions"]
        base_config = json.loads(Path(base["config_path"]).read_text(encoding="utf-8"))
        base_package = Path(base["package_dir"])
        members = {
            path.relative_to(base_package).as_posix(): path.read_bytes()
            for path in base_package.rglob("*") if path.is_file()
        }
    helper_path = repository / "src/rsna_knee/training/learning_path.py"
    if not helper_path.is_file():
        raise ValueError("learning-path helper source is unavailable")
    members["src/rsna_knee/training/learning_path.py"] = helper_path.read_bytes()
    names = tuple(sorted(members))
    hashes = {name: _sha256(members[name]) for name in names}

    output_dir.mkdir(parents=True, exist_ok=False)
    package_dir = output_dir / "package"
    package_dir.mkdir()
    archive_path = output_dir / "kaggle-training-learning-path-diagnostic.zip"
    inventory_path = output_dir / "inventory.json"
    config_out = output_dir / "kaggle_training_learning_path_diagnostic.json"
    notebook_out = output_dir / "kaggle_training_learning_path_diagnostic.ipynb"
    transport_dir = output_dir / "upload"
    transport_dir.mkdir()
    transport_package = transport_dir / "package"
    transport_package.mkdir()
    for name in names:
        target = package_dir.joinpath(*Path(name).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(members[name])
        upload_target = transport_package.joinpath(*Path(name).parts)
        upload_target.parent.mkdir(parents=True, exist_ok=True)
        upload_target.write_bytes(members[name])
    with ZipFile(archive_path, "x", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name in names:
            info = ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            archive.writestr(info, members[name])
    inventory = {
        "schema_version": "kaggle-learning-path-inventory-v1",
        "package_sha256": _sha256(archive_path.read_bytes()),
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "folds_sha256": EXPECTED_FOLDS_SHA256,
        "members": hashes,
        "exclusions": base_exclusions,
    }
    inventory_bytes = (json.dumps(inventory, indent=2, sort_keys=True) + "\n").encode("utf-8")
    inventory_path.write_bytes(inventory_bytes)
    config = base_config
    config.update({
        "diagnostic_kind": "scratch_learning_path_probe",
        "max_diagnostic_steps": 2,
        "expected_member_count": len(names),
        "expected_helper_sha256": hashes["src/rsna_knee/training/learning_path.py"],
        "expected_inventory_sha256": _sha256(inventory_bytes),
        "expected_archive_provenance": inventory["package_sha256"],
    })
    config_bytes = (json.dumps(config, indent=2, sort_keys=True) + "\n").encode("utf-8")
    config_out.write_bytes(config_bytes)
    (transport_dir / "inventory.json").write_bytes(inventory_bytes)
    (transport_dir / config_out.name).write_bytes(config_bytes)
    (transport_dir / "kaggle-training-learning-path-diagnostic.zip.b64").write_bytes(
        base64.b64encode(archive_path.read_bytes())
    )
    inventory_b64 = base64.b64encode(inventory_bytes).decode("ascii")
    code = f'''RUN_DIAGNOSTIC = False
if not RUN_DIAGNOSTIC:
    print({{"diagnostic": "disabled", "input_audit_studies": 8, "probe_studies": 2, "optimizer_steps_total": 2}})
else:
    import base64, hashlib, io, json, sys, zipfile
    from pathlib import Path

    def resolve_input_root(candidates, expected_sha256):
        for candidate in candidates:
            try:
                inventory_bytes = (candidate / 'inventory.json').read_bytes()
            except OSError:
                continue
            if hashlib.sha256(inventory_bytes).hexdigest() == expected_sha256:
                return candidate
        raise RuntimeError('input_mount_unavailable')

    INPUT_CANDIDATES = (
        Path('/kaggle/input/datasets/davidsulleiro/rsna-knee-five-study-reader-private'),
        Path('/kaggle/input/rsna-knee-five-study-reader-private'),
    )
    EXPECTED_INVENTORY_SHA256 = '{_sha256(inventory_bytes)}'
    input_root = resolve_input_root(INPUT_CANDIDATES, EXPECTED_INVENTORY_SHA256)
    competition_root = Path('/kaggle/input/competitions/rsna-knee-abnormality-detection/train_series')
    inventory_raw = (input_root / 'inventory.json').read_bytes()
    if inventory_raw != base64.b64decode('{inventory_b64}', validate=True):
        raise RuntimeError('inventory_content_mismatch')
    import torch
    torch.set_num_threads(1)
    inventory = json.loads(inventory_raw)
    archive_raw = base64.b64decode((input_root / 'kaggle-training-learning-path-diagnostic.zip.b64').read_bytes(), validate=True)
    if hashlib.sha256(archive_raw).hexdigest() != inventory['package_sha256']:
        raise RuntimeError('archive_hash_mismatch')
    work = Path('/kaggle/working/learning-path-diagnostic')
    work.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(archive_raw)) as archive:
        if set(archive.namelist()) != set(inventory['members']):
            raise RuntimeError('member_set_mismatch')
        for name, expected in inventory['members'].items():
            raw = archive.read(name)
            if hashlib.sha256(raw).hexdigest() != expected:
                raise RuntimeError('member_hash_mismatch')
        archive.extractall(work)
    sys.path.insert(0, str(work / 'src'))
    from rsna_knee.data.training_dataset import TrainingStudyDataset
    from rsna_knee.training.learning_path import diagnose_scratch_learning_path
    manifest = work / 'data/private/500-study-training-cohort.json'
    folds = work / 'data/private/500-study-training-folds.json'
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != inventory['manifest_sha256'] or hashlib.sha256(folds.read_bytes()).hexdigest() != inventory['folds_sha256']:
        raise RuntimeError('cohort_pin_mismatch')
    dataset = TrainingStudyDataset(manifest, folds, competition_root, partition='train', fold=0, output_size=256, max_slices=8)
    subset = dataset.bounded_training_subset(8)
    rows, input_means, input_stds = [], [], []
    duplicate_pairs = 0
    for index in range(8):
        images, masks, targets, _ = subset[index]
        values = [image.masked_select(mask) for image, mask in zip(images, masks)]
        input_means.append(sum(float(value.mean()) for value in values) / len(values))
        input_stds.append(sum(float(value.std(unbiased=False)) for value in values) / len(values))
        rows.append((images, masks, targets))
    for left in range(8):
        for right in range(left + 1, 8):
            a, am, _ = rows[left]
            b, bm, _ = rows[right]
            same = len(a) == len(b) and all(x.shape == y.shape and torch.equal(x, y) and torch.equal(mx, my) for x, mx, y, my in zip(a, am, b, bm))
            duplicate_pairs += int(same)
    probe = diagnose_scratch_learning_path(*rows[0][:2], *rows[1][:2], rows[0][2], steps=2)
    finite_inputs = all(torch.isfinite(value).all().item() for row in rows for series in row[0] for value in (series,))
    print({{"input_audit_study_count": 8, "probe_study_count": 2, "optimizer_steps_total": 2,
        "finite_inputs": finite_inputs, "unique_exact_input_pairs": 28 - duplicate_pairs,
        "duplicate_exact_input_pairs": duplicate_pairs, "input_mean_range": [min(input_means), max(input_means)],
        "input_std_range": [min(input_stds), max(input_stds)], "probe": probe}})
'''
    notebook = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [
                "# Eight-study scratch learning-path diagnostic\n",
                "\nPrepared only. Execution is disabled by default. If separately authorized and enabled, this loads only the unchanged fold-0 first eight training studies, audits aggregate input diversity over those eight, and performs exactly two optimizer steps with a fresh CPU scratch model on the first two studies. It emits aggregate diagnostics only; this is not validation or generalization evidence.\n",
            ]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(keepends=True)},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    notebook_out.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    return {
        **inventory, "inventory_sha256": _sha256(inventory_bytes), "archive_path": str(archive_path),
        "package_dir": str(package_dir), "transport_dir": str(transport_dir),
        "inventory_path": str(inventory_path), "notebook_path": str(notebook_out),
        "config_path": str(config_out), "members": hashes,
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
