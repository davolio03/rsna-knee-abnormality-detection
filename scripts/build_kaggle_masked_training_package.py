"""Build an offline, opt-in Kaggle package for masked soft-label training."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import os
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable


INVENTORY_SCHEMA = "masked-soft-kaggle-package-v1"
ENTRY_MODULES = (
    "rsna_knee.training.masked_soft",
    "rsna_knee.training.masked_preflight",
    # The generated notebook loads this runtime dependency by module-name string,
    # so the AST import-closure visitor cannot discover it from source imports.
    "rsna_knee.inference.offline_codec_bootstrap",
)
SOURCE_MODULE_ALLOWLIST = frozenset({
    "rsna_knee.training.masked_soft", "rsna_knee.training.masked_preflight",
    "rsna_knee.training.baseline", "rsna_knee.data.masked_soft_dataset",
    "rsna_knee.data.dicom_benchmark", "rsna_knee.data.dicom_preprocessing",
    "rsna_knee.data.preprocessing", "rsna_knee.data.training_dataset",
    "rsna_knee.labels.report_source_comparison", "rsna_knee.labels.full_cohort_prefilter",
    "rsna_knee.data.cohort", "rsna_knee.labels.subscription", "rsna_knee.labels.siwc_responses",
    "rsna_knee.validation.cohort", "rsna_knee.validation.multilabel",
    "rsna_knee.inference.offline_codec_bootstrap",
    "rsna_knee.models.resnet18", "rsna_knee.models.baseline",
    "rsna_knee.labels.siwc_auth", "rsna_knee.labels.prompt_comparison",
})
MANIFEST_SHA256 = "617ea1c267960b1e903b332a78388db3b6ab80d44374721df5a4c5e184e8961b"
FOLDS_SHA256 = "b816c7902a5d0caa68a9e818d3f913943bc61cd2e6db65c4de069f5cd94e15f9"
POLICY_SHA256 = "7a1b41d3d96c137859d83d78927fd020afba8ce4034cb0addbed67e730280662"
COHORT_SHA256 = "e88c61c22a463737f16951b7c76db03834034aecd1b31dc61b4ce495080b2be2"
BASELINE_FOLDS_SHA256 = "b82b376b0086201e3364b11a0203b62890e628442075a52ea4a9e09ee0e778ac"
WEIGHTS_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
WEIGHTS_SIZE = 46830571
CODEC_SHA256 = "2af0d5084dc70cb3daee2becf7a81aedb13093461aa64ceb532527c7701e3896"
CODEC_MANIFEST_SHA256 = "50f39d3f627b3dc06a8080ba51dd6424898ef34fa4c4e21a84593dd326eb776d"
HSR_SOURCE_SHA256 = "47aae9f98cd24fa1d5dd2c2e3433e01774f71e147377e34906cbafc8ef8b0462"
HSR_NOTEBOOK_SHA256 = "361ac91914b0ee5ef85aabb47da272cb238bb4ab730974a3566b5d46b520ab21"

ASSETS = {
    "full-cohort-soft-label-manifest.json": (
        "data/private/full-cohort-prefilter/run01/full-cohort-soft-label-manifest.json", MANIFEST_SHA256),
    "full-cohort-grouped-folds.json": (
        "data/private/full-cohort-prefilter/run01/full-cohort-grouped-folds.json", FOLDS_SHA256),
    "baseline-cohort.json": ("data/private/500-study-training-cohort.json", COHORT_SHA256),
    "baseline-folds.json": ("data/private/500-study-training-folds.json", BASELINE_FOLDS_SHA256),
    "fold0-inference-source.zip": (
        "data/private/fold0-submission/run12-partial-recovery-candidate-v5-final/fold0-inference-source.zip", HSR_SOURCE_SHA256),
    "fold0-inference-notebook.ipynb": (
        "data/private/fold0-submission/run12-partial-recovery-candidate-v5-final/kaggle_fold0_submission.ipynb", HSR_NOTEBOOK_SHA256),
}
UPLOAD_ASSETS = frozenset({
    "full-cohort-soft-label-manifest.json", "full-cohort-grouped-folds.json",
})


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_hash(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def _module_path(source_root: Path, module: str) -> Path | None:
    path = source_root.joinpath(*module.split(".")).with_suffix(".py")
    package_init = source_root.joinpath(*module.split("."), "__init__.py")
    return package_init if package_init.is_file() else (path if path.is_file() else None)


def _import_closure(source_root: Path) -> set[Path]:
    # Runtime imports in the explicitly used training path are declared here;
    # imports in unrelated helper functions are not part of this execution.
    pending = [*ENTRY_MODULES, "rsna_knee.training.baseline", "rsna_knee.models.resnet18",
               "rsna_knee.models.baseline",
               "rsna_knee.data.masked_soft_dataset", "rsna_knee.labels.full_cohort_prefilter"]
    found: set[Path] = set()
    while pending:
        module = pending.pop()
        if module not in SOURCE_MODULE_ALLOWLIST:
            raise ValueError("source_module_not_allowlisted")
        path = _module_path(source_root, module)
        if path is None or path in found:
            continue
        found.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]

        class Visitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                return

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                return

            def visit_Import(self, node: ast.Import) -> None:
                for alias in node.names:
                    if alias.name.startswith("rsna_knee."):
                        self._accept(alias.name)

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                if node.level:
                    parts = package.split(".")
                    base = ".".join(parts[:len(parts) - node.level + 1] + ([node.module] if node.module else []))
                else:
                    base = node.module or ""
                if base.startswith("rsna_knee."):
                    self._accept(base)
                    for alias in node.names:
                        candidate = f"{base}.{alias.name}"
                        if _module_path(source_root, candidate) is not None:
                            self._accept(candidate)

            def _accept(self, name: str) -> None:
                if name not in SOURCE_MODULE_ALLOWLIST:
                    raise ValueError(f"source_module_not_allowlisted:{name}")
                pending.append(name)

        Visitor().visit(tree)
    # Keep explicit package initializers for zip import, without including unrelated files.
    for file_path in tuple(found):
        parent = file_path.parent
        while parent != source_root.parent and parent.is_relative_to(source_root):
            init = parent / "__init__.py"
            if init.is_file():
                found.add(init)
            parent = parent.parent
    return found


def _zip_source(source_root: Path, paths: Iterable[Path]) -> tuple[bytes, dict[str, str]]:
    from io import BytesIO
    stream = BytesIO()
    members: dict[str, str] = {}
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(paths, key=lambda item: item.relative_to(source_root).as_posix()):
            relative = path.relative_to(source_root).as_posix()
            if not relative.endswith(".py") or relative.startswith("../"):
                raise ValueError("source_member_invalid")
            payload = path.read_bytes()
            info = zipfile.ZipInfo(relative, (2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, payload)
            members[relative] = _hash(payload)
    return stream.getvalue(), members


def _safe_output(output: Path, private_root: Path) -> tuple[Path, Path]:
    root = private_root.resolve(strict=True)
    if private_root.is_symlink() or any(parent.is_symlink() for parent in [output, *output.parents]):
        raise ValueError("output_symlink_forbidden")
    target = output.resolve(strict=False)
    if not target.is_relative_to(root) or target == root or target.exists():
        raise ValueError("output_path_invalid")
    return root, target


def _validate_codec(path: Path) -> None:
    if _file_hash(path) != (CODEC_SHA256, 7943426):
        raise ValueError("codec_asset_mismatch")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or "manifest.json" not in names:
            raise ValueError("codec_manifest_invalid")
        manifest_bytes = archive.read("manifest.json")
        if _hash(manifest_bytes) != CODEC_MANIFEST_SHA256:
            raise ValueError("codec_manifest_mismatch")
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        if manifest.get("target") != {
            "implementation": "CPython", "machine": "x86_64", "minimum_glibc": "2.17",
            "python": "3.13", "system": "Linux",
        }:
            raise ValueError("codec_target_mismatch")
        filenames = {item.get("filename") for item in manifest.get("packages", []) if isinstance(item, dict)}
        expected = {
            "pylibjpeg-2.1.0-py3-none-any.whl",
            "pylibjpeg_libjpeg-2.4.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
            "pylibjpeg_openjpeg-2.5.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
            "pyjpegls-1.5.1-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        }
        if filenames != expected or set(names) != {"manifest.json", *filenames}:
            raise ValueError("codec_wheel_inventory_mismatch")


def _notebook_source() -> str:
    return r'''from pathlib import Path
import base64, hashlib, importlib, io, json, sys, zipfile

# The package builder deliberately emits a planning-only notebook.
RUN_STAGE = "plan"  # plan | decode_preflight | full_training
RUN_FULL_TRAINING = False
OUTPUT_RUN_NAME = "run01"  # Change for each execution; never reuse an output directory.
RESUME_CHECKPOINT = None  # Set only to an explicitly selected mounted checkpoint file.
RESUME_CHECKPOINT_SHA256 = None  # Required SHA-256 of RESUME_CHECKPOINT; keep both unset on the first segment.
MAX_EPOCHS_PER_RUN = 2
MAX_RUNTIME_SECONDS = 28800
INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working")
PREPARED_ASSET_ROOT = None  # Optional exact mounted dataset directory.
COMPETITION_IMAGE_ROOT = None  # Optional exact competition image directory.
EXPECTED_INVENTORY_SHA256 = "__INVENTORY_SHA256__"
EXPECTED_SOURCE_SHA256 = "__SOURCE_SHA256__"
EXPECTED_CODEC_SHA256 = "__CODEC_SHA256__"
EXPECTED_CODEC_MANIFEST_SHA256 = "__CODEC_MANIFEST_SHA256__"

def fail(code):
    raise RuntimeError(code)

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024): h.update(chunk)
    return h.hexdigest()

def _verify_asset_root(root):
    root = Path(root)
    marker = root / "masked-training-inventory.json"
    if root.is_symlink() or marker.is_symlink() or not root.is_dir() or not marker.is_file():
        fail("prepared_input_missing_or_ambiguous")
    if digest(marker) != EXPECTED_INVENTORY_SHA256: fail("inventory_pin_mismatch")
    inventory = json.loads(marker.read_text(encoding="utf-8"))
    if inventory.get("schema_version") != "masked-soft-kaggle-package-v1": fail("inventory_schema_invalid")
    expected = ("masked-training-source.zip.b64", "full-cohort-soft-label-manifest.json",
                "full-cohort-grouped-folds.json", "resnet18-f37072fd.pth", "offline-codecs.zip.b64")
    if any((root / name).is_symlink() or not (root / name).is_file() for name in expected):
        fail("prepared_asset_missing")
    return root, inventory

def find_prepared_asset_root(input_root, explicit_asset_root=None):
    if explicit_asset_root is not None:
        return _verify_asset_root(explicit_asset_root)[0]
    input_root = Path(input_root)
    if input_root.is_symlink() or not input_root.is_dir(): fail("prepared_input_missing_or_ambiguous")
    roots = []
    # Exact bounded Kaggle layouts: flat datasets and datasets/<owner>/<dataset>.
    for child in input_root.iterdir():
        if child.is_dir() and (child / "masked-training-inventory.json").is_file(): roots.append(child)
    datasets_root = input_root / "datasets"
    if datasets_root.is_dir() and not datasets_root.is_symlink():
        for owner in datasets_root.iterdir():
            if not owner.is_dir() or owner.is_symlink(): continue
            for dataset in owner.iterdir():
                if dataset.is_dir() and not dataset.is_symlink() and (dataset / "masked-training-inventory.json").is_file():
                    roots.append(dataset)
    distinct = {item.resolve(strict=True) for item in roots}
    if len(distinct) != 1: fail("prepared_input_missing_or_ambiguous")
    return _verify_asset_root(next(iter(distinct)))[0]

def find_competition_root(input_root, explicit_image_root=None):
    if explicit_image_root is not None:
        candidate = Path(explicit_image_root)
        if candidate.is_symlink() or not candidate.is_dir(): fail("competition_series_input_missing")
        return candidate.resolve(strict=True)
    input_root = Path(input_root)
    candidates = []
    for prefix in (input_root, input_root / "competitions"):
        for folder in ("train_series", "train_images"):
            path = prefix / "rsna-knee-abnormality-detection" / folder
            if path.is_dir() and not path.is_symlink(): candidates.append(path.resolve(strict=True))
    candidates = list(dict.fromkeys(candidates))
    if not candidates: fail("competition_series_input_missing")
    if len(candidates) != 1: fail("competition_input_ambiguous")
    return candidates[0]

def verify_and_extract_source(root, inventory, working_root):
    archive_bytes = base64.b64decode((root / "masked-training-source.zip.b64").read_bytes(), validate=True)
    if hashlib.sha256(archive_bytes).hexdigest() != EXPECTED_SOURCE_SHA256: fail("source_pin_mismatch")
    with zipfile.ZipFile(__import__("io").BytesIO(archive_bytes)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(inventory["source_members"]): fail("source_inventory_mismatch")
        for name in names:
            if name.startswith("/") or ".." in Path(name).parts or not name.startswith("rsna_knee/") or not name.endswith(".py"):
                fail("source_member_invalid")
            if hashlib.sha256(archive.read(name)).hexdigest() != inventory["source_members"][name]: fail("source_member_pin_mismatch")
    working_root = Path(working_root)
    if working_root.is_symlink() or not working_root.is_dir(): fail("working_root_invalid")
    source_root = working_root / "masked-soft-source"
    if source_root.exists() or source_root.is_symlink(): fail("source_workdir_not_fresh")
    source_root.mkdir()
    with zipfile.ZipFile(__import__("io").BytesIO(archive_bytes)) as archive:
        for name in archive.namelist():
            destination = source_root.joinpath(*Path(name).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(name))
    return source_root

def verify_inputs(root, inventory):
    asset_pins = inventory.get("assets")
    if not isinstance(asset_pins, dict): fail("inventory_assets_invalid")
    checks = {"full-cohort-soft-label-manifest.json": inventory["cohort"]["manifest_sha256"],
              "full-cohort-grouped-folds.json": inventory["cohort"]["folds_sha256"],
              "resnet18-f37072fd.pth": inventory["model"]["weights_sha256"]}
    for name, expected in checks.items():
        path = root / name
        if digest(path) != expected: fail("prepared_asset_pin_mismatch")
    for name in checks:
        if asset_pins.get(name, {}).get("sha256") != checks[name]: fail("inventory_asset_pin_mismatch")
    codec_bytes = base64.b64decode((root / "offline-codecs.zip.b64").read_bytes(), validate=True)
    if hashlib.sha256(codec_bytes).hexdigest() != EXPECTED_CODEC_SHA256: fail("codec_asset_pin_mismatch")
    if asset_pins.get("offline-codecs.zip", {}).get("sha256") != EXPECTED_CODEC_SHA256:
        fail("inventory_asset_pin_mismatch")
    with zipfile.ZipFile(io.BytesIO(codec_bytes)) as codec:
        if hashlib.sha256(codec.read("manifest.json")).hexdigest() != EXPECTED_CODEC_MANIFEST_SHA256:
            fail("codec_manifest_pin_mismatch")
    source_b64 = base64.b64decode((root / "masked-training-source.zip.b64").read_bytes(), validate=True)
    if hashlib.sha256(source_b64).hexdigest() != EXPECTED_SOURCE_SHA256:
        fail("source_pin_mismatch")
    return codec_bytes

def _check_required_decoders(pydicom, UID):
    required = ("1.2.840.10008.1.2.4.57", "1.2.840.10008.1.2.4.70",
                "1.2.840.10008.1.2.4.80", "1.2.840.10008.1.2.4.81", "1.2.840.10008.1.2.4.90")
    for transfer_syntax in required:
        try: decoder = pydicom.pixels.get_decoder(UID(transfer_syntax))
        except NotImplementedError: fail("required_dicom_decoder_unavailable")
        if getattr(decoder, "is_available", False) is not True:
            fail("required_dicom_decoder_unavailable")

def run_notebook(*, input_root=None, working_root=None, asset_root=None,
                 image_root=None, stage=None, full_training=None,
                 module_loader=None, codec_installer=None, path_list=None,
                 emit=print):
    input_root = Path(input_root or INPUT_ROOT)
    working_root = Path(working_root or WORKING_ROOT)
    stage = RUN_STAGE if stage is None else stage
    full_training = RUN_FULL_TRAINING if full_training is None else full_training
    if stage not in {"plan", "decode_preflight", "full_training"}: fail("stage_invalid")
    if stage == "full_training" and full_training is not True: fail("training_not_explicitly_enabled")
    load = importlib.import_module if module_loader is None else module_loader
    search_path = sys.path if path_list is None else path_list
    assets, inventory = _verify_asset_root(asset_root) if asset_root is not None else _verify_asset_root(
        find_prepared_asset_root(input_root))
    competition_root = find_competition_root(input_root, image_root)
    codec_bytes = verify_inputs(assets, inventory)
    source_root = verify_and_extract_source(assets, inventory, working_root)
    if str(source_root) not in search_path: search_path.insert(0, str(source_root))
    runner = load("rsna_knee.training.masked_soft")
    if (not isinstance(OUTPUT_RUN_NAME, str) or not OUTPUT_RUN_NAME
            or Path(OUTPUT_RUN_NAME).name != OUTPUT_RUN_NAME): fail("output_run_name_invalid")
    output = working_root / "data" / "private" / "masked-soft-training" / OUTPUT_RUN_NAME
    config = runner.MaskedSoftConfig.from_mapping({
        "manifest_path": assets / "full-cohort-soft-label-manifest.json",
        "folds_path": assets / "full-cohort-grouped-folds.json",
        "image_root": competition_root, "output_dir": output,
        "fold": 0, "seed": 42, "epochs": 5, "max_steps": None,
        "output_size": 256, "max_slices": 8, "device": "cuda",
        "learning_rate": 0.001, "balance_mode": "unweighted",
        "weights_path": assets / "resnet18-f37072fd.pth",
        "weights_sha256": inventory["model"]["weights_sha256"],
        "weights_size": inventory["model"]["weights_size"],
        "resume_checkpoint": RESUME_CHECKPOINT,
        "resume_checkpoint_sha256": RESUME_CHECKPOINT_SHA256,
        "max_epochs_per_run": MAX_EPOCHS_PER_RUN,
        "max_runtime_seconds": MAX_RUNTIME_SECONDS,
    })
    plan = runner.plan_masked_soft_training(config)
    if plan.get("status") != "planned": fail("metadata_plan_blocked")
    if stage == "plan":
        emit(json.dumps({"stage": "plan", "training_study_count": plan["training_study_count"],
                         "validation_study_count": plan["validation_study_count"],
                         "status": plan["status"]}, sort_keys=True))
        return {"plan": plan}

    codec_bundle = working_root / "masked-offline-codecs.zip"
    if codec_bundle.exists() or codec_bundle.is_symlink(): fail("codec_workfile_not_fresh")
    codec_bundle.write_bytes(codec_bytes)
    bootstrap = load("rsna_knee.inference.offline_codec_bootstrap")
    installer = codec_installer or bootstrap.install_codec_bundle
    codec_target = working_root / "masked-codec-install"
    install_result = installer(codec_bundle, EXPECTED_CODEC_MANIFEST_SHA256, codec_target)
    if not isinstance(install_result, dict) or not install_result.get("target"):
        fail("codec_install_failed")
    installed_target = Path(install_result["target"])
    if installed_target.is_symlink() or not installed_target.is_dir(): fail("codec_install_failed")
    installed_target = installed_target.resolve(strict=True)
    if str(installed_target) not in search_path: search_path.insert(0, str(installed_target))

    pydicom = load("pydicom")
    uid_module = load("pydicom.uid")
    _check_required_decoders(pydicom, uid_module.UID)
    dataset_module = load("rsna_knee.data.masked_soft_dataset")
    preflight_module = load("rsna_knee.training.masked_preflight")
    dataset = dataset_module.FullCohortSoftDataset(
        config.manifest_path, config.folds_path, config.image_root,
        fold=config.fold, partition="train", output_size=config.output_size,
        max_slices=config.max_slices,
    )
    preflight = preflight_module.run_bounded_decode_preflight(dataset, max_studies=8)
    if stage == "decode_preflight":
        emit(json.dumps(preflight, sort_keys=True))
        return {"preflight": preflight}
    if preflight["status"] != "complete": fail("bounded_preflight_failed")
    training = runner.run_masked_soft_training(config, execute=True)
    emit(json.dumps(training, sort_keys=True))
    return {"training": training}

if __name__ == "__main__":
    try:
        run_notebook()
    except KeyboardInterrupt:
        raise
    except Exception:
        print(json.dumps({"status": "failed", "reason_code": "runtime_failed"}, sort_keys=True))
        sys.exit(2)
'''


def build_masked_training_package(repository_root: Path, output_dir: Path, *,
                                   private_root: Path | None = None,
                                   weights_path: Path | None = None,
                                   codec_path: Path | None = None,
                                   authorized_full_training: bool = False) -> dict[str, object]:
    if type(authorized_full_training) is not bool:
        raise ValueError("launch_mode_invalid")
    root = Path(repository_root).resolve(strict=True)
    private = Path(private_root or root / "data" / "private").resolve(strict=True)
    output_input = Path(output_dir)
    if not output_input.is_absolute():
        output_input = root / output_input
    parent = output_input.parent
    private_resolved, output = _safe_output(output_input, private)
    source_root = root / "src"
    source_paths = _import_closure(source_root)
    archive_bytes, members = _zip_source(source_root, source_paths)

    weight_path = Path(weights_path or Path.home() / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth").resolve(strict=True)
    if _file_hash(weight_path) != (WEIGHTS_SHA256, WEIGHTS_SIZE):
        raise ValueError("weights_asset_mismatch")
    codec = Path(codec_path or root / "data/private/hidden-submission-codecs/run07-bundle-final/rsna-knee-offline-dicom-codecs.zip").resolve(strict=True)
    _validate_codec(codec)

    assets_root = root / "data/private/masked-soft-training-preparation/assets"
    asset_bytes: dict[str, bytes] = {}
    for name, (relative, expected) in ASSETS.items():
        payload = (root / relative).read_bytes()
        if _hash(payload) != expected:
            raise ValueError("pinned_asset_mismatch")
        if name in UPLOAD_ASSETS:
            asset_bytes[name] = payload
    manifest = asset_bytes["full-cohort-soft-label-manifest.json"]
    folds = asset_bytes["full-cohort-grouped-folds.json"]
    if _hash(manifest) != MANIFEST_SHA256 or _hash(folds) != FOLDS_SHA256:
        raise ValueError("cohort_pin_mismatch")
    weight_payload = weight_path.read_bytes()
    codec_payload = codec.read_bytes()
    manifest_object = json.loads(manifest.decode("utf-8"))
    manifest_records = manifest_object.get("records")
    if type(manifest_records) is not list or len(manifest_records) != 4298:
        raise ValueError("cohort_manifest_count_mismatch")
    eligible_series_count = sum(len(record.get("series", [])) for record in manifest_records
                                if isinstance(record, dict) and isinstance(record.get("series"), list))
    if eligible_series_count != 23765:
        raise ValueError("cohort_series_count_mismatch")
    inventory: dict[str, object] = {
        "schema_version": INVENTORY_SCHEMA,
        "source_archive_sha256": _hash(archive_bytes),
        "source_members": members,
        "cohort": {"manifest_sha256": MANIFEST_SHA256, "folds_sha256": FOLDS_SHA256,
                   "prefilter_policy_sha256": POLICY_SHA256,
                   "baseline_cohort_sha256": COHORT_SHA256,
                   "baseline_folds_sha256": BASELINE_FOLDS_SHA256,
                   "eligible_study_count": 4298, "eligible_series_count": eligible_series_count,
                   "full_non_gold_series_count": 24035},
        "model": {"kind": "resnet18_imagenet1k_v1", "freeze_backbone": True,
                  "optimizer": "Adam", "learning_rate": 0.001,
                  "weights_sha256": WEIGHTS_SHA256, "weights_size": WEIGHTS_SIZE},
        "training_config": {"stage_default": "plan", "fold": 0, "seed": 42,
                            "epochs": 5, "output_size": 256, "max_slices": 8,
                            "balance_mode": "unweighted", "max_steps": None,
                            "learning_rate": 0.001},
        "assets": {name: {"sha256": _hash(payload), "size": len(payload)}
                   for name, payload in {**asset_bytes, "resnet18-f37072fd.pth": weight_payload,
                                         "offline-codecs.zip": codec_payload}.items()},
    }
    inventory_bytes = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    notebook_code = _notebook_source().replace("__INVENTORY_SHA256__", _hash(inventory_bytes)) \
        .replace("__SOURCE_SHA256__", _hash(archive_bytes)).replace("__CODEC_SHA256__", CODEC_SHA256) \
        .replace("__CODEC_MANIFEST_SHA256__", CODEC_MANIFEST_SHA256)
    if authorized_full_training:
        notebook_code = notebook_code.replace('RUN_STAGE = "plan"', 'RUN_STAGE = "full_training"') \
            .replace("RUN_FULL_TRAINING = False", "RUN_FULL_TRAINING = True")
    notebook = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {},
                           "outputs": [], "source": notebook_code.splitlines(keepends=True)}],
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": "3.13.0"}},
                "nbformat": 4, "nbformat_minor": 5}
    notebook_bytes = json.dumps(notebook, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    compile(notebook_code, "kaggle_masked_soft_training.py", "exec")

    # Exclusive fresh directory; cleanup only this invocation's newly-created output on failure.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    created: list[Path] = []
    try:
        upload = output / "upload"
        upload.mkdir()
        created.append(upload)
        files: dict[str, bytes] = {"masked-training-inventory.json": inventory_bytes,
                                   "masked-training-source.zip.b64": base64.b64encode(archive_bytes),
                                   "offline-codecs.zip.b64": base64.b64encode(codec_payload),
                                   "resnet18-f37072fd.pth": weight_payload,
                                   **asset_bytes}
        for name, payload in files.items():
            path = upload / name
            created.append(path)
            path.write_bytes(payload)
        notebook_name = ("kaggle_masked_soft_training_authorized.ipynb" if authorized_full_training
                         else "kaggle_masked_soft_training.ipynb")
        notebook_path = output / notebook_name
        created.append(notebook_path)
        notebook_path.write_bytes(notebook_bytes)
        upload_notebook = upload / notebook_path.name
        created.append(upload_notebook)
        upload_notebook.write_bytes(notebook_bytes)
        source_archive = output / "masked-training-source.zip"
        created.append(source_archive)
        source_archive.write_bytes(archive_bytes)
        # Readback every generated byte before returning success.
        if source_archive.read_bytes() != archive_bytes or notebook_path.read_bytes() != notebook_bytes:
            raise ValueError("package_readback_failed")
        if any((upload / name).read_bytes() != payload for name, payload in files.items()):
            raise ValueError("asset_readback_failed")
        return {"output_dir": str(output), "upload_dir": str(upload),
                "inventory_path": str(upload / "masked-training-inventory.json"),
                "archive_path": str(source_archive), "archive_sha256": _hash(archive_bytes),
                "notebook_path": str(notebook_path), "notebook_sha256": _hash(notebook_bytes),
                "inventory_sha256": _hash(inventory_bytes), "source_member_count": len(members),
                "source_members": members}
    except BaseException:
        for path in reversed(created):
            try:
                if path.is_dir():
                    path.rmdir()
                elif path.exists():
                    path.unlink()
            except OSError:
                pass
        try:
            output.rmdir()
        except OSError:
            pass
        raise


if __name__ == "__main__":
    repository = Path(__file__).resolve().parents[1]
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--authorized-full-training", action="store_true")
    args = parser.parse_args()
    target = args.output_dir or repository / "data/private/masked-soft-training-preparation/run01"
    result = build_masked_training_package(repository, target,
                                           authorized_full_training=args.authorized_full_training)
    print(json.dumps({key: result[key] for key in ("output_dir", "archive_sha256", "notebook_sha256", "source_member_count")}, sort_keys=True))
