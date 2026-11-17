"""Build a private, offline Kaggle package for E048 attention epoch five."""
from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from scripts.build_kaggle_submission_package import build_submission_package
from rsna_knee.inference.radimagenet_attention import (
    ATTENTION_MODEL_SOURCE_SHA256, EXPECTED_CHECKPOINT_SHA256, EXPECTED_CHECKPOINT_SIZE,
    EXPECTED_REPORT_SHA256, NATIVE_FINGERPRINT, RESNET_MODEL_SOURCE_SHA256,
    SOURCE_CHECKPOINT_SHA256, export_attention_model, validate_attention_report,
)


CODEC_MANIFEST_SHA256 = "50f39d3f627b3dc06a8080ba51dd6424898ef34fa4c4e21a84593dd326eb776d"
SOURCE_ROOT = "/kaggle/input/datasets/davidsulleiro/rsna-knee-attention-epoch05"
OUTPUT_RELATIVE = Path("data/private/attention-model-submission/run02")
SOURCE_EXECUTION = "disease-attention-mil/e048-run01/target_attention/epoch-05"
TRAINING_SOURCE = Path("data/private/resnet50-pretraining-preparation/run01/resnet50-recipe-source.zip")
E047_SOURCE_ARCHIVE = Path("data/private/radimagenet-model-submission/run01/radimagenet-inference-source.zip")
E047_SOURCE_ARCHIVE_SHA256 = "66a7036a3f8b2b8251a31d6af70f790274298bb2875e977ec036f127d53c23be"
E047_RADIMAGENET_SOURCE_SHA256 = "62470aefc5894350160e2ee62ed588b3d159d74f51917cf0b194c63501076138"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _private_file(path):
    path = Path(path)
    if not path.is_file() or any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError("private_artifact_unavailable")
    return path


def validate_private_artifacts(report_path, checkpoint_path):
    report_path, checkpoint_path = _private_file(report_path), _private_file(checkpoint_path)
    report_raw = report_path.read_bytes()
    checkpoint_hash = _sha(checkpoint_path.read_bytes())
    proof_path = _private_file(checkpoint_path.with_suffix(".sha256"))
    try:
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("checkpoint_proof_invalid") from None
    if (_sha(report_raw) != EXPECTED_REPORT_SHA256 or checkpoint_hash != SOURCE_CHECKPOINT_SHA256
            or checkpoint_path.stat().st_size != EXPECTED_CHECKPOINT_SIZE
            or type(proof) is not dict
            or set(proof) != {"schema", "arm", "checkpoint_sha256", "completed_epochs", "fingerprint"}
            or proof != {"schema": "disease-attention-mil-checkpoint-proof-v1",
                         "arm": "target_attention", "checkpoint_sha256": checkpoint_hash,
                         "completed_epochs": 5, "fingerprint": NATIVE_FINGERPRINT}):
        raise ValueError("source_artifact_pin_mismatch")
    report = json.loads(report_raw.decode("utf-8"))
    validate_attention_report(report)
    return report, report_raw, checkpoint_path, checkpoint_hash


def _extend_source_archive(base_raw, repository):
    model_path = Path(repository) / "src/rsna_knee/models/attention_mil.py"
    resnet_path = Path(repository) / "src/rsna_knee/models/resnet50.py"
    adapter_path = Path(repository) / "src/rsna_knee/inference/radimagenet_attention.py"
    for path in (model_path, resnet_path, adapter_path):
        if path.is_symlink() or not path.is_file():
            raise ValueError("source_symlink_forbidden")
    additions = {
        "rsna_knee/models/attention_mil.py": model_path.read_bytes(),
        "rsna_knee/models/resnet50.py": resnet_path.read_bytes(),
        "rsna_knee/inference/radimagenet_attention.py": adapter_path.read_bytes(),
    }
    if (_sha(additions["rsna_knee/models/attention_mil.py"]) != ATTENTION_MODEL_SOURCE_SHA256
            or _sha(additions["rsna_knee/models/resnet50.py"]) != RESNET_MODEL_SOURCE_SHA256):
        raise ValueError("frozen_model_source_mismatch")
    accepted_archive = _private_file(Path(repository) / E047_SOURCE_ARCHIVE)
    accepted_raw = accepted_archive.read_bytes()
    if _sha(accepted_raw) != E047_SOURCE_ARCHIVE_SHA256:
        raise ValueError("accepted_source_archive_pin_mismatch")
    with ZipFile(io.BytesIO(accepted_raw)) as frozen:
        frozen_names = frozen.namelist()
        if (len(frozen_names) != len(set(frozen_names))
                or "rsna_knee/inference/radimagenet.py" not in frozen_names):
            raise ValueError("accepted_source_archive_invalid")
        frozen_radimagenet = frozen.read("rsna_knee/inference/radimagenet.py")
    if _sha(frozen_radimagenet) != E047_RADIMAGENET_SOURCE_SHA256:
        raise ValueError("accepted_radimagenet_source_pin_mismatch")
    additions["rsna_knee/inference/radimagenet.py"] = frozen_radimagenet
    with ZipFile(io.BytesIO(base_raw)) as archive:
        names = archive.namelist()
        if (len(names) != len(set(names))
                or any(not name.startswith("rsna_knee/") or not name.endswith(".py")
                       or ".." in PurePosixPath(name).parts for name in names)):
            raise ValueError("source_archive_invalid")
        members = {name: archive.read(name) for name in names}
    if any(name in members and members[name] != raw for name, raw in additions.items()):
        raise ValueError("frozen_source_member_mismatch")
    members.update(additions)
    output = io.BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for name, raw in sorted(members.items()):
            ast.parse(raw, filename=name)
            info = ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, raw)
    return output.getvalue(), members


def _notebook_code(base_notebook, base_archive, source_archive, source_digest, *,
                   export_sha256, export_size, report_sha256, max_studies):
    code = "".join(base_notebook["cells"][1]["source"])
    old_payload = base64.b64encode(base_archive).decode("ascii")
    if code.count(old_payload) != 1 or code.count(_sha(base_archive)) < 1:
        raise ValueError("source_payload_ambiguous")
    code = code.replace(old_payload, base64.b64encode(source_archive).decode("ascii"))
    code = code.replace(_sha(base_archive), source_digest)
    pins = {"SOURCE_ROOT": SOURCE_ROOT, "CHECKPOINT_SHA256": export_sha256,
            "CHECKPOINT_SIZE": export_size, "REPORT_SHA256": report_sha256}
    locations = {}
    for node in ast.parse(code).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in pins:
                    locations.setdefault(target.id, []).append((node.lineno, node.end_lineno))
    lines = code.splitlines(keepends=True)
    if any(len(locations.get(name, [])) != 1 for name in pins):
        raise ValueError("notebook_pin_assignment_ambiguous")
    for name, value in pins.items():
        first, last = locations[name][0]
        lines[first - 1:last] = [f"{name} = {value!r}\n"]
    code = "".join(lines)
    code = code.replace("from rsna_knee.inference.fold0 import run_inference",
                        "from rsna_knee.inference.radimagenet_attention import run_attention_inference")
    code = code.replace("fold0-inference-source.zip", "attention-inference-source.zip")
    code = code.replace("fold0-inference-run01", "attention-inference-run02")
    code = code.replace("# Exact mounted resnet18-frozen-backbone/run01 directory.",
                        "# Exact mounted private E048 model/report dataset.")
    code = code.replace("SOURCE_ROOT = None", f"SOURCE_ROOT = {SOURCE_ROOT!r}")
    start = code.index("    summary = run_inference(")
    end = code.index("    destination_created = False", start)
    code = code[:start] + f'''    summary = run_attention_inference(
        COMPETITION_ROOT, source, REPORT_SHA256, CHECKPOINT_SHA256,
        '/kaggle/working/attention-inference-run02', checkpoint_size=CHECKPOINT_SIZE, device='cuda',
        max_studies=MAX_STUDIES, recovery_policy_version=RECOVERY_POLICY_VERSION)
''' + code[end:]
    start = code.index("    destination_created = False")
    end = code.index("    print(json.dumps(summary, sort_keys=True))", start)
    publication = code[start:end]
    code = (code[:start]
            + "    if MAX_STUDIES is None or summary['study_count'] == summary['test_study_count']:\n"
            + "".join("    " + line for line in publication.splitlines(keepends=True))
            + code[end:])
    code = code.replace("RUN_INFERENCE = True\n",
                        f"RUN_INFERENCE = True\nMAX_STUDIES = {max_studies!r}\n", 1)
    compile(code, "radimagenet-attention-submission", "exec")
    return code


def build_attention_submission_package(repository_root, output_dir, report_path, checkpoint_path,
                                       training_source, *, source_root=SOURCE_ROOT):
    """Create the exclusive local package; never overwrite an existing preparation."""
    repository = Path(repository_root).resolve(strict=True)
    output = Path(output_dir).resolve(strict=False)
    expected_output = (repository / OUTPUT_RELATIVE).resolve(strict=False)
    if output != expected_output or output.exists() or any(item.is_symlink() for item in (output, *output.parents)):
        raise ValueError("preparation_output_must_be_new")
    if (type(source_root) is not str or not source_root.startswith("/kaggle/input/datasets/")
            or ".." in PurePosixPath(source_root).parts):
        raise ValueError("source_mount_invalid")
    report, report_raw, checkpoint, checkpoint_hash = validate_private_artifacts(report_path, checkpoint_path)
    training_source = _private_file(training_source)
    training_raw = training_source.read_bytes()
    with ZipFile(io.BytesIO(training_raw)) as frozen:
        if "rsna_knee/models/resnet50.py" not in frozen.namelist():
            raise ValueError("frozen_model_source_missing")
        if frozen.read("rsna_knee/models/resnet50.py") != (repository / "src/rsna_knee/models/resnet50.py").read_bytes():
            raise ValueError("frozen_model_source_mismatch")
    with TemporaryDirectory() as temporary:
        base_dir = Path(temporary) / "base"
        base = build_submission_package(
            repository, base_dir, source_root=source_root,
            checkpoint_sha256=checkpoint_hash, checkpoint_size=checkpoint.stat().st_size,
            report_sha256=EXPECTED_REPORT_SHA256, codec_manifest_sha256=CODEC_MANIFEST_SHA256,
            run_inference=True, recovery_policy_version=1)
        base_archive = (base_dir / "fold0-inference-source.zip").read_bytes()
        base_notebook = json.loads(Path(base["notebook"]).read_text(encoding="utf-8"))
    source_archive, members = _extend_source_archive(base_archive, repository)
    source_digest = _sha(source_archive)
    output.mkdir(parents=True, exist_ok=False)
    export_path = output / "attention-model-epoch-05.pt"
    export = export_attention_model(
        checkpoint, Path(report_path), export_path,
        source_checkpoint_sha256=checkpoint_hash, report_sha256=EXPECTED_REPORT_SHA256)
    (output / "attention-inference-source.zip").write_bytes(source_archive)
    (output / "training-report.json").write_bytes(report_raw)
    variants = []
    for label, cap in (("canary", 3), ("submission", None)):
        notebook = json.loads(json.dumps(base_notebook))
        notebook["cells"][0]["source"] = [
            "# RSNA Knee: E048 target-specific attention inference\n",
            "Offline inference only. This package contains epoch-five model tensors without optimizer/RNG state; "
            "the report is aggregate provenance. The attention adapter preserves trained slice and series "
            "attention, 224px input, eight slices, microbatch two, and 12-target logits.\n",
        ]
        code = _notebook_code(base_notebook, base_archive, source_archive, source_digest,
                              export_sha256=export["sha256"], export_size=export["size"],
                              report_sha256=EXPECTED_REPORT_SHA256, max_studies=cap)
        notebook["cells"][1]["source"] = code.splitlines(keepends=True)
        variants.append((label, notebook))
    inventory = {
        "schema": "radimagenet-attention-submission-package-v1",
        "source_execution": SOURCE_EXECUTION,
        "source_archive_sha256": source_digest, "source_archive_member_count": len(members),
        "model_export_sha256": export["sha256"], "model_export_size": export["size"],
        "source_checkpoint_sha256": checkpoint_hash, "report_sha256": EXPECTED_REPORT_SHA256,
        "native_training_fingerprint": NATIVE_FINGERPRINT,
        "attention_model_source_sha256": ATTENTION_MODEL_SOURCE_SHA256,
        "resnet_model_source_sha256": RESNET_MODEL_SOURCE_SHA256,
        "codec_manifest_sha256": CODEC_MANIFEST_SHA256,
        "source_root": source_root, "remote_executed": False, "submitted": False,
        "canary_rows_cap": 3, "full_rows_cap": None,
        "private_artifacts": ["aggregate training report", "model-only epoch-five tensors"],
    }
    result = {**inventory, "source_archive": str(output / "attention-inference-source.zip"),
              "model_export": str(export_path), "report": str(output / "training-report.json")}
    for label, notebook in variants:
        path = output / f"kaggle_radimagenet_attention_{label}.ipynb"
        path.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
        result[label] = str(path)
        result[f"{label}_sha256"] = _sha(path.read_bytes())
    (output / "inventory.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--training-source", type=Path, default=None)
    args = parser.parse_args()
    root = args.repository.resolve(strict=True)
    report = args.report or root / "data/private/disease-attention-mil/e048-run01/training-report.json"
    checkpoint = args.checkpoint or root / "data/private/disease-attention-mil/e048-run01/target_attention/checkpoint-epoch-05.pt"
    source = args.training_source or root / TRAINING_SOURCE
    output = args.output or root / OUTPUT_RELATIVE
    print(json.dumps(build_attention_submission_package(
        root, output, report, checkpoint, source), sort_keys=True))
