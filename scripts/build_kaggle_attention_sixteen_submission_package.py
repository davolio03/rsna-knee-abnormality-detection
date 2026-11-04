"""Build a private, offline Kaggle package for the completed E050 attention16 model."""
from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_kaggle_submission_package import build_submission_package
from rsna_knee.inference.attention_sixteen import (
    ATTENTION_MODEL_SOURCE_SHA256, NATIVE_FINGERPRINT, RESNET_MODEL_SOURCE_SHA256,
    SOURCE_CHECKPOINT_SHA256, SOURCE_CHECKPOINT_SIZE, SOURCE_REPORT_SHA256,
    export_attention16_model, sanitized_attention16_provenance,
)


CODEC_MANIFEST_SHA256 = "50f39d3f627b3dc06a8080ba51dd6424898ef34fa4c4e21a84593dd326eb776d"
SOURCE_ROOT = "/kaggle/input/datasets/davidsulleiro/rsna-knee-attention-sixteen-epoch05"
OUTPUT_RELATIVE = Path("data/private/attention-sixteen-submission/run01")
REPORT_PATH = Path("data/private/attention-sixteen-slices/local16-attention-run01/training-report.json")
CHECKPOINT_PATH = Path("data/private/attention-sixteen-slices/local16-attention-run01/target_attention/checkpoint-epoch-05.pt")
PROOF_PATH = CHECKPOINT_PATH.with_suffix(".sha256")
ACCEPTED_ARCHIVE = Path("data/private/radimagenet-model-submission/run01/radimagenet-inference-source.zip")
ACCEPTED_ARCHIVE_SHA256 = "66a7036a3f8b2b8251a31d6af70f790274298bb2875e977ec036f127d53c23be"
ACCEPTED_RADIMAGENET_SHA256 = "62470aefc5894350160e2ee62ed588b3d159d74f51917cf0b194c63501076138"


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
    if _sha(report_raw) != SOURCE_REPORT_SHA256:
        raise ValueError("source_artifact_pin_mismatch")
    checkpoint_hash = _sha(checkpoint_path.read_bytes())
    proof_path = _private_file(checkpoint_path.with_suffix(".sha256"))
    try:
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
        report = json.loads(report_raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("source_artifact_proof_invalid") from None
    expected_proof = {"schema": "disease-attention-mil-checkpoint-proof-v1", "arm": "target_attention",
                      "checkpoint_sha256": SOURCE_CHECKPOINT_SHA256, "completed_epochs": 5,
                      "fingerprint": NATIVE_FINGERPRINT}
    if (checkpoint_hash != SOURCE_CHECKPOINT_SHA256 or checkpoint_path.stat().st_size != SOURCE_CHECKPOINT_SIZE
            or type(proof) is not dict or proof != expected_proof):
        raise ValueError("source_artifact_pin_mismatch")
    sanitized = sanitized_attention16_provenance(report)
    sanitized_raw = (json.dumps(sanitized, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return report, report_raw, checkpoint_path, checkpoint_hash, sanitized_raw


def _extend_source_archive(base_raw, repository):
    repository = Path(repository)
    model_path = repository / "src/rsna_knee/models/attention_mil.py"
    resnet_path = repository / "src/rsna_knee/models/resnet50.py"
    adapter_path = repository / "src/rsna_knee/inference/attention_sixteen.py"
    for path in (model_path, resnet_path, adapter_path):
        if path.is_symlink() or not path.is_file():
            raise ValueError("source_symlink_forbidden")
    additions = {
        "rsna_knee/models/attention_mil.py": model_path.read_bytes(),
        "rsna_knee/models/resnet50.py": resnet_path.read_bytes(),
        "rsna_knee/inference/attention_sixteen.py": adapter_path.read_bytes(),
    }
    if (_sha(additions["rsna_knee/models/attention_mil.py"]) != ATTENTION_MODEL_SOURCE_SHA256
            or _sha(additions["rsna_knee/models/resnet50.py"]) != RESNET_MODEL_SOURCE_SHA256):
        raise ValueError("frozen_model_source_mismatch")
    accepted = _private_file(repository / ACCEPTED_ARCHIVE)
    accepted_raw = accepted.read_bytes()
    if _sha(accepted_raw) != ACCEPTED_ARCHIVE_SHA256:
        raise ValueError("accepted_source_archive_pin_mismatch")
    with ZipFile(io.BytesIO(accepted_raw)) as frozen:
        names = frozen.namelist()
        if len(names) != len(set(names)) or "rsna_knee/inference/radimagenet.py" not in names:
            raise ValueError("accepted_source_archive_invalid")
        radimagenet_raw = frozen.read("rsna_knee/inference/radimagenet.py")
    if _sha(radimagenet_raw) != ACCEPTED_RADIMAGENET_SHA256:
        raise ValueError("accepted_radimagenet_source_pin_mismatch")
    additions["rsna_knee/inference/radimagenet.py"] = radimagenet_raw
    with ZipFile(io.BytesIO(base_raw)) as archive:
        names = archive.namelist()
        if (len(names) != len(set(names)) or any(not name.startswith("rsna_knee/")
                or not name.endswith(".py") or ".." in PurePosixPath(name).parts for name in names)):
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
                   export_sha256, export_size, report_sha256, max_studies, source_root):
    code = "".join(base_notebook["cells"][1]["source"])
    old_payload = base64.b64encode(base_archive).decode("ascii")
    if code.count(old_payload) != 1 or code.count(_sha(base_archive)) < 1:
        raise ValueError("source_payload_ambiguous")
    code = code.replace(old_payload, base64.b64encode(source_archive).decode("ascii"))
    code = code.replace(_sha(base_archive), source_digest)
    pins = {"SOURCE_ROOT": source_root, "CHECKPOINT_SHA256": export_sha256,
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
                        "from rsna_knee.inference.attention_sixteen import run_attention16_inference")
    code = code.replace("# Inference only: attach the private output of execution 361175523.",
                        "# Inference only: attach the private E050 attention16 model-only dataset.")
    code = code.replace("# Exact mounted resnet18-frozen-backbone/run01 directory.",
                        "# Exact mounted E050 private model-only dataset.")
    code = code.replace("fold0-inference-source.zip", "attention16-inference-source.zip")
    code = code.replace("fold0-inference-run01", "attention16-inference-run01")
    start = code.index("    summary = run_inference(")
    end = code.index("    print(json.dumps(summary, sort_keys=True))", start)
    code = code[:start] + f'''    summary = run_attention16_inference(
        COMPETITION_ROOT, source, REPORT_SHA256, CHECKPOINT_SHA256,
        '/kaggle/working/attention16-inference-run01', checkpoint_size=CHECKPOINT_SIZE, device='cuda',
        max_studies=MAX_STUDIES, recovery_policy_version=RECOVERY_POLICY_VERSION)
    if MAX_STUDIES is None and summary['study_count'] == summary['test_study_count']:
        result = Path('/kaggle/working/attention16-inference-run01/submission.csv')
        destination = Path('/kaggle/working/submission.csv')
        destination_created = False
        try:
            with destination.open('xb') as stream:
                destination_created = True
                stream.write(result.read_bytes())
            if destination.read_bytes() != result.read_bytes():
                raise ValueError('submission_publication_readback_failed')
        except Exception:
            if destination_created:
                try:
                    destination.unlink(missing_ok=True)
                except OSError:
                    pass
            raise RuntimeError('submission_publication_failed') from None
    print(json.dumps(summary, sort_keys=True))
''' + code[end + len("    print(json.dumps(summary, sort_keys=True))\n"):]
    code = code.replace("RUN_INFERENCE = True\n", f"RUN_INFERENCE = True\nMAX_STUDIES = {max_studies!r}\n", 1)
    compile(code, "attention16-submission", "exec")
    return code


def build_attention16_submission_package(repository_root, output_dir, report_path, checkpoint_path, *,
                                         source_root=SOURCE_ROOT):
    repository = Path(repository_root).resolve(strict=True)
    output = Path(output_dir).resolve(strict=False)
    allowed_outputs = {
        (repository / OUTPUT_RELATIVE).resolve(strict=False),
        (repository / OUTPUT_RELATIVE.parent / "run02").resolve(strict=False),
    }
    if (output not in allowed_outputs or output.exists()
            or any(item.is_symlink() for item in (output, *output.parents))):
        raise ValueError("preparation_output_must_be_new")
    if (type(source_root) is not str or not source_root.startswith("/kaggle/input/datasets/")
            or ".." in PurePosixPath(source_root).parts):
        raise ValueError("source_mount_invalid")
    report, report_raw, checkpoint, checkpoint_hash, sanitized_raw = validate_private_artifacts(report_path, checkpoint_path)
    sanitized_report_hash = _sha(sanitized_raw)
    with TemporaryDirectory() as temporary:
        base_dir = Path(temporary) / "base"
        base = build_submission_package(
            repository, base_dir, source_root=source_root,
            checkpoint_sha256=SOURCE_CHECKPOINT_SHA256, checkpoint_size=SOURCE_CHECKPOINT_SIZE,
            report_sha256=sanitized_report_hash, codec_manifest_sha256=CODEC_MANIFEST_SHA256,
            run_inference=True, recovery_policy_version=1)
        base_archive = (base_dir / "fold0-inference-source.zip").read_bytes()
        base_notebook = json.loads(Path(base["notebook"]).read_text(encoding="utf-8"))
    source_archive, members = _extend_source_archive(base_archive, repository)
    source_digest = _sha(source_archive)
    output.mkdir(parents=True, exist_ok=False)
    export_path = output / "attention-model-epoch-05.pt"
    exported = export_attention16_model(checkpoint, report_path, export_path)
    (output / "attention-inference-source.zip").write_bytes(source_archive)
    (output / "training-report.json").write_bytes(sanitized_raw)
    variants = []
    for label, cap in (("canary", 3), ("submission", None)):
        notebook = json.loads(json.dumps(base_notebook))
        notebook["cells"][0]["source"] = [
            "# RSNA Knee: E050 target-attention sixteen-slice inference\n",
            "Offline inference only. The uploaded model contains epoch-five tensors only; the adjacent report "
            "contains sanitized aggregate provenance. Preprocessing uses 224px, sixteen slices, RadImageNet "
            "normalization, and microbatch two; target-specific slice/series attention and canonical target order "
            "are preserved. No training, labels, identifiers, or validation records are included.\n",
        ]
        code = _notebook_code(base_notebook, base_archive, source_archive, source_digest,
                              export_sha256=exported["sha256"], export_size=exported["size"],
                              report_sha256=sanitized_report_hash, max_studies=cap,
                              source_root=source_root)
        notebook["cells"][1]["source"] = code.splitlines(keepends=True)
        variants.append((label, notebook))
    inventory = {
        "schema": "radimagenet-attention16-submission-package-v1",
        "source_execution": source_root, "source_archive_sha256": source_digest,
        "source_archive_member_count": len(members), "model_export_sha256": exported["sha256"],
        "model_export_size": exported["size"], "source_checkpoint_sha256": checkpoint_hash,
        "source_checkpoint_size": checkpoint.stat().st_size, "source_report_sha256": SOURCE_REPORT_SHA256,
        "sanitized_report_sha256": sanitized_report_hash, "native_training_fingerprint": NATIVE_FINGERPRINT,
        "attention_model_source_sha256": ATTENTION_MODEL_SOURCE_SHA256,
        "resnet_model_source_sha256": RESNET_MODEL_SOURCE_SHA256,
        "codec_manifest_sha256": CODEC_MANIFEST_SHA256, "max_slices": 16, "output_size": 224,
        "normalization": "radimagenet_2x_minus_1", "slice_microbatch": 2,
        "pooling": "target_specific_slice_attention_then_series_attention_logits",
        "remote_executed": False, "submitted": False, "canary_rows_cap": 3,
        "full_rows_cap": None, "uploaded_files": ["attention-model-epoch-05.pt", "training-report.json"],
        "private_artifacts": ["model-only epoch-five tensors", "sanitized aggregate provenance"],
    }
    result = {**inventory, "source_archive": str(output / "attention-inference-source.zip"),
              "model_export": str(export_path), "report": str(output / "training-report.json")}
    for label, notebook in variants:
        path = output / f"kaggle_attention16_{label}.ipynb"
        path.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
        result[label] = str(path)
        result[f"{label}_sha256"] = _sha(path.read_bytes())
    (output / "inventory.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--source-output-root", type=str, default=SOURCE_ROOT)
    args = parser.parse_args()
    repository = args.repository.resolve(strict=True)
    result = build_attention16_submission_package(
        repository, args.output or repository / OUTPUT_RELATIVE,
        args.report or repository / REPORT_PATH, args.checkpoint or repository / CHECKPOINT_PATH,
        source_root=args.source_output_root)
    print(json.dumps(result, sort_keys=True))
