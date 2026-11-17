"""Build private canary/full Kaggle notebooks for the completed RadImageNet model."""
from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

from scripts.build_kaggle_submission_package import build_submission_package
from rsna_knee.inference.radimagenet import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_CHECKPOINT_SIZE,
    EXPECTED_REPORT_SHA256,
    NATIVE_FINGERPRINT,
    TRAINING_SOURCE_SHA256,
    validate_radimagenet_report,
)


CODEC_MANIFEST_SHA256 = "50f39d3f627b3dc06a8080ba51dd6424898ef34fa4c4e21a84593dd326eb776d"
SOURCE_EXECUTION = "local-radimagenet-final-epochs/final-run01"


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _private_file(path):
    path = Path(path)
    if not path.is_file() or any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("private_artifact_unavailable")
    return path


def validate_private_artifacts(report_path, checkpoint_path, training_source):
    """Check exact local report/checkpoint/source pins before building any package."""
    report_path = _private_file(report_path)
    checkpoint_path = _private_file(checkpoint_path)
    training_source = _private_file(training_source)
    report_raw = report_path.read_bytes()
    if _sha(report_raw) != EXPECTED_REPORT_SHA256:
        raise ValueError("report_pin_mismatch")
    report = json.loads(report_raw.decode("utf-8"))
    validate_radimagenet_report(report)
    checkpoint_hash = _sha(checkpoint_path.read_bytes())
    if (checkpoint_hash != EXPECTED_CHECKPOINT_SHA256 or report["checkpoint_sha256"] != checkpoint_hash
            or checkpoint_path.stat().st_size != EXPECTED_CHECKPOINT_SIZE):
        raise ValueError("checkpoint_pin_mismatch")
    training_raw = training_source.read_bytes()
    if _sha(training_raw) != TRAINING_SOURCE_SHA256:
        raise ValueError("training_source_pin_mismatch")
    return report, report_raw, checkpoint_hash, checkpoint_path.stat().st_size, training_raw


def _extend_source_archive(base_raw, repository, training_raw):
    with ZipFile(io.BytesIO(training_raw)) as frozen:
        names = frozen.namelist()
        if (len(names) != len(set(names))
                or any(not name.startswith("rsna_knee/") or not name.endswith(".py")
                       or ".." in PurePosixPath(name).parts for name in names)):
            raise ValueError("training_source_invalid")
        frozen_resnet50 = frozen.read("rsna_knee/models/resnet50.py")
    model_path = Path(repository) / "src/rsna_knee/models/resnet50.py"
    adapter_path = Path(repository) / "src/rsna_knee/inference/radimagenet.py"
    if model_path.is_symlink() or adapter_path.is_symlink():
        raise ValueError("source_symlink_forbidden")
    model_raw, adapter_raw = model_path.read_bytes(), adapter_path.read_bytes()
    if model_raw != frozen_resnet50:
        raise ValueError("frozen_model_source_mismatch")
    with ZipFile(io.BytesIO(base_raw)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("source_archive_invalid")
        members = {name: archive.read(name) for name in names}
    additions = {
        "rsna_knee/models/resnet50.py": model_raw,
        "rsna_knee/inference/radimagenet.py": adapter_raw,
    }
    if any(name in members for name in additions):
        raise ValueError("source_member_collision")
    members.update(additions)
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for name, raw in sorted(members.items()):
            ast.parse(raw, filename=name)
            info = ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, raw)
    return buffer.getvalue(), members


def _notebook_code(base_notebook, base_archive, source_archive, archive_sha256, *, report_sha256,
                   checkpoint_sha256, checkpoint_size, fingerprint, max_studies):
    code = "".join(base_notebook["cells"][1]["source"])
    old_payload = base64.b64encode(base_archive).decode("ascii")
    if code.count(old_payload) != 1 or code.count(hashlib.sha256(base_archive).hexdigest()) < 1:
        raise ValueError("source_payload_ambiguous")
    code = code.replace(old_payload, base64.b64encode(source_archive).decode("ascii"))
    code = code.replace(hashlib.sha256(base_archive).hexdigest(), archive_sha256)
    code = code.replace("CHECKPOINT_SHA256 = None", f"CHECKPOINT_SHA256 = {checkpoint_sha256!r}")
    code = code.replace("CHECKPOINT_SIZE = None", f"CHECKPOINT_SIZE = {checkpoint_size!r}")
    code = code.replace("REPORT_SHA256 = None", f"REPORT_SHA256 = {report_sha256!r}")
    code = code.replace("from rsna_knee.inference.fold0 import run_inference",
                        "from rsna_knee.inference.radimagenet import run_radimagenet_inference")
    code = code.replace("fold0-inference-source.zip", "radimagenet-inference-source.zip")
    code = code.replace("fold0-inference-run01", "radimagenet-inference-run01")
    start = code.index("    summary = run_inference(")
    end = code.index("    destination_created = False", start)
    code = code[:start] + f'''    summary = run_radimagenet_inference(
        COMPETITION_ROOT, source, REPORT_SHA256, CHECKPOINT_SHA256,
        '/kaggle/working/radimagenet-inference-run01', device='cuda',
        max_studies=MAX_STUDIES, recovery_policy_version=RECOVERY_POLICY_VERSION)
''' + code[end:]
    start = code.index("    destination_created = False")
    end = code.index("    print(json.dumps(summary, sort_keys=True))", start)
    publication = code[start:end]
    code = (code[:start]
            + "    if MAX_STUDIES is None or summary['study_count'] == summary['test_study_count']:\n"
            + "".join("    " + line for line in publication.splitlines(keepends=True))
            + code[end:])
    code = code.replace("execution 361175523", "completed RadImageNet execution; no retraining")
    code = code.replace("RUN_INFERENCE = True\n",
                        f"RUN_INFERENCE = True\nMAX_STUDIES = {max_studies!r}\n", 1)
    compile(code, "radimagenet-submission", "exec")
    return code


def build_radimagenet_submission_package(repository_root, output_dir, report_path, checkpoint_path,
                                         training_source, *, source_root):
    repository = Path(repository_root).resolve(strict=True)
    output = Path(output_dir)
    if output.exists() or any(part.is_symlink() for part in (output, *output.parents)):
        raise ValueError("preparation_output_must_be_new")
    if (type(source_root) is not str or not source_root.startswith("/kaggle/input/")
            or ".." in PurePosixPath(source_root).parts):
        raise ValueError("source_mount_invalid")
    report, report_raw, checkpoint_sha, checkpoint_size, training_raw = validate_private_artifacts(
        report_path, checkpoint_path, training_source)
    with TemporaryDirectory() as temp_dir:
        base_dir = Path(temp_dir) / "base"
        base_result = build_submission_package(
            repository, base_dir, source_root=source_root,
            checkpoint_sha256=checkpoint_sha, checkpoint_size=checkpoint_size,
            report_sha256=EXPECTED_REPORT_SHA256, codec_manifest_sha256=CODEC_MANIFEST_SHA256,
            run_inference=True, recovery_policy_version=1,
        )
        base_archive = (base_dir / "fold0-inference-source.zip").read_bytes()
        base_notebook = json.loads(Path(base_result["notebook"]).read_text(encoding="utf-8"))
    source_archive, members = _extend_source_archive(base_archive, repository, training_raw)
    source_digest = _sha(source_archive)
    variants = []
    for label, maximum in (("canary", 3), ("submission", None)):
        notebook = json.loads(json.dumps(base_notebook))
        notebook["cells"][0]["source"] = [
            "# RSNA Knee: pinned RadImageNet ResNet-50 inference\n",
            "Internet OFF; use the official competition, existing offline DICOM codecs, and the exact private "
            "checkpoint/report inputs. The model uses 224px preprocessing, eight selected slices, mean-slice "
            "then mean-series logits, and a 12-target sigmoid output. The capped canary writes the root CSV "
            "only when every visible test row was processed.\n",
        ]
        code = _notebook_code(base_notebook, base_archive, source_archive, source_digest,
                              report_sha256=EXPECTED_REPORT_SHA256, checkpoint_sha256=checkpoint_sha,
                              checkpoint_size=checkpoint_size, fingerprint=NATIVE_FINGERPRINT,
                              max_studies=maximum)
        notebook["cells"][1]["source"] = code.splitlines(keepends=True)
        variants.append((label, notebook))

    output.mkdir(parents=True, exist_ok=False)
    (output / "radimagenet-inference-source.zip").write_bytes(source_archive)
    (output / "checkpoint-epoch-05.pt").write_bytes(Path(checkpoint_path).read_bytes())
    (output / "training-report.json").write_bytes(report_raw)
    inventory = {
        "schema": "radimagenet-submission-package-v1", "source_execution": SOURCE_EXECUTION,
        "source_archive_sha256": source_digest, "source_archive_member_count": len(members),
        "checkpoint_sha256": checkpoint_sha, "checkpoint_size": checkpoint_size,
        "report_sha256": EXPECTED_REPORT_SHA256, "training_source_sha256": TRAINING_SOURCE_SHA256,
        "native_training_fingerprint": NATIVE_FINGERPRINT,
        "codec_manifest_sha256": CODEC_MANIFEST_SHA256,
        "source_root": source_root, "remote_executed": False, "submitted": False,
        "canary_rows_cap": 3, "full_rows_cap": None,
    }
    result = {**inventory, "source_archive": str(output / "radimagenet-inference-source.zip"),
              "checkpoint": str(output / "checkpoint-epoch-05.pt"),
              "report": str(output / "training-report.json")}
    for label, notebook in variants:
        path = output / f"kaggle_radimagenet_{label}.ipynb"
        path.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
        result[label] = str(path)
        result[f"{label}_sha256"] = _sha(path.read_bytes())
    (output / "inventory.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
