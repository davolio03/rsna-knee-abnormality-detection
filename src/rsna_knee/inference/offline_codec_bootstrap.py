"""Install a pinned DICOM codec wheel bundle without network or dependency resolution."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from email import message_from_bytes
from pathlib import Path


_PACKAGES = (
    ("pylibjpeg", "2.1.0", "pylibjpeg-2.1.0-py3-none-any.whl", ("py3-none-any",)),
    ("pylibjpeg-libjpeg", "2.4.0", "pylibjpeg_libjpeg-2.4.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl", ("cp313-cp313-manylinux_2_17_x86_64", "cp313-cp313-manylinux2014_x86_64")),
    ("pylibjpeg-openjpeg", "2.5.0", "pylibjpeg_openjpeg-2.5.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl", ("cp313-cp313-manylinux_2_17_x86_64", "cp313-cp313-manylinux2014_x86_64")),
    ("pyjpegls", "1.5.1", "pyjpegls-1.5.1-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl", ("cp313-cp313-manylinux_2_17_x86_64", "cp313-cp313-manylinux2014_x86_64")),
)
_MAX_BUNDLE_BYTES = 20 * 1024 * 1024


def _require_runtime(manifest: dict) -> None:
    target = manifest.get("target")
    if not isinstance(target, dict):
        raise ValueError("codec_manifest_invalid")
    libc_name, libc_version = platform.libc_ver()
    match = re.match(r"^(\d+)\.(\d+)", libc_version or "")
    libc = tuple(map(int, match.groups())) if match else (0, 0)
    if (platform.python_implementation() != target.get("implementation")
            or tuple(sys.version_info[:2]) != (3, 13)
            or target.get("python") != "3.13"
            or platform.system() != target.get("system")
            or platform.machine().lower() != target.get("machine")
            or libc_name.lower() != "glibc"
            or libc < (2, 17)):
        raise ValueError("codec_runtime_incompatible")

    # Compiled wheels require NumPy 2, but replacing Kaggle's existing NumPy can
    # invalidate its preinstalled Torch stack. Fail before pip rather than resolve it.
    try:
        from importlib.metadata import version
        numpy_version = version("numpy")
        major = int(numpy_version.split(".", 1)[0])
    except Exception:
        raise ValueError("codec_numpy_requirement_unmet") from None
    if major != 2:
        raise ValueError("codec_numpy_requirement_unmet")


def _wheel_metadata(path: Path) -> tuple[str, str, list[str], list[str]]:
    try:
        with zipfile.ZipFile(path) as wheel:
            metadata_names = [name for name in wheel.namelist()
                              if name.endswith(".dist-info/METADATA")]
            wheel_names = [name for name in wheel.namelist()
                           if name.endswith(".dist-info/WHEEL")]
            if len(metadata_names) != 1 or len(wheel_names) != 1:
                raise ValueError
            metadata = message_from_bytes(wheel.read(metadata_names[0]))
            wheel_metadata = message_from_bytes(wheel.read(wheel_names[0]))
            tags = wheel_metadata.get_all("Tag", [])
            return (metadata.get("Name", ""), metadata.get("Version", ""),
                    metadata.get_all("Requires-Dist", []), tags)
    except Exception:
        raise ValueError("codec_wheel_metadata_invalid") from None


def install_codec_bundle(bundle_root, expected_manifest_sha256: str, target_dir, *, working_root=None) -> dict:
    """Validate and install the four pinned wheels into a new target directory."""
    target = Path(target_dir)
    supplied = Path(bundle_root)
    if supplied.is_file():
        if supplied.is_symlink() or supplied.stat().st_size > _MAX_BUNDLE_BYTES:
            raise ValueError("codec_bundle_invalid")
        expected_members = {"manifest.json", *(entry[2] for entry in _PACKAGES)}
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(supplied) as archive:
                infos = archive.infolist()
                names = [info.filename for info in infos]
                if len(names) != len(set(names)) or set(names) != expected_members:
                    raise ValueError("codec_bundle_members_invalid")
                if sum(info.file_size for info in infos) > _MAX_BUNDLE_BYTES:
                    raise ValueError("codec_bundle_size_exceeded")
                for info in infos:
                    mode = (info.external_attr >> 16) & 0o170000
                    if info.is_dir() or mode == 0o120000 or info.file_size > _MAX_BUNDLE_BYTES:
                        raise ValueError("codec_bundle_members_invalid")
                with tempfile.TemporaryDirectory(prefix="codec-bundle-", dir=target.parent) as temporary:
                    stage = Path(temporary)
                    for name in names:
                        (stage / name).write_bytes(archive.read(name))
                    return install_codec_bundle(stage, expected_manifest_sha256, target,
                                                working_root=working_root)
        except zipfile.BadZipFile:
            raise ValueError("codec_bundle_invalid") from None
    if not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256 or ""):
        raise ValueError("codec_manifest_pin_invalid")
    root_path = supplied
    if root_path.is_symlink():
        raise ValueError("codec_bundle_invalid")
    root = root_path.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("codec_bundle_invalid")
    expected_members = {"manifest.json", *(entry[2] for entry in _PACKAGES)}
    if {path.name for path in root.iterdir()} != expected_members:
        raise ValueError("codec_bundle_members_invalid")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("codec_manifest_missing")
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_sha256:
        raise ValueError("codec_manifest_pin_mismatch")
    try:
        manifest = json.loads(manifest_bytes)
    except Exception:
        raise ValueError("codec_manifest_invalid") from None
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "kaggle-offline-codec-bundle-v1":
        raise ValueError("codec_manifest_invalid")
    _require_runtime(manifest)
    packages = manifest.get("packages")
    if not isinstance(packages, list) or len(packages) != len(_PACKAGES):
        raise ValueError("codec_manifest_invalid")

    wheel_paths = []
    total_size = 0
    for item, (name, version, filename, expected_tags) in zip(packages, _PACKAGES):
        if (not isinstance(item, dict) or item.get("name") != name
                or item.get("version") != version or item.get("filename") != filename
                or Path(filename).name != filename):
            raise ValueError("codec_manifest_invalid")
        wheel_path = root / filename
        if wheel_path.is_symlink() or not wheel_path.is_file():
            raise ValueError("codec_wheel_integrity_failed")
        raw = wheel_path.read_bytes()
        if (len(raw) != item.get("size")
                or hashlib.sha256(raw).hexdigest() != item.get("sha256")):
            raise ValueError("codec_wheel_integrity_failed")
        actual_name, actual_version, requires_dist, tags = _wheel_metadata(wheel_path)
        if (actual_name.lower().replace("_", "-") != name
                or actual_version != version
                or requires_dist != item.get("requires_dist")
                or len(tags) != len(expected_tags)
                or set(tags) != set(expected_tags)):
            raise ValueError("codec_wheel_metadata_mismatch")
        if not item.get("source_url", "").startswith("https://files.pythonhosted.org/packages/"):
            raise ValueError("codec_manifest_invalid")
        total_size += len(raw)
        wheel_paths.append(wheel_path)
    if total_size > _MAX_BUNDLE_BYTES:
        raise ValueError("codec_bundle_size_exceeded")

    if target.exists() or target.is_symlink():
        raise ValueError("codec_target_must_be_new")
    if working_root is not None:
        working = Path(working_root).resolve(strict=True)
        resolved_target = target.resolve(strict=False)
        if not resolved_target.is_relative_to(working):
            raise ValueError("codec_target_outside_working")
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
               "--disable-pip-version-check", "--no-cache-dir", "--target", str(target),
               *(str(path) for path in wheel_paths)]
    environment = dict(os.environ)
    environment.update({"PIP_CONFIG_FILE": os.devnull, "PIP_NO_INDEX": "1",
                        "PIP_DISABLE_PIP_VERSION_CHECK": "1"})
    try:
        subprocess.run(command, check=True, capture_output=True, text=True,
                       timeout=300, env=environment)
    except Exception:
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target, ignore_errors=True)
        raise RuntimeError("codec_install_failed") from None
    return {"packages": [item[0] for item in _PACKAGES], "target": str(target)}
