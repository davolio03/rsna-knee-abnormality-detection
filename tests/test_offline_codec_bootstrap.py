"""Fail-closed tests for the private offline DICOM codec bootstrap."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from contextlib import contextmanager, ExitStack
from pathlib import Path
from unittest.mock import patch

from rsna_knee.inference import offline_codec_bootstrap as bootstrap


PACKAGE_DATA = (
    ("pylibjpeg", "2.1.0", "pylibjpeg-2.1.0-py3-none-any.whl", ("py3-none-any",)),
    ("pylibjpeg-libjpeg", "2.4.0", "pylibjpeg_libjpeg-2.4.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl", ("cp313-cp313-manylinux_2_17_x86_64", "cp313-cp313-manylinux2014_x86_64")),
    ("pylibjpeg-openjpeg", "2.5.0", "pylibjpeg_openjpeg-2.5.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl", ("cp313-cp313-manylinux_2_17_x86_64", "cp313-cp313-manylinux2014_x86_64")),
    ("pyjpegls", "1.5.1", "pyjpegls-1.5.1-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl", ("cp313-cp313-manylinux_2_17_x86_64", "cp313-cp313-manylinux2014_x86_64")),
)


class OfflineCodecBootstrapTests(unittest.TestCase):
    @contextmanager
    def _compatible_runtime(self):
        with ExitStack() as stack:
            stack.enter_context(patch.object(bootstrap.platform, "python_implementation", return_value="CPython"))
            stack.enter_context(patch.object(bootstrap.platform, "system", return_value="Linux"))
            stack.enter_context(patch.object(bootstrap.platform, "machine", return_value="x86_64"))
            stack.enter_context(patch.object(bootstrap.sys, "version_info", (3, 13, 2)))
            stack.enter_context(patch.object(bootstrap.platform, "libc_ver", return_value=("glibc", "2.39")))
            yield

    def _bundle(self, root: Path):
        packages = []
        for name, version, filename, tags in PACKAGE_DATA:
            normalized = name.replace("-", "_")
            dist = f"{normalized}-{version}.dist-info"
            wheel = root / filename
            metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n\n"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr(f"{dist}/METADATA", metadata)
                archive.writestr(f"{dist}/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\n" +
                                 "".join(f"Tag: {tag}\n" for tag in tags))
                archive.writestr("payload.txt", "synthetic wheel fixture")
            raw = wheel.read_bytes()
            packages.append({
                "name": name,
                "version": version,
                "filename": filename,
                "size": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "requires_dist": [],
                "source_url": f"https://files.pythonhosted.org/packages/example/{filename}",
            })
        manifest = {
            "schema_version": "kaggle-offline-codec-bundle-v1",
            "target": {"implementation": "CPython", "python": "3.13", "system": "Linux", "machine": "x86_64", "minimum_glibc": "2.17"},
            "packages": packages,
        }
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        return path, hashlib.sha256(path.read_bytes()).hexdigest(), packages

    def test_valid_bundle_installs_only_explicit_offline_wheels_into_new_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, digest, packages = self._bundle(root)
            target = root / "working" / "codec-target"
            with self._compatible_runtime(), patch.object(bootstrap.subprocess, "run") as run:
                result = bootstrap.install_codec_bundle(root, digest, target)
            self.assertEqual(result["packages"], [item[0] for item in PACKAGE_DATA])
            self.assertEqual(run.call_count, 1)
            command = run.call_args.args[0]
            self.assertIn("--no-index", command)
            self.assertIn("--no-deps", command)
            self.assertIn("--target", command)
            self.assertEqual(command[command.index("--target") + 1], str(target))
            self.assertEqual(
                [Path(item).name for item in command[command.index("--target") + 2:]],
                [item[2] for item in PACKAGE_DATA],
            )
            self.assertFalse(manifest.is_symlink())

    def test_real_pinned_bundle_accepts_actual_multiline_wheel_tags(self):
        bundle = (Path(__file__).resolve().parents[1] /
                  "data/private/hidden-submission-codecs/run07-bundle-final/rsna-knee-offline-dicom-codecs.zip")
        self.assertTrue(bundle.is_file(), "private HSR08 codec bundle required for integration fixture")
        with zipfile.ZipFile(bundle) as archive:
            manifest_bytes = archive.read("manifest.json")
        digest = hashlib.sha256(manifest_bytes).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "working" / "codec-target"
            with self._compatible_runtime(), patch.object(bootstrap.subprocess, "run") as run:
                result = bootstrap.install_codec_bundle(bundle, digest, target, working_root=root / "working")
            self.assertEqual(result["packages"], [item[0] for item in PACKAGE_DATA])
            self.assertEqual(run.call_count, 1)

    def test_single_zip_bundle_allows_only_the_manifest_and_four_pinned_wheels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, digest, packages = self._bundle(root)
            archive_path = root / "bundle.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.write(root / "manifest.json", "manifest.json")
                for item in packages:
                    archive.write(root / item["filename"], item["filename"])
            target = root / "working" / "codec-target"
            with self._compatible_runtime(), patch.object(bootstrap.subprocess, "run") as run:
                result = bootstrap.install_codec_bundle(archive_path, digest, target)
            self.assertEqual(result["packages"], [item[0] for item in PACKAGE_DATA])
            self.assertEqual(run.call_count, 1)

    def test_zip_with_an_unlisted_member_is_rejected_before_installer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, digest, packages = self._bundle(root)
            archive_path = root / "bundle.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.write(root / "manifest.json", "manifest.json")
                for item in packages:
                    archive.write(root / item["filename"], item["filename"])
                archive.writestr("extra.txt", "not allowed")
            with patch.object(bootstrap.subprocess, "run") as run:
                with self.assertRaisesRegex(ValueError, "codec_bundle_members_invalid"):
                    bootstrap.install_codec_bundle(archive_path, digest, root / "target")
            run.assert_not_called()

    def test_modified_manifest_is_rejected_before_installer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, digest, _ = self._bundle(root)
            (root / "manifest.json").write_text("{}\n", encoding="utf-8")
            with patch.object(bootstrap.subprocess, "run") as run:
                with self.assertRaisesRegex(ValueError, "codec_manifest_pin_mismatch"):
                    bootstrap.install_codec_bundle(root, digest, root / "target")
            run.assert_not_called()

    def test_missing_or_tampered_wheel_is_rejected_before_installer(self):
        for tamper in ("missing", "bytes"):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _, digest, packages = self._bundle(root)
                wheel = root / packages[0]["filename"]
                if tamper == "missing":
                    wheel.unlink()
                else:
                    wheel.write_bytes(wheel.read_bytes() + b"x")
                with self._compatible_runtime(), patch.object(bootstrap.subprocess, "run") as run:
                    expected = "codec_bundle_members_invalid" if tamper == "missing" else "codec_wheel_integrity_failed"
                    with self.assertRaisesRegex(ValueError, expected):
                        bootstrap.install_codec_bundle(root, digest, root / "target")
                run.assert_not_called()

    def test_wrong_runtime_platform_is_rejected_before_installer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, digest, _ = self._bundle(root)
            with patch.object(bootstrap.platform, "python_implementation", return_value="CPython"), \
                 patch.object(bootstrap.platform, "system", return_value="Linux"), \
                 patch.object(bootstrap.platform, "machine", return_value="aarch64"), \
                 patch.object(bootstrap.sys, "version_info", (3, 13, 2)), \
                 patch.object(bootstrap.platform, "libc_ver", return_value=("glibc", "2.39")), \
                 patch.object(bootstrap.subprocess, "run") as run:
                with self.assertRaisesRegex(ValueError, "codec_runtime_incompatible"):
                    bootstrap.install_codec_bundle(root, digest, root / "target")
            run.assert_not_called()

    def test_existing_numpy_below_two_is_rejected_without_dependency_replacement(self):
        import importlib.metadata
        for numpy_version in ("1.26.4", "3.0.0"):
            with self.subTest(numpy_version=numpy_version), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _, digest, _ = self._bundle(root)
                with self._compatible_runtime(), patch.object(importlib.metadata, "version", return_value=numpy_version), \
                     patch.object(bootstrap.subprocess, "run") as run:
                    with self.assertRaisesRegex(ValueError, "codec_numpy_requirement_unmet"):
                        bootstrap.install_codec_bundle(root, digest, root / "target")
                run.assert_not_called()

    def test_installer_failure_removes_owned_target_and_never_creates_submission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, digest, _ = self._bundle(root)
            target = root / "working" / "codec-target"

            def fail_after_creating_target(command, **kwargs):
                target.mkdir(parents=True)
                (target / "partial.txt").write_text("partial", encoding="utf-8")
                raise OSError("private installer diagnostic")

            with self._compatible_runtime(), \
                 patch.object(bootstrap.subprocess, "run", side_effect=fail_after_creating_target):
                with self.assertRaisesRegex(RuntimeError, "codec_install_failed") as raised:
                    bootstrap.install_codec_bundle(root, digest, target)
            self.assertNotIn("private installer diagnostic", str(raised.exception))
            self.assertFalse(target.exists())
            self.assertFalse((root / "submission.csv").exists())


if __name__ == "__main__":
    unittest.main()
