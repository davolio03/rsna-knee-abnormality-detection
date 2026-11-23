import os
import shutil
import tarfile
import tempfile
import unittest
from email.parser import Parser
from pathlib import Path

import setuptools
from setuptools import build_meta

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        tomllib = None


ROOT = Path(__file__).resolve().parents[1]
SETUPTOOLS_VERSION = tuple(
    int(part) for part in setuptools.__version__.split(".")[:2]
)


def requirement_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


@unittest.skipIf(tomllib is None, "metadata tests require Python 3.11+ or test-only tomli")
class PackageInstallationMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    def test_project_metadata_is_minimal_and_python_compatible(self) -> None:
        project = self.metadata["project"]
        self.assertEqual(project["name"], "rsna-knee-abnormality-detection")
        self.assertEqual(project["version"], "1.0.0")
        self.assertEqual(project["requires-python"], ">=3.10")
        self.assertEqual(project["readme"], "README.md")
        self.assertEqual(project.get("dependencies", []), [])

    def test_optional_dependencies_match_existing_workload_requirements(self) -> None:
        optional = self.metadata["project"]["optional-dependencies"]
        self.assertEqual(optional["images"], requirement_lines(ROOT / "requirements-images.txt"))
        self.assertEqual(optional["siwc"], requirement_lines(ROOT / "requirements-siwc.txt"))

    def test_setuptools_discovers_only_src_layout_package_and_schema(self) -> None:
        setuptools = self.metadata["tool"]["setuptools"]
        self.assertEqual(setuptools["package-dir"], {"": "src"})
        self.assertFalse(setuptools["include-package-data"])
        self.assertEqual(
            setuptools["packages"]["find"]["include"],
            ["rsna_knee", "rsna_knee.*"],
        )
        self.assertEqual(
            setuptools["package-data"]["rsna_knee"],
            ["labels/codex_label_output.schema.json"],
        )
        self.assertTrue(
            (ROOT / "src/rsna_knee/labels/codex_label_output.schema.json").is_file()
        )

    def test_source_distribution_manifest_is_a_package_only_allowlist(self) -> None:
        manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
        directives = [line.strip() for line in manifest.splitlines() if line.strip()]
        self.assertIn("include README.md pyproject.toml", directives)
        self.assertIn("recursive-include src/rsna_knee *.py", directives)
        self.assertIn(
            "include src/rsna_knee/labels/codex_label_output.schema.json", directives
        )
        for path in ("data", "scripts", "tests", "notebooks", "configs", "experiments", "outputs"):
            self.assertIn(f"prune {path}", directives)
        self.assertIn("global-exclude *.py[cod]", directives)

    def test_generated_build_metadata_is_ignored(self) -> None:
        ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
        for pattern in ("build/", "dist/", "*.egg-info/"):
            self.assertIn(pattern, ignored.splitlines())

@unittest.skipIf(
    SETUPTOOLS_VERSION < (64, 0),
    f"sdist artifact check requires setuptools>=64; found {setuptools.__version__}",
)
class PackageArtifactPolicyTests(unittest.TestCase):
    def test_built_sdist_excludes_unlisted_json_and_private_root_files(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rsna-sdist-policy-") as temp_name:
            temp = Path(temp_name)
            shutil.copy2(ROOT / "pyproject.toml", temp / "pyproject.toml")
            shutil.copy2(ROOT / "MANIFEST.in", temp / "MANIFEST.in")
            shutil.copy2(ROOT / "README.md", temp / "README.md")
            shutil.copytree(ROOT / "src/rsna_knee", temp / "src/rsna_knee")
            private_package_json = temp / "src/rsna_knee/data/private_packaging_fixture.json"
            private_package_json.write_text('{"sentinel": "DO-NOT-PUBLISH"}\n', encoding="utf-8")
            private_root_file = temp / "data/private/packaging_fixture.txt"
            private_root_file.parent.mkdir(parents=True)
            private_root_file.write_text("PRIVATE-PACKAGING-SENTINEL\n", encoding="utf-8")
            legacy_script = temp / "scripts/packaging_fixture.py"
            legacy_script.parent.mkdir()
            legacy_script.write_text("# private compatibility fixture\n", encoding="utf-8")

            output = temp / "dist"
            output.mkdir()
            original_cwd = Path.cwd()
            try:
                os.chdir(temp)
                archive_name = build_meta.build_sdist(str(output))
            finally:
                os.chdir(original_cwd)

            with tarfile.open(output / archive_name, "r:gz") as archive:
                self.assertTrue(
                    archive_name.startswith("rsna-knee-abnormality-detection-1.0.0."),
                    archive_name,
                )
                metadata = Parser().parsestr(
                    archive.extractfile(f"{archive_name.removesuffix('.tar.gz')}/PKG-INFO")
                    .read()
                    .decode("utf-8")
                )
                self.assertEqual(metadata["Name"], "rsna-knee-abnormality-detection")
                self.assertEqual(metadata["Version"], "1.0.0")
                files = {
                    member.name.split("/", 1)[1]
                    for member in archive.getmembers()
                    if member.isfile() and "/" in member.name
                }

        self.assertIn("src/rsna_knee/labels/codex_label_output.schema.json", files)
        self.assertNotIn("src/rsna_knee/data/private_packaging_fixture.json", files)
        self.assertFalse(any(path.startswith(("data/", "scripts/", "tests/")) for path in files))
        self.assertFalse(any("DO-NOT-PUBLISH" in path or "PRIVATE-PACKAGING-SENTINEL" in path for path in files))


if __name__ == "__main__":
    unittest.main()
