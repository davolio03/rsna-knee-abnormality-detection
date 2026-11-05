import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = {
    "scripts/data/prepare_attention16.py": "scripts/prepare_local16_attention.py",
    "scripts/training/train_attention16.py": "scripts/train_local16_attention.py",
    "scripts/evaluation/export_attention16_validation.py": "scripts/export_local16_validation.py",
    "scripts/review/prepare_attention16_sample.py": "scripts/prepare_local16_review_sample.py",
    "scripts/packaging/build_attention16_submission.py": "scripts/build_kaggle_attention_sixteen_submission_package.py",
}


def load_wrapper(relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(f"entrypoint_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load wrapper: {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WorkflowEntrypointTests(unittest.TestCase):
    def test_import_is_inert_and_does_not_mutate_sys_path(self) -> None:
        original_path = list(sys.path)
        for wrapper_path in ENTRYPOINTS:
            with self.subTest(wrapper=wrapper_path), mock.patch.object(subprocess, "run") as run:
                module = load_wrapper(wrapper_path)
                self.assertEqual(sys.path, original_path)
                run.assert_not_called()
                self.assertTrue(Path(module.LEGACY_COMMAND).is_absolute())

    def test_main_forwards_arguments_interpreter_working_directory_and_exit_code(self) -> None:
        arguments = ["--help", "--device", "cuda", "value with spaces"]
        for wrapper_path, legacy_path in ENTRYPOINTS.items():
            with self.subTest(wrapper=wrapper_path):
                module = load_wrapper(wrapper_path)
                result = mock.Mock(returncode=17)
                with mock.patch.object(module.sys, "executable", "test-python"), mock.patch.object(
                    module.subprocess, "run", return_value=result
                ) as run:
                    exit_code = module.main(arguments)
                self.assertEqual(exit_code, 17)
                run.assert_called_once_with(
                    [
                        "test-python",
                        str((ROOT / legacy_path).resolve()),
                        *arguments,
                    ],
                    cwd=ROOT,
                    check=False,
                )

    def test_main_uses_process_arguments_when_argv_is_omitted(self) -> None:
        for wrapper_path in ENTRYPOINTS:
            with self.subTest(wrapper=wrapper_path):
                module = load_wrapper(wrapper_path)
                result = mock.Mock(returncode=0)
                with mock.patch.object(module.sys, "argv", [wrapper_path, "--help"]), mock.patch.object(
                    module.subprocess, "run", return_value=result
                ) as run:
                    self.assertEqual(module.main(), 0)
                self.assertEqual(run.call_args.args[0][-1], "--help")


if __name__ == "__main__":
    unittest.main()
