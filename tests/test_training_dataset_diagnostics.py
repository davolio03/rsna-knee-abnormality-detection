import json
import io
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

try:
    import numpy as np
    import torch
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("optional NumPy/PyTorch training dependencies are not installed") from exc

from rsna_knee.labels.subscription import LABEL_COLUMNS


class TrainingDatasetDiagnosticPropagationTests(unittest.TestCase):
    def make_inputs(self, root):
        image_root = root / "images"
        (image_root / "synthetic-study" / "synthetic-series").mkdir(parents=True)
        record = {
            "labels": {name: 0 for name in LABEL_COLUMNS},
            "study_id": "synthetic-study",
            "series": [{"series_id": "synthetic-series"}],
        }
        return image_root, record

    def make_config(self, root):
        from rsna_knee.training.baseline import TrainingConfig

        for name in ("manifest.json", "folds.json"):
            (root / name).write_text("{}", encoding="utf-8")
        return TrainingConfig.from_mapping({
            "mode": "overfit", "manifest_path": str(root / "manifest.json"),
            "folds_path": str(root / "folds.json"), "image_root": str(root / "images"),
            "output_dir": str(root / "output"), "fold": 0, "seed": 42,
            "epochs": 1, "max_steps": 1, "overfit_limit": 1,
            "output_size": 8, "max_slices": 1, "device": "cpu",
            "learning_rate": 0.001, "feature_width": 8,
        })

    def make_dataset(self, root, error):
        from rsna_knee.data.training_dataset import TrainingStudyDataset

        image_root, record = self.make_inputs(root)
        def fail(*_args, **_kwargs):
            raise error
        with mock.patch(
            "rsna_knee.data.training_dataset._load_artifacts",
            return_value=([record], {"synthetic-study": 1}, "digest"),
        ):
            return TrainingStudyDataset(
                root / "manifest.json", root / "folds.json", image_root,
                partition="train", fold=0, output_size=8, max_slices=1,
                preprocessor=fail,
            )

    def test_preprocessor_reason_survives_dataset_runner_and_notebook_diagnostic_gate(self):
        from rsna_knee.data.dicom_preprocessing import DicomPreprocessingError
        from rsna_knee.training.baseline import TrainingRunError, run_training

        secret = "C:/private/study-identifier/series/path"
        with TemporaryDirectory() as temp:
            root = Path(temp)
            config = self.make_config(root)
            dataset = self.make_dataset(
                root, DicomPreprocessingError("pixel_decode_failed", f"private {secret}"),
            )
            with self.assertRaises(TrainingRunError) as caught:
                run_training(
                    config, execute=True,
                    dataset_factory=lambda *_args: dataset,
                )

        error = caught.exception
        self.assertEqual(error.reason_code, "training_failed")
        self.assertEqual(error.diagnostics["stage"], "training")
        self.assertEqual(error.diagnostics["data_reason_code"], "pixel_decode_failed")
        self.assertNotIn(secret, str(error))
        self.assertNotIn(secret, repr(error.diagnostics))

        notebook = json.loads(Path("notebooks/kaggle_first_fold_baseline.ipynb").read_text(encoding="utf-8"))
        code = "".join(next(cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code"))
        namespace = {"__name__": "first_fold_diagnostic_test"}
        exec(compile(code, "notebooks/kaggle_first_fold_baseline.ipynb", "exec"), namespace)
        diagnostics = namespace["safe_failure_diagnostics"](error)
        self.assertEqual(diagnostics["data_reason_code"], "pixel_decode_failed")
        self.assertNotIn(secret, repr(diagnostics))

        from scripts.train_baseline import main

        with TemporaryDirectory() as temp:
            config_path = Path(temp) / "config.json"
            config_path.write_text("{}", encoding="utf-8")
            stderr = io.StringIO()
            with mock.patch("scripts.train_baseline.TrainingConfig.from_mapping", return_value=object()), \
                 mock.patch("scripts.train_baseline.run_training", side_effect=error), \
                 redirect_stderr(stderr):
                exit_code = main(["--config", str(config_path), "--execute"])
        event = json.loads(stderr.getvalue())
        self.assertEqual(exit_code, 2)
        self.assertEqual(event["diagnostics"]["data_reason_code"], "pixel_decode_failed")
        self.assertNotIn(secret, stderr.getvalue())

    def test_unknown_or_spoofed_preprocessor_reason_is_not_propagated(self):
        from rsna_knee.data.dicom_preprocessing import DicomPreprocessingError
        from rsna_knee.data.training_dataset import TrainingDatasetError

        secret = "private path and study identifier"
        for error in (
            DicomPreprocessingError(secret, secret),
            type("SpoofedDicomError", (DicomPreprocessingError,), {})(
                "pixel_decode_failed", secret,
            ),
            type("SpoofedError", (ValueError,), {"reason_code": "pixel_decode_failed"})(secret),
        ):
            with TemporaryDirectory() as temp:
                root = Path(temp)
                dataset = self.make_dataset(root, error)
                with self.assertRaises(TrainingDatasetError) as caught:
                    dataset[0]
            self.assertEqual(caught.exception.reason_code, "study_load_failed")
            self.assertIsNone(getattr(caught.exception, "data_reason_code", None))
            self.assertNotIn(secret, str(caught.exception))

    def test_runner_diagnostic_schema_rejects_unknown_cause_code_and_extra_fields(self):
        from rsna_knee.training.baseline import _validate_failure_diagnostics

        base = {
            "stage": "training", "exception_family": "value", "exception_chain": ["value"],
            "data_reason_code": "pixel_decode_failed",
        }
        self.assertEqual(_validate_failure_diagnostics(base), base)
        self.assertIsNone(_validate_failure_diagnostics({**base, "data_reason_code": "private_path"}))
        self.assertIsNone(_validate_failure_diagnostics({**base, "study_id": "private"}))


if __name__ == "__main__":
    unittest.main()
