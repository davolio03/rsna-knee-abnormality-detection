"""Bounded synthetic reproductions for the fold-zero production submission path.

No competition data, trained checkpoint, network access, or Kaggle runtime is used.
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import time
import tracemalloc
import unittest
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

try:
    import pydicom
    from PIL import Image
    from pydicom.encaps import encapsulate
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, JPEG2000Lossless, CTImageStorage, generate_uid
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("pydicom is required for synthetic real-DICOM composition") from exc

from rsna_knee.inference import fold0
from rsna_knee.inference.fold0 import InferenceError, TARGETS, TestStudyDataset, run_inference
from rsna_knee.inference import public_training_audit
from rsna_knee.inference.public_training_audit import (
    dataset_constructor_injection,
    run_public_training_inference_audit,
    stage_training_metadata,
    validate_public_training_plan,
)
from scripts.smoke_submission import validate_submission


def _training_report():
    return {
        "status": "completed", "mode": "fold", "fold": 0,
        "training_study_count": 333, "validation_study_count": 167,
        "steps_completed": 1665, "training_steps_planned": 1665, "epochs_requested": 5,
        "model_kind": "resnet18_imagenet1k_v1",
        "input_sha256": {
            "manifest": "e88c61c22a463737f16951b7c76db03834034aecd1b31dc61b4ce495080b2be2",
            "folds": "b82b376b0086201e3364b11a0203b62890e628442075a52ea4a9e09ee0e778ac"},
        "config": {"fold": 0, "seed": 42, "model_seed": 42, "output_size": 256,
                   "max_slices": 8, "slice_microbatch": 2, "freeze_backbone": True,
                   "epochs": 5, "learning_rate": .001, "model_kind": "resnet18_imagenet1k_v1"},
        "model_initialization": {
            "architecture": "torchvision_resnet18", "initialization": "imagenet1k_v1",
            "model_seed": 42, "freeze_backbone": True, "trainable_parameter_count": 6156,
            "head_trainable_parameter_count": 6156, "slice_microbatch": 2,
            "activation_checkpointing": False, "batchnorm_running_statistics": "frozen_eval_affine_frozen",
            "weights_sha256": "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
            "weights_size": 46830571}}


class _SyntheticModel:
    def to(self, _device):
        return self

    def eval(self):
        return self

    def __call__(self, images, masks):
        return torch.zeros((1, len(TARGETS)), dtype=torch.float32)


class _FailingModel(_SyntheticModel):
    def __call__(self, images, masks):
        raise RuntimeError("synthetic-private-runtime-detail")


def _write_dicom(path: Path, *, z: int, pixels=None, frames: int = 1, transfer_syntax=ExplicitVRLittleEndian):
    path.parent.mkdir(parents=True, exist_ok=True)
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = transfer_syntax
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = CTImageStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = path.parts[-3]
    dataset.SeriesInstanceUID = path.parts[-2]
    dataset.Modality = "CT"
    dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    dataset.ImagePositionPatient = [0, 0, z]
    dataset.PixelSpacing = [0.7, 0.8]
    dataset.Rows = 8
    dataset.Columns = 8
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.RescaleSlope = 1
    dataset.RescaleIntercept = 0
    if frames > 1:
        dataset.NumberOfFrames = frames
    if pixels is None:
        pixels = np.arange(64, dtype=np.uint16).reshape(8, 8) + z
    pixels = np.asarray(pixels, dtype="<u2")
    if transfer_syntax == JPEG2000Lossless:
        encoded = io.BytesIO()
        Image.fromarray(pixels).save(encoded, format="JPEG2000", irreversible=False, no_jp2=True)
        encoded_frame = encoded.getvalue()
        dataset.PixelData = encapsulate([encoded_frame])
        dataset["PixelData"].is_undefined_length = True
    else:
        dataset.PixelData = pixels.tobytes()
    dataset.save_as(path, enforce_file_format=True)


class SubmissionFailureReproductionTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.competition = self.root / "competition"
        self.source = self.root / "source"
        self.image_root = self.competition / "test_series"
        self.source.mkdir()
        report_path = self.source / "report.json"
        report_path.write_text(json.dumps(_training_report()), encoding="utf-8")
        self.report_sha256 = hashlib.sha256(report_path.read_bytes()).hexdigest()
        # This is deliberately not a checkpoint: the loader is replaced below and
        # the bytes must never reach torch.load or a model deserializer.
        (self.source / "checkpoint.pt").write_bytes(b"synthetic-placeholder")

    def _csv_metadata(self, study_ids, series_by_study):
        self.competition.mkdir(parents=True, exist_ok=True)
        with (self.competition / "test.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["StudyInstanceUID"])
            writer.writerows((study_id,) for study_id in study_ids)
        with (self.competition / "test_series.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["StudyInstanceUID", "SeriesInstanceUID"])
            for study_id, series_ids in series_by_study.items():
                writer.writerows((study_id, series_id) for series_id in series_ids)

    def _write_series(self, study_id, series_id, *, count=2, frames=1):
        directory = self.image_root / study_id / series_id
        for index in range(count):
            _write_dicom(directory / f"slice-{index}.dcm", z=index + 1, frames=frames)

    def _run(self, output, model=None, *, recovery_policy_version=None):
        loader = patch.object(fold0, "load_fold0_model", return_value=model or _SyntheticModel())
        return loader, lambda: run_inference(
            self.competition, self.source, "a" * 64, 21, self.report_sha256, output,
            device="cpu", recovery_policy_version=recovery_policy_version)

    def test_real_dicom_runner_csv_and_smoke_validator_compose(self):
        studies = ["1.2.826.0.1.3680043.10.543.100", "1.2.826.0.1.3680043.10.543.101"]
        series = {studies[0]: ["1.2.826.0.1.3680043.10.543.200"],
                  studies[1]: ["1.2.826.0.1.3680043.10.543.201"]}
        self._csv_metadata(studies, series)
        for study, series_ids in series.items():
            for series_id in series_ids:
                self._write_series(study, series_id)
        output = self.root / "inference-output"
        loader, invoke = self._run(output)
        stdout = io.StringIO()
        with loader, contextlib.redirect_stdout(stdout):
            summary = invoke()
        self.assertEqual((summary["study_count"], summary["series_count"]), (2, 2))
        self.assertEqual(summary["status"], "completed")
        submission = output / "submission.csv"
        validate_submission(submission, expected_ids=studies)
        with submission.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(rows[0], ["StudyInstanceUID", *TARGETS])
        self.assertEqual([row[0] for row in rows[1:]], studies)
        self.assertTrue(all(len(row) == 13 and all(0 <= float(v) <= 1 for v in row[1:]) for row in rows[1:]))
        self.assertNotIn(str(studies[0]), stdout.getvalue())

    def test_real_jpeg2000_dicom_roundtrip_through_production_runner(self):
        study = "1.2.826.0.1.3680043.10.543.107"
        series_id = "1.2.826.0.1.3680043.10.543.207"
        self._csv_metadata([study], {study: [series_id]})
        directory = self.image_root / study / series_id
        expected = np.arange(64, dtype=np.uint16).reshape(8, 8) + 1
        _write_dicom(directory / "slice-0.dcm", z=1, pixels=expected, transfer_syntax=JPEG2000Lossless)
        first = pydicom.dcmread(directory / "slice-0.dcm")
        fragments = list(pydicom.encaps.generate_frames(first.PixelData))
        self.assertEqual(len(fragments), 1)
        self.assertTrue(fragments[0].startswith(b"\xff\x4f"), "JPEG2000 DICOM frame must be a raw codestream")
        np.testing.assert_array_equal(first.pixel_array, expected)
        _write_dicom(directory / "slice-1.dcm", z=2, transfer_syntax=JPEG2000Lossless)
        output = self.root / "jpeg2000-output"
        loader, invoke = self._run(output)
        with loader, contextlib.redirect_stdout(io.StringIO()):
            summary = invoke()
        self.assertEqual((summary["study_count"], summary["series_count"]), (1, 1))
        validate_submission(output / "submission.csv", expected_ids=[study])

    def test_partial_recovery_real_dicom_clean_parity_and_truncated_pixel_journal(self):
        study = "1.2.826.0.1.3680043.10.543.220"
        series_id = "1.2.826.0.1.3680043.10.543.221"
        self._csv_metadata([study], {study: [series_id]})
        self._write_series(study, series_id, count=3)
        strict_output = self.root / "strict-clean"
        recovery_output = self.root / "recovery-clean"
        strict_loader, strict_run = self._run(strict_output)
        recovery_loader = patch.object(fold0, "load_fold0_model", return_value=_SyntheticModel())
        with strict_loader, contextlib.redirect_stdout(io.StringIO()):
            strict_run()
        with recovery_loader, contextlib.redirect_stdout(io.StringIO()):
            run_inference(self.competition, self.source, "a" * 64, 21, self.report_sha256,
                          recovery_output, device="cpu", recovery_policy_version=1)
        self.assertEqual((strict_output / "submission.csv").read_bytes(),
                         (recovery_output / "submission.csv").read_bytes())

        broken = pydicom.dcmread(self.image_root / study / series_id / "slice-1.dcm")
        broken.PixelData = broken.PixelData[:-2]
        broken.save_as(self.image_root / study / series_id / "slice-1.dcm", enforce_file_format=True)
        strict_bad_output = self.root / "strict-bad"
        strict_bad_loader, strict_bad_run = self._run(strict_bad_output)
        with strict_bad_loader, contextlib.redirect_stdout(io.StringIO()), self.assertRaises(InferenceError) as raised:
            strict_bad_run()
        self.assertEqual(raised.exception.reason_code, "pixel_decode_failed")
        self.assertFalse((strict_bad_output / "submission.csv").exists())
        partial_output = self.root / "recovery-partial"
        loader = patch.object(fold0, "load_fold0_model", return_value=_SyntheticModel())
        with loader, contextlib.redirect_stdout(io.StringIO()):
            summary = run_inference(self.competition, self.source, "a" * 64, 21,
                                    self.report_sha256, partial_output, device="cpu",
                                    recovery_policy_version=1)
        self.assertEqual(summary["study_count"], 1)
        validate_submission(partial_output / "submission.csv", expected_ids=[study])
        journal_text = (partial_output / "recovery-journal.json").read_text(encoding="utf-8")
        journal = json.loads(journal_text)
        self.assertEqual(journal["studies"][0]["slice_reasons"], [
            {"series_ordinal": 1, "slice_ordinal": 2, "reason_code": "pixel_payload_invalid"},
        ])
        self.assertNotIn(study, journal_text)
        self.assertNotIn(series_id, journal_text)

        mixed_study = "1.2.826.0.1.3680043.10.543.224"
        bad_series = "1.2.826.0.1.3680043.10.543.225"
        good_series = "1.2.826.0.1.3680043.10.543.226"
        self._csv_metadata([mixed_study], {mixed_study: [bad_series, good_series]})
        self._write_series(mixed_study, bad_series, count=2)
        self._write_series(mixed_study, good_series, count=2)
        for path in (self.image_root / mixed_study / bad_series).glob('*.dcm'):
            dataset = pydicom.dcmread(path)
            dataset.PixelData = dataset.PixelData[:-2]
            dataset.save_as(path, enforce_file_format=True)
        mixed_output = self.root / "recovery-bad-series-with-good-series"
        mixed_loader = patch.object(fold0, "load_fold0_model", return_value=_SyntheticModel())
        with mixed_loader, contextlib.redirect_stdout(io.StringIO()):
            mixed_summary = run_inference(self.competition, self.source, "a" * 64, 21,
                                          self.report_sha256, mixed_output, device="cpu",
                                          recovery_policy_version=1)
        validate_submission(mixed_output / "submission.csv", expected_ids=[mixed_study])
        self.assertEqual((mixed_summary['series_skipped'], mixed_summary['slice_skipped']), (1, 2))
        mixed_journal = json.loads((mixed_output / 'recovery-journal.json').read_text(encoding='utf-8'))
        self.assertEqual(mixed_journal['studies'][0]['series_reasons'], [
            {'series_ordinal': 1, 'reason_code': 'series_files_missing'},
        ])
        self.assertEqual([item['slice_ordinal'] for item in mixed_journal['studies'][0]['slice_reasons']], [1, 2])
        self.assertNotIn(mixed_study, json.dumps(mixed_journal))

    def test_missing_test_report_column_is_not_required_by_fold0_dataset(self):
        study = "1.2.826.0.1.3680043.10.543.102"
        series_id = "1.2.826.0.1.3680043.10.543.202"
        self._csv_metadata([study], {study: [series_id]})
        self._write_series(study, series_id)
        data = TestStudyDataset(self.competition / "test.csv", self.competition / "test_series.csv", self.image_root)
        self.assertEqual(len(data), 1)
        self.assertEqual(tuple(data[0][0][0].shape), (2, 1, 256, 256))

    def test_missing_series_directory_fails_closed_without_csv_or_identifier_output(self):
        study = "1.2.826.0.1.3680043.10.543.103"
        series_id = "1.2.826.0.1.3680043.10.543.203"
        self._csv_metadata([study], {study: [series_id]})
        self.image_root.mkdir(parents=True)
        output = self.root / "missing-series-output"
        loader, invoke = self._run(output)
        stdout = io.StringIO()
        with loader, contextlib.redirect_stdout(stdout), self.assertRaises(InferenceError) as raised:
            invoke()
        self.assertEqual(raised.exception.reason_code, "series_path_invalid")
        self.assertFalse((output / "submission.csv").exists())
        diagnostic = json.loads((output / "inference-diagnostics.json").read_text(encoding="utf-8"))
        self.assertEqual(diagnostic["reason_code"], "series_path_invalid")
        self.assertNotIn(study, stdout.getvalue())
        self.assertNotIn(series_id, stdout.getvalue())

    def test_unsupported_multiframe_fails_safely_without_publishing_csv(self):
        study = "1.2.826.0.1.3680043.10.543.104"
        series_id = "1.2.826.0.1.3680043.10.543.204"
        self._csv_metadata([study], {study: [series_id]})
        self._write_series(study, series_id, count=1, frames=2)
        output = self.root / "multiframe-output"
        loader, invoke = self._run(output)
        with loader, contextlib.redirect_stdout(io.StringIO()), self.assertRaises(InferenceError) as raised:
            invoke()
        self.assertEqual(raised.exception.reason_code, "unsupported_frame_count")
        self.assertFalse((output / "submission.csv").exists())

    def test_model_runtime_error_is_categorized_and_not_published(self):
        study = "1.2.826.0.1.3680043.10.543.105"
        series_id = "1.2.826.0.1.3680043.10.543.205"
        self._csv_metadata([study], {study: [series_id]})
        self._write_series(study, series_id)
        output = self.root / "model-failure-output"
        loader, invoke = self._run(output, _FailingModel())
        stdout = io.StringIO()
        with loader, contextlib.redirect_stdout(stdout), self.assertRaises(InferenceError) as raised:
            invoke()
        self.assertEqual(raised.exception.reason_code, "unexpected_error")
        diagnostic = json.loads((output / "inference-diagnostics.json").read_text(encoding="utf-8"))
        self.assertEqual((diagnostic["stage"], diagnostic["exception_family"]), ("model", "runtime"))
        self.assertNotIn("synthetic-private-runtime-detail", stdout.getvalue())
        self.assertFalse((output / "submission.csv").exists())

    def test_recovery_policy_keeps_model_and_provenance_errors_fatal(self):
        study = "1.2.826.0.1.3680043.10.543.222"
        series_id = "1.2.826.0.1.3680043.10.543.223"
        self._csv_metadata([study], {study: [series_id]})
        self._write_series(study, series_id)
        output = self.root / "recovery-model-failure"
        loader, invoke = self._run(output, _FailingModel(), recovery_policy_version=1)
        with loader, contextlib.redirect_stdout(io.StringIO()), self.assertRaises(InferenceError) as raised:
            invoke()
        self.assertEqual(raised.exception.reason_code, "unexpected_error")
        journal = json.loads((output / "recovery-journal.json").read_text(encoding="utf-8"))
        self.assertEqual(journal["failure"]["stage"], "model")
        self.assertFalse((output / "submission.csv").exists())

        report = json.loads((self.source / "report.json").read_text(encoding="utf-8"))
        report["fold"] = 1
        (self.source / "report.json").write_text(json.dumps(report), encoding="utf-8")
        bad_digest = hashlib.sha256((self.source / "report.json").read_bytes()).hexdigest()
        provenance_output = self.root / "recovery-provenance-failure"
        loader = patch.object(fold0, "load_fold0_model", return_value=_SyntheticModel())
        with loader, contextlib.redirect_stdout(io.StringIO()), self.assertRaises(InferenceError) as raised:
            run_inference(self.competition, self.source, "a" * 64, 21, bad_digest,
                          provenance_output, device="cpu", recovery_policy_version=1)
        self.assertEqual(raised.exception.reason_code, "source_report_provenance_mismatch")
        journal = json.loads((provenance_output / "recovery-journal.json").read_text(encoding="utf-8"))
        self.assertEqual(journal["failure"]["stage"], "checkpoint")

    def test_permission_denied_series_is_fatal_with_valid_sibling_and_no_csv(self):
        study = "1.2.826.0.1.3680043.10.543.227"
        denied_series = "1.2.826.0.1.3680043.10.543.228"
        valid_series = "1.2.826.0.1.3680043.10.543.229"
        self._csv_metadata([study], {study: [denied_series, valid_series]})
        self._write_series(study, denied_series)
        self._write_series(study, valid_series)
        real_preprocessor = fold0.preprocess_dicom_series
        def permission_denied(path, **kwargs):
            if Path(path).name == denied_series:
                raise PermissionError("PRIVATE")
            return real_preprocessor(path, **kwargs)
        output = self.root / "permission-denied-recovery"
        loader = patch.object(fold0, "load_fold0_model", return_value=_SyntheticModel())
        preprocessor = patch.object(fold0, "preprocess_dicom_series", side_effect=permission_denied)
        with loader, preprocessor, contextlib.redirect_stdout(io.StringIO()), self.assertRaises(InferenceError):
            run_inference(self.competition, self.source, "a" * 64, 21, self.report_sha256,
                          output, device="cpu", recovery_policy_version=1)
        self.assertFalse((output / "submission.csv").exists())
        journal = json.loads((output / "recovery-journal.json").read_text(encoding="utf-8"))
        self.assertEqual(journal["failure"]["reason_code"], "unexpected_error")
        self.assertEqual(journal["failure"]["exception_family"], "io")
        self.assertNotIn("PRIVATE", json.dumps(journal))
        self.assertNotIn(study, json.dumps(journal))

        (self.source / "report.json").write_text(json.dumps(_training_report()), encoding="utf-8")
        valid_report_digest = hashlib.sha256((self.source / "report.json").read_bytes()).hexdigest()
        checkpoint_output = self.root / "recovery-checkpoint-failure"
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(InferenceError) as raised:
            run_inference(self.competition, self.source, "0" * 64, 21, valid_report_digest,
                          checkpoint_output, device="cpu", recovery_policy_version=1)
        self.assertEqual(raised.exception.reason_code, "checkpoint_invalid")
        journal = json.loads((checkpoint_output / "recovery-journal.json").read_text(encoding="utf-8"))
        self.assertEqual(journal["failure"]["stage"], "checkpoint")

    def test_duplicate_metadata_and_existing_output_are_rejected_without_overwrite(self):
        study = "1.2.826.0.1.3680043.10.543.106"
        series_id = "1.2.826.0.1.3680043.10.543.206"
        self._csv_metadata([study, study], {study: [series_id]})
        with self.assertRaisesRegex(ValueError, "^duplicate_study$"):
            TestStudyDataset(self.competition / "test.csv", self.competition / "test_series.csv", self.image_root)
        self._csv_metadata([study], {study: [series_id],
                                     "1.2.826.0.1.3680043.10.543.299": ["1.2.826.0.1.3680043.10.543.399"]})
        self.image_root.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "^series_metadata_invalid$"):
            TestStudyDataset(self.competition / "test.csv", self.competition / "test_series.csv", self.image_root)
        self._csv_metadata([study], {study: [series_id]})
        self._write_series(study, series_id)
        output = self.root / "collision-output"
        output.mkdir()
        existing = output / "submission.csv"
        existing.write_bytes(b"preserve-existing-output")
        loader, invoke = self._run(output)
        with loader, contextlib.redirect_stdout(io.StringIO()), self.assertRaises(InferenceError) as raised:
            invoke()
        self.assertEqual(raised.exception.reason_code, "output_must_be_new")
        self.assertEqual(existing.read_bytes(), b"preserve-existing-output")

    def test_scale_runner_cpu_stub_preprocessor_only(self):
        """Exercise row/grouping overhead only; this is not a decode/GPU benchmark."""
        count = 1300
        studies = [f"1.2.826.0.1.3680043.10.543.{1000 + index}" for index in range(count)]
        series = {study: [f"1.2.826.0.1.3680043.10.543.{3000 + index}"] for index, study in enumerate(studies)}
        self._csv_metadata(studies, series)
        self.image_root.mkdir(parents=True)
        for study, series_ids in series.items():
            (self.image_root / study / series_ids[0]).mkdir(parents=True)
        array = np.zeros((1, 1, 256, 256), dtype=np.float32)
        mask = np.ones_like(array, dtype=np.uint8)
        fake = type("SyntheticPreprocessed", (), {"images": array, "valid_mask": mask})()
        output = self.root / "scale-output"
        loader, invoke = self._run(output)
        tracemalloc.start()
        started = time.perf_counter()
        with loader, patch.object(fold0, "preprocess_dicom_series", return_value=fake), contextlib.redirect_stdout(io.StringIO()):
            summary = invoke()
        elapsed = time.perf_counter() - started
        _, peak_python_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.assertEqual((summary["study_count"], summary["series_count"]), (count, count))
        validate_submission(output / "submission.csv", expected_ids=studies)
        with (output / "submission.csv").open(encoding="utf-8") as stream:
            self.assertEqual(sum(1 for _ in stream), count + 1)
        # Keep measurements available in the test report without printing IDs or rows.
        self.assertGreater(elapsed, 0)
        self.assertGreater(peak_python_bytes, 0)
        print(json.dumps({"scale_studies": count, "scale_series": count,
                          "scale_elapsed_seconds": round(elapsed, 3),
                          "scale_peak_python_tracemalloc_bytes": peak_python_bytes,
                          "scale_scope": "stub preprocessor and model; excludes DICOM decode and GPU"}, sort_keys=True))

class PublicTrainingAuditInjectionRegressions(unittest.TestCase):
    def test_dataset_constructor_is_redirected_and_restored_after_success(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            production_root = root / "competition"
            staging = root / "staging"
            training_root = root / "train_series"
            calls = []

            def original(test_csv, series_csv, image_root):
                calls.append((test_csv, series_csv, image_root))
                return "real-dataset"

            namespace = SimpleNamespace(TestStudyDataset=original)
            with dataset_constructor_injection(
                namespace, production_root, staging / "test.csv",
                staging / "test_series.csv", training_root,
            ):
                factory = namespace.TestStudyDataset
                self.assertEqual(
                    factory(production_root / "test.csv", production_root / "test_series.csv",
                            production_root / "test_series"),
                    "real-dataset",
                )
            self.assertIs(namespace.TestStudyDataset, original)
            self.assertEqual(calls, [(staging / "test.csv", staging / "test_series.csv", training_root)])

    def test_dataset_constructor_is_restored_after_runner_failure(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            original = lambda *args: None
            namespace = SimpleNamespace(TestStudyDataset=original)
            with self.assertRaisesRegex(RuntimeError, "synthetic runner failure"):
                with dataset_constructor_injection(
                    namespace, root / "competition", root / "staging/test.csv",
                    root / "staging/test_series.csv", root / "train_series",
                ):
                    self.assertIsNot(namespace.TestStudyDataset, original)
                    raise RuntimeError("synthetic runner failure")
            self.assertIs(namespace.TestStudyDataset, original)

    def test_dataset_constructor_rejects_unexpected_production_arguments(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            original = lambda *args: None
            namespace = SimpleNamespace(TestStudyDataset=original)
            with dataset_constructor_injection(
                namespace, root / "competition", root / "staging/test.csv",
                root / "staging/test_series.csv", root / "train_series",
            ):
                with self.assertRaisesRegex(ValueError, "production_dataset_arguments_unexpected"):
                    namespace.TestStudyDataset(root / "wrong/test.csv", root / "competition/test_series.csv",
                                               root / "competition/test_series")
            self.assertIs(namespace.TestStudyDataset, original)

    def test_staged_training_metadata_contains_only_plan_ids_and_no_labels(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            source = root / "train_series.csv"
            source.write_text(
                "StudyInstanceUID,SeriesInstanceUID,Anatomical_Plane,PrivateLabel\n"
                "study-a,series-a,Sagittal,SECRET\n"
                "study-b,series-b,Coronal,SECRET\n"
                "unselected,series-c,Axial,SECRET\n",
                encoding="utf-8",
            )
            series_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            plan = {
                "schema_version": "public-training-inference-plan-v1",
                "source_plan_sha256": "a" * 64,
                "series_metadata_sha256": series_sha,
                "population_study_count": 4407,
                "fresh_study_count": 3849,
                "excluded_study_count": 558,
                "seed": 42,
                "study_count": 2,
                "series_count": 2,
                "plane_counts": {"Axial": 0, "Coronal": 1, "Sagittal": 1},
                "studies": [
                    {"study_id": "study-a", "series": [{"series_id": "series-a", "plane": "Sagittal"}]},
                    {"study_id": "study-b", "series": [{"series_id": "series-b", "plane": "Coronal"}]},
                ],
            }
            self.assertEqual(validate_public_training_plan(
                plan, expected_source_plan_sha256="a" * 64,
                expected_series_sha256=series_sha, expected_studies=2,
                expected_series=2, expected_plane_counts={"Axial": 0, "Coronal": 1, "Sagittal": 1}),
                ("study-a", "study-b"))
            altered = json.loads(json.dumps(plan))
            altered["studies"][0]["labels"] = {"ACL": 1}
            with self.assertRaises(ValueError):
                validate_public_training_plan(altered, expected_source_plan_sha256="a" * 64,
                    expected_series_sha256=series_sha, expected_studies=2, expected_series=2,
                    expected_plane_counts={"Axial": 0, "Coronal": 1, "Sagittal": 1})
            test_csv, series_csv, study_ids = stage_training_metadata(
                plan, source, root / "stage", expected_source_plan_sha256="a" * 64,
                expected_series_sha256=series_sha, expected_studies=2, expected_series=2,
                expected_plane_counts={"Axial": 0, "Coronal": 1, "Sagittal": 1},
            )
            self.assertEqual(study_ids, ("study-a", "study-b"))
            self.assertEqual(test_csv.read_text(encoding="utf-8"),
                             "StudyInstanceUID\nstudy-a\nstudy-b\n")
            staged = series_csv.read_text(encoding="utf-8")
            self.assertEqual(staged, "StudyInstanceUID,SeriesInstanceUID\nstudy-a,series-a\nstudy-b,series-b\n")
            self.assertNotIn("SECRET", staged + test_csv.read_text(encoding="utf-8"))

    def test_actual_runner_composes_with_narrow_training_dataset_injection(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            competition = root / "competition"
            training_root = root / "train_series"
            stage = root / "staging"
            studies = ("1.2.826.0.1.3680043.10.543.901",)
            series = ("1.2.826.0.1.3680043.10.543.902",)
            training_root.joinpath(studies[0], series[0]).mkdir(parents=True)
            for index in range(2):
                _write_dicom(training_root / studies[0] / series[0] / f"slice-{index}.dcm", z=index + 1)
            source_csv = root / "train_series.csv"
            source_csv.write_text(
                "StudyInstanceUID,SeriesInstanceUID,Anatomical_Plane\n"
                f"{studies[0]},{series[0]},Sagittal\n", encoding="utf-8",
            )
            series_sha = hashlib.sha256(source_csv.read_bytes()).hexdigest()
            plan = {
                "schema_version": "public-training-inference-plan-v1",
                "source_plan_sha256": "b" * 64,
                "series_metadata_sha256": series_sha,
                "population_study_count": 4407,
                "fresh_study_count": 3849,
                "excluded_study_count": 558,
                "seed": 42,
                "study_count": 1,
                "series_count": 1,
                "plane_counts": {"Axial": 0, "Coronal": 0, "Sagittal": 1},
                "studies": [{"study_id": studies[0],
                             "series": [{"series_id": series[0], "plane": "Sagittal"}]}],
            }
            source = root / "source"
            source.mkdir()
            report_path = source / "report.json"
            report_path.write_text(json.dumps(_training_report()), encoding="utf-8")
            report_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()
            (source / "checkpoint.pt").write_bytes(b"synthetic-placeholder")
            output = root / "audit-output"
            original_dataset = fold0.TestStudyDataset
            with (patch.object(fold0, "load_fold0_model", return_value=_SyntheticModel()),
                  patch.object(public_training_audit, "load_public_training_plan", return_value=plan),
                  contextlib.redirect_stdout(io.StringIO())):
                summary = run_public_training_inference_audit(
                    plan_archive=root / "synthetic-plan.zip", expected_plan_sha256="d" * 64,
                    train_series_csv=source_csv, training_image_root=training_root,
                    competition_root=competition, source_root=source,
                    checkpoint_sha256="c" * 64, checkpoint_size=21, report_sha256=report_sha,
                    output_root=output, fold0_module=fold0, runner=run_inference, device="cpu",
                )
            validate_submission(output / "inference" / "submission.csv", expected_ids=studies)
            self.assertIs(fold0.TestStudyDataset, original_dataset)
            self.assertEqual(list(output.glob("training-input-*")), [])
            audit_report = (output / "audit-summary.json").read_text(encoding="utf-8")
            self.assertNotIn(studies[0], audit_report)
            self.assertEqual((summary["status"], summary["study_count"], summary["series_count"]),
                             ("complete", 1, 1))


if __name__ == "__main__":
    unittest.main()
