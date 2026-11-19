# E031: Fold-zero submission failure reproduction

## Status

SR01 and SR02 are complete as performed; SR03 records the final bounded interpretation. The parent-owned independent scoped verification passed. No production behavior changed, no Kaggle execution or submission occurred, and the historical hidden failure cause remains unknown.

## Question and hypothesis

Which local failure classes can be reproduced when the fold-zero production dataset, preprocessing, inference runner, CSV writer, and smoke validator are composed? The working hypothesis is only that some boundaries may be locally reproduced; no local or public evidence is assumed to explain the hidden `NotebookThrewException`.

## Baseline and changed variable

Baseline: preserved enabled run05 package/source matches the current source package (19 members). Its visible production inference and CSV checks succeeded on 3 studies / 15 series. The subsequent hidden execution reported a generic `NotebookThrewException`; no specific cause was exposed. The later 100-study / 546-series CPU preprocessing and 20-study / 81-series GPU model diagnostics succeeded, but neither exercised the complete production runner-plus-CSV composition at hidden scale.

Changed variable for SR01: add isolated tests that compose the current production runner and validation path on temporary synthetic inputs; stub only trained-checkpoint loading and model forward. No production code, preprocessing, checkpoint, or remote artifact changed.

## Public implementation context

The parent reviewed the [competition leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard), [second-place write-up](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/2nd-place-gold-solution-vision-llm-soft-labels-an), [fifth-place write-up](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/5th-place-solution), the [second-place source repository at commit `4f168e3894c8ef12ed454b26ae7f9bc495c332a0`](https://github.com/azamat1ch/rsna-knee-2nd-place/tree/4f168e3894c8ef12ed454b26ae7f9bc495c332a0), and its [prediction notebook](https://www.kaggle.com/code/azamat1ch/rsna-knee-2nd-place-predict). These sources provide comparator context only.

The reviewed second-place implementation pins JPEG Lossless/SV1 and JPEG2000 Lossless decoder wheels and performs availability/self-tests. Its first-choice path uses four cores and all-or-nothing behavior, with optional timed specialist fallback. Its DICOM path can discard a failed slice, retain other usable series after a series header/decode failure, or fail if no series remain; this is not evidence that it masks every failure. The write-up describes about 4,403 usable cases, multiple 2.5D ConvNeXt-T / EfficientNetV2-S attention models at 384 resolution, and vision/report soft-label teachers, unlike this project's frozen ResNet-18, eight selected slices at 256 resolution and mean pooling. The authors state the clean rewrite was bit-identical across 48 training runs and later achieved the same score; these claims were not reproduced here. The fifth-place write-up indicated code/weights were forthcoming, so no comparable implementation was verified. Public methods and reported scores are not directly comparable and do not establish a cause for this project's exception.

## Validation protocol and SR01 results

Synthetic pydicom-generated Explicit VR Little Endian files are written only under test temporary directories. The positive case runs the genuine `TestStudyDataset`, preprocessing, `run_inference`, CSV publication/readback and `validate_submission`; only checkpoint loading and model output are stubbed. It validates two synthetic studies / two series, expected CSV columns/order and 12 finite target probabilities per study in [0,1]. Negative cases check missing-series and unsupported-multiframe failures, safe model-runtime failure, duplicate metadata and output collision without overwriting, failure diagnostics without identifier leakage, and the metadata path where no report column is required. These prove only the asserted synthetic conditions.

Worker evidence:

- Focused `PYTHONPATH=src .venv-training/Scripts/python.exe -m unittest tests.test_submission_failure_reproductions`: 7 passed in 7.855s.
- Scoped `PYTHONPATH=src .venv-training/Scripts/python.exe -m unittest tests.test_fold0_submission tests.test_kaggle_submission_package tests.test_smoke_submission tests.test_dicom_preprocessing tests.test_preprocessing tests.test_resnet18 tests.test_submission_failure_reproductions`: 88 passed, no skips, 15.568s.
- Stub-only 1,300-study / 1,300-series CPU bookkeeping case: 4.921s; Python `tracemalloc` peak 5,541,528 bytes. This excludes real DICOM decode, trained model, GPU, and total-process RSS.

Parent spot-check, a separate run from the worker measurements: 7 focused tests passed in 10.623s; scale case 7.303s and Python `tracemalloc` peak 5,563,491 bytes. These runs vary and are not a performance benchmark. The parent spot-check is distinct from independent SR02 verification.

Worker and independent local pydicom availability probes found JPEG Lossless SV1 unavailable (`plugins=[]`) and JPEG2000 Lossless available via Pillow; the independent probe used pydicom 3.0.2. No compressed bytes were decoded, so codec availability is not codec decode evidence and is not evidence about Kaggle hidden runtime codecs. No trained checkpoint was deserialized and no actual Kaggle hidden execution was repeated. The independent verifier first tried an incorrect decoder-introspection import path, corrected it to `pydicom.pixels.get_decoder`; this had no suite impact.

## Independent SR02 verification

Parent reports the independent verifier observed the exact seven-module scoped command PASS (88 tests, no skips, 19.373s). The genuine positive composition was confirmed with only checkpoint loader/model forward stubbed. The independent scale case took 5.73s with Python `tracemalloc` peak 5,541,712 bytes. This is a separate run from both worker and parent spot-check measurements. The decoder availability check used pydicom 3.0.2; JPEG Lossless/SV1 had no plugin, JPEG2000 Lossless was available via Pillow, and no compressed bytes were decoded. An initial introspection import-path error was corrected to `pydicom.pixels.get_decoder` and did not affect the test suite. Independent `git diff --check` exited 0 with only pre-existing LF/CRLF warnings. These observations complete SR02; SR03 interpretation below incorporates them.

## Interpretation and limits

The positive composition shows that this local synthetic uncompressed path can traverse the production dataset/preprocessing/runner/CSV/validator interfaces when the model and checkpoint loader are stubbed. The negative tests demonstrate selected safe-failure and no-publication conditions. Neither result identifies the historical failure. Public comparator methods suggest decoder/input handling as a hypothesis, while resource/hidden-I/O and other runtime causes remain open. The local codec probe makes compressed decoding an explicit evidence gap, not a proven defect.

The 1,300-row case measures Python bookkeeping with stub preprocessing/model only, not actual image decoding, GPU execution, checkpoint loading, RSS, or hidden Kaggle resource behavior. The public 3-study/15-series success, 500-study training context, 100-study preprocessing audit, 20-study GPU model audit, and local synthetic tests each exercise different contracts; their results should not be combined as proof of full hidden-scale submission readiness.

No root cause has been established. No production fix, remote execution, new submission, or trained-checkpoint deserialization occurred. SR02 independent verification and SR03 final handoff are complete as recorded below.

## Sources

- [Competition leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard)
- [Second-place solution write-up](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/2nd-place-gold-solution-vision-llm-soft-labels-an)
- [Fifth-place solution write-up](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/5th-place-solution)
- [Second-place repository, pinned commit](https://github.com/azamat1ch/rsna-knee-2nd-place/tree/4f168e3894c8ef12ed454b26ae7f9bc495c332a0)
- [Second-place prediction notebook](https://www.kaggle.com/code/azamat1ch/rsna-knee-2nd-place-predict)
- [Existing submission history and run05 evidence](E029_fold0_kaggle_submission.md)
- [Training compatibility and bounded model/runtime evidence](E030_training_compatibility_audit.md)
