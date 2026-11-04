# E051 — E050 attention-16 submission package preparation

**Status:** corrected `run02` package prepared locally and the exact model/report pair uploaded to a private Kaggle dataset Version 1. A matching notebook draft exists, but its imported source and attached inputs are not verified because the editor is persistently loading with execution controls disabled. No notebook version, canary, full inference, or submission exists. The earlier local `run01` remains unchanged and must not be used with the newly created dataset slug.

## Objective and boundary

Prepare an offline Kaggle inference package for the completed E050 target-attention-16 epoch-five checkpoint. Preserve the E050 sixteen-slice preprocessing, attention architecture, canonical target order, and E048/E049 eight-slice artifacts. This record covers local package preparation and structural/model verification only; it does not claim a Kaggle result or competition score.

## Pinned inputs and package

| Artifact | Evidence |
|---|---|
| E050 source checkpoint | `data/private/attention-sixteen-slices/local16-attention-run01/target_attention/checkpoint-epoch-05.pt`; 214,808,723 bytes; SHA-256 `631536a30d0da51c0f6872fb45fac15b5bcb0f6185b5a53708eed1ef2be42f21` |
| E050 aggregate report | `data/private/attention-sixteen-slices/local16-attention-run01/training-report.json`; SHA-256 `8c9c83a8d7947f082f4cdb6a0f07f1ed3a956c7b451952c1ceaacf0764e51e38` |
| Model-only export | `data/private/attention-sixteen-submission/run02/attention-model-epoch-05.pt`; 94,646,113 bytes; SHA-256 `95de045ad077691a099bf7426605bfa433eb1a0abcfcd0d321250f43611f67b5` |
| Sanitized report | `data/private/attention-sixteen-submission/run02/training-report.json`; SHA-256 `14c7c5ceb92fe0c2debfceb699eeba006074f62afcd09042979c331fcdcd78f8` |
| Offline inference source | `data/private/attention-sixteen-submission/run02/attention-inference-source.zip`; 24 code-only members; SHA-256 `f42334e1281f3f2ac1a5e21f8989f577555882d075a6d689284010d6404189bb` |
| Native configuration | 224px, up to 16 slices, microbatch 2, RadImageNet normalization, target-specific slice then series attention, canonical 12-target order |

The source checkpoint was hash/size/proof checked before export. The uploaded model contains model tensors and fixed provenance only: optimizer/RNG state, training caches, raw DICOMs, study identifiers, and local output paths are excluded. The report is sanitized aggregate provenance, not row-level validation data. Upload only the model export and sanitized report to the private model dataset; the notebooks embed the code-only source ZIP.

## Prepared notebooks and runtime inputs

- Canary: `data/private/attention-sixteen-submission/run02/kaggle_attention16_canary.ipynb` (maximum three studies; SHA-256 `94c3d50ea6f2641d36b7cbb29c5d56e178c488765bac3b396261de94fac3d7b0`).
- Full inference: `data/private/attention-sixteen-submission/run02/kaggle_attention16_submission.ipynb` (uncapped; use only after canary review; SHA-256 `bff2dc7712f35658d748643d861f25b54a6bb909be3d300b1788ba3119ea8532`).
- Private model dataset mount expected by both run02 notebooks: `/kaggle/input/datasets/davidsulleiro/rsna-knee-attention16-epoch05`.
- Also attach the official RSNA Knee competition input and the pinned offline DICOM codec dataset. Keep Internet off and GPU enabled; P100 if available.

The canary does not publish a competition-root submission file. The full notebook publishes only after every visible test study is processed. The private dataset [rsna-knee-attention16-epoch05](https://www.kaggle.com/datasets/davidsulleiro/rsna-knee-attention16-epoch05) was created as Version 1 with exactly `attention-model-epoch-05.pt` and `training-report.json` (94.65 MB total).

## Native notebook handoff status — 2026-11-04

A Kaggle notebook draft titled `rsna-knee-attention16-epoch05-submission-run01` exists at [the notebook editor](https://www.kaggle.com/code/davidsulleiro/rsna-knee-attention16-epoch05-submission-run01/edit). The editor reports zero saved versions, and the run02 source contents and input persistence have not been independently read back. A prior-day import of the local run01 notebook has uncertain source persistence and is not evidence that the corrected run02 notebook is saved.

On 2026-11-04, the Chrome 3 editor remained in a persistent Loading state. Import Notebook, Run All, and Start Session were disabled; a safe reload did not resolve the state. An editor for an already accepted notebook showed the same Loading state. An attempt to attach the official competition briefly showed “Remove Competition,” but after reload the sidebar showed no inputs, so competition attachment is unconfirmed. A draft-mode Script request may have had no effect; its outcome is unknown. The editor displayed a reCAPTCHA connection message: “No se ha podido establecer conexión con el servicio reCAPTCHA. Comprueba tu conexión a Internet y vuelve a cargar la página para ver otro reCAPTCHA.” Its cause was not determined, and no CAPTCHA attempt or security workaround was made.

The screenshot evidence is `data/private/attention-sixteen-submission/browser-proof/editor-loading-20261104.jpg`. Treat this as a UI/access-state blocker only; it is not evidence of a model, package, or inference-code failure. No canary, full execution, competition submission, or score is claimed.

The earlier `run01` notebooks retain the old `rsna-knee-attention-sixteen-epoch05` mount and a stale, nonfunctional E048 execution/ResNet-18 template comment; preserve them unchanged and do not upload them. Fresh `run02` threads the actual source-output root into both notebook pins and inventory, and replaces the stale comment. Its model export, sanitized report and source ZIP are byte-identical to run01; only package metadata/notebooks and the output identity differ.

## Verification

- Strict-TDD RED for this correction: `_notebook_code` rejected a `source_root` override and CLI `--help` did not expose `--source-output-root`.
- GREEN: `PYTHONPATH=src data/private/local-cuda-env/Scripts/python.exe -m unittest tests.test_attention_sixteen_submission` — **9 tests passed**.
- E049 compatibility regressions: `PYTHONPATH=src data/private/local-cuda-env/Scripts/python.exe -m unittest tests.test_radimagenet_attention_submission_inference tests.test_radimagenet_attention_submission_package` — **11 tests passed**.
- Synthetic DICOM preprocessing regressions: `PYTHONPATH=src data/private/local-cuda-env/Scripts/python.exe -m unittest tests.test_dicom_preprocessing` — **41 tests passed**, including explicit sixteen-slice selection. This is not an end-to-end native Kaggle DICOM inference.
- The corrected builder command was `PYTHONPATH=src data/private/local-cuda-env/Scripts/python.exe scripts/build_kaggle_attention_sixteen_submission_package.py --output data/private/attention-sixteen-submission/run02 --source-output-root /kaggle/input/datasets/davidsulleiro/rsna-knee-attention16-epoch05`. Inventory and both notebook assignments contain that exact mount root. Both notebooks compile and contain no stale template execution/ResNet-18 comment. The unchanged source ZIP imported all 18 modules in isolation; model/report/source bytes match run01. The model-only export strictly reloads as `RadImageNetAttentionStudyModel` with 324 state entries. Parent additionally verified the source/report pins, accepted-source closure, strict sixteen-slice forward, 31 related tests, 49 DICOM tests, and an eight-test spot check.
- No local GPU benchmark, raw-test DICOM inference, Kaggle runtime, hidden-set outcome, or leaderboard score is claimed. The E049 private score `0.848` belongs to its separate eight-slice candidate and is not a control for E050/E051.

## Next step

After the Kaggle editor is accessible again, read back the saved run02 notebook source and attached inputs before the first run. The next planned step is the three-row canary; inspect native logs/output before deciding whether to continue. This record does not claim that a canary/full run or submission occurred.
