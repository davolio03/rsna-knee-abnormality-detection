# E051 — E050 attention-16 submission package preparation

**Status:** corrected `run02` package prepared locally; its exact model/report pair is in private dataset Version 1. Canary Version 1 / execution `365679535` passed for all 3 studies. Full notebook Version 2 / execution `365680900` completed uncapped inference over all 3 visible test studies and wrote the root `submission.csv`; parent selected this exact version and file and clicked Submit once. The competition UI reports “Notebook Running (after deadline)” with no score. Hidden-set completion, full hidden coverage, ranking eligibility, and score remain unverified. The earlier local `run01` remains unchanged and must not be used.

## Objective and boundary

Prepare an offline Kaggle inference package for the completed E050 target-attention-16 epoch-five checkpoint. Preserve the E050 sixteen-slice preprocessing, attention architecture, canonical target order, and E048/E049 eight-slice artifacts. This record covers local package preparation, canary and visible-test inference verification, and the one observed competition submission action; it does not claim hidden-set completion or a score.

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

## Native three-study canary — 2026-11-04

The saved notebook is titled `E051 attention16 canary3studies`; its Version 1 / execution `365679535` [completed successfully](https://www.kaggle.com/code/davidsulleiro/rsna-knee-attention16-epoch05-submission-run01?scriptVersionId=365679535). Native result: 3/3 studies completed, total elapsed 60.4 seconds, inference 29.8794 seconds, `canary=true`, `max_slices=16`, output 224px, microbatch 2, checkpoint/report pins matched, zero skipped series/slices, and `submission_written=false`. Screenshot: `data/private/attention-sixteen-submission/browser-proof/canary-success-365679535.jpg`. Parent reimported the corrected local run02 notebook and read back its Markdown and Code, verifying `RUN_INFERENCE=True`, `MAX_STUDIES=3`, and `SOURCE_ROOT=/kaggle/input/datasets/davidsulleiro/rsna-knee-attention16-epoch05`. The official competition, private model dataset Version 1, and offline codec dataset were attached and read back; Internet is OFF, GPU is T4x2, and 29 hours remained in the native quota at launch.

A historical progress snapshot at 39.8 seconds showed startup at 10 seconds, codecs ready at 13 seconds, decoder availability at 20.5 seconds, and preprocessing study 1/3 at 29.2 seconds. The later terminal result above supersedes that snapshot.

## Native full inference and competition status — 2026-11-04

Full notebook Version 2 / execution `365680900` completed successfully in 62.0 seconds total, including 31.362422858 seconds inference. It ran uncapped (`MAX_STUDIES=None`, `canary=false`), processed all 3/3 visible test studies with 16-slice inference, skipped zero series and slices, and wrote the root `submission.csv`. This confirms the visible-test inference path only; it is not evidence of full hidden-test coverage.

Parent selected that exact full Version 2 and its root CSV in the native dialog, then clicked Submit once. The official [competition Submissions page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/submissions) shows “RSNA Knee Attention16 Epoch05 Submission Run01 - E051 attention16 full inference epoch05” as “Notebook Running (after deadline)” with no score. Hidden-set completion, hidden coverage, ranking eligibility, and score are not established. Do not resubmit while this native run remains pending.

### Earlier editor-loading observation

Earlier on 2026-11-04, the Chrome 3 editor was stuck Loading with Import Notebook, Run All, and Start Session disabled; a safe reload did not resolve it, and the old accepted notebook editor showed the same state. Competition attachment was not confirmed after reload, a draft-mode Script request had unknown effect, and a reCAPTCHA connection message was visible. The cause was not established; no CAPTCHA attempt or workaround was made. The later Chrome 4 editor recovered and parent successfully reimported/read back run02 and verified inputs. Preserve the earlier screenshot at `data/private/attention-sixteen-submission/browser-proof/editor-loading-20261104.jpg` as historical evidence; do not treat the earlier state as current or infer its root cause.

The earlier `run01` notebooks retain the old `rsna-knee-attention-sixteen-epoch05` mount and a stale, nonfunctional E048 execution/ResNet-18 template comment; preserve them unchanged and do not upload them. Fresh `run02` threads the actual source-output root into both notebook pins and inventory, and replaces the stale comment. Its model export, sanitized report and source ZIP are byte-identical to run01; only package metadata/notebooks and the output identity differ.

## Verification

- Strict-TDD RED for this correction: `_notebook_code` rejected a `source_root` override and CLI `--help` did not expose `--source-output-root`.
- GREEN: `PYTHONPATH=src data/private/local-cuda-env/Scripts/python.exe -m unittest tests.test_attention_sixteen_submission` — **9 tests passed**.
- E049 compatibility regressions: `PYTHONPATH=src data/private/local-cuda-env/Scripts/python.exe -m unittest tests.test_radimagenet_attention_submission_inference tests.test_radimagenet_attention_submission_package` — **11 tests passed**.
- Synthetic DICOM preprocessing regressions: `PYTHONPATH=src data/private/local-cuda-env/Scripts/python.exe -m unittest tests.test_dicom_preprocessing` — **41 tests passed**, including explicit sixteen-slice selection. This is not an end-to-end native Kaggle DICOM inference.
- The corrected builder command was `PYTHONPATH=src data/private/local-cuda-env/Scripts/python.exe scripts/build_kaggle_attention_sixteen_submission_package.py --output data/private/attention-sixteen-submission/run02 --source-output-root /kaggle/input/datasets/davidsulleiro/rsna-knee-attention16-epoch05`. Inventory and both notebook assignments contain that exact mount root. Both notebooks compile and contain no stale template execution/ResNet-18 comment. The unchanged source ZIP imported all 18 modules in isolation; model/report/source bytes match run01. The model-only export strictly reloads as `RadImageNetAttentionStudyModel` with 324 state entries. Parent additionally verified the source/report pins, accepted-source closure, strict sixteen-slice forward, 31 related tests, 49 DICOM tests, and a nine-test root/package spot check in 6.516 seconds.
- Local checks did not establish a GPU benchmark or native DICOM outcome; the later native canary and full visible-test results are recorded above. No hidden-set outcome or leaderboard score is claimed. The E049 private score `0.848` belongs to its separate eight-slice candidate and is not a control for E050/E051.

## Next step

Await the terminal state on the already-submitted competition run; do not resubmit. Record any later native result without inferring score, ranking eligibility, or hidden coverage beyond the evidence shown.
