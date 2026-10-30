# E026 — First full-fold Kaggle scratch baseline (V1 and diagnostic run02 failed)

- **Date:** 2026-10-17
- **Status:** Version 1 failed with unknown cause and no artifacts. Separately authorized diagnostic run02 failed after 400.7 seconds with 0 B output; its safe event still localizes only to training. Both single-run authorizations are consumed; do not restart or launch another run.
- **Parent:** E025
- **Hypothesis:** The unchanged scratch study-level CNN can complete one full
  deterministic development fold on the existing Kaggle GPU runtime.
- **Changed variable:** Scope increases from the separate eight-study E025
  optimization smoke test to the predeclared full fold; no model, preprocessing,
  data, or split changes.
- **Frozen protocol:** fold 0, seed 42, scratch initialization, 333 training and
  167 validation studies, five epochs, 1,665 planned steps, CUDA only, 256-pixel
  inputs, at most eight slices per series, feature width 64, learning rate
  0.001. `max_steps` and `overfit_limit` are null. No Gold-58, fold search,
  checkpoint resume, or repeated run.
- **Inputs:** Existing private Version 3 package with pinned manifest SHA-256
  `e88c61c22a463737f16951b7c76db03834034aecd1b31dc61b4ce495080b2be2`, folds
  SHA-256 `b82b376b0086201e3364b11a0203b62890e628442075a52ea4a9e09ee0e778ac`,
  and the existing competition `train_series` mount. Notebook verifies the
  embedded inventory and all 24 unpacked member hashes before source import;
  remote ZIP archive bytes are not independently verified.
- **Artifacts:** `notebooks/kaggle_first_fold_baseline.ipynb`,
  `configs/kaggle_first_fold_baseline.json`, and ignored private output directory
  `/kaggle/working/kaggle-first-fold-baseline/run01`. The notebook defaults
  training off and validates the complete plan before any fit. The runner saves
  a final checkpoint only; it does not provide resume or per-epoch checkpoints.
- **Validation:** Full fixed fold; report all 12 per-target ROC-AUCs, macro AUC,
  and missing-class targets. This is provisional silver-label agreement, not
  clinical truth or final generalization evidence.
- **Launch observation:** Parent created the dedicated private notebook
  “RSNA Knee First Fold Baseline” and verified owner-only visibility, existing
  private Version 3 and competition-image inputs, and T4x2 selection. The enabled
  code cell was read back in full from the clipboard and matched SHA-256
  `a8535e7a7752b38509cd7106334cf2346a7658a909e37a09e2344da40b91b3f1`. Save
  Version -> Save & Run All was clicked once. Private notebook version 1,
  `E026 Fold 0 - 333 train 167 val - 5 epochs`, was accepted and observed running
  at 2026-10-17 01:46 Europe/Madrid (approximately 2026-10-16 23:46:48 UTC).
  Version 1's execution identifier is `359170889`; the private status viewer is
  [Version 1 status](https://www.kaggle.com/code/davidsulleiro/rsna-knee-first-fold-baseline?scriptVersionId=359170889).
  A later read-only observation confirmed Version 1 failed after 258.3 seconds
  with 0 B output. At 247.3 seconds, logs emitted `training_blocked` and
  `training_failed`, followed by Papermill `RuntimeError(training_failed)` at
  notebook main line 232. The draft editor session is off.
- **Results:** No metrics, report, predictions, or checkpoint were produced.
  The actual exception, failure stage, and completed step count are unknown;
  the prior wrapper suppressed the cause with `from None`. Do not infer GPU
  out-of-memory or any specific failure step. The Version 1 authorization was
  consumed. A later, separate user authorization permitted exactly one
  diagnostic relaunch with a fresh package; that authorization is now also
  consumed. Do not retry or restart either execution.
- **Diagnostic follow-up:** Locally, the training runner now retains only an
  allowlisted exception family and bounded exception-family chain, coarse
  execution stage, and validated internal source location for unexpected exceptions. The notebook and local CLI emit
  those fields only after strict schema/allowlist checks; raw messages, paths,
  IDs, reports, and locals remain excluded. Synthetic regression tests verify
  the redaction and fail-closed behavior. This code was not present in private
  Kaggle package Version 3 or the failed Version 1, and cannot retroactively
  identify that run's cause. The separately authorized diagnostic run02 below
  used a matching provenance-preserving package version but also failed without
  preserving the safe categorical cause from dataset loading. The local fix below
  is not retroactive and was not present in either private run.
  New local notebook cell SHA-256: `837f39df3b4f7bb9d4085dd7d36c1cfa2193f005ca61e239826154aea920171a`
  (18,249 bytes); the historical launched-cell hash above remains unchanged.
- **Launch evidence:** Screenshot
  `data/private/kaggle-first-fold-background/batch-running-v1.jpg` (110,768
  bytes; SHA-256
  `a4b1647f7205d394bc0d28a39672d621b733b16b5c05f676c2e4d68c472bc281`).
- **Preparation checks:** Strict TDD observed five expected missing-artifact
  test errors before notebook/config creation; six synthetic notebook guard
  tests then passed. A later regression RED showed a failed preflight emitted a
  diagnostic but returned normally; the notebook now re-raises only allowlisted
  safe reason codes, and validates the complete report before success output.
  Ten synthetic notebook guard tests now pass, including exact parity of the
  ignored launch cell with the checked-in notebook. The focused
  runner/component/fold/validation/preprocessing, package, and notebook suite
  passed (88 tests). Independent parent verification passed 88 focused tests and
  the legacy full suite (297 tests, seven optional skips). The private Version 1
  subsequently failed as documented above. The CSV registry contains older
  rows with inconsistent column widths; the new
  E026 row matches the 28-column header and was checked independently.
- **Diagnostic verification:** Strict TDD observed a RED because the runner
  had no `diagnostics` metadata; the targeted regression then passed. The
  focused suite passed 92 tests. The legacy suite passed 298/299 tests with
  eight optional skips; its sole failure was the existing Windows child-process
  timeout cleanup test (`test_windows_timeout_kills_synthetic_child_process`),
  even after using a writable workspace temp directory. Notebook JSON, syntax,
  empty outputs, config parity, exact local launch-cell parity, and
  `git diff --check` passed.
- **Conclusion:** Both full-fold runs failed without a reliable model-performance
  result. Version 1's underlying cause remains unknown. Run02 establishes only a
  value-family failure at the training call site; the exact cause remains
  unknown, and the local correction does not retrospectively fix either run.

## Authorized diagnostic relaunch (run02) — failed; cause detail incomplete

- **Authorization and scope:** The user explicitly authorized one new private
  diagnostic run using the existing Kaggle account/session, private dataset and
  competition attachment. No model, fold, cohort, or training-setting changes;
  no Gold-58 use. This second one-run authorization is consumed. No retry or
  restart is authorized.
- **Package:** The exact locally audited ZIP was uploaded once as private
  dataset Version 4. The success dialog confirmed completion and 22 private
  files in the `data/` and `src/` layout. The competition dataset remained
  attached. ZIP SHA-256:
  `1e454db9f9aba9d574cb9e37a7323166e3b5ab8be89dcd80c47e986014950cff`;
  inventory SHA-256:
  `7b1b1bd51357f7b6a7e4ad7b2c88edaf11fde3a768b5c4425f9cc592a278da73`.
  Manifest and fold hashes remain unchanged from the values above.
- **Notebook and source:** A new private notebook, “RSNA Knee First Fold
  Diagnostic Run02,” was created after the old editor remained stuck loading.
  Its explicit package input is Version 4; visibility is private. The full
  edited code cell was read back from the clipboard (17,884 characters), SHA-256
  `b3d9428c2f380e92e1e99f0bff76ef30e7ea93bb398b4d68972c845e2cc93e06`. It
  differs from the prepared notebook only by `RUN_TRAINING = False` to
  `RUN_TRAINING = True`. The output remains the distinct `run02` directory.
- **Launch:** One Save Version -> Save & Run All action was accepted. Version
  name: `E026 Run02 - fold0 - safe diagnostics`; notebook Version 1, execution
  ID `359359350`; status URL:
  [private run02 status](https://www.kaggle.com/code/davidsulleiro/rsna-knee-first-fold-diagnostic-run02?scriptVersionId=359359350).
  Launch was observed at approximately 14:37 Europe/Madrid on 2026-10-17; the
  exact launch time was not captured. At 14:38:09 Europe/Madrid, the selected
  viewer showed “Running for 32.6s,” T4x2, Latest Container Image, and 0 B
  output. No completion, failure, diagnostic event, or training result was
  observed. The interactive session was off while the versioned execution
  remained running.
- **Evidence:** Ignored local provenance record:
  `data/private/kaggle-first-fold-background/run02-launch-provenance.json`.
  Proof files: `data/private/kaggle-first-fold-background/run02-preparation-v2/dataset-v4-proof.jpg`
  (SHA-256 `d86b5b239fca821ea02495626f91523ccb0ec3088c5c237f05af2034443df97d`),
  `data/private/kaggle-first-fold-background/run02-preparation-v2/executed-cell.py`
  (SHA-256 `b3d9428c2f380e92e1e99f0bff76ef30e7ea93bb398b4d68972c845e2cc93e06`),
  and `data/private/kaggle-first-fold-background/run02-preparation-v2/run02-running-proof.jpg`
  (SHA-256 `ca77fdbe115d19797fb727777ec198b61dd30c93c2bfff422c85a38e63f3a082`).
- **Interpretation at launch:** The initial launch snapshot was not a successful
  training result. The terminal evidence is recorded below; do not infer the
  exact cause from the snapshot or later coarse diagnostics.

## Terminal run02 result and local follow-up

- **Observed terminal result:** Parent-observed execution `359359350` failed
  after 400.7 seconds with 0 B output. At 390.4 seconds, its safe event reported
  `stage=training`, `exception_family=value`, `exception_chain=[value]`, and
  source `training.baseline.run_training:477`. The exact deployed source line
  evaluates `_sample_loss(model, train_dataset[row_index], loss_function,
  device)`. The trusted frame collector allows `_sample_loss`, but reported the
  caller line instead. This points to failure while evaluating the call
  arguments, most plausibly `train_dataset[row_index]` before `_sample_loss`
  entry; it does not prove a specific decoder/preprocessor failure or exact
  `ValueError` cause. Do not infer a numeric-loss or OOM cause.
- **Evidence limits:** The event's family and call-site are observed, not the
  wrapped exception text or exact root cause. No metrics, report, predictions,
  or checkpoint were produced. The run02 authorization is consumed; no relaunch.
- **Local correction:** The dataset now carries only a `DicomPreprocessingError`
  with an exact allowlisted type and known categorical code through a safe
  `TrainingDatasetError`; the runner revalidates and reports that code through
  the shared diagnostic validator used by the CLI and notebook. Unknown codes,
  other exception classes, arbitrary fields, messages, paths, and identifiers
  remain excluded. This is a local code/test correction only; no package rebuild,
  remote upload, or training run occurred. It does not identify the historical
  run02 cause.
- **Verification:** Strict TDD observed RED before production changes (focused
  regression suite: 3 tests, 1 failure and 1 error due to missing propagated
  category/schema support). After the fix the focused suite passed 98 tests.
  Independent parent verification also passed the 98-test focused suite and
  separately passed all 3 dataset-diagnostic tests; no code or privacy blockers
  were found.
  `git diff --check`, notebook JSON/syntax/output/default checks, and synthetic
  end-to-end dataset → runner → CLI/notebook diagnostic checks passed. The
  ignored prior launch-cell snapshot was preserved unchanged; no historical
  package or launch artifact was rewritten.

## Run03 controlled retry — terminal failure recorded

- **Authorization and scope:** The user authorized canceling the one queued run03 execution and one identical controlled retry; both actions are complete and that training authorization is consumed. No further retry or restart is authorized. The frozen run03 training protocol and Version 5 package were unchanged.
- **Historical first execution:** Execution `359370509` was canceled while queued with 0 B output. The cancellation proof remains `data/private/kaggle-first-fold-background/run03-preparation/run03-canceled-proof.jpg` (SHA-256 `7e57df480af1bbf7018f92e50665770b55cbbf85e6b86d7f17592101100774a8`). Its earlier queued snapshot remains a valid historical observation.
- **Controlled retry:** Parent observed the one retry, execution `359386220`, fail after 289.4 seconds with 0 B output. At 279.1 seconds the safe event reported `data_reason_code=inconsistent_orientation`, family `training`, source `run_training:492`. This gives the allowlisted coarse category only; no measured deltas, failing series identity, metrics, or checkpoint are established. No further training retry.

## Authorized CPU orientation-header audit — completed

- **Scope:** One private CPU-only header audit was authorized and completed. It reads only `ImageOrientationPatient` headers from fold-0 training studies; it does not train, inspect labels/reports, decode pixels, use a GPU, or change the fixed `1e-5` comparison threshold. No new upload or package change was made.
- **Notebook and inputs:** Parent imported the corrected private notebook “RSNA Knee Orientation Header Audit,” Version 1, execution `359458891`: [private audit status](https://www.kaggle.com/code/davidsulleiro/rsna-knee-orientation-header-audit?scriptVersionId=359458891). The private reader dataset was explicitly pinned to Version 5 and the official competition input remained attached. Accelerator was `None` (CPU). The prepared notebook remains disabled by default; its parent-enabled full cell (the single `RUN_AUDIT = False` → `True` change) was read back as 33,845 characters, SHA-256 `c920522b9a6b4f5e6e65ea67ada6d1f52f811a01302bc1e5b2901326b264154d`. One Save & Run All was accepted.
- **Observed result:** The ID-free completed summary reported `complete` / nonpartial; 12,319 headers read in 115.273 seconds; first threshold-exceeding series ordinal 355 was fully read (43 headers); 23 headers exceeded the unchanged `1e-5` component-delta threshold; zero invalid orientations and zero header-read failures. Maximum component delta/spread was `0.000010000000000010001`. Maximum row, column, and normal angular deltas from the reference were `0`, `0.0005712049259344931`, and `0.0005593304958631258` degrees. This is a bounded training-header measurement through the first rejected series, not a cohort-wide distribution or clinical validation. No IDs or paths from the DICOMs were retained.
- **Initial viewer observation:** Before completion, the selected viewer showed `Running` for 12.1 seconds, `Accelerator None`, and 0 B output. This historical snapshot did not itself establish a measurement. The single audit authorization is consumed; no retry or restart.
- **Artifacts and verification:** Prepared disabled notebook SHA-256 `11035efcdc23699dd4be77b2e54a67ca3c86909a1b3099eacca80f7d66c02a7c`; paste-ready cell SHA-256 `16be5d49ba7057cc9430ba1a774d0eb0d769c549877d126d71113adbd8d6ba8d`; embedded tested helper SHA-256 `f3e2d78472c3b0efc24babc85c1f00cfc49496999aaea73addaf1c7d25d1c6dd`. Correction commit: `1a40c51` (`fix(data): fail closed in orientation audit`). The writer's focused suite passed 121 tests; independent verification passed 56 checks and parent verification passed 23 checks. The initial viewer proof is `data/private/kaggle-first-fold-background/orientation-audit-run01/kaggle-running.png` (SHA-256 `F247F6FA7B2E65ED6E1E9E4E60B26A35DF27496810E71542C88F79DF39F7F5D`); no DICOM data was opened locally.

## Local orientation-comparison boundary correction

The user authorized a local correction after the audit showed a delta represented just above the fixed `1e-5` comparison boundary. The preprocessing path now allows at most two ULPs at the operand scale for orientation-component comparisons, solely to absorb binary-float representation/subtraction rounding. The scientific `1e-5` threshold is unchanged; spacing and projected-position comparisons are unchanged. Synthetic regression tests reproduce the boundary, reject a materially larger delta and an actual rotation, and confirm spacing/position remain strict. No package rebuild, data access, GPU work, or remote rerun was performed at that time. The focused runner below passed 125 tests; `git diff --check` passed.

## Authorized CPU preprocessing canary — local artifacts prepared

- **Scope and status:** The user separately authorized one private CPU preprocessing canary before any training. Local preparation and parent verification are complete; the parent uploaded and launched the V4 candidate once, and execution `359467569` completed successfully. This was preprocessing of one bounded series, not training. No real DICOM pixels were read locally. V2 and V3 preparations were preserved, never uploaded, and superseded after independent verification; V4 is the launched candidate.
- **Bounded target:** The notebook derives training-series ordinal 355 from the pinned manifest order and fold-0 assignments, then requires exactly 43 direct DICOM files. It verifies every inventoried package member plus the embedded inventory, manifest, and fold hashes before selecting the target or importing the packaged preprocessor. It calls only `preprocess_dicom_series` with the existing `output_size=256` and `max_slices=8` settings on CPU. It emits only an ID-free summary with shape, finite/range status, counts, hashes, and elapsed time; it saves no images or metadata. A failed preflight or preprocessing path prints one allowlisted reason and raises rather than returning success. No labels, reports, validation studies, Gold-58, model, cohort, fold, filter, or threshold changes are involved.
- **Package provenance:** Fresh V6 ZIP `data/private/kaggle-first-fold-background/orientation-preprocessing-canary-run01-v4/kaggle-first-fold-diagnostic.zip`, SHA-256 `226ec46ab3ec944d67e2537b4ba35598eae90be826de875de9710baa97183d82`; inventory SHA-256 `06fb3d5b59b5e8dd5f0ff3c0de57ed0307233be9753a510c190a07860a2a4586`; 22 unique members. Compared with historical V5, the only changed member is `src/rsna_knee/data/dicom_preprocessing.py`; the pinned manifest/fold hashes remain `e88c61c22a463737f16951b7c76db03834034aecd1b31dc61b4ce495080b2be2` / `b82b376b0086201e3364b11a0203b62890e628442075a52ea4a9e09ee0e778ac`. All archive members matched inventory and extracted files; no disallowed data/checkpoint/bytecode members were present.
- **Canary artifacts:** Corrected disabled-default notebook `data/private/kaggle-first-fold-background/orientation-preprocessing-canary-run01-v4/preprocessing-canary-run01.ipynb`, SHA-256 `0569630e45be74ffe80f1d223f5489ec4248181dfc07e5b3eca79d1b916fcea8`; config SHA-256 `eed2bd57e30711b2bf6fef9d3071c383ac22f06d5eb6867c336ec6542de19fa1`; default cell canonical (LF) SHA-256 `54e17fa50ab33e16c80a3f99a1aa1994e3302c0f42fe603c6b9908208e1cfde4`; sole `RUN_CANARY = False` → `True` enabled-cell canonical (LF) SHA-256 `3c063afa6b59a63b09b437c07c6c9561e2c337c4af232688512ae18a6704bb3a`. Fresh output is `/kaggle/working/orientation-preprocessing-canary/run01`; expected private dataset mount remains `/kaggle/input/datasets/davidsulleiro/rsna-knee-five-study-reader-private` and official competition input remains the existing training-series mount. V2/V3 folders are preserved locally but were never uploaded or launched.
- **Verification and corrections:** Strict-TDD RED reproduced the initial builder gap, config hash-domain mismatch, missing/unlisted mounted files, and lost `inconsistent_orientation` reason. The latest focused preprocessing/audit/package/canary/notebook suite ran 100 tests: 99 passed and one directory-symlink test was skipped because this sandbox cannot create directory symlinks. The enabled notebook executes the actual packaged preprocessor on synthetic DICOMs; success emits aggregate-only JSON and an above-threshold synthetic case emits the exact safe `inconsistent_orientation` reason. V4 now hashes the embedded base64 config in the same byte domain as its verifier and rejects symlink roots/entries and any non-inventory mounted file before imports. ZIP/tree/inventory hashes, JSON and Python syntax, default-off status, exact one-line enable flip, V5 member delta, and prohibited archive-member scan were read back. During local verification no real data, GPU, or remote execution was used. Do not retry or run training.
- **Parent-verified upload and launch:** Parent independently passed 52 package/notebook checks with one symlink-capability skip and separately re-ran the 12-test canary suite with one skip. Parent created private reader dataset Version 6 from the V4 ZIP once (22 files; notes identify the orientation roundoff boundary fix). A new private notebook, “RSNA Knee Preprocessing Canary Run01,” explicitly pinned Version 6 and the official competition input, with visibility private and Accelerator `None` (CPU). The enabled cell was read back in full (44,992 characters), SHA-256 `3c063afa6b59a63b09b437c07c6c9561e2c337c4af232688512ae18a6704bb3a`; it is the prepared cell with only `RUN_CANARY = False` changed to `True`. One Save & Run All was accepted: notebook Version 1, execution `359467569`, version name `CPU canary run01 - orientation roundoff fix`, [private canary status](https://www.kaggle.com/code/davidsulleiro/rsna-knee-preprocessing-canary-run01?scriptVersionId=359467569). Initial viewer snapshot showed `Running` for 11.8 seconds, CPU, and 0 B output. No completion, failure, canary summary, or training was observed. The one canary authorization is consumed; no restart or retry.
- **Terminal result:** Parent verified a successful 23.8-second run and one aggregate JSON output at `/kaggle/working/orientation-preprocessing-canary/run01/preprocessing-summary.json`. The JSON reports `status=completed`, `elapsed_seconds=2.651`, `finite_values=true`, `fold=0`, `training_study_count=333`, `validation_study_count=167`, `series_ordinal=355`, `input_file_count=43`, `max_slices=8`, `output_size=256`, `image_shape=[8,1,256,256]`, and normalized range `[0.0,1.0]`. Manifest SHA-256 `e88c61c22a463737f16951b7c76db03834034aecd1b31dc61b4ce495080b2be2`, folds SHA-256 `b82b376b0086201e3364b11a0203b62890e628442075a52ea4a9e09ee0e778ac`, package inventory SHA-256 `06fb3d5b59b5e8dd5f0ff3c0de57ed0307233be9753a510c190a07860a2a4586`, and package ZIP SHA-256 `226ec46ab3ec944d67e2537b4ba35598eae90be826de875de9710baa97183d82` match the pinned candidate. This demonstrates successful preprocessing for this one selected series only; it is not evidence that the entire cohort preprocesses, and produced no training, checkpoint, AUC, or model metrics. The initial `Running`/0 B snapshot above remains a valid earlier observation. The single canary authorization is consumed; no rerun, retry, or training is authorized.
- **Evidence file:** Parent's success screenshot is expected at `data/private/kaggle-first-fold-background/orientation-preprocessing-canary-run01-v4/kaggle-canary-success.png`; it was not present in the local workspace at the time of this record, so no screenshot hash is claimed.


## Run04 preparation — one new private full-fold training authorization

- **Authorization and scope:** The user explicitly authorized exactly one new private Kaggle GPU training run (`run04`) using the existing reader dataset Version 6 and official competition input. This worker performed local artifact preparation only. At the time of this preparation entry, no upload, launch, training, or output was claimed; the later parent-verified launch is recorded below. No automatic retry/restart. Gold-58 remains excluded; preserve fold 0 / seed 42, 333/167 studies, scratch model, five epochs / 1,665 planned steps, CUDA, 256 pixels, max eight slices, width 64, learning rate 0.001, and all existing data/model/fold/protocol boundaries.
- **Provenance-preserving preparation:** Reused the trusted run03 notebook/config templates, transplanted the exact V6 inventory and ZIP provenance pins, and changed only the output path to `/kaggle/working/kaggle-first-fold-baseline/run04`. The V6 archive was reused byte-for-byte; no ZIP build or upload occurred. ZIP SHA-256 `226ec46ab3ec944d67e2537b4ba35598eae90be826de875de9710baa97183d82`; inventory SHA-256 `06fb3d5b59b5e8dd5f0ff3c0de57ed0307233be9753a510c190a07860a2a4586`; 22 members. Compared with V5, the only changed member is `src/rsna_knee/data/dicom_preprocessing.py`. Manifest/fold hashes remain `e88c61c22a463737f16951b7c76db03834034aecd1b31dc61b4ce495080b2be2` / `b82b376b0086201e3364b11a0203b62890e628442075a52ea4a9e09ee0e778ac`.
- **Artifacts:** Ignored folder `data/private/kaggle-first-fold-background/run04-preparation/`. Notebook `kaggle_first_fold_baseline_run04.ipynb`, SHA-256 `31cda55cf9d9cf912dc8db371cd6c6dab6b187090b62d652f453a92baf6416ca`; config `kaggle_first_fold_baseline_run04.json`, SHA-256 `c506749e6f555af4bf7f7f2ea48baa657e9455658463600abc5168622e6c56f2`; disabled cell `run04-disabled-cell.py`, 16,297 chars, SHA-256 `0b4b23f2749a47687b11ecdd46b5fc1afdf3eb0afcd5c515cff167f2e778d7e0`; one-flag enabled cell `run04-enabled-cell.py`, 16,296 chars, SHA-256 `8f4a5558d47fc0b291a729ce68611984ae9367ea7fc73c6558daa39e47952807`. The notebook is default-off, output-free, and exact code diff from run03 is limited to output path and V6 inventory/archive pins. Provenance JSON records the same details. Historical run01–run03 and V1–V6 assets remain unchanged.
- **Verification and boundary:** Read back the V6 ZIP/inventory and verified all 22 members; embedded inventory bytes, manifest/fold pins, expected member count, run config and all frozen parameters match. Notebook JSON and disabled/enabled Python cells parse; one exact `RUN_TRAINING = False` → `True` toggle was verified. The focused command `PYTHONPATH=src .venv-training\Scripts\python.exe -m unittest tests.test_training_runner tests.test_training_components tests.test_training_folds tests.test_validation_multilabel tests.test_preprocessing tests.test_dicom_preprocessing tests.test_kaggle_training_package tests.test_kaggle_first_fold_notebook tests.test_training_dataset_diagnostics` passed 102 tests. `git diff --check` passed before the tracked documentation update. No DICOMs, reports, pixels, GPU, network, or remote service were used.
- **Preparation status:** Prepared locally and independently verified by the parent before its single launch, recorded below. No training result, metric, checkpoint, or full-cohort readiness was established at preparation time.


## Run04 launch — terminal outcome pending

- **Parent-verified launch:** One private notebook, “RSNA Knee First Fold Training Run04,” Version 1, execution `359470765`, was launched from the prepared run04 artifact: [private run04 status](https://www.kaggle.com/code/davidsulleiro/rsna-knee-first-fold-training-run04?scriptVersionId=359470765). Reader dataset Version 6 and the official competition input were explicitly pinned; notebook visibility was private, T4x2 selected, and interactive session off. Parent accepted exactly one Save Version → Save & Run All action, version name `E026 Run04 - fold0 - V6 roundoff fix`. No new dataset upload occurred.
- **Enabled source readback:** The full enabled cell (16,296 characters) was read back and matched the prepared `run04-enabled-cell.py` SHA-256 `8f4a5558d47fc0b291a729ce68611984ae9367ea7fc73c6558daa39e47952807`; the only change from the prepared disabled cell was `RUN_TRAINING = False` to `True`.
- **Initial state only:** Parent's initial viewer snapshot showed `Running` for 10.5 seconds, GPU T4x2, and 0 B output. This does not establish training progress, completed epochs/steps, metrics, checkpoint, success, or failure. Terminal outcome remains pending; do not retry or restart.
- **Verification and evidence:** Parent reports 15 notebook/package tests and 18 independent checks passed; the local focused suite passed 102 tests. Initial running screenshot: `data/private/kaggle-first-fold-background/run04-preparation/kaggle-run04-running.png`, SHA-256 `ec4ccc0dcc110672b4b140e062bcf63ab71675a4f85b6bfa96efab6029dc8003`. The run04 authorization is consumed by this single launch. Preserve all prior artifacts and do not start another run.
