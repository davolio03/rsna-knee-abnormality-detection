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

## Newly authorized run03 preparation — pending parent upload and launch

- **Authorization:** The user explicitly authorized exactly one additional private run03 after proposing a package rebuild with the local dataset-reason fix. This authorization applies only to this single attempt; it does not authorize any retry or restart. Parent owns all Kaggle/browser operations.
- **Preparation:** The existing audited builder produced a fresh 22-member package from current allowlisted source, pinned manifest/folds, plus a new disabled-default run03 notebook and matching config. No historical run01/run02 package, notebook, proof, or provenance was changed. The fixed fold0/seed42 scratch protocol and existing private dataset mount are unchanged.
- **Artifacts:** ZIP `data/private/kaggle-first-fold-background/run03-preparation/kaggle-first-fold-diagnostic.zip`, SHA-256 `774aed89cba781017a130abe607bffc077d15a4e6b91fb2ef112d4d27b246ede`; inventory `.../inventory.json`, SHA-256 `96016ffefdb4e96ac36119d943156b07c70caa3a20618a39459bad2eb9e94d82`; notebook `.../kaggle_first_fold_baseline_run03.ipynb`, SHA-256 `c90a64e48589ba6a4376f5fdab85de264c251c30d2c421fbab6d0d33e4e966f2`; matching config `.../kaggle_first_fold_baseline_run03.json`, SHA-256 `031b369c1cf78d417727524a8b9be4156bf932d0cfe94fc58ce0d3c52b0f8ece`. Full local hashes and bounded verification are in ignored `data/private/kaggle-first-fold-background/run03-preparation/run03-preparation-provenance.json`.
- **Verification:** The archive, inventory, and extracted build folder agree for all 22 unique members; manifest/fold bytes and hashes match pinned source. Isolated ZIP import used the packaged source. A synthetic lazy dataset failure traversed `run_training` and the prepared notebook event, retaining only `pixel_decode_failed` without a sentinel message/path; no GPU or real data. The focused suite passed 98 tests. Notebook/config JSON, syntax, config parity, empty outputs, and disabled default passed.
- **Current boundary:** Local preparation is complete. No package upload or run03 launch has occurred. Parent must independently verify before any remote action; R3-02/R3-03 remain pending.
