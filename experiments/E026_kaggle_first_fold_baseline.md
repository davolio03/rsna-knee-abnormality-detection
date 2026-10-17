# E026 — First full-fold Kaggle scratch baseline (failed; cause unknown)

- **Date:** 2026-10-17
- **Status:** Inconclusive failure; no metrics or checkpoint. Historical run ended; no retry authorized.
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
  out-of-memory or any specific failure step. The one-run authorization is
  consumed; do not retry or restart.
- **Diagnostic follow-up:** Locally, the training runner now retains only an
  allowlisted exception family and bounded exception-family chain, coarse
  execution stage, and validated internal source location for unexpected exceptions. The notebook and local CLI emit
  those fields only after strict schema/allowlist checks; raw messages, paths,
  IDs, reports, and locals remain excluded. Synthetic regression tests verify
  the redaction and fail-closed behavior. This code was not present in private
  Kaggle package Version 3 or the failed Version 1, and cannot retroactively
  identify that run's cause. A future diagnostic run would require separately
  authorized remote work and a matching provenance-preserving package version.
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
- **Conclusion:** The single private batch failed with its underlying cause
  unavailable. No model-performance or adoption conclusion can be drawn; any
  additional remote diagnostic execution requires new explicit authorization.
