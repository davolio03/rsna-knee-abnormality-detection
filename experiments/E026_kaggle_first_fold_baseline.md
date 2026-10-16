# E026 — First full-fold Kaggle scratch baseline (running; results pending)

- **Date:** 2026-10-17
- **Status:** running; result pending
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
  A read-only viewer check later showed “Running for 67.0s”, T4x2 / Latest
  Container Image, and 0 B output; no training result or failure was present.
  The draft editor session is off; the separate batch is active.
- **Results:** Plan validation, fitting, metrics, report, checkpoint, and
  predictions remain pending. This record makes no training-result claim. The
  one-run authorization is consumed; do not retry or restart. Next step is
  read-only inspection of this same Version 1 execution.
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
  batch was accepted and is running; metrics and outputs are still pending. The
  CSV registry contains older rows with inconsistent column widths; the new
  E026 row matches the 28-column header and was checked independently.
- **Conclusion:** Pending. No adoption decision is made before the authorized
  private batch result is observed and independently reconciled.
