# E026 — First full-fold Kaggle scratch baseline (prepared; not run)

- **Date:** 2026-10-17
- **Status:** pending private batch execution
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
- **Results:** None. Local preparation is complete, but the Kaggle batch has not
  been started from this task. Runtime, metrics, model output, and checkpoint
  are pending; do not infer them from the E025 smoke test.
- **Preparation checks:** Strict TDD observed five expected missing-artifact
  test errors before notebook/config creation; six synthetic notebook guard
  tests then passed. A later regression RED showed a failed preflight emitted a
  diagnostic but returned normally; the notebook now re-raises only allowlisted
  safe reason codes, and validates the complete report before success output.
  Ten synthetic notebook guard tests now pass, including exact parity of the
  ignored launch cell with the checked-in notebook. The focused
  runner/component/fold/validation/preprocessing, package, and notebook suite
  passed (88 tests). Independent parent review and the Kaggle batch remain
  pending at this record's preparation stage. The CSV registry contains older
  rows with inconsistent column widths; the new E026 row matches the 28-column
  header and was checked independently.
- **Conclusion:** Pending. No adoption decision is made before the authorized
  private batch result is observed and independently reconciled.
