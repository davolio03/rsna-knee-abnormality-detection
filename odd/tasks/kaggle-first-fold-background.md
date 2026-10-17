# Diagnose and safely expose the first-fold batch failure

## Objective and authorization
Improve bounded, privacy-safe diagnostics for the already-authorized first-fold notebook after its private Kaggle Version 1 failed. Identify a cause only if supported by local reproducible evidence or existing safe artifacts. Do not relaunch training, inspect bulk medical data, use Gold-58, access remote services, upload, push, or create a PR. Existing private V3 package inventory and frozen first-fold protocol remain unchanged.

## Reconciled evidence
A read-only Kaggle observation on 2026-10-17 confirms private Version 1 / scriptVersionId `359170889` failed after 258.3 seconds with 0 B output. At 247.3 seconds the logs emitted `training_blocked` / `training_failed`; Papermill then surfaced `RuntimeError(training_failed)` from notebook main line 232. The notebook currently re-raises only the safe reason with `from None`, hiding the underlying cause. No metrics/checkpoint exist. The actual cause and completed training step are unknown; do not infer GPU OOM or a failing stage. Historical one-run authorization is consumed; no retry/restart.

## Frozen constraints
Fold 0 / seed 42; 333 train and 167 validation studies; scratch model; five epochs / 1,665 planned steps; CUDA only; 256 pixels; max eight slices; width 64; learning rate 0.001. Preserve notebook/config, V3 24-member inventory and hashes, manifest/fold hashes, all data privacy, and Gold exclusion. Notebook remains disabled by default. No model, prompt, cohort, fold, or protocol changes.

## Tasks and routing
- [x] FB-01: Prepare clean standalone output-free notebook/config and strict-TDD synthetic artifact checks. Completed and committed as `ce9f40a0f465c3c32e0d249a577a3256174b1f4a`.
- [x] FB-02 / launch record: Parent launched one private Version 1 after source hash readback; launch records are preserved in E026, registry, project state, and ignored provenance. Local docs commit `dfabc25`.
- [x] FB-03 / result inspection: Read-only evidence confirms failure after 258.3 seconds, 0 B output, safe terminal reason only. Root cause remains unknown and no result artifacts exist. Do not relaunch.
- [x] FD-01: Added allowlisted exception family and maximum-four family chain, fixed stage, and validated internal module/function/line diagnostics. Notebook and CLI expose only validated fields; adversarial tests verify messages, paths, IDs, arbitrary metadata, and raw locals cannot leak. Observed RED (missing `diagnostics`) before production edits; then targeted tests passed. Route: delegated direct. ROADMAP unchanged.
- [x] FD-02 (diagnostic review only): Local code inspection confirmed the generic `training_failed` catch discards cause details, but no local safe artifact can determine the historical cause. Historical root cause is NOT resolved. Updated E026, registry, and PROJECT_STATE; any diagnostic remote run needs separate authorization and a provenance-preserving package update. Independent verification remains pending.

## Acceptance and checks
Focused runner: `PYTHONPATH=src .venv-training/Scripts/python.exe -m unittest tests.test_training_runner tests.test_training_components tests.test_training_folds tests.test_validation_multilabel tests.test_preprocessing tests.test_dicom_preprocessing tests.test_kaggle_training_package tests.test_kaggle_first_fold_notebook`.
Observed: focused suite passed 92 tests. Legacy suite ran 299 tests with 8 skips and one known Windows child-timeout cleanup failure; using a writable workspace TEMP/TMP removed other environment failures but did not resolve that timeout. Notebook JSON/syntax, output-free status, config and local launch-cell parity, and `git diff --check` passed. No remote execution/training. Independent verification is pending with parent.

## Delivery and recovery
Use the existing `feat/gold58-prompt-comparison-runner` branch and one local conventional work-unit commit for the cohesive diagnostics+tests+docs change; no AI attribution. Stage only owned files. Preserve dirty `odd/tasks/siwc-chatgpt-plan-labeler.md` and untracked `.atl/`. RDD remains OFF; do not enable review. Rollback only owned diagnostic changes. Implementation work-unit commit: `7410c96474616b5e46dedda06db2ff8d011c522f` (`fix(training): expose redacted baseline failure diagnostics`, 8 tracked implementation/test/experiment files). This task-tracker evidence is recorded in a follow-up docs commit because `odd/` is ignored by the repository rules. Mirror the full current tracker to Engram topic `odd/kaggle-first-fold-background/tasks` and read back both. No remote data, upload, push, PR, or training relaunch.
