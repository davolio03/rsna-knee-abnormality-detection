# Diagnose and safely expose the first-fold batch failure

## Objective and authorization
Prepare a provenance-preserving diagnostic package and notebook for one newly authorized private first-fold Kaggle run, after private Version 1 failed. Parent owns all browser/Kaggle UI work. This agent may perform local preparation only; it must not access remote services, authenticate, upload, launch, inspect bulk medical data, use Gold-58, push, or create a PR. Preserve the old V3/V1 evidence and artifacts.

## Reconciled evidence
A read-only Kaggle observation on 2026-10-17 confirms private Version 1 / scriptVersionId `359170889` failed after 258.3 seconds with 0 B output. At 247.3 seconds the logs emitted `training_blocked` / `training_failed`; Papermill then surfaced `RuntimeError(training_failed)` from notebook main line 232. The notebook currently re-raises only the safe reason with `from None`, hiding the underlying cause. No metrics/checkpoint exist. The actual cause and completed training step are unknown; do not infer GPU OOM or a failing stage. Historical one-run authorization is consumed; no retry/restart.

## Frozen constraints
Fold 0 / seed 42; 333 train and 167 validation studies; scratch model; five epochs / 1,665 planned steps; CUDA only; 256 pixels; max eight slices; width 64; learning rate 0.001. Preserve notebook/config, V3 24-member inventory and hashes, manifest/fold hashes, all data privacy, and Gold exclusion. Notebook remains disabled by default. No model, prompt, cohort, fold, or protocol changes.

## Tasks and routing
- [x] R01 (delegated direct): Prepared fresh allowlisted package, inventory, matching config, and separate first-fold notebook with exact new inventory/archive pins. Notebook remains disabled by default. Preserved existing package/notebook versions, manifest/fold bytes and hashes, and frozen protocol. Validated all ZIP member hashes and isolated diagnostic import; no remote action. Commit `dbe03e3`.
- [x] R02 (parent-owned): Parent uploaded the prepared package once as private dataset Version 4 and launched exactly one run in a new private notebook. The fixed fold-0/seed-42 protocol, dataset/mount and competition attachment were preserved; no Gold/model/fold/cohort/protocol changes.
- [x] R03 (local recording): Launch evidence has been recorded from the parent-verified remote observation in E026, PROJECT_STATE, registry, and ignored provenance. At the 14:38:09 Europe/Madrid snapshot, run02 was still running (32.6s, T4x2, 0 B); the training result remains pending. No cause, completion, metrics, checkpoint, or outputs are inferred; do not restart. Commit `8ad00e2`.
- [x] FB-01: Prepare clean standalone output-free notebook/config and strict-TDD synthetic artifact checks. Completed and committed as `ce9f40a0f465c3c32e0d249a577a3256174b1f4a`.
- [x] FB-02 / launch record: Parent launched one private Version 1 after source hash readback; launch records are preserved in E026, registry, project state, and ignored provenance. Local docs commit `dfabc25`.
- [x] FB-03 / result inspection: Read-only evidence confirms failure after 258.3 seconds, 0 B output, safe terminal reason only. Root cause remains unknown and no result artifacts exist. Do not relaunch.
- [x] FD-01: Added allowlisted exception family and maximum-four family chain, fixed stage, and validated internal module/function/line diagnostics. Notebook and CLI expose only validated fields; adversarial tests verify messages, paths, IDs, arbitrary metadata, and raw locals cannot leak. Observed RED (missing `diagnostics`) before production edits; then targeted tests passed. Route: delegated direct. ROADMAP unchanged.
- [x] FD-02 (diagnostic review only): Local code inspection confirmed the generic `training_failed` catch discards cause details, but no local safe artifact can determine the historical cause. Historical root cause is NOT resolved. Updated E026, registry, and PROJECT_STATE; any diagnostic remote run needs separate authorization and a provenance-preserving package update. Independent verification passed after FD-03: 92 initial focused tests and 28 final runner/notebook tests; direct no-GPU OOM reproduction reports memory without sentinel leakage.
- [x] FD-03: Corrected CUDA OOM classification by checking `isinstance` against the actual `torch.cuda.OutOfMemoryError` type (its `__module__` is `torch`, not `torch.cuda`) without type-name spoofing. Source location now keeps the deepest allowlisted internal frame. Actual OOM and nested-frame tests both demonstrated RED before implementation and GREEN afterward. No GPU operation or remote training was run.

## Acceptance and checks
Focused runner: `PYTHONPATH=src .venv-training/Scripts/python.exe -m unittest tests.test_training_runner tests.test_training_components tests.test_training_folds tests.test_validation_multilabel tests.test_preprocessing tests.test_dicom_preprocessing tests.test_kaggle_training_package tests.test_kaggle_first_fold_notebook`.
Observed before FD-03: focused suite passed 92 tests. After FD-03: focused suite passed 94 tests. Legacy suite ran 299 tests with 8 skips and one known Windows child-timeout cleanup failure; using a writable workspace TEMP/TMP removed other environment failures but did not resolve that timeout. Notebook JSON/syntax, output-free status, config and local launch-cell parity, and `git diff --check` passed. No remote execution/training. Independent verification passed after FD-03; parent independently reran 28 runner/notebook tests successfully.
R01 verification: focused runner passed 95 tests; new package was independently read back with all 22 ZIP members matching the inventory and extracted build folder, and isolated ZIP import identified the actual `torch.cuda.OutOfMemoryError` as `memory` without GPU work or sentinel leakage. The generated notebook JSON and Python cell parse; outputs are empty; the fixed fold/seed/counts/steps/CUDA/model settings and dataset mount match the prior protocol. `git diff --check` passed.

R01 private artifacts (ignored; parent upload only after independent readback):
- Dataset-root folder: `data/private/kaggle-first-fold-background/run02-preparation-v2/package/` (exactly the 22 inventoried members).
- Provenance ZIP: `data/private/kaggle-first-fold-background/run02-preparation-v2/kaggle-first-fold-diagnostic.zip` — SHA-256 `1e454db9f9aba9d574cb9e37a7323166e3b5ab8be89dcd80c47e986014950cff`.
- Inventory: `data/private/kaggle-first-fold-background/run02-preparation-v2/inventory.json` — SHA-256 `7b1b1bd51357f7b6a7e4ad7b2c88edaf11fde3a768b5c4425f9cc592a278da73`.
- Notebook: `data/private/kaggle-first-fold-background/run02-preparation-v2/kaggle_first_fold_baseline_run02.ipynb` (disabled default; distinct output `run02`).
- Matching config: `data/private/kaggle-first-fold-background/run02-preparation-v2/kaggle_first_fold_baseline_run02.json` (kept outside the inventoried dataset root to avoid self-reference).
- Manifest/folds remain `e88c61c22a463737f16951b7c76db03834034aecd1b31dc61b4ce495080b2be2` / `b82b376b0086201e3364b11a0203b62890e628442075a52ea4a9e09ee0e778ac`.

R02/R03 remote launch record (parent-verified; execution result pending): private
dataset Version 4 upload completed once with 22 files and the competition dataset
attached. New private notebook `RSNA Knee First Fold Diagnostic Run02`, Version 1,
execution `359359350`, version name `E026 Run02 - fold0 - safe diagnostics`;
status URL:
https://www.kaggle.com/code/davidsulleiro/rsna-knee-first-fold-diagnostic-run02?scriptVersionId=359359350
The explicit package input is dataset Version 4, and GPU selection is T4x2. The
17,884-character enabled code-cell clipboard readback is SHA-256
`b3d9428c2f380e92e1e99f0bff76ef30e7ea93bb398b4d68972c845e2cc93e06`; the only
change from the prepared notebook was `RUN_TRAINING = False` to `True`. At the
14:38:09 Europe/Madrid snapshot it was running for 32.6 seconds with Latest
Container Image and 0 B output. The interactive session was off; no run result
or failure was available. Approximate launch time: 14:37 Europe/Madrid (exact
time not captured). The single-use authorization is consumed; no retry.
Ignored proof files and SHA-256 values are documented in E026 and
`data/private/kaggle-first-fold-background/run02-launch-provenance.json`.

## Delivery and recovery
Use the existing `feat/gold58-prompt-comparison-runner` branch and local conventional work-unit commits; no AI attribution. Stage only owned files. Preserve dirty `odd/tasks/siwc-chatgpt-plan-labeler.md` and untracked `.atl/`. RDD remains OFF; do not enable review. Rollback only owned diagnostic changes. Diagnostics implementation commits: `7410c96474616b5e46dedda06db2ff8d011c522f` (`fix(training): expose redacted baseline failure diagnostics`) and `9ff31f51e893abd1c8550c1476fc0bd1985a3e66` (`fix(training): classify CUDA out-of-memory exceptions`). The latter adds actual-type OOM classification and deepest trusted source-frame selection. Work-unit commits: `dbe03e3` (package preparation) and `8ad00e2` (R03 launch record). Task-tracker evidence is recorded in follow-up docs commits because `odd/` is ignored by repository rules. Mirror the full current tracker to Engram topic `odd/kaggle-first-fold-background/tasks` and read back both. The parent performed the authorized private package upload and single run02 launch recorded above; this agent did not access the remote service. No raw data download, push, PR, or further run/restart.
