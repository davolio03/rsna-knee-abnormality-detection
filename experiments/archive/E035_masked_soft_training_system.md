# E035 — Implement the opt-in masked soft-label training path

## Hypothesis and scope

An isolated opt-in runner can consume the verified E034 soft-label artifacts without changing the legacy integer-label dataset or binary training runner. Synthetic CPU tests can verify optimizer/objective behavior, private checkpoint/report writes, and planning safety without loading DICOM images or model weights. No real-image training, quality result, or model-performance claim is made.

## Baseline and changed variables

- Baseline binary dataset and training runner remain behaviorally unchanged; only a shared pure image-loading/tensor-validation helper was extracted from `data/training_dataset.py` and reused by both dataset adapters.
- New default objective is unweighted soft BCE, reduced over observed target cells only. Unknowns remain masked; the denominator is observed-cell count, never 12 or the sum of class weights.
- Optional `training_only_class_weights` mode uses the exact positive and negative weights already computed from each fold's training partition. Validation BCE/Brier remain unweighted.
- The production default uses frozen `resnet18_imagenet1k_v1` and `torch.optim.Adam`, matching the existing frozen baseline family. It requires an explicitly configured local weights file matching SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` and 46,830,571 bytes; there is no download or random-initialization fallback. Planning hashes the file only and never deserializes it.
- New runner is separate from the old baseline runner. It plans by default and needs an explicit `--execute` opt-in to train and save a model-state artifact in a fresh private output directory.

## Frozen inputs and validation protocol

- Manifest SHA-256: `617ea1c267960b1e903b332a78388db3b6ab80d44374721df5a4c5e184e8961b`.
- Fold artifact SHA-256: `b816c7902a5d0caa68a9e818d3f913943bc61cd2e6db65c4de069f5cd94e15f9`.
- Prefilter policy module SHA-256: `7a1b41d3d96c137859d83d78927fd020afba8ce4034cb0addbed67e730280662`.
- Original 500-study C2 cohort SHA-256: `e88c61c22a463737f16951b7c76db03834034aecd1b31dc61b4ce495080b2be2`; folds SHA-256: `b82b376b0086201e3364b11a0203b62890e628442075a52ea4a9e09ee0e778ac`.
- Protected HSR12 source ZIP SHA-256: `47aae9f98cd24fa1d5dd2c2e3433e01774f71e147377e34906cbafc8ef8b0462`; notebook SHA-256: `361ac91914b0ee5ef85aabb47da272cb238bb4ab730974a3566b5d46b520ab21`.
- Loader verifies exact raw manifest/fold pins, expected provenance and target order, rejects excluded-row/audit mismatches, validates probabilities/masks/series metadata, and recomputes the exact grouped fold/weight artifact before dataset or model construction.
- Report-hash grouping remains a duplicate-report proxy, not evidence of patient or site isolation. Source probabilities are not established as calibrated.

## Results

- Strict TDD RED was observed for the corrective findings: wrong random/AdamW production defaults, a masked extreme logit contaminating weighted loss, observed weighted overflow escaping the update guard, and unexpected CLI exception leakage. The corresponding focused tests now pass.
- Focused training-module run: `PYTHONPATH=src .venv-training/Scripts/python.exe -m unittest tests.test_masked_soft_training` — **16 passed**. Parent's later spot-check of both `tests.test_masked_soft_training` and `tests.test_masked_soft_dataset` independently ran **20 tests, all passed**.
- Applicable regression command: `PYTHONPATH=src .venv-training/Scripts/python.exe -m unittest tests.test_masked_soft_training tests.test_masked_soft_dataset tests.test_training_components tests.test_training_runner tests.test_training_dataset_diagnostics tests.test_resnet18 tests.test_full_cohort_prefilter tests.test_training_folds tests.test_training_cohort` — **85 tests run, 84 passed and 1 existing Windows symlink test skipped**.
- `py_compile` passed for all changed/new Python modules and tests. `git diff --check` passed; Git emitted line-ending normalization warnings but no whitespace errors.
- Parent independently verified the correction: frozen ImageNet1K V1 plus Adam and weight metadata/pin; finite-safe masking and zero gradient for an extreme masked logit; observed extreme weighted overflow yields `loss_nonfinite` before optimizer update; unexpected private runtime errors return only safe JSON `training_unavailable`, exit 2, with no marker or traceback. Parent's nine-module run was 85 tests (84 passed, 1 existing Windows symlink skip); the two-module focused spot-check was 20 passed.
- A pinned fold-0 metadata plan with an explicitly supplied local weights file completed without creating output, inspecting images, or constructing a model: 2,850 training studies / 2,796 groups; 1,448 validation studies / 1,379 groups. The plan verified the file's hash/size only. With no weights path, planning reports `blocked/offline_weights_path_required`.
- Synthetic injected CPU harness traversed the pinned dataset adapter with fake image arrays and a tiny 12-logit model in both unweighted and training-only-weight modes. It performed an optimizer step, computed unweighted validation BCE/Brier from observed soft targets, wrote and read back a private report/checkpoint in a temporary directory, then the temporary directory was removed. A zero-observation case raised before optimizer update and cleaned only its newly created run directory.
- No actual DICOM/pixel data, checkpoint, pretrained weights, or production model were loaded; the offline weights file was read only for streaming SHA-256/size verification. Synthetic CPU training used injected arrays and a tiny model. No competition image training occurred. No persistent masked-training output directory was created.
- Post-check hashes confirmed the E034 manifest/folds/policy module, old C2 cohort/folds and protected HSR12 ZIP/notebook match their recorded pins. No files were changed in those protected areas.

## Limits and next step

This establishes software-path behavior on synthetic CPU inputs and the integrity of the metadata/weights plan only. Parent independent verification passed MST.4, including the three correction findings; all eight protected hashes remained unchanged. It does not establish image availability/quality, clinical validity, label calibration, generalization, improved performance, or suitability of the balance heuristic. No real-image QC/training or Kaggle packaging/upload/launch occurred. No broad test discovery was run, and no Git commit could be created because `.git` is read-only.
