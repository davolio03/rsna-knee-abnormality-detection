# E045 — Local RadImageNet epoch three continuation

## Hypothesis and baseline
Continue the same fold-0 RadImageNet model for exactly one additional epoch from the verified epoch-two checkpoint. Baseline is epoch two: checkpoint SHA-256 `bd1abee90e19256292a57d04d31eced4341ad7207c403125d03b150cdeacaf72`, `completed_epochs=2`, native fingerprint `5c08ab6be89773802ce4f18fa2209a7e85ec88f2703b5caaded9bd940025a920`.

## Changed variable and expected mechanism
Only the requested optimizer training epoch changes: continue the exact model/Adam/RNG/history state from epoch two to epoch three. Keep the frozen 29-file recipe, source-bound fingerprint, cohort/folds/labels, selected cache arrays, preprocessing, model and precision policy unchanged. Native `_read_resume` remains authoritative and is called before any new output is created; the native training call receives the explicitly SHA-pinned epoch-two checkpoint. One-run cap stops at three of five epochs.

## Validation protocol
The adapter's default resume selector remains epoch one for backward-compatible epoch-two planning. `--resume-epoch 2` selects only the audited epoch-two checkpoint/report/provenance, validates the exact checkpoint SHA, completed-epoch/history evidence and native fingerprint, and plans a fresh output under `data/private/local-radimagenet-epoch03/epoch03-run01`. Actual continuation requires both `--resume-epoch 2 --train`; no arbitrary “latest checkpoint” discovery is allowed. Revalidate all seven cache archives/coverage and pinned CUDA runtime/GPU memory, enforce FP32 with TF32/autocast/cuDNN benchmarking off, and retain a separate provenance sidecar binding the resume checkpoint hash/path and parent evidence. The parent, not this adapter task, launches the full epoch after independent verification.

## Results and status
Strict TDD observed RED for the epoch-two resume selection before implementation. The pinned CUDA command `data/private/local-cuda-env/Scripts/python.exe -m unittest tests.test_local_radimagenet_epoch_adapter tests.test_local_masked_tensor_cache tests.test_local_masked_cache_chunk_plan tests.test_kaggle_local_masked_cache_export tests.test_local_radimagenet_checkpoint_migration` passed **34 tests** in 120.617 seconds. Coverage includes checkpoint SHA/report/history/fingerprint selection, wrong SHA and malformed-history refusal, fresh output selection, native-call resume arguments, precision policy, and one-epoch caps; the existing cache/export/migration regressions remained green.

The epoch-three plan-only CLI passed in the pinned CUDA runtime: `data/private/local-cuda-env/Scripts/python.exe scripts/train_local_radimagenet_epoch.py --resume-epoch 2` with `PYTHONPATH=src`. It reported all 4,298 cohort studies (2,850 train / 1,448 validation), the exact epoch-two checkpoint SHA, two completed history entries, native fingerprint, one-epoch cap, and fresh epoch-three output path. `compileall` and `git diff --check` passed. No epoch-three training was launched; no new checkpoint or validation metrics exist. No performance or model-quality conclusion is available.

## Next step
Parent independently verifies the adapter and plan, then launches exactly one epoch with `data/private/local-cuda-env/Scripts/python.exe scripts/train_local_radimagenet_epoch.py --train --resume-epoch 2` and `PYTHONPATH=src`. Expected checkpoint: `data/private/local-radimagenet-epoch03/epoch03-run01/checkpoint-epoch-03.pt`; report under that output directory and sibling provenance `epoch03-run01-provenance.json`. Preserve epoch-one and epoch-two artifacts unchanged.
