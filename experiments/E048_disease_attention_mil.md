# E048 — Disease-specific hierarchical attention MIL (prepared)

**Status:** local preparation and synthetic contract tests complete; neither a cache-backed GPU smoke nor an epoch has been run. There are no model-quality results yet.

## Question and controlled comparison

Does target-specific learned weighting over slices and series improve the fixed fold-0 soft-report-label objective over equal-mean pooling, when both arms start fresh from the same pinned RadImageNet ResNet-50 recipe?

| Element | Frozen protocol |
|---|---|
| Baseline | Fresh RadImageNet ResNet-50, equal mean of slice and series logits |
| Treatment | Same backbone, seeded classifier, trainable layer4 and training recipe; add zero-initialized 2048-to-12 slice and series score heads |
| Initialization | Original pinned RadImageNet weights; fresh independent construction for each arm; seed 42; no epoch-five warm-start |
| Data | Verified local full-cohort soft-label cache; study-level fold 0; Gold-58 excluded |
| Inputs | 224 pixels, at most eight selected slices, native RadImageNet normalization, masked spatial pooling |
| Optimization | Adam, head LR 1e-3, layer4 LR 1e-5, five epochs; frozen BatchNorm |
| Outcomes | Masked soft BCE, Brier score, per-target metrics and observed-cell counts |

At initialization, zero attention scores imply uniform weights, so treatment logits are expected to match equal-mean logits within the explicit floating-point tolerance in the model test. This is not an exact-bit guarantee. Attention weights are not clinical explanations. Fractional targets remain soft; no binary AUC or threshold is defined.

## Prepared commands

From the repository root in PowerShell:

```powershell
$env:PYTHONPATH = 'src'
$python = 'data/private/local-cuda-env/Scripts/python.exe'
& $python scripts/train_local_attention_mil.py
& $python scripts/train_local_attention_mil.py --smoke --output data/private/disease-attention-mil/e048-smoke01
& $python scripts/train_local_attention_mil.py --train --output data/private/disease-attention-mil/e048-run01
```

The default is plan-only: it checks pinned labels, folds, cache archives and pretrained-weight bytes, but does not create the output directory, download weights, instantiate a model, or train. `--smoke` performs one cached training-study forward/backward/Adam update in RAM and writes no checkpoint. `--train` explicitly runs both fresh five-epoch arms. Training requires a new private output path; it refuses overwrite.

The run plan records each arm through `not_started → started → checkpointed → complete`, with atomic transitions. An interrupted matching run can be resumed only with `--train --resume`: every started/checkpointed/complete arm must have a verified latest epoch checkpoint; a `not_started` arm must have no arm directory and may be started fresh. This permits a valid control checkpoint to resume while the treatment arm is still unstarted, including when the control is already complete. The run fingerprint and all started-arm checkpoint proofs are checked before dataset creation, model construction, deserialization, or output mutation. SHA verifies bytes only; resume also validates checkpoint schema/status, arm, expected epoch, complete epoch-history length and metric/count consistency, finite strict model-state names/shapes/dtypes, Adam group identity/LRs/options/parameter mapping and state shapes/finiteness/expected update counts, plus RNG structure/device lengths before restoring any state. The atomically written checkpoint for each epoch is retained, so a prior checkpoint is not silently replaced by an incomplete later write. A missing/corrupt latest proof fails closed instead of falling back. An arm marked `started` without a complete epoch checkpoint fails closed; it is not silently retrained or deleted. Preserve it for inspection and, if starting over is explicitly authorized, use a new output directory. Do not point an arm at a historical checkpoint.

## Preparation evidence

- Focused tests: `tests.test_radimagenet_attention` and `tests.test_radimagenet_attention_training` cover uniform-init equivalence within tolerance, target-wise mask sums, empty-mask rejection, a known nonuniform weighted value, slice/series permutation invariance, finite attention and classifier gradients, invalid-input rejection, plan-only no-write behavior, failed cache validation, and soft-target observed counts.
- Result: 17 focused model/training tests pass on CPU. The checkpoint tamper test includes fail-closed subcases for missing proof, implicit fresh restart, history gaps/nonfinite values and cell-count mismatch, incomplete status, model NaN/dtype/shape mismatch, optimizer group/LR/options/state mismatch or corruption, and missing/malformed RNG. Paired-resume tests confirm a checkpointed or completed control can resume with the treatment arm still `not_started`; a started arm without a checkpoint still fails before data/model/output mutation. This verifies synthetic model and preparation contracts only; it does not establish cache readability on GPU, runtime cost, or model quality.
- Final plan-only command `& $python scripts/train_local_attention_mil.py --output data/private/disease-attention-mil/e048-plan-check-lifecycle` completed successfully in about 55 seconds after checkpoint and pair-lifecycle hardening. It verified the local seven-archive cache, fold-0 data counts, pinned pretrained weights, final source/config fingerprint, and initial `not_started` arm states. Neither the plan-check directory nor the default training output directory was created.
- Requested regression command ran 37 tests successfully: `tests.test_resnet50_recipes`, `tests.test_resnet50_training`, `tests.test_local_radimagenet_epoch_adapter`, `tests.test_local_masked_tensor_cache`, and `tests.test_local_masked_cache_chunk_plan`.
- Static checks: compileall and CLI `--help` succeeded. `git diff --check` emitted no whitespace errors for tracked changes; the new files are untracked until parent integrates them.
- Actual experiment result: **not run**. No epochs, checkpoints, or predictions were produced. Do not interpret this preparation as validated attention performance.

## Next step

The GPU smoke remains unrun: the local device showed 5,994 MiB free and 0% utilization, but the CUDA-process query returned five PIDs as `[Insufficient Permissions]`. Therefore, no-concurrent-job status could not be established safely. After independent review and confirming the GPU is idle with sufficient free VRAM, run the one-study smoke once and record exact elapsed time and peak memory. Only a separately authorized `--train` run can produce the paired five-epoch comparison.
