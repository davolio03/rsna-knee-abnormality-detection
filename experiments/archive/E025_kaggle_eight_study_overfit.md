# E025 - Kaggle eight-study overfit smoke test

## Outcome

The bounded real-image training run completed on a fresh Kaggle GPU kernel using
private dataset Version 3. The fixed training subset's mean BCE decreased, but
this train-only smoke test supplies no validation, prediction, AUC, or
independent generalization evidence. Keep the scratch model provisional; no
full memorization or model-quality claim is supported.

## Hypothesis and baseline

- **Date:** 2026-10-17, Europe/Madrid.
- **Type:** Bounded private real-image training smoke test.
- **Question / hypothesis:** Can the existing fold-bound scratch model complete
the explicitly capped eight-study fit on the mounted private package without
changing the frozen preprocessing, model, or split?
- **Baseline:** No prior real-image fit existed. E024's synthetic CUDA probe
  established only that a small deterministic GPU operation could execute.
- **Changed variable:** The user manually uploaded private dataset Version 3,
  which was mounted for this run; the runtime-only launch flag was changed from
  its default `False` to `True` for this authorized fit. No source model,
  preprocessing, labels, or fold allocation changed.
- **Expected mechanism:** Decode and optimize the fixed eight-study training
  subset for at most 40 optimizer steps/five epochs on CUDA, then compare mean
  BCE on that same training subset before and after optimization.

## Protocol and environment

The private Version 3 package contained 24 files. Before importing training
code, the ignored launch cell checked every unpacked member against the
SHA-256 mapping embedded from the local transfer inventory, including the
manifest, folds, config, notebook, and source files. The local ZIP digest is
inventory provenance only: the remote ZIP archive bytes were not checked. The
launch cell defaults to `RUN_TRAINING = False`; for this run only, the parent
explicitly enabled it after mounting and auditing the package. The scratch model,
PP-01/PP-02 preprocessing, fold allocation, labels, and training code were
unchanged. Source package code was from local commit `f79c0cb`; the exact
ignored launch-cell source hash is recorded in `BASELINE.md`.

The frozen run was overfit mode, fold 0, seed 42, maximum eight training
studies, five epochs, 40 optimizer steps, 256-pixel inputs, maximum eight slices
per series, CUDA only. The report indicates eight training studies and all 40
steps completed. No validation partition was used. Runtime environment reported
Python `3.13.15`, Linux, PyTorch `2.11.0+cu128`, and NumPy `2.1.3`. The exact GPU
model is not recorded here. No Gold-58 evaluation, prediction generation,
submission, or code/package installation occurred.

## Results

| Measure | Observed |
|---|---:|
| Training subset | 8 studies |
| Fold / seed | 0 / 42 |
| Epoch / step cap | 5 / 40; 40 steps completed |
| Input size / slice cap | 256 pixels / 8 per series |
| Mean BCE, same training subset, start | `0.6906174942851067` |
| Mean BCE, same training subset, end | `0.48356025479733944` |
| Model runtime | `83.31482125599996` seconds |
| Wrapper runtime | `83.430137425` seconds |
| Peak CUDA allocation | `222188032` bytes |
| Validation / predictions / AUC | None / none / none |
| Gold-58 | Not used |

The endpoint mean BCE was lower, but some individual loss values rose during
the run. This does not establish monotone convergence, complete memorization,
validation performance, generalization, clinical utility, or a strong baseline.
The outcome is therefore **inconclusive for model quality**, although the
bounded execution path completed.

## Artifacts and execution notes

Parent-observed ID-free report:
`data/private/kaggle-eight-study-overfit/observed-training-run01/report.json`
(2,336 bytes; SHA-256
`a2871673660d462a9e6a0d27a50c4dd3ac5f4f9c81300873a7255bbc063cf0e3`).
Private checkpoint:
`data/private/kaggle-eight-study-overfit/observed-training-run01/checkpoint.pt`
(100,087 bytes; SHA-256
`b09f222057dd8fe8c1886ab3c107e8127264b43450019e4e2367cf6eba38518e`). The
parent verified remote and downloaded artifact hashes and the local copies.
Browser download events timed out although both files were saved; no retry was
made. The GPU UI confirmed the session stopped.

Notebook source/results remain a draft. The attempted Save Version action was
rejected before execution because notebook publication was not explicitly
authorized; no save-version operation ran, and all three historical dataset
versions were preserved. This publication decision is separate from the
completed training authorization. Seek separate user consent before publishing
a notebook version. No further training, full-fold run, or validation is
authorized by this smoke test.
