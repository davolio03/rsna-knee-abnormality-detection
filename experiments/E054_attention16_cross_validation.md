# E054 — 3-fold cross-validation of the attention16 model

- **Date:** 2026-11-17
- **Status:** running
- **Parent:** E050 (fold-0 training run), E052 (fold-0 validation export)

## Hypothesis

The single-fold validation of E050, together with the late leaderboard score of E051, does not
show how stable the model is across data splits. Training the unchanged E050 recipe on all three
report-grouped folds will give a cross-validated ROC-AUC with a spread across folds. That number
becomes the reference for later controlled changes.

## Changed variable

Only the held-out fold changes: 0, 1, 2. The recipe stays fixed: RadImageNet ResNet-50 with
target-specific slice and series attention; 16 slices at 224 px; Adam with head LR 1e-3 and
`layer4` LR 1e-5; frozen BatchNorm; 5 epochs; epoch 5 chosen in advance; seed 42; masked soft
BCE. The cache, manifest and fold file are the same as in E050.

Fold 0 is not retrained. `FoldCacheDataset` reproduces the original E050 partition exactly
(2,850 / 1,448 studies, in the same order), so the E050 epoch-5 checkpoint
(`631536a3…2f21`) serves as the fold-0 model.

| Fold | Train studies | Validation studies |
|---:|---:|---:|
| 0 | 2,850 | 1,448 |
| 1 | 2,865 | 1,433 |
| 2 | 2,881 | 1,417 |

## Validation protocol (fixed before results)

Each held-out study is scored by the epoch-5 model of its fold. For each target, the label is
the observed soft label binarized at 0.5; unobserved cells are excluded
(`rsna_knee.validation.masked_auc`). The protocol reports:

- per-fold macro AUC, with mean ± sample std across folds;
- pooled out-of-fold macro AUC;
- per-target AUC, as mean across folds and pooled.

Gold-58 is excluded. The labels come from an LLM, so this is a proxy metric, not an estimate
against expert labels.

## Execution

- Code: `scripts/cv_attention16.py` and `src/rsna_knee/training/cross_validation.py`. A frozen
  copy of the code for the run, with its base commit, is kept in `data/private/cv-attention16/code/`.
- Environment: Python 3.11.9, torch 2.11.0+cu128, numpy 2.1.3, RTX 3060 Laptop (6 GB).
- Private outputs: `data/private/cv-attention16/`, containing fold checkpoints, per-study
  out-of-fold predictions and metrics.

## Results

Pending.

## Conclusion

Pending.
