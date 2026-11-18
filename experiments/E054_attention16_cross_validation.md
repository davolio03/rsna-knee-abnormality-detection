# E054 — 3-fold cross-validation of the attention16 model

- **Date:** 2026-11-17
- **Status:** completed (2026-11-18)
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

**Macro ROC-AUC: 0.861 ± 0.003** (mean ± std over 3 folds) · pooled out-of-fold: 0.860 · 4,298 studies

| Target | Fold 0 | Fold 1 | Fold 2 | Mean ± std | Pooled OOF | Positives / n |
|---|---:|---:|---:|---:|---:|---:|
| ACL | 0.816 | 0.819 | 0.841 | 0.825 ± 0.013 | 0.825 | 954 / 3,987 |
| MCL | 0.813 | 0.805 | 0.796 | 0.805 ± 0.008 | 0.800 | 679 / 3,912 |
| Medial Meniscus | 0.876 | 0.873 | 0.854 | 0.868 ± 0.012 | 0.867 | 2,330 / 4,108 |
| Lateral Meniscus | 0.801 | 0.841 | 0.826 | 0.823 ± 0.020 | 0.822 | 1,116 / 3,917 |
| Medial OA | 0.883 | 0.885 | 0.881 | 0.883 ± 0.002 | 0.880 | 1,648 / 3,469 |
| Lateral OA | 0.855 | 0.858 | 0.869 | 0.861 ± 0.007 | 0.859 | 1,259 / 3,280 |
| PF OA | 0.848 | 0.841 | 0.831 | 0.840 ± 0.009 | 0.839 | 2,022 / 3,600 |
| Effusion | 0.845 | 0.836 | 0.834 | 0.839 ± 0.006 | 0.837 | 2,606 / 3,919 |
| Synovitis | 0.949 | 0.897 | 0.904 | 0.917 ± 0.028 | 0.920 | 551 / 701 |
| Baker's | 0.884 | 0.875 | 0.902 | 0.887 ± 0.014 | 0.886 | 1,075 / 2,256 |
| Contusion | 0.894 | 0.874 | 0.879 | 0.883 ± 0.010 | 0.882 | 1,080 / 3,304 |
| Fracture | 0.908 | 0.889 | 0.908 | 0.901 ± 0.011 | 0.902 | 315 / 2,482 |
| **Macro** | 0.864 | 0.858 | 0.860 | **0.861 ± 0.003** | 0.860 | |

Runtime on the RTX 3060 Laptop: about 75 min per epoch (fold 2: 22,585 s for 5 epochs). Fold 1
was interrupted after epoch 2 when Windows logged off and suspended overnight, and resumed from
the verified epoch-2 checkpoint (`--resume`, same fingerprint). Scoring took 1,479 / 831 / 678 s
for folds 0 / 1 / 2.

## Conclusion

The baseline is stable across splits: macro AUC 0.861 ± 0.003, with the pooled out-of-fold value
(0.860) in agreement. The fold-0 result of E050 (0.864) was representative, not a lucky split.
Per-target spread is larger (up to ±0.028 for Synovitis, n = 701), so per-target comparisons
need care. MCL, Lateral Meniscus and ACL are the weakest targets.

This is the reference for later changes: a candidate should beat 0.861 by clearly more than the
fold spread, consistently across folds. A T4 replicate of folds 0 and 1 on Kaggle is running to
measure the hardware effect before experiments mix local and Kaggle GPUs.
