# E055 — Fine-tune backbone layer3 as well as layer4

- **Date:** 2026-11-19
- **Status:** completed — adopt (see [D036](../docs/DECISIONS.md#d036--fine-tune-layer3-and-layer4-2026-11-19))
- **Parent:** E054 (3-fold cross-validated baseline, macro AUC 0.861 ± 0.003)

## Hypothesis

RadImageNet's mid-level features (layer3) are generic across radiology. Letting layer3 adapt to
knee MRI should sharpen fine local findings, so the 3-fold macro AUC should rise above the E054
baseline, most of all for the weakest fine-detail targets (MCL, Lateral Meniscus, ACL).

## Changed variable

Only the trainable backbone: layer3 and layer4 instead of layer4 alone
(`--trainable layer3`, `rsna_knee.training.cross_validation.unfreeze_layer3`). Layer3 joins the
backbone optimizer group at the same learning rate (1e-5). BatchNorm stays frozen. Everything
else is the E054 recipe: target attention, 16 slices at 224 px, Adam with head LR 1e-3, 5 epochs,
epoch 5 chosen in advance, seed 42, masked soft BCE, same cache, manifest and folds. The byte-pinned
model files are unchanged; the change is applied to the model instance.

## Validation protocol (fixed before results)

Same as E054 ([D034](../docs/DECISIONS.md#d034--cross-validation-protocol-2026-11-17)): epoch 5
of each fold scores its held-out fold; per-fold macro AUC with mean ± sample std, pooled
out-of-fold macro AUC, and per-target AUC against observed LLM labels binarized at 0.5.

**Decision rule.** Adopt layer3 fine-tuning only if the mean macro AUC improves on 0.861 by more
than the E054 fold spread (> 0.003) **and** improves on every fold. A smaller or mixed change counts
as no improvement; the simpler layer4-only recipe stays.

## Execution

- Code: commit `c9552e8`. Fold 2 runs on the local RTX 3060 Laptop
  (`data/private/cv-attention16-layer3/`); folds 0 and 1 run in parallel on a Kaggle T4 x2
  session (`scripts/build_kaggle_cv_kernel.py --folds 0 1 --trainable layer3`). E054 showed that
  the two GPU types give identical results for this recipe.
- Canary (40 studies, local): peak GPU memory 0.56 GB vs. 0.45 GB for the baseline; training
  about 40% slower per study.
- Runtime: local fold 2 trained in 28,714 s (8.0 h) and scored in 676 s. On Kaggle, folds 0 and 1
  trained in 33,225 s and 33,405 s (9.2 h) and scored in about 1,300 s each, all in one session
  (kernel `rsna-knee-attention16-cv-layer3` version 1); the expected second session was not needed.
- Private outputs: `data/private/cv-attention16-layer3/` (predictions for all folds, summary,
  local fold-2 checkpoints, and the Kaggle fold-0 epoch-5 checkpoint and logs in `kaggle-run01/`).

## Results

Macro ROC-AUC **0.864 ± 0.003** (mean ± std over 3 folds; pooled out-of-fold 0.863), against
0.861 ± 0.003 (pooled 0.860) for E054.

| Fold | E054 (layer4) | E055 (layer3 + layer4) | Change |
|---|---:|---:|---:|
| 0 | 0.8643 | 0.8671 | +0.0028 |
| 1 | 0.8579 | 0.8622 | +0.0043 |
| 2 | 0.8604 | 0.8634 | +0.0030 |
| **Mean** | 0.8608 | **0.8642** | **+0.0034** |

Per target (mean over folds; full table in [RESULTS.md](../docs/RESULTS.md)):

| Target | E054 | E055 | Change |
|---|---:|---:|---:|
| ACL | 0.825 | 0.832 | +0.006 |
| MCL | 0.805 | 0.810 | +0.005 |
| Medial Meniscus | 0.868 | 0.870 | +0.003 |
| Lateral Meniscus | 0.823 | 0.827 | +0.005 |
| Medial OA | 0.883 | 0.885 | +0.002 |
| Lateral OA | 0.861 | 0.863 | +0.002 |
| PF OA | 0.840 | 0.843 | +0.003 |
| Effusion | 0.839 | 0.844 | +0.005 |
| Synovitis | 0.917 | 0.914 | −0.003 |
| Baker's | 0.887 | 0.901 | +0.015 |
| Contusion | 0.883 | 0.882 | −0.001 |
| Fracture | 0.901 | 0.900 | −0.001 |

## Conclusion

**Adopt.** The mean macro AUC rises by 0.0034, above the 0.003 threshold, and every fold improves,
so the pre-registered rule is met. The margin is small: it is about one fold standard deviation,
and the rule passes by only 0.0004. As predicted, the weakest fine-detail targets gain (ACL, MCL,
Lateral Meniscus +0.005 to +0.006); the largest gain is Baker's cyst (+0.015). Synovitis,
Contusion and Fracture change by −0.001 to −0.003, within their fold spread.

Cost: about 30–40% more training time per fold and 0.1 GB more GPU memory; inference is
unchanged. The fold-0 model was submitted as [E056](E056_attention16_layer3_submission.md).

## Gold-58 check (2026-11-20)

Scored later on the 58 expert-labelled Gold studies, as declared in
[E058](E058_attention16_seventh_place_labels.md) under
[D037](../docs/DECISIONS.md#d037--gold-58-for-expert-evaluation-and-final-training-2026-11-20).
No fold model saw Gold-58.

| | Fold 0 | Fold 1 | Fold 2 | Ensemble (mean logits) |
|---|---:|---:|---:|---:|
| E054 (layer4) | 0.858 | 0.860 | 0.857 | 0.865 |
| E055 (layer3 + layer4) | 0.869 | 0.858 | 0.862 | **0.871** |

Layer3 fine-tuning also wins on expert labels (+0.006 for the ensemble). The fold-0 models agree
with the Kaggle private scores of the same models: 0.858 vs 0.862 (E051) and 0.869 vs 0.868 (E056).
With 58 studies this is noisy, but the direction is consistent with CV and the leaderboard.
