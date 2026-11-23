# E058 — Train on the 7th-place image-and-report labels

- **Date:** 2026-11-20
- **Status:** completed; adopted (D038)
- **Parent:** E055 (layer3 fine-tuning, cached report labels)

## Hypothesis

Top teams' main gain came from training labels that an LLM read from the images as well as the
report. The 7th-place labels score 0.952 macro AUC on Gold-58 (their author's measurement); public
report-only labels like ours score about 0.90. Training the E055 recipe on them should raise the
AUC against expert labels.

## Changed variable

Only the training targets: `songqizhou/rsna-knee-llm-labels`, file `8_final_label.csv`
(CC BY-NC-SA 4.0; 4,349 non-Gold training studies, 12 soft targets per study, no unobserved
cells; SHA-256 `fce2e4e4…e26532`). Half of each value is the author's LLM label and half the
out-of-fold prediction of their image models. It covers all 4,298 studies of our cohort.
Everything else is E055: layer3 + layer4, target attention, 16 slices at 224 px, 5 epochs, epoch 5
fixed, seed 42, same cache and folds. Option `--labels` (commit after `a68bcc3`) swaps the targets
in training only; held-out folds keep the cached report labels.

## Pre-analysis (before training)

Scored without training against these labels (binarized at 0.5, all cells), the existing OOF
predictions give E054 0.917 and E055 0.921 macro AUC; E055 is better on every fold (+0.004). Our
report labels mark far more positives than these labels: Synovitis 79% vs 16% of observed cells,
Lateral OA 38% vs 8%, PF OA 56% vs 22%, while ranking the studies similarly (AUC 0.89–0.98 between
the two sources). Our labels leave 24.5% of cells unobserved, including 84% of Synovitis cells.

## Evaluation (fixed before results)

1. **Primary — Gold-58 (expert labels, D037).** The ensemble of the three fold models (mean
   logits, as in E057) is scored on the 58 Gold studies, for E058 and for E055. No fold model saw
   Gold-58. Per-fold Gold-58 AUCs are reported too.
2. **Secondary — CV.** OOF macro AUC against the 7th-place labels and against our cached report
   labels. A drop against our report labels is expected and is not a criterion.
3. **External — Kaggle.** Late submission of the E058 ensemble, compared with E057.

E054 is scored on Gold-58 as well, to check the E055 adoption against expert labels. That check
is informative only; it does not change D036.

Gold-58 samples come from a separate cache built with the same preprocessing
(`scripts/build_extra_cache_kernel.py`, Kaggle CPU, 111 studies: 58 Gold, 51 training studies
outside our cohort that the 7th-place labels cover, and 2 controls). Both controls matched the
existing cache array for array.

**Decision rule.** Adopt the 7th-place labels if the Gold-58 ensemble macro AUC of E058 exceeds
E055's **and** the private Kaggle score of the E058 ensemble exceeds E057's. If the two disagree,
the result is inconclusive.

## Execution

Folds 0 and 1 on a Kaggle T4 x2 session, fold 2 on the local RTX 3060 Laptop, as in E055.

Runtime: Kaggle folds 0 and 1 took 9.3 h of training and 21 min of prediction each (9.7 h wall
time for the session); local fold 2 took 8.1 h of training and 11 min of prediction.

## Results

**Primary — Gold-58 (58 studies, expert labels), macro AUC.**

| Run | Fold 0 | Fold 1 | Fold 2 | Ensemble |
|---|---:|---:|---:|---:|
| E055 (report labels) | 0.869 | 0.858 | 0.861 | 0.871 |
| E058 (7th-place labels) | 0.890 | 0.895 | 0.896 | **0.897** |

Every E058 fold model beats every E055 fold model. Per target (ensembles), E058 gains on 8 of 12
targets, most on Synovitis (+0.084, 0.749 → 0.833), Effusion (+0.073), MCL (+0.057) and Medial
Meniscus (+0.053), and loses on PF OA (−0.030), Lateral OA (−0.027), Lateral Meniscus (−0.021)
and Baker's (−0.011). With 58 studies, single-target differences are noisy.

**Secondary — CV (OOF macro AUC, mean of folds).**

| Labels used for scoring | E055 | E058 |
|---|---:|---:|
| Our cached report labels (observed cells) | 0.864 | 0.844 |
| 7th-place labels (all cells, binarized at 0.5) | 0.921 | 0.966 |

As expected, E058 scores lower against the labels it was not trained on and higher against the
ones it was trained on, so CV cannot settle the question; it does show that the models disagree
with our report labels mostly where the label sources disagree.

**External — Kaggle.** Package: `scripts/build_attention16_ensemble_notebook.py --experiment E058`;
private datasets `rsna-knee-attention16-labels7-fold{0,1,2}`, export SHA-256 `a85e7600…62a471`,
`1a6f7823…3d9a7b`, `4d24d68a…f8aaa2`. The first push (2026-11-20) was refused: the weekly 30 h GPU
quota was used up by the training runs. On 2026-11-21 kernel
`rsna-knee-attention16-labels7-ensemble-submission` failed twice (versions 1 and 2) in the
bootstrap, 8 s after start, with the generic `bootstrap_failed` code. A copy of the same notebook
in a new kernel, `rsna-knee-labels7-bootstrap-diag` version 4, whose only change prints the
traceback in the bootstrap error branch, ran cleanly: three members loaded, 3/3 visible studies, no
skipped series or slices, 36 s, model pin `a85e7600…`. The cause of the original kernel's failure
is unknown. The late submission was made from that version (reference 58686652).

| Submission | Public | Private |
|---|---:|---:|
| E057 (E055 ensemble, report labels) | 0.902 | 0.877 |
| **E058 (ensemble, 7th-place labels)** | **0.932** | **0.919** |

## Conclusion

Both criteria of the decision rule are met: Gold-58 ensemble macro AUC 0.897 vs 0.871, and Kaggle
private 0.919 vs 0.877 (+0.042; public +0.030). The change of training labels is the largest gain
in the project so far, bigger than every architecture change since E049. Gold-58 predicted the
direction, and it underestimated the size of the gain. The 7th-place labels are adopted as the
training targets (D038).

Attribution: the labels are by songqizhou, Kaggle dataset `songqizhou/rsna-knee-llm-labels`,
licensed CC BY-NC-SA 4.0. Models trained on them inherit the non-commercial, share-alike terms.
