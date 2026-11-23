# E059 — Final models on all labelled data

- **Date:** 2026-11-22
- **Status:** completed
- **Parent:** E058 (7th-place labels, adopted in D038)

## Question

Does the E058 ensemble improve when its members train on every labelled study instead of two
thirds of the cohort, and when more members are averaged? This is the final submission of the
project, so it changes several things at once; it is not a controlled comparison.

## Changes from E058

1. **Training data:** every labelled study, 4,407 in all, instead of about 2,865 per fold model.

   | Source | Studies | Targets |
   |---|---:|---|
   | Cohort (16-slice cache) | 4,298 | 7th-place labels |
   | Training studies outside the cohort (extra cache) | 51 | 7th-place labels |
   | Gold-58 (extra cache) | 58 | expert labels from `train.csv` (D037) |

2. **Members:** four final models with seeds 42, 43, 44 and 45. The seed sets the head
   initialization and a fixed random study order (`numpy` `default_rng(seed).permutation`); E058
   used the cohort manifest order, which is also a fixed arbitrary order.
3. Everything else is the E058 recipe: layer3 + layer4, target attention, 16 slices at 224 px, 5
   epochs, epoch 5 fixed in advance.

Code: `scripts/cv_attention16.py train-final` (commit `65a5a39`), with `FinalCacheDataset` and
`train_final` in `src/rsna_knee/training/cross_validation.py`. No held-out data exists, so the
per-epoch validation numbers come from 16 training studies and only show that training runs.

## Checks before training

- With the real data, the training set holds 4,407 distinct studies (cohort 4,298, Gold 58,
  extra 51), each with its label. Gold samples carry the expert labels, and seeds 42 and 43 give
  different orders.
- The E055 fold fingerprints are unchanged by the refactor.
- A 6-study run on the local GPU completed 5 epochs. Resuming from epoch 3 gave weights identical
  to an uninterrupted run (maximum difference 0.0).

## Evaluation (fixed before results)

Gold-58 cannot be used: every final model trains on it. The only test is the late Kaggle score.

- **Primary submission:** mean logits of 7 models, the 4 final models and the 3 E058 fold models.
- **Secondary submission:** mean logits of the 4 final models only.
- Both are compared with E058 (public 0.932 / private 0.919). No other variant is submitted, and
  nothing is tuned on the scores.

## Execution

- **Local (RTX 3060 Laptop):** seeds 42 and 43 ran one after the other, 8.6 h and 8.9 h.
- **Kaggle (T4 x2, `rsna-knee-attention16-final`):** seeds 44 and 45 ran in parallel. Version 1
  stopped at the 11.25 h limit after epoch 4; version 2 restored those checkpoints (epoch-4 SHA-256
  unchanged) and trained epoch 5 in 2.4 h. GPU quota used: about 13.7 h.
- All four models trained on 4,407 studies (cohort 4,298, extra 51, Gold 58). Epoch-5 training BCE
  was 0.2584–0.2585 for every seed, and the weights differ between seeds.
- Model-only exports (`scripts/build_attention16_weight_swap.py`) were uploaded as private datasets
  `rsna-knee-attention16-final-seed{42,43,44,45}`. Notebooks were built with
  `scripts/build_attention16_ensemble_notebook.py --experiment E059` (7 members, kernel
  `rsna-knee-attention16-final-ensemble7-submission`) and `--experiment E059-final4` (4 members,
  kernel `rsna-knee-attention16-final4-submission`). Both ran cleanly on the first attempt.

## Results

Late Kaggle submissions (2026-11-22):

| Submission | Members | Public | Private |
|---|---:|---:|---:|
| E058 (reference) | 3 fold models | 0.932 | 0.919 |
| **E059 primary** | 4 final + 3 E058 folds | **0.934** | **0.923** |
| E059 secondary | 4 final | 0.935 | 0.924 |

## Conclusion

Both pre-declared submissions beat E058: the primary by +0.002 public and +0.004 private, the
secondary by +0.003 and +0.005. The gain is small; it combines more training data (all labelled
studies, Gold-58 included), more members and different study orders, so no single cause can be
named. The secondary scores 0.001 above the primary on both splits, which is within noise. Under
the rule fixed before the results, the primary (7-member) ensemble is the final submission of the
project.
