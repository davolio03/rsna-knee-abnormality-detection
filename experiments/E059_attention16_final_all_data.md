# E059 — Final models on all labelled data

- **Date:** 2026-11-22
- **Status:** running
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

Seeds 42 and 43 run one after the other on the local RTX 3060 Laptop (about 11 h each). Seeds 44
and 45 run in parallel on a Kaggle T4 x2 session (about 12 h each; the 11.25 h session limit
needs one resumed session).

## Results

Pending.

## Conclusion

Pending.
