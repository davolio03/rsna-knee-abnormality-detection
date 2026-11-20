# E057 — Late submission of the 3-fold E055 ensemble

- **Date:** 2026-11-20
- **Status:** completed
- **Parent:** E055 (layer3 fold models), E056 (single fold-0 submission)

## Question

Does averaging the three E055 fold models beat the single fold-0 model (E056: public 0.891 /
private 0.868)? The three models are already trained; each saw two thirds of the studies, and
together they cover all 4,298 training studies.

Expectation, fixed before the score: a small gain from lower variance, of the order of the
leaderboard noise. No local estimate exists, because every study is in the training set of two of
the three members; the leaderboard is the only test.

## Package

- Members: the E055 epoch-5 models of folds 0, 1 and 2, each exported with
  `scripts/build_attention16_weight_swap.py` into the E051 `run02` export format (94,646,113
  bytes each):

  | Fold | Trained on | Export SHA-256 | Private dataset |
  |---|---|---|---|
  | 0 | Kaggle T4 | `f4fa4e24…1391d7` | `rsna-knee-attention16-layer3-fold0` |
  | 1 | Kaggle T4 | `7bd7bc45…9b5a2c` | `rsna-knee-attention16-layer3-fold1` |
  | 2 | RTX 3060 Laptop | `a06dc46d…5a9341` | `rsna-knee-attention16-layer3-fold2` |

- Notebook: `scripts/build_attention16_ensemble_notebook.py` adds one block to the E056 notebook.
  It wraps `attention_sixteen.load_attention16_model` so the pinned loader loads all three
  members, each with its own SHA-256 and size check, and returns a module that averages their
  logits. The rest of the E051 inference code is byte-identical; each study is preprocessed once.
- A local check loaded the three real exports through the wrapper and confirmed that the output
  equals the mean of the members' logits. `tests/test_attention16_ensemble_notebook.py` covers
  the wrapper with stand-in models.

## Execution

Private kernel `rsna-knee-attention16-layer3-ensemble-submission`, version 1: three members loaded,
3/3 visible test studies, no skipped series or slices, 34 s (E056: about 30 s). Submitted from that
version with the Kaggle CLI on 2026-11-20 (late submission, reference 58606738).

## Results

| Submission | Public | Private |
|---|---:|---:|
| E051 (layer4, fold 0) | 0.885 | 0.862 |
| E056 (layer3, fold 0) | 0.891 | 0.868 |
| E057 (layer3, 3-fold ensemble) | **0.902** | **0.877** |

## Conclusion

The ensemble beats the single fold-0 model by +0.011 public and +0.009 private, more than the
expected noise-sized gain and more than layer3 fine-tuning itself added (+0.006). It matches the
Gold-58 check in E055, where the E055 ensemble also beat every member. The 3-fold ensemble becomes
the reference submission; later submissions are compared with it (E058).
