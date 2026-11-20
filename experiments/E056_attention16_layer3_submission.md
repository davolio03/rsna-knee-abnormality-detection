# E056 — Late submission of the E055 layer3 fold-0 model

- **Date:** 2026-11-19
- **Status:** completed — public 0.891 / private 0.868
- **Parent:** E055 (layer3 fine-tuning), E051 (submission package)

## Question

Does the layer3 gain seen in cross-validation (E055, +0.003 macro AUC) carry over to the Kaggle
hidden test? The fold-0 model is the direct counterpart of the E050 model behind E051: same
training studies (2,850), same recipe, only the trainable backbone differs.

Expectation, fixed before the score: the CV gain is smaller than typical leaderboard noise, so the
score alone can neither confirm nor reject E055. CV stays the primary evidence.

## Package

- Source checkpoint: E055 fold 0, epoch 5, trained on Kaggle (SHA-256 `25f836ed…aed8bf`, from
  `data/private/cv-attention16-layer3/kaggle-run01/fold0/target_attention/`).
- `scripts/build_attention16_weight_swap.py` copies the E051 `run02` model export and sanitized
  report and replaces only the model tensors (35 of 324 tensors change: layer3, layer4 and the
  attention heads). The E051 inference code is byte-identical, so the export header still names
  the E050 source checkpoint; this record is the authoritative provenance. The export reloads
  strictly with the E051 loader and reproduces the source weights.
- Export: `data/private/attention-sixteen-submission/layer3-fold0-run01/`, SHA-256
  `f4fa4e24…1391d7`, 94,646,113 bytes; uploaded as the private dataset
  `davidsulleiro/rsna-knee-attention16-layer3-fold0`.
- Notebook: the E051 `run02` full-inference notebook with only `SOURCE_ROOT` and
  `CHECKPOINT_SHA256` changed (`data/private/attention-sixteen-submission/layer3-fold0-kernel/`),
  pushed as the private kernel `rsna-knee-attention16-layer3-fold0-submission`. Inputs: the
  competition data, the model dataset and the offline DICOM codec dataset. Internet off, T4.

## Execution

Kernel version 1 completed: 3/3 visible test studies, uncapped, no skipped series or slices, root
`submission.csv` written. Submitted from that version with the Kaggle CLI on 2026-11-19 (late
submission, reference 58582253).

## Results

| Submission | Public | Private |
|---|---:|---:|
| E051 (layer4, fold 0) | 0.885 | 0.862 |
| E056 (layer3 + layer4, fold 0) | **0.891** | **0.868** |
| Change | +0.006 | +0.006 |

## Conclusion

**Consistent with E055.** Both leaderboard splits improve by 0.006, the same direction as the CV
gain (+0.003 on fold 0, +0.0034 mean). This is the best late score so far. One submission cannot
separate the gain from leaderboard noise, so the evidence for layer3 fine-tuning is still the
cross-validation in E055; the leaderboard result does not contradict it. The E056 model replaces
E051 as the current model.
