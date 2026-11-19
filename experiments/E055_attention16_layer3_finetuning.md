# E055 — Fine-tune backbone layer3 as well as layer4

- **Date:** 2026-11-19
- **Status:** running
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
  about 40% slower per study. A Kaggle fold is expected to exceed one 11.25-hour session and
  resume in a second one.

## Results

Pending.

## Conclusion

Pending.
