# E058 — Train on the 7th-place image-and-report labels

- **Date:** 2026-11-20
- **Status:** running
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
fixed, seed 42, same cache and folds. Option `--labels` (commit after `ce5a335`) swaps the targets
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

**Decision rule.** Adopt the 7th-place labels if the Gold-58 ensemble macro AUC of E058 exceeds
E055's **and** the private Kaggle score of the E058 ensemble exceeds E057's. If the two disagree,
the result is inconclusive.

## Execution

Folds 0 and 1 on a Kaggle T4 x2 session, fold 2 on the local RTX 3060 Laptop, as in E055.

## Results

Pending.

## Conclusion

Pending.
