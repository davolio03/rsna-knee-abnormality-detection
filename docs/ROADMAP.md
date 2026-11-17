# Roadmap

## Done

| Area | Outcome |
|---|---|
| Competition contract | Rules, metric, submission format and offline-inference constraints documented ([COMPETITION.md](COMPETITION.md)). |
| Data understanding | Metadata inventory, DICOM geometry characterization and Gold-58 isolation ([DATA.md](DATA.md)). |
| Labels | Public LLM report labels adopted as masked soft targets. Own LLM-labeling clients built and piloted, but not used for the final model. |
| Validation design | 3-fold report-hash-grouped multilabel split. Gold-58 excluded. |
| Data layer | Verified 16-slice tensor cache for all 4,298 studies (61 GB). |
| Modeling | Progression from ResNet-18 (0.620 private) to RadImageNet ResNet-50 with target attention and 16 slices (0.862 private). |
| Inference | Offline, pinned Kaggle notebook package. |

## Next

1. **Cross-validated baseline (E054).** Run the current recipe on all 3 folds and report
   per-target ROC-AUC with mean ± std. Every later change is measured against this number.
2. **Label quality.** Finish the blinded E053 review. Then decide whether a Gold-58 evaluation of
   the label source is worth changing [D004](DECISIONS.md#d004--gold-58-is-reserved).
3. **Controlled ablations on CV.** Change one variable at a time: slice count, pooling,
   backbone trainability, class weighting and epochs.
4. **Ensembling.** Average the three fold models at inference, which costs no extra training.
   Then add seeds.
5. **Self-contained submission build.** Build the inference archive only from repository sources.
