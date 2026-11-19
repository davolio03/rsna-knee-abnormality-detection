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
| Baseline (E054) | 3-fold CV macro ROC-AUC 0.861 ± 0.003; see [RESULTS.md](RESULTS.md). |
| Layer3 fine-tuning (E055) | 3-fold CV macro ROC-AUC 0.864 ± 0.003, better on every fold; adopted ([D036](DECISIONS.md#d036--fine-tune-layer3-and-layer4-2026-11-19)). |
| Compute | Folds run in parallel on Kaggle T4 x2 sessions and the local GPU (`scripts/build_kaggle_cv_kernel.py`). |

## Next

1. **Label quality.** Finish the blinded E053 review. Then decide whether a Gold-58 evaluation of
   the label source is worth changing [D004](DECISIONS.md#d004--gold-58-is-reserved).
2. **Controlled ablations on CV.** Change one variable at a time against E055: slice count,
   pooling, class weighting and epochs.
3. **Ensembling.** Average the three fold models at inference, which costs no extra training.
   Then add seeds.
4. **Self-contained submission build.** Build the inference archive only from repository sources.
