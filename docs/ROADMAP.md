# Roadmap

## Done

| Area | Outcome |
|---|---|
| Competition contract | Rules, metric, submission format and offline-inference constraints documented ([COMPETITION.md](COMPETITION.md)). |
| Data understanding | Metadata inventory, DICOM geometry characterization and Gold-58 isolation ([DATA.md](DATA.md)). |
| Labels | Public LLM report labels adopted as masked soft targets. Own LLM-labeling clients built and piloted, but not used for the final model. |
| Validation design | 3-fold report-hash-grouped multilabel split. Gold-58 excluded. |
| Data layer | Verified 16-slice tensor cache for all 4,298 studies (61 GB). |
| Modeling | Progression from ResNet-18 (0.620 private) to RadImageNet ResNet-50 with target attention and 16 slices (0.868 private with layer3 fine-tuning, E056; 0.877 with the 3-fold ensemble, E057; 0.919 with the 7th-place labels, E058). |
| Inference | Offline, pinned Kaggle notebook package. |
| Baseline (E054) | 3-fold CV macro ROC-AUC 0.861 ± 0.003; see [RESULTS.md](RESULTS.md). |
| Layer3 fine-tuning (E055) | 3-fold CV macro ROC-AUC 0.864 ± 0.003, better on every fold; adopted ([D036](DECISIONS.md#d036--fine-tune-layer3-and-layer4-2026-11-19)). |
| 7th-place labels (E058) | Gold-58 ensemble macro ROC-AUC 0.897 vs 0.871; Kaggle private 0.919 vs 0.877; adopted ([D038](DECISIONS.md#d038--train-on-the-7th-place-labels-2026-11-21)). |
| Compute | Folds run in parallel on Kaggle T4 x2 sessions and the local GPU (`scripts/build_kaggle_cv_kernel.py`). |

## Next

1. **Final model.** Train on all data with the 7th-place labels (adopted in
   [D038](DECISIONS.md#d038--train-on-the-7th-place-labels-2026-11-21)): the 4,298 cohort studies,
   the 51 extra labelled studies and Gold-58 with its expert labels, as D037 allows.
2. **Controlled ablations on CV.** Change one variable at a time against E055: slice count,
   pooling, class weighting and epochs.
3. **Ensembling.** The three fold models are averaged at inference (E057, E058). Next: add seeds.
4. **Self-contained submission build.** Build the inference archive only from repository sources.
