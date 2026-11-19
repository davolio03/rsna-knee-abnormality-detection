# Project State

- **Last updated:** 2026-11-18
- **Phase:** Post-competition consolidation. The competition closed on 2026-10-22, and every model
  submission was scored afterwards as a late submission.
- **Current model:** RadImageNet ResNet-50 with target-specific slice and series attention, 16
  slices per series, trained with masked soft-label BCE for 5 epochs
  ([E050](../experiments/E050_local_sixteen_slice_attention_preparation.md),
  [E051](../experiments/E051_attention16_model_submission.md)).
- **Best Kaggle score (late):** public 0.885 / private 0.862.
- **Validation:** 3-fold study-grouped cross-validation, using ROC-AUC against observed
  report-derived labels binarized at 0.5 ([E054](../experiments/E054_attention16_cross_validation.md)).
  Baseline: **macro AUC 0.861 ± 0.003** (pooled OOF 0.860); see [RESULTS.md](RESULTS.md).
- **Labels:** public LLM report labels (`stevenleehans/rsna-knee-llm-report-labels` v6). These
  are not expert labels.
- **Gold-58:** reserved. It is excluded from training, validation and model selection
  ([D004](DECISIONS.md#d004--gold-58-is-reserved)).

## Open items

1. Run controlled experiments on the CV protocol. Kaggle T4 x2 sessions and the local GPU give
   identical results, so folds can run in parallel on both (`scripts/build_kaggle_cv_kernel.py`).
2. Assess label quality: the qualified blinded review of the E053 disagreement sample is pending.
3. Run controlled ablations on the CV protocol, for example 8 vs. 16 slices and attention vs. mean
   pooling. The leaderboard history so far confounds these changes.
4. Make the submission build self-contained. It currently reuses a frozen source archive from
   `data/private/` for part of the inference code.
5. Ask the organizers how the MIRA license applies to derived labels and weights before
   distributing any artifact.

## Where things are

- Method, results and reproduction steps: [README](../README.md) and [RESULTS.md](RESULTS.md).
- Durable decisions: [DECISIONS.md](DECISIONS.md).
- Full research history, including the detailed decision log and all superseded code: git tag
  `competition-archive-2026-11` and [`experiments/archive/`](../experiments/archive/).
