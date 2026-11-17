# E033 — Compare report-extracted labels with frozen C2 labels

## Hypothesis

The original Steven v1 report-label probabilities can be joined to the frozen C2 training cohort for a descriptive, paired label-source comparison without changing the cohort, folds, training reader, model, or Gold-58 policy.

## Baseline and changed variable

- Baseline: the pinned 500-study C2 cohort, 12 binary targets, and frozen grouped folds (validation counts 167 / 163 / 170).
- Changed variable: comparison reference only—Steven's dataset Version 6 original `llm_labels_full.csv`; no training variable changed.
- The other dataset files (v2 and v4 blend) were not extracted or analyzed.

## Frozen inputs and provenance

- Source card: https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels, Version 6.
- Downloaded archive: 426,025 bytes; SHA-256 `573f11c299a30b15c13f0f19368543f2d0ee81810809541b188ed83b359a9aa0`.
- Selected source file: `llm_labels_full.csv`, 534,442 bytes; SHA-256 `96444cd3c7d396bae4a71454317d9cbe0689759b5278b30939781e04d0774537`.
- C2 cohort manifest: `data/private/500-study-training-cohort.json`, SHA-256 `e88c61c22a463737f16951b7c76db03834034aecd1b31dc61b4ce495080b2be2`.
- Frozen folds: `data/private/500-study-training-folds.json`, SHA-256 `b82b376b0086201e3364b11a0203b62890e628442075a52ea4a9e09ee0e778ac`.
- Gold exclusion inventory: 58 study IDs, SHA-256 `e0e584d475a7e8b0749ffcce70267ac19a2e2549338f98f50ed515f8fabce3a1`; excluded before parsing source probabilities. No Gold labels or metrics were used.
- The card describes `0.5` as “not addressed”; exact label-generation model and prompt are undisclosed. The author-reported 0.878 Gold AUC is unverified and was not computed or used. Card CC0 metadata is not a legal determination and does not supersede competition terms.
- Private staged provenance JSON: 959 bytes; SHA-256 `75949376c349c324dd1d95d0628ef3d1b90aafb3b800cbfb0ece27f562d746fe`.

## Protocol

1. Verify archive, selected CSV, cohort, fold, and Gold-ID byte pins and exact target order.
2. Read only `StudyInstanceUID` from the official training CSV to establish its 4,407-study inventory. Require the source to cover that exact inventory once each; exclude the exact 58 Gold IDs before parsing probabilities.
3. Compare only the fixed 500 C2 studies. Treat `p = 0.5` as masked/not addressed; for descriptive confusion and agreement, define public positive as `p > 0.5`. Report coverage, C2/public prevalence on the same addressed subset, confusion counts, agreement, and mean absolute probability difference, overall and by frozen fold.
4. Do not interpret either source as ground truth; do not compute AUC or choose a source from these paired comparisons.
5. Write raw probability/mask rows and study keys only to ignored private output. Keep the future training-comparison plan separate: same cohort/folds/model and C2 binary AUC plus masked Steven soft-target BCE/Brier; this is prepared, not implemented or run.

## Results

The source covered 4,407 / 4,407 official studies; 58 Gold rows were excluded before probability parsing, leaving 4,349 non-Gold rows. All 500 C2 studies joined; frozen fold membership remained 167 / 163 / 170. The table reports agreement and probability distance only on addressed paired studies; coverage varies substantially and is especially low for Synovitis. These are descriptive label-source comparisons, not accuracy estimates.

Across the 6,000 target-study cells in the paired cohort, 4,473 (74.55%) were addressed and 1,527 (25.45%) were masked. Micro agreement over addressed cells was 0.954 (TP 1,587; FP 164; FN 42; TN 2,680); this pools targets with different prevalences and coverage and is not a quality score.

| Target | Addressed / 500 | Not addressed | Agreement | Mean absolute probability difference | C2 prevalence* | Public binary prevalence* |
|---|---:|---:|---:|---:|---:|---:|
| ACL | 464 | 36 | 0.931 | 0.120 | 0.289 | 0.241 |
| Baker's | 254 | 246 | 1.000 | 0.081 | 0.437 | 0.437 |
| Contusion | 376 | 124 | 0.856 | 0.173 | 0.176 | 0.314 |
| Effusion | 460 | 40 | 0.996 | 0.108 | 0.635 | 0.630 |
| Fracture | 264 | 236 | 0.981 | 0.066 | 0.091 | 0.110 |
| Lateral Meniscus | 446 | 54 | 0.975 | 0.084 | 0.229 | 0.253 |
| Lateral OA | 383 | 117 | 0.914 | 0.133 | 0.313 | 0.394 |
| MCL | 444 | 56 | 0.982 | 0.079 | 0.176 | 0.176 |
| Medial Meniscus | 473 | 27 | 0.956 | 0.107 | 0.554 | 0.577 |
| Medial OA | 408 | 92 | 0.944 | 0.115 | 0.407 | 0.453 |
| PF OA | 418 | 82 | 0.959 | 0.107 | 0.495 | 0.536 |
| Synovitis | 83 | 417 | 1.000 | 0.095 | 0.807 | 0.807 |

\*Prevalence is computed only over the addressed subset (`p != 0.5`); undefined when no rows are addressed. Public binary label uses the predeclared `p > 0.5` threshold.

## Artifacts and verification

- Private source staging and provenance: `data/private/report-label-source-comparison/steven-v6/` (ignored).
- Private aggregate, ID-keyed soft-target overlay, and future comparison plan: `data/private/report-label-source-comparison/results-v1/` (ignored).
- Implementation: `src/rsna_knee/labels/report_source_comparison.py`; CLI: `scripts/compare_report_label_sources.py`.
- Synthetic regression suite: `tests/test_report_label_source_comparison.py` (6 tests passed).
- The existing training dataset reader, C2 manifest/folds, model, and training metrics were not changed. No training, Kaggle operation, DICOM load, checkpoint load, Gold metric, or model-performance conclusion was produced.
- Private result artifacts: aggregate summary 19,399 bytes / SHA-256 `529052d87cb3c009a8f5da28ba53b02d3e83b91ba72bf3b63e53015ed9b23056`; ID-keyed soft-label overlay 260,046 bytes / SHA-256 `5eb0c385eb6fa95a060d8e550e7ea806009378bb1336d994f8c26da40e9e07cc`; future comparison plan 1,175 bytes / SHA-256 `77dfc466ec33c861c6169ce1dde1fbe3708f69e6d91a4300629341b918b7c94e`.
- Independent verification recomputed the non-Gold aggregates from the approved source and matched 4,473 addressed / 1,527 masked cells, TP 1,587 / FP 164 / FN 42 / TN 2,680, agreement `4267/4473 = 0.9539458976`, all per-target/fold counts, all pins, private output hashes, and unchanged C2/fold/HSR12 bytes. The independent command passed 23 tests with 1 Windows symlink skip; parent spot-check passed 6 comparison tests (0.094s). A separate broader 65-test run including subscription-label tests remains unclassified (2 failures, 3 errors, 3 skips), with no baseline comparison; do not attribute those failures to this change.

## Conclusion and limits

The exact source-to-official-inventory join and 500-study paired comparison are independently verified under pinned inputs. Agreement is not ground truth, observed probabilities may have undisclosed provenance, and per-target coverage differences make aggregate agreement unsuitable as a source-quality ranking. The controlled model comparison has not been trained; any later adapter/training work must be separately scoped and authorized. This stage's `training_authorized_in_this_task: false` is not a permanent prohibition. The current integer-casting reader must not consume this overlay.
