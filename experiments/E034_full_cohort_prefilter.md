# E034 — Prepare the full non-Gold soft-label cohort

## Hypothesis and scope

The pinned original Steven v1 probability file can support a full non-Gold metadata prefilter while preserving all partial labels, grouping normalized duplicate reports, and checking class support in deterministic folds. This is a preparation artifact only: no model training, image loading, image-quality assessment, training-core change, or model comparison occurred.

## Baseline and changed variables

- Baseline C2 cohort and fold files were byte-pinned and rechecked before/after preparation: cohort SHA-256 `e88c61c22a463737f16951b7c76db03834034aecd1b31dc61b4ce495080b2be2`; folds SHA-256 `b82b376b0086201e3364b11a0203b62890e628442075a52ea4a9e09ee0e778ac`.
- Source: Steven report-label dataset Version 6, original `llm_labels_full.csv` only; SHA-256 `96444cd3c7d396bae4a71454317d9cbe0689759b5278b30939781e04d0774537`.
- Gold-58 exclusion file SHA-256 `e0e584d475a7e8b0749ffcce70267ac19a2e2549338f98f50ed515f8fabce3a1`; the CSV reader materializes row fields, but those 58 rows were excluded before probability numeric conversion or report normalization/hashing/use.
- Official inventory: 4,407 studies; 4,349 non-Gold studies; 24,035 non-Gold series.
- No training variable changed. No new labels, label calls, expert evidence, or Gold-derived metrics were added.

## Frozen prefilter and validation protocol

1. Validate the source, Gold list and baseline C2/fold byte pins. Verify exact source coverage against the official study-ID inventory, then exclude exact Gold IDs before numeric probability conversion or report normalization/hashing/use. The standard CSV reader materializes row fields; no claim is made that those raw CSV fields were never decoded.
2. Require complete, unique series metadata for every non-Gold study. Metadata eligibility is not image eligibility; image existence, decoding and visual quality were not inspected.
3. Preserve every original probability and mask; exactly `0.5` is unknown. Retain studies with at least one observed target; record all-unknown rows separately with `all_targets_unknown` rather than mapping unknown to negative.
4. Build 24 group-level incidence columns (observed positive and negative per target); unknowns contribute neither. Group using the existing case-fold/collapse-whitespace report SHA-256 proxy, then run the existing deterministic iterative multilabel allocator with 3 splits and seed 42.
5. Require positive and negative support for all 12 targets in every validation fold and its complementary training partition. Compute weights from each training partition only: `w_c = max(1, min(3, sqrt(max(n_pos,n_neg) / n_c)))`. Validation remains unweighted and unresampled. This capped formula is an explicit heuristic, not an empirically validated training choice.
6. Record the intended future masked soft-target loss, but do not connect the overlay to the existing integer-casting reader or run training.

## Results

The source join yielded **4,349** non-Gold studies. **51** had all 12 targets unknown and were quarantined from supervised fold membership; **4,298** studies with at least one addressed target were retained, with partial masks preserved. Their normalized-report grouping produced **4,175** groups. All 24,035 series metadata rows were present and validated. This establishes metadata eligibility only, not usable pixels.

Fold sizes are study counts; group counts confirm the partition was made at report-group level.

| Fold | Training studies / groups | Validation studies / groups |
|---:|---:|---:|
| 0 | 2,850 / 2,796 | 1,448 / 1,379 |
| 1 | 2,865 / 2,780 | 1,433 / 1,395 |
| 2 | 2,881 / 2,774 | 1,417 / 1,401 |

The following ranges are the min–max support across the three folds. Prevalence and split sizes are not treated as quality measures.

| Target | Addressed / 4,349 | Unknown | Validation positive | Validation negative | Training positive | Training negative | Positive weight | Negative weight |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ACL | 3,987 | 362 | 317–319 | 988–1,030 | 635–637 | 2,003–2,045 | 1.773–1.793 | 1.000 |
| Baker's | 2,256 | 2,093 | 357–360 | 381–404 | 715–718 | 777–800 | 1.042–1.056 | 1.000 |
| Contusion | 3,304 | 1,045 | 360 | 726–770 | 720 | 1,454–1,498 | 1.421–1.442 | 1.000 |
| Effusion | 3,919 | 430 | 867–870 | 416–465 | 1,736–1,739 | 848–897 | 1.000 | 1.392–1.431 |
| Fracture | 2,482 | 1,867 | 105 | 708–748 | 210 | 1,419–1,459 | 2.599–2.636 | 1.000 |
| Lateral Meniscus | 3,917 | 432 | 372 | 919–949 | 744 | 1,852–1,882 | 1.578–1.590 | 1.000 |
| Lateral OA | 3,280 | 1,069 | 419–421 | 659–682 | 838–840 | 1,339–1,362 | 1.263–1.273 | 1.000 |
| MCL | 3,912 | 437 | 224–230 | 1,053–1,104 | 449–455 | 2,129–2,180 | 2.177–2.191 | 1.000 |
| Medial Meniscus | 4,108 | 241 | 772–782 | 583–607 | 1,548–1,558 | 1,171–1,195 | 1.000 | 1.138–1.153 |
| Medial OA | 3,469 | 880 | 548–550 | 582–634 | 1,098–1,100 | 1,187–1,239 | 1.039–1.062 | 1.000 |
| PF OA | 3,600 | 749 | 673–676 | 513–544 | 1,346–1,349 | 1,034–1,065 | 1.000 | 1.125–1.142 |
| Synovitis | 701 | 3,648 | 183–185 | 42–60 | 366–368 | 90–108 | 1.000 | 1.841–2.022 |

Weights above are the proposed fold-specific training-only heuristics, not trained or performance-tested values. Synovitis has particularly low coverage; every target nevertheless had positive and negative observed support in each validation and training partition.

## Private artifacts and input pins

All row-level IDs, report hashes, probabilities, masks, assignments, per-fold weights and audit reasons are under ignored `data/private/full-cohort-prefilter/run01/` only.

- `full-cohort-soft-label-manifest.json`: 6,883,212 bytes; SHA-256 `617ea1c267960b1e903b332a78388db3b6ab80d44374721df5a4c5e184e8961b`.
- `full-cohort-grouped-folds.json`: 1,636,413 bytes; SHA-256 `b816c7902a5d0caa68a9e818d3f913943bc61cd2e6db65c4de069f5cd94e15f9`.
- `aggregate-summary.json`: 4,676 bytes; SHA-256 `6f90ab99c7eef4af73ab1574aca790fad364fe2e4a67bdd5e7ce173aef898513`.
- Official `train.csv` SHA-256 `8ca2203c0e9d61c080c7a314c7cdb51c1b03a1d9eb4770819f7f34af53ef4e33`.
- Official `train_series.csv` SHA-256 `573c1d80772bf41211c91b149c95677385a1c22d63f485c347f1b46c0177aef3`.
- Prefilter policy module SHA-256 at run time `7a1b41d3d96c137859d83d78927fd020afba8ce4034cb0addbed67e730280662`.
- Baseline and Gold hashes are as listed above and are also bound inside the private manifest.

The aggregate CLI output contains counts, target coverage, and per-fold class supports only; it does not print row IDs or reports. The existing C2 source and fold pins were rechecked after preparation. The bundle is staged first, written only to a fresh private directory, then checked for exact file inventory and complete byte readback; directory rename was denied by the Windows runtime, so atomic directory publication is not claimed. Failed output writes trigger cleanup only of the directory created by that invocation. Independent verification confirmed all manifest records, groups, series, fold allocation/support, weight formulas, privacy, and output/source pins; the old C2/folds and protected HSR notebook ZIP remained unchanged.

## Verification and limits

- RED observed before implementation: the focused test import failed because the module did not yet exist. GREEN: the focused synthetic suite passed 8 tests.
- Required regression command: `PYTHONPATH=src .venv-training/Scripts/python.exe -m unittest tests.test_full_cohort_prefilter tests.test_report_label_source_comparison tests.test_training_folds tests.test_training_cohort tests.test_training_dataset_diagnostics` — **31 passed, 1 skipped** (the existing Windows symlink case).
- `py_compile` passed for the module and CLI. `git diff --check` passed; only pre-existing LF/CRLF notices appeared. `.git` remained read-only; no commit/branch was created.
- The CLI completed locally in about 1.8 seconds with `preparation_only_not_trained`. No network, Kaggle, image load, DICOM, checkpoint, new LLM call, expert adjudication, or training occurred.
- The normalized report hash only reduces duplicate-report leakage risk; it does not prove patient-level or site-level isolation. Image availability and image quality remain unknown. Label provenance/model/prompt remain undisclosed, and the balance formula is heuristic.
- Independent parent verification: PASS for the checks listed above; parent spot-check: 8 focused tests passed. The exact applicable suite ran 31 tests with 1 existing Windows symlink skip; compile and diff-check passed. No broad discovery suite was run. This stage is not a trained model comparison or model-performance claim.
- The Gold boundary is precise: CSV row fields are materialized by the standard reader, but the excluded rows do not undergo probability numeric conversion or report normalization/hashing/use. No Gold outcome metrics were calculated.
