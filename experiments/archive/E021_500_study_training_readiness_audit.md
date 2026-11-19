# E021 - 500-study training-readiness audit

## Outcome

The 500 silver-labeled studies have enough per-target class support to plan an initial classifier pilot, conditional on image access, preprocessing and a leakage-aware split. This audit does **not** establish sufficient sample size for a final classifier or certify clinical label accuracy. No training or additional labeling was performed.

## Objective and baseline

- **Date:** 2026-10-16, Europe/Madrid.
- **Type:** Descriptive readiness audit, not a model-training experiment.
- **Question / hypothesis:** Does the completed E020 cohort support initial development splits for all 12 targets, and what limitations must be addressed before training?
- **Baseline:** The first 500 eligible non-Gold studies, selected in CSV order from 4,349 eligible studies; 3,849 remain outside the labeled cohort. This is not a random sample.
- **Changed variable:** None. Existing inputs, labels, prompts, evidence and production code were kept unchanged.
- **Expected mechanism:** Class-support diagnostics reveal sparse targets; duplicate-report sensitivity reveals potential dependencies; report/series metadata comparisons screen limited distribution proxies, not clinical representativeness.

## Protocol and boundaries

Local standard-library calculations reused the actual checkpoint loader/schema validator, `eligible_reports`, strict source-evidence validator and `iterative_multilabel_folds`. Gold-58 IDs were used only for exclusion before report/target analysis; no Gold report, target value, frozen split assignment or classifier evaluation was used. Eight complete input/source file hashes were recorded privately and checked before/after analysis and on readback; their values are not published here.

Per-target counts describe pseudo-labels, not verified disease prevalence. Report-length, word-count, line-count and series-count summaries use linear-interpolated quantiles at `(n-1)*p` (type 7). Duplicate sensitivity compares exact report text and Python casefold-plus-whitespace-normalized text; punctuation is retained. No identifiers, report text, evidence text, row-level predictions or split assignments are exported.

Diagnostic fold configurations were predeclared: seeds 17, 42 and 123; three and five folds; study-only versus whole normalized-report groups. The existing iterative allocator uses group-wise OR labels for group allocation, after which original study labels are counted. This group-balancing heuristic is not study-weight-aware. All seeds are reported without selecting the best seed; five-fold fold 0 is illustrative approximately 80/20 development geometry, not a frozen final split.

Saved provenance remains 64 legacy, 408 literal-v1, zero literal-v2 and 28 fragment-v3 records, with boundaries at 64/472/472. Stratum summaries are descriptive only: different sizes and ordered-cohort confounding prohibit causal prompt ranking. E020's operational completion is not extended into a clinical-quality claim.

## Integrity and target support

The checkpoint contains 500 unique studies matching the ordered eligible prefix, with zero Gold overlap. All 500 records pass the actual loader, all 6,000 target items pass validation, and all 4,374 nonempty evidence strings occur literally in their source reports. All positive labels have nonempty evidence; neither nonemptiness nor literal containment proves relevance or clinical correctness.

Positive counts out of 500 are:

- ACL: 135; MCL: 78.
- Medial Meniscus: 262; Lateral Meniscus: 103.
- Medial OA: 166; Lateral OA: 121; PF OA: 207.
- Effusion: 292; Synovitis: 67.
- Baker's: 111; Contusion: 67; Fracture: 24.

Every target retains positives and negatives in training and validation across all 48 diagnostic folds. Fracture is the sparsest target: five-fold validation contains only 4-5 positives, versus eight in three-fold validation. Both-class coverage establishes feasibility of calculating class-dependent metrics, not precise estimates or sufficient learning signal. Grouped fold sizes range from 97-106 studies for five folds and 163-171 for three folds; no final assignment was persisted or chosen.

## Label quality and uncertainty limitations

Uncertainty means, medians and counts at thresholds 0.3 and 0.5 are recorded separately for positive and negative pseudo-labels in the local report. These values are model self-reported estimates of the probability that the binary label is wrong, as defined in D004; they are not calibrated or independently validated error rates. No automatic filtering, relabeling or class dropping follows.

Synovitis has 433 negative labels, of which 423 have empty evidence. This is an important quality-review priority, but does not demonstrate that those labels are wrong or establish disease absence. A negative/unmentioned report result is not equivalent to verified clinical absence. No expert adjudication or independent clinical accuracy estimate was available for this cohort.

## Duplicate-report sensitivity

- Exact text: five repeated-report groups contain 15 cohort studies, with 10 duplicate excess rows and largest group size seven.
- Casefold/whitespace normalization: seven repeated-report groups contain 19 studies, with 12 duplicate excess rows and largest group size seven.
- No target-label disagreement was found within these repeated-report groups.
- Exact cross-cohort/remainder overlap: 14 shared report groups cover 21 cohort and 76 remainder studies.
- Normalized cross-cohort/remainder overlap: 15 shared groups cover 24 cohort and 82 remainder studies.

Study-only fold allocation separates 5-6 normalized repeated-report groups across folds; whole-report-group allocation separates none. These are potential dependency groups, not proven duplicate MRI acquisitions or shared patients. No patient identifier is available to establish patient independence. If the cohort is expanded later, existing report-group overlap with the remainder must be considered before splitting.

## Representativeness and image readiness

Series metadata covers all 500 cohort and 3,849 remainder studies: 2,784 versus 21,251 series rows. Median series count is five in both groups, with means 5.568 versus 5.521. Median report length is 998 versus 973 characters. Plane shares are broadly similar descriptively: sagittal 40.84% versus 40.43%, coronal 35.38% versus 35.32%, axial 23.78% versus 24.26%. Fluid-sensitive series shares are 57.44% versus 57.50%; fat-suppression shares match those values in this metadata. Every study in both groups has metadata for all three planes and both flag values.

These are series-level shares unless explicitly stated as study coverage. Similar report/series proxies do not establish label prevalence in the unlabeled remainder, language mix, scanner/site coverage, patient independence or representative disease distribution.

No candidate image-extension files were found under local `data`. This does not imply images are absent from Kaggle. The historical 50-study/273-series/8,605-instance header characterization does not verify image availability, decoding or preprocessing for these 500 studies; no retained row manifest connecting the samples was verified. No new DICOM payload was inspected.

## Verification and artifacts

Independent stdlib recomputation confirmed integrity, all target/evidence/uncertainty statistics, duplicate/report-overlap counts, metadata denominators and unchanged hashes. All 48 recorded folds passed arithmetic and class-coverage checks. Two of the 12 configurations were independently replayed exactly: study-only and normalized-report-group five-fold allocation at seed 42; all class counts and report-group crossing counts matched. The other ten configurations received arithmetic checks, not independent allocation replay.

Aggregate privacy scans, JSON/HTML structural readback and `git diff --check` passed. Existing overlong-evidence preference warnings preserved full evidence and did not invalidate the audit. Production code/tests were unchanged, so no functional suite was rerun; the previously known Windows timeout-cleanup failure and unavailable directory-symlink check remain historical limitations, not newly tested outcomes.

Local outputs are ignored and **not committed**: [HTML report](../data/outputs/500-training-readiness-audit.html) and [aggregate JSON](../data/outputs/500-training-readiness-audit.json). The HTML contains per-target/fold tables and expandable aggregate detail; full-input hashes remain local. A Canvas workspace association was unavailable, so this is an HTML fallback, not a claimed Canvas render.

## Conclusion and next step

Do not label more studies solely to increase the headline sample count. First resolve authorized image access/series selection and preprocessing, examine pseudo-label quality, and define a dependency-aware development split outside Gold-58. A grouped three-fold pilot offers eight fracture positives per validation fold rather than 4-5 with five folds, at the cost of fewer training studies per fold; this is a planning tradeoff, not a selected validation protocol.

Once those prerequisites are addressed, train a baseline and learning curves with increasing training subsets against unchanged validation. Only then can measured generalization and rare-target uncertainty guide whether more labeling is useful. Gold-58 remains excluded from classifier training and model selection. No final sample-sufficiency claim, uncertainty-based exclusion rule or downstream model decision is made by this audit.
