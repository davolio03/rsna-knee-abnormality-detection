# Project Roadmap

## Objective

Build a reproducible, leakage-resistant system that generates or evaluates training labels from reports and predicts 12 abnormalities from complete knee MRI studies. Optimize for robust validation and eventual compatibility with the official competition inference environment. Phases are a research sequence, not authorization to download data, use external services, or change validation policy; verify constraints and obtain required access first.

## Phase status

| Phase | Status | Depends on |
|---|---|---|
| -1 — Project / Agent Infrastructure | Complete | None |
| 0 — Competition Setup & Submission Contract | Complete (2026-10-10) | Phase -1 |
| 1 — Dataset Understanding / EDA | In progress (light non-Gold CSV metadata inventory tranche complete 2026-10-10) | Phase 0 |
| 2 — Report-to-Label System | Planned | Phase 0; coordinate with Phase 3 evaluation policy |
| 3 — Validation Design | Planned; critical | Phase 0; inform Phases 2 and 5 |
| 4 — MRI Preprocessing & Efficient Data Layer | Planned | Phase 0–1 |
| 5 — Strong End-to-End Baseline | Planned | Phases 2–4 |
| 6 — Slice Aggregation / MIL / Pooling | Planned | Phase 5 |
| 7 — Multi-Series / Multi-Plane Modeling | Planned | Phases 4–6 |
| 8 — Target-Specific Error Analysis | Planned | Phase 5; iterates with later phases |
| 9 — Label Noise / Silver-Label Learning | Planned | Phases 2–3, 5, and 8 |
| 10 — Encoder / Pretraining Experiments | Planned | Phases 5 and 8 |
| 11 — Systematic Optimization | Planned | Stable experiment tracking and credible baselines (Phases 3, 5) |
| 12 — Ensembling | Planned | Phases 5–11; diversity evidence |
| 13 — Kaggle Inference Pipeline | Planned | Phase 0 contract; chosen model(s) from Phases 11–12 |
| 14 — Final Selection | Planned | OOF/CV evidence, inference feasibility, Phase 13 |

## Phase -1 — Project / Agent Infrastructure (complete)

**Goal:** Establish repository structure, durable context, scientific workflow expectations, data safeguards, and experiment tracking before project execution.

**Questions:** Are durable project documents linked and internally consistent? Are raw data and generated artifacts protected by default? Can future work distinguish known project context from locally verified observations?

**Outputs:** Root agent guidance, roadmap, operational project state, decision log, competition and data notes, experiment conventions/registry, package boundaries, and documentation placeholders.

**Acceptance:** Repository structure exists; `AGENTS.md`, roadmap, project state, decisions log, and experiment registry exist; data safety rules are present; current state names Phase 0 as next; no local dataset inspection or modeling is represented as complete.

**Dependencies:** None. Phase -1 bootstrap is complete as of 2026-10-01.

## Phase 0 — Competition Setup & Submission Contract (complete, 2026-10-10)

**Goal:** Understand the official rules, data contract, metric, and inference/submission requirements before serious modeling.

**Questions:** What are the current rules, restrictions on external data/models/services, runtime and hardware limits, internet policy, submission schema and limits, final-selection process, deadlines, and winning-solution obligations? What exact files/columns and metric behavior are supplied? What are the host's documented terms for report labeling with LLMs? Can the required Kaggle access be confirmed?

**Outputs:** Source-linked verified facts in `docs/COMPETITION.md`; schema/column confirmation in `docs/DATA.md`; verified environment/setup notes; a minimal structural submission path and corresponding check only after the contract is understood.

**Acceptance:** Official current constraints and schema are documented with sources and dates; metric interpretation is confirmed; access and environment limitations are recorded; a structurally valid submission contract is understood. No serious modeling is required. This acceptance is met by the source-linked contract, explicit unknowns, local synthetic validator, and one successful constant-prediction smoke submission (submission/scoring-path evidence only; no model-performance claim).

**Dependencies:** Phase -1. Completion gates data-dependent EDA and informed pipeline work, but is not data-access authorization. Explicitly unresolved prerequisites for later inference/deployment include full local schema/data-quality inspection, representative full-size runtime and hardware feasibility, and Kaggle API/CLI access. Phase 1 remains gated on separately authorized data access. The MIRA/winner-license ambiguity remains a blocker to distribution or commercial deployment of competition-derived artifacts, not private research. Do not infer additional verification from this status.

## Phase 1 — Dataset Understanding / EDA

**Status (2026-10-10):** The bounded light CSV metadata inventory for non-Gold training studies is complete and recorded in `docs/DATA.md`. This tranche used study keys and documented series metadata only; it is not completion of Phase 1. Its three-study test split is too small to establish reliable distribution alignment or representativeness, and no corpus/scanner/site or image-level audit was performed. Continue with other Phase 1 questions only within separately authorized data scope.

**Goal:** Characterize the actual dataset and data quality without prematurely optimizing models.

**Questions:** How many studies/series/slices exist? What are the distributions of image resolution, orientation, anatomical plane, fluid sensitivity, fat suppression, and sequences? What heterogeneity or missingness exists? Are scanner/site signals inferable? What report languages and label prevalence occur in the expert-labeled subset? Are duplicates or anomalies present?

**Outputs:** Reproducible inspection/visualization utilities as justified; actual summary tables/plots; updates to `docs/DATA.md` with sample sizes, limitations, and provenance.

**Acceptance:** Data claims are backed by reproducible local summaries; units and missingness are defined; no model optimization is mixed into exploratory characterization; data-sensitive output remains untracked unless specifically approved.

**Dependencies:** Phase 0 access and schema verification. Coordinate with Phase 3 before using labels to shape a validation set.

## Phase 2 — Report-to-Label System

**Goal:** Develop a controlled, reproducible approach for deriving useful silver labels from reports while preserving label provenance and evaluation integrity.

**Questions:** What are the clinical definitions of all 12 targets? How should negation, uncertainty, historical findings, multilingual text, and missing evidence be represented? Are hard, soft, or confidence-weighted targets appropriate? How do deterministic extraction, LLM extraction, translation, or hybrid methods compare? How should prompt strategies be compared under D004 without leaking outer-fold evidence into prompt iteration?

**Outputs:** Versioned target definitions, structured output schema, report preprocessing/extraction artifacts, prompt/configuration snapshots where used, adjudication protocol, and per-target label-quality evaluation.

**Acceptance:** Outputs are traceable to source reports and extraction versions; uncertainty/negation behavior is tested; evaluation is controlled and documented; no prompt/model selection leakage is hidden; external processing has a concrete purpose and satisfies verified rules.

**Dependencies:** Phase 0 rule verification; coordinate with Phase 3, whose evaluation policy constrains label-quality comparisons. This phase does not depend on itself. The host clarification about LLM use is project bootstrap context until its official evidence is recorded.

**Potential variables:** Model, prompt, target definitions, few-shot examples, schema, confidence, uncertainty policy, self-consistency, translation strategy, and deterministic/LLM hybridization. Avoid casually tuning all choices against only 58 studies.

## Phase 3 — Validation Design (critical)

**Goal:** Freeze an honest, reproducible validation design that estimates generalization and preserves complete-study grouping.

**Questions:** What split structure best reflects intended generalization? Is scanner/site leakage plausible? Are duplicate studies or reports grouped? Is multi-label stratification useful without undermining grouping? Gold-58's authorized role is a short, predeclared report-labeling prompt-strategy comparison using nested study-level cross-validation under D004; how should the folds and selection protocol be implemented without leakage? How should macro ROC-AUC handle missing/constant targets and match the official metric?

**Outputs:** Versioned split assignments or deterministic split-generation config; leakage/duplicate analysis; metric implementation and tests; explicit decision record for label/evaluation roles.

**Acceptance:** No study crosses a leakage-sensitive fold boundary; grouping rationale and limitations are documented; all 12 per-target AUCs and aggregate score are reproducible; validation changes require a recorded rationale. For Gold-58 prompt comparisons, freeze candidates, protocol, and metrics before outer-fold evaluation; outer folds assess the inner-selection procedure and are not an independent final test set. Check per-target prevalence and positive/negative counts and confirm fold feasibility before any evaluation run.

**Dependencies:** Phase 0 metric/schema; informs label evaluation in Phase 2 and model comparison in Phase 5. Do not freeze folds from intuition alone.

## Phase 4 — MRI Preprocessing & Efficient Data Layer

**Goal:** Build a reliable, efficient representation of MRI studies appropriate to the data and constrained development hardware.

**Questions:** How to decode DICOM robustly and order slices? Which intensity normalization, orientation, spatial handling, crop, and resize choices are defensible? How to select series and use metadata? What cache format and location reduce I/O while controlling storage? How can experiments run within approximately 6 GB local VRAM without requiring an unnecessary full raw-data download?

**Outputs:** Tested preprocessing components, versioned configurations, cache/index conventions, timing/storage measurements, and data documentation updates.

**Acceptance:** Study/series/slice identity is preserved; ordering and transformations are reproducible; cached outputs can be traced to source/config versions; representative edge cases are checked; memory/I/O costs are measured.

**Dependencies:** Phase 0 schema and Phase 1 characterization. Prefer small representative samples where authorized; avoid collecting raw data beyond actual need.

## Phase 5 — Strong End-to-End Baseline

**Goal:** Establish the first credible reproducible model pipeline and baseline.

**Questions:** What is a simple, compute-feasible image model and study-level aggregation strategy? Which labels are usable? Does it produce stable out-of-fold predictions and valid 12-target outputs?

**Outputs:** Versioned baseline configuration, training/inference code, OOF predictions, per-target and macro metrics, runtime/VRAM observations, and experiment record.

**Acceptance:** Fixed validation, traceable labels and preprocessing, study-level predictions with 12 probabilities, reproducible OOF metrics, and recorded compute costs. No score becomes baseline without evidence and review of per-target behavior.

**Dependencies:** Phases 2–4. Candidate starting points include 2D/2.5D encoders and simple study-level pooling; avoid unjustified model scale.

## Phase 6 — Slice Aggregation / MIL / Pooling

**Goal:** Test how slice-level evidence should combine into study predictions.

**Questions:** Does mean pooling dilute sparse findings? How do max, top-k, attention/gated attention, MIL, or lightweight transformer aggregation compare? Is benefit target-dependent?

**Outputs:** Controlled experiment records and per-target comparisons using the same split, labels, and encoder where possible.

**Acceptance:** Comparisons change a defined aggregation variable; compute cost and uncertainty are retained; adopted methods improve robust evidence rather than a single noisy metric.

**Dependencies:** Phase 5 baseline and Phase 3 validation. Feed useful failures into Phase 8.

## Phase 7 — Multi-Series / Multi-Plane Modeling

**Goal:** Determine how series and anatomical planes contribute and should be fused.

**Questions:** Are sagittal, coronal, and axial information complementary by target? Should encoders be shared or plane-specific? Are plane embeddings, series attention, target-specific aggregation, or cross-series fusion useful? What happens when metadata is incomplete?

**Outputs:** Series-selection/fusion experiments, ablations, missing-series behavior, and per-target analysis.

**Acceptance:** Series identity and metadata are handled consistently; ablations isolate fusion choices; inference-time availability matches training assumptions.

**Dependencies:** Phases 4–6 and adequate validation evidence.

## Phase 8 — Target-Specific Error Analysis

**Goal:** Replace aggregate-only optimization with evidence-driven per-target diagnosis.

**Questions:** Which targets are limited by label quality, class imbalance, sequence relevance, calibration, or architecture? Which error patterns matter for macro AUC? Are apparent errors data artifacts or true model failures?

**Outputs:** Per-target metrics/plots, error cohorts with privacy-safe identifiers, prioritized hypotheses, and links to experiments.

**Acceptance:** Every proposed intervention is tied to observed failure evidence; any protected evaluation asset use follows documented policy; negative findings are retained.

**Dependencies:** Phase 5 baseline; repeats after material changes in Phases 6–12.

## Phase 9 — Label Noise / Silver-Label Learning

**Goal:** Determine whether and how noisy or weak labels can improve training without corrupting validation.

**Questions:** Do gold/silver weights, confidence weighting, soft targets, sample weighting, smoothing, robust losses, focal/BCE comparisons, teacher-student methods, or pseudo-labels help? Are gains stable across targets and folds?

**Outputs:** Label-versioned controlled experiments, noise analyses, and documented safeguards against pseudo-label feedback loops.

**Acceptance:** No validation labels leak into training; label provenance and transformations are versioned; pseudo-label decisions use fixed validation and preserve independent evidence.

**Dependencies:** Phases 2–3, credible model from Phase 5, and diagnostic hypotheses from Phase 8.

## Phase 10 — Encoder / Pretraining Experiments

**Goal:** Compare useful encoders and pretraining sources against the baseline with compute-aware evaluation.

**Questions:** Do ImageNet-pretrained, DINO-style, medical-pretrained, or feasible self-supervised encoders improve target performance? Are gains worth storage, runtime, and inference constraints?

**Outputs:** Encoder comparison experiments, source/license/rule records where applicable, and cost/performance analysis.

**Acceptance:** Comparisons share validation and controlled training conditions; external pretraining compliance is verified; compute costs are measured and considered.

**Dependencies:** Phase 5 baseline and Phase 8 hypotheses; Phase 0 official-rule verification for external assets.

## Phase 11 — Systematic Optimization

**Goal:** Use a stable experiment system to evaluate model and data choices efficiently.

**Questions:** Which configurations merit search? How to prevent repeated or confounded experiments? Are seeds and compute budgets sufficient to interpret variation?

**Outputs:** Reusable experiment configuration/launch conventions and traceable comparisons for experiment ID, code revision, labels, preprocessing, validation, encoder, resolution, slices, pooling, seed, macro and target AUC, runtime, VRAM, and leaderboard score when submitted.

**Acceptance:** Every meaningful run is reproducible from recorded inputs; comparisons identify changed variables; failed/negative runs remain discoverable; search is bounded by realistic compute and useful hypotheses.

**Dependencies:** Phases 3 and 5; informed by Phases 6–10.

## Phase 12 — Ensembling

**Goal:** Combine only strong and sufficiently diverse models when evidence supports a robust gain.

**Questions:** Do probability or rank averaging help? Are global or per-target weights justified? How diverse are errors? Does complexity fit inference limits?

**Outputs:** OOF ensemble comparisons, diversity analysis, fixed weight/config records, and inference cost estimate.

**Acceptance:** Ensemble choices use out-of-fold predictions without fitting to held-out evaluation; benefit and stability are assessed per target; near-duplicate models are not added without evidence.

**Dependencies:** Competitive individual models from Phases 5–11 and official constraints from Phase 0.

## Phase 13 — Kaggle Inference Pipeline

**Goal:** Produce a reliable, reproducible, competition-compatible inference path.

**Questions:** How are artifacts loaded, test studies read/preprocessed, predictions ensembled, and `submission.csv` generated? Are all constraints met without manual intervention or prohibited network use?

**Outputs:** Competition notebook or inference system, artifact manifest, submission schema checks, runtime/memory measurements, and documented execution procedure.

**Acceptance:** End-to-end execution satisfies the verified official contract, produces one valid prediction row per required study with the exact target columns, and is reproducible without undocumented manual steps.

**Dependencies:** Phase 0 contract and selected pipeline/model from Phases 11–12. Structural smoke tests may precede final model selection, but do not substitute for final validation.

## Phase 14 — Final Selection

**Goal:** Select a robust final solution using validation evidence and operational feasibility.

**Questions:** Which candidates are stable across folds/seeds and targets? Does the ensemble add meaningful diversity? Can the final system execute within official constraints? Is public-LB movement consistent with OOF evidence?

**Outputs:** Selection rationale, final artifact/config manifest, reproducibility record, and documented limitations.

**Acceptance:** Selection relies primarily on OOF/CV evidence, stability, per-target behavior, diversity, and compute feasibility. Public leaderboard is secondary; public-LB overfitting is explicitly considered.

**Dependencies:** Phases 3, 11–13. Final decision remains evidence-based and constraint-aware.
