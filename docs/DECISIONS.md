# Decision Log

This log contains durable project decisions, not untested hypotheses. Source provenance is explicit; decisions based on bootstrap context are not presented as independently verified competition rules.

## D001 — Project workflow

- **Date:** 2026-10-01
- **Status:** Accepted
- **Decision:** Use Codex as the primary interactive orchestrator with Gentle AI ODD as the normal workflow. Use native subagents selectively when independent fresh context materially helps. Do not use SDD by default.
- **Rationale:** Keep project direction coherent while using independent context only when useful.
- **Evidence / source:** Project bootstrap context.
- **Consequences:** Follow repository-specific guidance together with the installed ODD workflow; this repository does not duplicate orchestration machinery.
- **Revisit when:** The user explicitly changes the workflow or the environment makes the current workflow unavailable.

## D002 — Repository as durable source of truth

- **Date:** 2026-10-01
- **Status:** Accepted
- **Decision:** Version-controlled repository artifacts are authoritative for durable project knowledge; Engram supports continuity but is not the sole source of scientific decisions.
- **Rationale:** Important evidence must remain inspectable and recoverable with the code.
- **Evidence / source:** Project bootstrap context.
- **Consequences:** Store durable decisions and experiment evidence in repository files; treat memory as supplementary.
- **Revisit when:** Repository persistence or collaboration requirements materially change.

## D003 — Scientific experiment discipline

- **Date:** 2026-10-01
- **Status:** Accepted
- **Decision:** Meaningful ML changes require explicit experimental evidence before becoming the new baseline.
- **Rationale:** Avoid decisions based solely on intuition or noisy leaderboard movement.
- **Evidence / source:** Project bootstrap context.
- **Consequences:** Record hypotheses, baselines, changed variables, validation, results, and conclusions for meaningful experiments.
- **Revisit when:** Evidence shows the experiment convention is insufficient or wasteful.

## D004 - Gold-58 nested prompt-strategy evaluation

- **Date:** 2026-10-01; policy revised 2026-10-11
- **Status:** Accepted; candidate-set identification is recorded in `DATA.md`; expert-label provenance and identity remain unverified from label values.
- **Decision:** Gold-58 may be used to compare a short, predeclared set of report-labeling prompt strategies using nested study-level cross-validation, only after exact prompt text and protocol are approved and frozen and external report transfer is separately confirmed permissible. Use `gpt-6-luna` with `model_reasoning_effort=high` through Codex CLI and the user's ChatGPT subscription OAuth path; this is a planned configuration, not verified entitlement or transfer authorization, and does not use an API key. Define the output field `uncertainty` as the model's estimated probability that its binary label is incorrect (`0` predicts the label is correct; `1` predicts the label is incorrect); this is an error probability, not a generic epistemic-uncertainty scale or a calibration guarantee. Compute balanced accuracy per target then macro-average the 12 scores. Select by mean inner-validation macro balanced accuracy; resolve exact ties in fixed simplicity order: baseline, evidence workflow, interpretation policy. Report the secondary macro Brier score for the error probability versus the label-error indicator descriptively only (lower is better); it never selects the prompt. Inner folds select among candidates; outer folds evaluate the complete inner-selection procedure and must not influence prompt iteration. Outer-fold results are not a separate independent final test set. Do not use Gold-58 for model training or downstream classifier model selection unless separately authorized.
- **Rationale:** A bounded nested design permits prompt-strategy comparison while keeping outer-fold evidence separate from prompt iteration and stating its limits honestly.
- **Evidence / source:** Explicit user protocol decisions recorded 2026-10-11. The aggregate-only feasibility screen recorded in `PROJECT_STATE.md` reports positive and negative representation for all 12 targets in every configured outer and inner validation fold; MCL is sparse (9 positives overall; 2 in each inner validation fold). No candidate IDs, label values, or fold memberships are recorded here.
- **Consequences:** Aggregate fold feasibility is established for the configured split, but does not establish metric precision, label validity, or clinical validity. Preserve study-level grouping and all applicable competition licensing, external-service, and data-privacy constraints; confirm subscription entitlement for the exact model/effort and report-transfer permissibility before any run. Outer-fold evidence must not guide prompt iteration. Do not treat nested cross-validation as an untouched final test set.
- **Revisit when:** The user changes the policy, fold feasibility cannot be established, or verified dataset evidence changes the assumed candidate set or label provenance.
## D005 — Hardware-aware development

- **Date:** 2026-10-01
- **Status:** Accepted; hardware unverified
- **Decision:** Prefer approaches that can be iterated realistically with approximately 6 GB local VRAM while allowing Kaggle compute for justified workloads.
- **Rationale:** Project context identifies a constrained local GPU environment.
- **Evidence / source:** Project bootstrap context; exact device and available resources require environment verification.
- **Consequences:** Record memory/runtime costs and avoid assuming larger models are better.
- **Revisit when:** The available local or competition compute environment changes.

## D006 — LLM report processing is a candidate technique

- **Date:** 2026-10-01
- **Status:** Host evidence verified 2026-10-02; conditional host permission only. The planned Codex subscription route for the Gold-58 prompt comparison remains subject to entitlement and report-transfer checks.
- **Decision:** The competition host's conditional statement covers commercially hosted LLM report processing subject to the competition rules and conditions recorded below. For the planned Gold-58 prompt comparison only, the selected route is Codex CLI using ChatGPT subscription OAuth (no API), targeting `gpt-6-luna` at `model_reasoning_effort=high`; this selection is not proof of current entitlement or authorization to transmit reports.
- **Rationale:** Report-derived labels are one of the project's coupled problems.
- **Evidence / source:** Competition-host-authored discussion with a visible Competition Host badge, [Use of Commercially Hosted LLMs](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965), dated 2026-08-09; explicit host reply dated 2026-08-27. The host allows API report reading/label generation subject to the competition rules, reasonable access/cost, and provider terms. The rules have a winner-license wording conflict recorded in `COMPETITION.md`.
- **Consequences:** This is conditional competition permission, not project authorization to transmit reports. Before any Gold-58 prompt run, verify subscription entitlement for the exact model/effort and resolve rule, provider-term, data-handling, and project authorization constraints for the proposed transfer. The rule page's conflicting license language does not change D006 into blanket permission. Candidate text remains unapproved, unfrozen, and unrun.
- **Revisit when:** Exact subscription entitlement and applicable provider/access terms are verified and report-transfer permission is confirmed; revisit further if the MIRA license or winner-license inconsistency is clarified.

## D007 — Use Codex CLI for a bounded subscription-label pilot

- **Date:** 2026-10-10
- **Status:** Pilot route selected and locally implemented; no report transmission or feasibility result claimed.
- **Decision:** Evaluate report-label extraction through Codex CLI using the user's ChatGPT subscription OAuth entitlement, rather than a separately billed OpenAI API key. The pilot is limited to explicitly capped training rows whose 12 target fields are all blank and whose study keys are absent from the verified Gold exclusion list.
- **Rationale:** The user authorized a small non-Gold subscription-backed pilot; local CPU/GPU LLM inference is infeasible.
- **Evidence / source:** Explicit user authorization and CP-01 local contract tests. No real report was sent during implementation. The CLI adapter invokes only `codex exec`; subscription authentication and actual inference remain unverified and are CP-02 prerequisites.
- **Consequences:** Keep generated labels and study keys in ignored local `data/outputs/` only. Never log report content or CLI payloads. Generated labels remain provisional and unvalidated against independent ground truth. This decision does not resolve provider terms, MIRA licensing, or the winner-license wording conflict recorded in `COMPETITION.md`; do not distribute outputs.
- **Revisit when:** CP-02 operational evidence is recorded, subscription auth is verified, or applicable competition/provider terms change.
