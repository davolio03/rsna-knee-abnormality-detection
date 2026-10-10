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

- **Date:** 2026-10-01; policy revised 2026-10-10
- **Status:** Accepted; candidate-set identification is recorded in `DATA.md`; expert-label provenance and identity remain unverified from label values.
- **Decision:** Gold-58 may be used to compare a short, predeclared set of report-labeling prompt strategies using nested study-level cross-validation. Freeze the candidate prompts, selection protocol, and metrics before evaluating outer folds. Use inner folds to select among candidates; outer folds evaluate the complete inner-selection procedure and must not influence prompt iteration. The outer-fold results estimate that selection procedure and are not a separate independent final test set. Do not use Gold-58 for model training or downstream classifier model selection unless separately authorized.
- **Rationale:** A bounded nested design permits prompt-strategy comparison while keeping outer-fold evidence separate from prompt iteration and stating its limits honestly.
- **Evidence / source:** Explicit user authorization on 2026-10-10. Candidate IDs and label values were not accessed for this policy update. The candidate-set identification and its limitations are recorded in `DATA.md`.
- **Consequences:** Before any evaluation run, inspect per-target prevalence and positive/negative counts and confirm that the planned study-level folds are feasible; this prerequisite has not been checked. Preserve study-level grouping and all applicable competition licensing, external-service, and data-privacy constraints. Do not treat nested cross-validation as an untouched final test set.
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
- **Status:** Host evidence verified 2026-10-02; use remains conditional, provider/workflow unselected
- **Decision:** The competition host permits LLM APIs to read reports and generate labels, subject to the competition rules and conditions recorded below. The exact provider, model, prompt, and service strategy remain undecided.
- **Rationale:** Report-derived labels are one of the project's coupled problems.
- **Evidence / source:** Competition-host-authored discussion with a visible Competition Host badge, [Use of Commercially Hosted LLMs](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965), dated 2026-08-09; explicit host reply dated 2026-08-27. The host allows API report reading/label generation subject to the competition rules, reasonable access/cost, and provider terms. The rules have a winner-license wording conflict recorded in `COMPETITION.md`.
- **Consequences:** This is conditional competition permission, not project authorization to transmit reports. Do not process reports externally until a provider/workflow is selected and rule, provider-term, data-handling, and project authorization constraints are resolved. The rule page's conflicting license language does not change D006 into blanket permission.
- **Revisit when:** The MIRA license and winner-license inconsistency are clarified, provider terms and access/cost are assessed, and a concrete workflow is proposed for explicit authorization.

## D007 — Use Codex CLI for a bounded subscription-label pilot

- **Date:** 2026-10-10
- **Status:** Pilot route selected and locally implemented; no report transmission or feasibility result claimed.
- **Decision:** Evaluate report-label extraction through Codex CLI using the user's ChatGPT subscription OAuth entitlement, rather than a separately billed OpenAI API key. The pilot is limited to explicitly capped training rows whose 12 target fields are all blank and whose study keys are absent from the verified Gold exclusion list.
- **Rationale:** The user authorized a small non-Gold subscription-backed pilot; local CPU/GPU LLM inference is infeasible.
- **Evidence / source:** Explicit user authorization and CP-01 local contract tests. No real report was sent during implementation. The CLI adapter invokes only `codex exec`; subscription authentication and actual inference remain unverified and are CP-02 prerequisites.
- **Consequences:** Keep generated labels and study keys in ignored local `data/outputs/` only. Never log report content or CLI payloads. Generated labels remain provisional and unvalidated against independent ground truth. This decision does not resolve provider terms, MIRA licensing, or the winner-license wording conflict recorded in `COMPETITION.md`; do not distribute outputs.
- **Revisit when:** CP-02 operational evidence is recorded, subscription auth is verified, or applicable competition/provider terms change.
