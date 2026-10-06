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

## D004 - Reserve Gold-58 for final evaluation

- **Date:** 2026-10-01; usage policy clarified 2026-10-02
- **Status:** Accepted; count and study/label mapping not independently verified
- **Decision:** Reserve the 58 expert-labeled studies exclusively for final evaluation. Do not use them for training, prompt optimization, model selection, or tuning.
- **Rationale:** Their scarcity makes evaluation-role leakage and policy drift especially consequential.
- **Evidence / source:** Explicit user direction on 2026-10-02; the count and expert-label mapping remain bootstrap-context facts, not independently verified dataset findings.
- **Consequences:** Once the studies and labels are identified under an authorized scope, keep them segregated from development and use them only for final evaluation. No such identification or label access is claimed in this decision update.
- **Revisit when:** The user changes the policy or verified dataset evidence shows that the assumed count/label mapping is incorrect.
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
