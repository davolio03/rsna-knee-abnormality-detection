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
- **Status:** Host evidence verified 2026-10-02; conditional host permission only. The user approved and froze the Gold-58 prompt candidates and nested study-level protocol on 2026-10-11. Gold-58 remains excluded from ordinary labeling and is not authorized for training. On 2026-10-14, after reviewing the SIWC-specific terms, the user authorized a bounded local SIWC non-Gold labeling run; sign-in completed with ChatGPT plan use enabled. The user has now separately authorized a scalable runner, then a fresh canary, then all eligible non-Gold inferences. The scalable implementation is pending local verification; this authorization does not claim that the fresh canary or full-cohort run has occurred.
- **Decision:** The competition host's conditional statement supports commercially hosted LLM report processing subject to competition rules, reasonable accessibility, minimal cost, and provider terms. For the Gold-58 prompt comparison, the selected route remains Codex CLI with ChatGPT subscription OAuth, targeting `gpt-6-luna` at `model_reasoning_effort=high`; this Codex `exec`/structured-output route remains on hold pending clarification of its applicable terms and remaining data/licensing conditions. Separately, OpenAI's Sign in with ChatGPT Terms (effective 2026-09-29) govern the local SIWC app: they allow eligible ChatGPT-plan use in a user-controlled local runtime for the authenticated user, including expressly authorized automations. The user expressly authorized a three-step local SIWC non-Gold rollout on 2026-10-14: implement a scalable resumable runner, perform a fresh canary, then infer all eligible non-Gold reports. The first step is local implementation only; live provider calls remain separate and will occur only in the later authorized steps. This is not authorization for Gold-58, another user's requests, token sharing, usage-limit circumvention, or any request batch larger than 14 records. The SIWC route uses the account-visible model slug; it does not accept a `reasoning_effort` parameter.
- **Rationale:** Report-derived labels are one of the project's coupled problems.
- **Evidence / source:** Competition-host-authored discussion with a visible Competition Host badge, [Use of Commercially Hosted LLMs](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965), dated 2026-08-09; explicit host reply dated 2026-08-27. The host discusses API report reading/label generation subject to competition rules, reasonable access/cost, and provider terms; it is conditional and does not resolve other licenses. OpenAI's [Sign in with ChatGPT Terms](https://openai.com/policies/sign-in-with-chatgpt-terms/), effective 2026-09-29, expressly allows user-authorized automations in a local runtime controlled by the authenticated user and governs SIWC plan usage. OpenAI's [SIWC models and inference guide](https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference) documents the public Responses API flow. The account reported `ChatGPT plan use: enabled`; actual Luna availability must still be confirmed from that account's visible model catalog. The [Codex CLI reference](https://learn.chatgpt.com/docs/cli/reference) documents non-interactive `codex exec` and JSONL output but does not resolve the separate Codex CLI terms question. The competition's MIRA license scope and winner-license wording conflict remain documented in `COMPETITION.md`.
- **Consequences:** The SIWC permission is narrowly for the authenticated user's expressly authorized, local non-Gold labeling run; send only eligible non-Gold training reports, keep results in ignored local outputs, and use the required `store=false`, `stream=true`, and completed-response checks. The scalable runner may process the full preflighted eligible non-Gold cohort across multiple sequential windows, with no more than 14 records in a window/request batch. Use fresh, distinct output/checkpoint and payload-free diagnostics paths; resume only from the explicitly selected matching paths and validated checkpoint, while preserving the earlier 14-record output untouched. Stop on provider-denied or usage-limit errors; do not retry or bypass provider limits. Gold-58 must remain excluded. This does not authorize distribution of reports or derived artifacts, public deployment, or use of Gold-58 for training/model selection; preserve applicable competition, privacy, and licensing requirements. Keep the separate Gold-58 Codex CLI comparison on hold pending clarification of that route's terms and the remaining external-data conditions. No conclusion is made that the MIRA derivative-work restriction or winner-license ambiguity is resolved.
- **Revisit when:** OpenAI changes the SIWC terms or usage limits, the user's authorization or intended cohort changes, the Codex CLI route receives applicable terms clarification, or organizers clarify the MIRA/winner-license questions.

## D007 — Use Codex CLI for a bounded subscription-label pilot

- **Date:** 2026-10-10
- **Status:** Pilot route selected and locally implemented; no report transmission or feasibility result claimed.
- **Decision:** Evaluate report-label extraction through Codex CLI using the user's ChatGPT subscription OAuth entitlement, rather than a separately billed OpenAI API key. The pilot is limited to explicitly capped training rows whose 12 target fields are all blank and whose study keys are absent from the verified Gold exclusion list.
- **Rationale:** The user authorized a small non-Gold subscription-backed pilot; local CPU/GPU LLM inference is infeasible.
- **Evidence / source:** Explicit user authorization and CP-01 local contract tests. No real report was sent during implementation. The CLI adapter invokes only `codex exec`; subscription authentication and actual inference remain unverified and are CP-02 prerequisites.
- **Consequences:** Keep generated labels and study keys in ignored local `data/outputs/` only. Never log report content or CLI payloads. Generated labels remain provisional and unvalidated against independent ground truth. This decision does not resolve provider terms, MIRA licensing, or the winner-license wording conflict recorded in `COMPETITION.md`; do not distribute outputs.
- **Revisit when:** CP-02 operational evidence is recorded, subscription auth is verified, or applicable competition/provider terms change.

## D008 - Treat evidence length as a warning, not an acceptance rule

- **Date:** 2026-10-11
- **Status:** Accepted by explicit user authorization.
- **Decision:** Evidence strings longer than 200 characters trigger one fixed privacy-safe nonfatal warning per response; retain the complete evidence string without truncation. The 200-character value is an operational preference threshold only and has no established empirical basis. Keep exact target-set/object shape, boolean labels, evidence string type, and finite uncertainty in [0,1] strict.
- **Rationale:** G58-10 stopped after 86 successful checkpoint entries when one output exceeded the local evidence-length preference. The validator guard was stricter than the authorized output contract; discarding or truncating evidence would alter the response. Warning-only handling permits the already-authorized explicit resume to reuse validated cached successes while preserving all other validation boundaries.
- **Evidence / source:** Explicit user decision and one-resume authorization recorded 2026-10-11; G58-10 terminal outcome is recorded in `odd/tasks/gold58-prompt-comparison-runner.md` and E007. No raw evidence, report, identifier, or prediction is included here. No empirical study establishes 200 characters as a correctness or privacy boundary.
- **Consequences:** The warning contains no evidence value, target, or study identifier. A warning does not validate the label, evidence relevance, model quality, clinical validity, or metric validity. Frozen prompts, output schema, checkpoint fingerprint, model/effort, folds, metrics, and generation contract remain unchanged. Exactly one resume is authorized from the 86-success checkpoint, with up to 320 remaining of 406 planned calls; do not archive, rename, rotate, reset, or delete the current checkpoint/key, and do not automatically retry after any new failure. This operational authorization does not resolve unrelated provider terms, competition rules, licensing, or data-transfer questions recorded in D006.
- **Revisit when:** A future user decision changes the evidence-length preference, another output-contract validation boundary is implicated, or evidence establishes a justified length limit.

## D009 - Persist SIWC refresh credentials only

- **Date:** 2026-10-14
- **Status:** Accepted and implemented locally; no live OAuth validation performed.
- **Decision:** Persist the SIWC refresh token with only required non-secret account, grant, token-type, issuer, and expiry metadata. Validate the initial ID token during sign-in, then discard it. Keep access tokens process-local, refresh on demand in each fresh CLI process, and expose them only through the existing `with_plan_access_token` callback boundary. Do not trust legacy persisted access tokens, including records whose expiry is still in the future.
- **Rationale:** The combined serialized token response exceeded the Windows Credential Manager per-credential payload limit; refresh-token-only storage keeps protected-operation credentials transient and reduces persisted payload size.
- **Evidence / source:** Explicit user authorization and local synthetic regression tests. No real credentials, OAuth requests, or provider behavior were inspected or exercised for this change.
- **Consequences:** Preserve rotated refresh credentials and granted plan scopes before making refreshed access usable. A missing refresh token, required plan scope, or failed persistence remains fail-closed. Optional refresh ID tokens are validated when present and never persisted; omission does not preserve a stale token. Legacy records are sanitized when rewritten after refresh.
- **Revisit when:** The provider changes the refresh contract or new local evidence demonstrates that the persisted metadata set is insufficient.
