# Feature: Gold-58 Prompt Candidate Drafts

## Objective
Prepare a compact, controlled set of report-labeling prompt candidates and a predeclared comparison protocol for human review before any Gold-58 evaluation.

## Why
The local nested 3x3 splitter and aggregate feasibility screen are complete. D004 permits only a short predeclared strategy set, selected within outer-training data; outer folds must not guide prompt iteration.

## Scope
- Draft three prompt candidates from the current Codex CLI baseline.
- Keep target names, model/runtime settings, input reports, and output schema fixed; change one strategy block per candidate.
- Document which candidate is baseline and which changes evidence workflow or interpretation guidance.
- Apply the user-approved shared uncertainty semantics and comparison protocol consistently to the prompt set.
- Pin the planned labeling run to GPT-6 Luna with high reasoning effort through Codex CLI, subject to local CLI capability/entitlement verification; capture usage only when explicitly reported by the client.
- Clearly mark prompt text as unapproved/unfrozen until the user reviews the exact final text; do not infer runtime entitlement from API model documentation.
- Do not run prompts, call Codex or external APIs, inspect or access Gold reports, generate labels, or use outer-fold evidence.

## Constraints
- Preserve the current JSON fields: `label`, exact quoted `evidence`, and `uncertainty` in [0,1].
- Define `uncertainty` as the estimated probability that the binary label is incorrect; use this identically in all candidates. Do not equate probability of error 1 with maximum epistemic uncertainty.
- Primary metric: macro balanced accuracy across the 12 targets; select by mean inner-validation macro balanced accuracy. Tie-break deterministically in favor of the simpler prompt (baseline, then evidence workflow, then interpretation policy).
- Secondary descriptive metric: macro Brier score for uncertainty against the label-error indicator; lower is better, but it does not select the prompt.
- Outer-fold results evaluate the complete inner-selection procedure only; do not tune candidates or metrics from outer results.
- Record model, effort, CLI version, elapsed runtime, and client-reported usage if available; mark usage unavailable if not explicitly exposed, and do not estimate or convert subscription usage into API cost.
- Preserve the existing no-API authentication path and fail closed if the installed Codex client rejects the model/effort setting; do not make a live model call to verify it in this task.
- Do not invent clinical definitions; flag them as a dependency if existing competition/project definitions are insufficient.
- Keep Gold-58 IDs, reports, row-level labels, and fold membership out of artifacts.
- Direct local commits to `main` are preferred; no push or PR.
- RDD remains clone-local off. No native review invocation.
- Unit tests are applicable to runner argument construction; prompt inference is not part of this task.

## Tasks
- [x] Read current prompt implementation and project policy, then create baseline and two controlled candidate variants.
- [x] Verify output contract consistency and document unresolved decisions/risks; do not execute candidates.
- [x] Update all candidate prompts with the same approved uncertainty meaning and revise the baseline description to disclose the shared addition.
- [x] Pin the Codex CLI command to `gpt-6-luna` and `model_reasoning_effort=high`; add focused tests asserting the exact arguments without launching inference.
- [x] Document the fixed Luna/high configuration, macro balanced accuracy selection, descriptive macro Brier score, inner-fold tie-break, and usage reporting limits; retain explicit no-run/no-Gold-access status.
- [x] Reconcile D004 and PROJECT_STATE with the approved protocol and local CLI capability evidence; do not claim subscription availability without proof.
- [x] Read back artifacts and run structural/privacy checks; record results and commit the completed work unit locally on main.
- [x] Correct independent-review finding: state 0 = the model estimates zero probability that the label is incorrect; state 1 = it estimates probability one that the label is incorrect; remove Candidate 3's obsolete uncertainty-policy conditional.

## Acceptance criteria
- Three identifiable prompt templates, with their shared uncertainty clause and the baseline's small departure from the exact current runner prompt clearly called out.
- Candidate variants and changed factors are explicit and comparable.
- All templates retain same inputs and JSON output schema.
- Model target is fixed to Luna/high for this comparison, with local Codex subscription compatibility clearly verified or explicitly left as a prerequisite.
- Primary metric, secondary metric, fold selection, and deterministic tie-break are explicit and consistent with D004.
- Artifact states not run; exact candidate text remains pending final user approval, and no evaluation can start before that approval and runtime compatibility checks.
- Uncertainty prompt wording and Brier target use the same probability-of-error interpretation; probability 1 is not described as maximum epistemic uncertainty.
- No private Gold content or identifiers included.

## Route and verification
- Route: delegated direct; mapping spans four or more project files, and the runner/tests/docs are one coordinated change for a single writer.
- Inspect current source/docs and local CLI help only; no model invocation, credential inspection, API use, or Gold access.
- Structural verification of exact schema/target placeholders and metrics definitions; no prompt invocation.
- TDD: no project-level strict-TDD setting was found, and the prior explicit strict-TDD choice applied to the nested-split feature; run focused ordinary unit tests for this runner change. No inference test applies to configuration wiring.

## Progress
- [x] Draft artifact created and read back.
- [x] Structural/safety checks passed (`git diff --check`; placeholder/schema consistency checked; no private Gold content included).
- [x] Work-unit commit recorded: `689a67a052d63862a21ddc09af94fe1d2680d9b8` (`docs(labels): draft Gold-58 prompt candidates`).
- [x] User approved adding the same uncertainty definition to all candidate prompts and accepted the prior Luna/high plus metric recommendation.
- [x] Applied the shared uncertainty contract across all three draft prompts; Candidate 1 now explicitly discloses the one-sentence baseline delta.
- [x] Pinned the Codex command and added exact Windows-shim and POSIX command-vector tests; CLI help supports `--model` and generic `--config`, but subscription entitlement remains unverified.
- [x] Reconciled E007, D004, DATA, D006/COMPETITION, and PROJECT_STATE; aggregate fold feasibility is recorded without IDs or membership.
- [x] Verification: `python -m unittest discover -s tests -p test_subscription_labels.py` — 22 tests passed; `git diff --check` — passed; structural checks confirmed 4 common-sentence occurrences, selected metric/config details, and no stale fold-feasibility claim.
- [x] No Gold reports, labels, IDs, or memberships were accessed; no prompt/model/API invocation or report transfer occurred. Subscription entitlement, usage telemetry, and report-transfer authorization remain unresolved prerequisites.
- [x] Work-unit commit recorded: `3b409db840303e1cfa119fecacfa2616d68bb304` (`feat(labels): pin Luna prompt comparison protocol`).
- [x] Independent review found inconsistent wording between probability of error and “maximally uncertain,” plus a stale Candidate 3 conditional; user confirmed the intended probability-of-error interpretation on 2026-10-11 and E007/D004 now use that meaning.

## Next step
Ask the user to review the corrected exact candidate text. Keep it unapproved, unfrozen, and unrun until explicit review and unresolved runtime/transfer prerequisites are satisfied.
