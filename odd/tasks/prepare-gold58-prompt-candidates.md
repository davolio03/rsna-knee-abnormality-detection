# Feature: Gold-58 Prompt Candidate Drafts

## Objective
Prepare a compact, controlled set of report-labeling prompt candidates for human review before any Gold-58 evaluation.

## Why
The local nested 3x3 splitter and aggregate feasibility screen are complete. D004 permits only a short predeclared strategy set, selected within outer-training data; outer folds must not guide prompt iteration.

## Scope
- Draft three prompt candidates from the current Codex CLI baseline.
- Keep target names, model settings, input reports, and output schema fixed; change one strategy block per candidate.
- Document which candidate is baseline and which changes evidence workflow or interpretation guidance.
- Clearly mark drafts as unapproved/unfrozen until the user reviews them and approves the metric/selection protocol.
- Do not run prompts, call Codex or external APIs, inspect or access Gold reports, generate labels, or use outer-fold evidence.

## Constraints
- Preserve the current JSON fields: `label`, exact quoted `evidence`, and `uncertainty` in [0,1].
- Do not invent clinical definitions; flag them as a dependency if existing competition/project definitions are insufficient.
- Keep Gold-58 IDs, reports, row-level labels, and fold membership out of artifacts.
- Direct local commits to `main` are preferred; no push or PR.
- RDD remains clone-local off. No native review invocation.
- TDD is not applicable to draft-only prompt documentation; verify schema compatibility structurally, no runtime execution.

## Tasks
- [x] Read current prompt implementation and project policy, then create baseline and two controlled candidate variants.
- [x] Verify output contract consistency and document unresolved decisions/risks; do not execute candidates.

## Acceptance criteria
- Three identifiable prompt templates; baseline matches current behavior.
- Candidate variants and changed factors are explicit and comparable.
- All templates retain same inputs and JSON output schema.
- Artifact states not approved/not frozen, and no evaluation can start before user review and protocol freeze.
- No private Gold content or identifiers included.

## Route and verification
- Route: delegated direct; preparation reading and multi-section prompt artifact should be handled together by one writer.
- Read-only review of current code/docs; structural verification of exact schema/target placeholders; no prompt invocation.
- No runtime test applies because this task creates prompt drafts only; executing a prompt is explicitly out of scope.

## Progress
- [x] Draft artifact created and read back.
- [x] Structural/safety checks passed (`git diff --check`; placeholder/schema consistency checked; no private Gold content included).
- [x] Work-unit commit recorded: `689a67a052d63862a21ddc09af94fe1d2680d9b8` (`docs(labels): draft Gold-58 prompt candidates`).

## Next step
Present `experiments/E007_gold58_prompt_candidates.md` for user review. Candidate set remains unapproved/unfrozen. Only after acceptance, freeze the candidate set plus selection and metrics protocol before any nested Gold evaluation.
