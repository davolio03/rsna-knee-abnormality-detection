# E007 — Gold-58 prompt-strategy candidates

> **PROMPT TEXT APPROVED — FIXED IN LOCAL REGISTRY — ONE AUTHORIZED RUN FAILED.** The user approved the exact candidate prompt text documented here and later explicitly authorized one execution of this fixed comparison. That invocation used the existing ChatGPT OAuth route through Codex CLI and terminated before a valid aggregate summary; no comparison metrics are available. This status records the user authorization and observed run only, not a legal or competition-permissibility conclusion.

## Purpose and boundary

Prepare a small, controlled set of report-labeling prompt strategies for possible comparison under D004. Candidate 1 is the current runner prompt plus the same shared uncertainty sentence used by Candidates 2 and 3; this is a small, explicit baseline delta. Candidate 2 adds only an evidence-workflow block; Candidate 3 adds only a report-interpretation block. Input placeholders, target list, model/runtime settings, and output schema are held constant.

The fixed Codex CLI target is `gpt-6-luna` with `model_reasoning_effort=high`, set explicitly by the local runner. The user authorized one execution through the existing ChatGPT OAuth session, not an API key or separately billed API path. Official Codex documentation describes non-interactive `exec` and ChatGPT authentication (`https://learn.chatgpt.com/docs/non-interactive-mode`, `https://learn.chatgpt.com/docs/auth`). These sources and the local `codex login status` method check establish the selected workflow/authentication method, not an independent provider-terms, competition-rule, licensing, or clinical-validity determination.

## Current implementation baseline

The exact prompt text in `src/rsna_knee/labels/subscription.py` is assembled as follows (with the runtime target list and report represented by placeholders):

```text
Extract provisional binary findings for these targets:
TARGETS:
${TARGET_LIST}

Return only one JSON object with each target mapped to {label:boolean, evidence:string, uncertainty:number from 0 to 1}. Evidence must be a short exact report excerpt (maximum 200 characters); use an empty excerpt when unsupported. Do not add keys.

REPORT:
${REPORT}
```

This reproduces the current runner wording, with runtime interpolation replaced by `${TARGET_LIST}` and `${REPORT}`. Candidate 1 additionally receives the common uncertainty sentence below, so it is a deliberately small departure from the exact current runner prompt. The Python validator additionally enforces exactly the 12 `LABEL_COLUMNS`, exact target-level keys `label`, `evidence`, `uncertainty`, JSON boolean labels, evidence length at most 200 characters, and numeric uncertainty in [0,1].

## Shared contract and fixed inputs

Every candidate below uses exactly these placeholders:

- `${TARGET_LIST}` — the existing ordered list of 12 target names.
- `${REPORT}` — one report supplied at inference time.

Every target must map to exactly this object and no other keys:

```json
{
  "label": true,
  "evidence": "exact report excerpt of at most 200 characters, or empty string when unsupported",
  "uncertainty": 0.0
}
```

The example values illustrate types only; they are not target labels or a calibration guide. `label` remains a JSON boolean, `evidence` an exact report excerpt of at most 200 characters (or empty when unsupported), and `uncertainty` is the model's estimated probability that its binary label is incorrect. This is an estimated error probability, not a generic epistemic-uncertainty scale or a calibration guarantee. Every candidate uses this exact shared instruction:

> For each target, set uncertainty to your estimated probability that its binary label is incorrect: 0 means you estimate the label is correct, and 1 means you estimate the label is incorrect.

The sentence defines the requested meaning, not a calibration guarantee.

## Candidate 1 — Current baseline

**Changed factor:** only the common uncertainty sentence relative to the current runner prompt; no additional strategy block. **Status:** exact prompt text approved and held fixed in the local registry; the comparison invocation failed before any candidate-specific result was available.

```text
Extract provisional binary findings for these targets:
TARGETS:
${TARGET_LIST}

Return only one JSON object with each target mapped to {label:boolean, evidence:string, uncertainty:number from 0 to 1}. Evidence must be a short exact report excerpt (maximum 200 characters); use an empty excerpt when unsupported. Do not add keys.

For each target, set uncertainty to your estimated probability that its binary label is incorrect: 0 means you estimate the label is correct, and 1 means you estimate the label is incorrect.

REPORT:
${REPORT}
```

## Candidate 2 — Evidence-first workflow

**Changed factor:** one added evidence-workflow block; the shared instructions, inputs, targets, and output schema are otherwise unchanged. The user approved it as a fixed comparison candidate, not as an established improvement. The comparison invocation failed; candidate-specific execution and results are unknown.

```text
Extract provisional binary findings for these targets:
TARGETS:
${TARGET_LIST}

Return only one JSON object with each target mapped to {label:boolean, evidence:string, uncertainty:number from 0 to 1}. Evidence must be a short exact report excerpt (maximum 200 characters); use an empty excerpt when unsupported. Do not add keys.

For each target, set uncertainty to your estimated probability that its binary label is incorrect: 0 means you estimate the label is correct, and 1 means you estimate the label is incorrect.

Evidence workflow:
For each target, first locate the exact report passage that may support or contradict that target, then decide the binary label from the report. Do not infer a finding from unrelated text. If no supporting or contradicting passage is present, use an empty evidence string and do not invent evidence. Keep the required output schema unchanged.

REPORT:
${REPORT}
```

## Candidate 3 — Report-interpretation policy

**Changed factor:** one added interpretation-policy block; the shared instructions, inputs, targets, and output schema are otherwise unchanged. The user approved this exact text as a comparison candidate, not as a clinical definition or validated labeling policy. The words “requires approval” inside the candidate block are preserved verbatim as part of that approved prompt text; they are not evidence that the candidate remains unapproved. The comparison invocation failed; candidate-specific execution and results are unknown.

```text
Extract provisional binary findings for these targets:
TARGETS:
${TARGET_LIST}

Return only one JSON object with each target mapped to {label:boolean, evidence:string, uncertainty:number from 0 to 1}. Evidence must be a short exact report excerpt (maximum 200 characters); use an empty excerpt when unsupported. Do not add keys.

For each target, set uncertainty to your estimated probability that its binary label is incorrect: 0 means you estimate the label is correct, and 1 means you estimate the label is incorrect.

Report interpretation policy (illustrative hypothesis; requires approval):
Interpret statements in context. Distinguish explicit negation from a positive finding; do not treat a negated finding as present. Do not convert tentative wording into a positive finding without explicit support. Do not treat a historical or resolved finding as a current finding unless the report explicitly establishes that it remains current. When the report contains no relevant evidence, return the required binary label as false and use empty evidence; the schema has no abstention value. Use the shared probability-of-error definition for the uncertainty field. Do not invent clinical criteria or target definitions. Keep the required output schema unchanged.

REPORT:
${REPORT}
```

## Comparison controls and open decisions

- The intended candidate differences are: shared uncertainty sentence only (Candidate 1's small baseline delta), one additional evidence-workflow block (Candidate 2), or one additional interpretation-policy block (Candidate 3). No examples, target definitions, few-shot demonstrations, or schema changes are introduced.
- The 12 target names, report text, Codex CLI/runtime, `gpt-6-luna`, `model_reasoning_effort=high`, output parsing, and all other inference settings must be held fixed if evaluated. Subscription entitlement for this exact model/effort and authorization to transfer reports remain prerequisites, not established facts.
- The current schema forces every target to true/false. It cannot express abstention, and an absent report mention becomes `false`; this may conflate “not reported” with “not present.” Do not change that schema in this candidate set. Any alternate treatment requires a separate user decision and a revised, re-reviewed candidate set.
- Candidate 3's interpretation-policy block is a hypothesis, not a clinical definition. Candidate results cannot establish clinical validity by themselves.
- The repository materials do not provide complete operational clinical definitions for all targets. Do not infer or fabricate such definitions. If target criteria are needed to label consistently, obtain and approve a source-backed definition set before evaluation.
- No labeled non-Gold development/reference set has been verified. Therefore, do not tune these prompts from pilot outputs or labels. Operational success of the prior pilot is not evidence of label quality.
- **Primary metric:** compute balanced accuracy separately for each of the 12 targets in each validation fold, as the mean of sensitivity and specificity, then macro-average the 12 target scores. Select a candidate by its mean inner-validation macro balanced accuracy. The documented aggregate feasibility screen found all 12 targets represented with positive and negative examples in every configured outer and inner validation fold; MCL remains sparse (9 positives overall and 2 in each inner validation fold), so report per-target results and interpret its estimates cautiously.
- **Tie-break:** if mean inner-validation macro balanced accuracy ties, choose the simpler candidate in this fixed order: Candidate 1 baseline, Candidate 2 evidence workflow, Candidate 3 interpretation policy.
- **Secondary descriptive metric:** report macro Brier score for uncertainty against the binary error indicator: per target, average `(uncertainty - 1[label is incorrect])^2` over validation studies, then average the 12 target scores. Lower is better. This metric is descriptive only and never selects a candidate.
- Report per-target and aggregate metrics for each fold as well as means across inner validation folds. Outer-fold results evaluate the complete inner-selection procedure only; never use them to alter candidates, metrics, or tie-breaking.
- Record planned model, effort, CLI version, elapsed runtime, and client-reported usage only when explicit telemetry is observable. The current CLI help exposes no usage flag and the runner consumes only `agent_message` text, discarding event metadata; do not invent usage parsing or infer tokens. If explicit usage telemetry is not available, mark it unavailable; do not estimate usage or convert subscription consumption into API cost.
- Before any Gold-58 comparison, freeze the approved candidate text, metric set, inner-fold selection/tie-break procedure, and reporting protocol under D004. Only inner training/validation may select a candidate. Outer folds evaluate the complete selection procedure; their outcomes must not guide prompt changes. These nested results are not an independent final test set.

## Approval and execution gate

The exact prompt text is user-approved and held fixed by the local candidate registry. The user explicitly authorized one comparison invocation using the existing ChatGPT OAuth workflow; the local confirmation flags record that assertion but do not verify or grant it. The invocation failed before producing a valid summary, and a second invocation is not authorized by the one-run approval. Keep legal/provider/competition constraints distinct from the observed workflow and execution; no conclusion about them is made here.

## Authorized execution attempt

- **Preflight:** `PYTHONPATH=src python -m unittest discover -s tests -v` passed (76 tests); all approved input, manifest, checkpoint, key, output, and failure paths were confirmed ignored. `codex login status` reported ChatGPT authentication; no account identifier or token was inspected. Aggregate-only local validation confirmed 58 input rows, 12 binary targets, 58 unique manifest-matched keys, and both classes in every 3 outer and 9 inner evaluation fold (144 target-fold checks; minimum positive count 2, minimum negative count 3). No report, study key, or row-level label was printed by the preflight.
- **Invocation:** one execution of `python scripts/run_gold58_prompt_comparison.py` with the authorized private input, canonical private manifest, `--confirm-transfer-prerequisites`, and `--subscription-oauth-confirmed`; fixed runner settings remained the documented candidates, nested 3x3 split, `gpt-6-luna`/high, and up to 406 planned requests. No source, prompt, split, or model changes occurred.
- **Outcome:** runner exited 1 after approximately 40.6 seconds with its fixed message `candidate evaluation failed; details suppressed`. Safe checkpoint metadata contains 2 completed coordinates and 1 failed coordinate, with only `stage=response_validation` and `cause_code=invalid_response` recorded for the failure. No raw stderr, provider payload, report, ID, or checkpoint coordinate was inspected. No valid aggregate summary or comparison metrics were produced; later coordinates were not issued by this invocation. Preserve the checkpoint/key and do not retry without fresh explicit authorization.
- **Interpretation boundary:** this was one failed nested-CV comparison attempt, not a successful experiment result, independent test, or clinical validation. Do not use any outer-fold result to revise the frozen candidates.

## Provenance

- Baseline prompt: `src/rsna_knee/labels/subscription.py`.
- Policy: `docs/DECISIONS.md` D004.
- Phase context: `docs/ROADMAP.md`, Phases 2–3.
- Operational pilot context: `experiments/E001_codex_subscription_label_pilot.md` and `experiments/E006_codex_subscription_utf8_batch.md`; both distinguish operational feasibility from label validity.
