# E007 — Gold-58 prompt-strategy candidate drafts

> **DRAFT — UNFROZEN — NOT APPROVED — NOT RUN.** These are candidates for human review only. They do not authorize any inference or evaluation. No Gold-58 report, label, identifier, or fold membership was accessed to prepare this document.

## Purpose and boundary

Prepare a small, controlled set of report-labeling prompt strategies for possible comparison under D004. The current runner implementation is the baseline. Candidate 2 adds only an evidence-workflow block; Candidate 3 adds only a report-interpretation block. Input placeholders, target list, model/runtime settings, and output schema are held constant.

The Codex CLI prompt call does not set an explicit model in this function; model resolution may depend on the user's CLI configuration or session. If these drafts are later approved, record the CLI version, resolved model when observable, and relevant non-secret runtime settings, and verify that they are identical across candidates. Do not copy secrets or private configuration values into experiment records.

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

This is the baseline verbatim in meaning and wording, with only runtime interpolation replaced by `${TARGET_LIST}` and `${REPORT}`. The current Python validator additionally enforces exactly the 12 `LABEL_COLUMNS`, exact target-level keys `label`, `evidence`, `uncertainty`, JSON boolean labels, evidence length at most 200 characters, and numeric uncertainty in [0,1].

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

The example values illustrate types only; they are not target labels or a calibration guide. `label` remains a JSON boolean, `evidence` an exact report excerpt of at most 200 characters (or empty when unsupported), and `uncertainty` a number in [0,1]. The semantic meaning and calibration of uncertainty remain unresolved and require an explicit policy decision before freezing.

## Candidate 1 — Current baseline

**Changed factor:** none. **Status:** baseline draft, not approved or frozen.

```text
Extract provisional binary findings for these targets:
TARGETS:
${TARGET_LIST}

Return only one JSON object with each target mapped to {label:boolean, evidence:string, uncertainty:number from 0 to 1}. Evidence must be a short exact report excerpt (maximum 200 characters); use an empty excerpt when unsupported. Do not add keys.

REPORT:
${REPORT}
```

## Candidate 2 — Evidence-first workflow

**Changed factor:** one added evidence-workflow block; the shared instructions, inputs, targets, and output schema are otherwise unchanged. This is a hypothesis to test, not an approved improvement.

```text
Extract provisional binary findings for these targets:
TARGETS:
${TARGET_LIST}

Return only one JSON object with each target mapped to {label:boolean, evidence:string, uncertainty:number from 0 to 1}. Evidence must be a short exact report excerpt (maximum 200 characters); use an empty excerpt when unsupported. Do not add keys.

Evidence workflow:
For each target, first locate the exact report passage that may support or contradict that target, then decide the binary label from the report. Do not infer a finding from unrelated text. If no supporting or contradicting passage is present, use an empty evidence string and do not invent evidence. Keep the required output schema unchanged.

REPORT:
${REPORT}
```

## Candidate 3 — Report-interpretation policy

**Changed factor:** one added interpretation-policy block; the shared instructions, inputs, targets, and output schema are otherwise unchanged. This is an illustrative candidate hypothesis requiring approval, not a clinical definition or approved labeling policy.

```text
Extract provisional binary findings for these targets:
TARGETS:
${TARGET_LIST}

Return only one JSON object with each target mapped to {label:boolean, evidence:string, uncertainty:number from 0 to 1}. Evidence must be a short exact report excerpt (maximum 200 characters); use an empty excerpt when unsupported. Do not add keys.

Report interpretation policy (illustrative hypothesis; requires approval):
Interpret statements in context. Distinguish explicit negation from a positive finding; do not treat a negated finding as present. Preserve the report's uncertainty rather than converting tentative wording into a certain positive. Do not treat a historical or resolved finding as a current finding unless the report explicitly establishes that it remains current. When the report contains no relevant evidence, return the required binary label as false and use empty evidence; the schema has no abstention value. Use the uncertainty field to indicate ambiguity only after its meaning and scale have been approved; do not invent clinical criteria or target definitions. Keep the required output schema unchanged.

REPORT:
${REPORT}
```

## Comparison controls and open decisions

- The only intended candidate differences are: no change (Candidate 1), one evidence-workflow block (Candidate 2), or one interpretation-policy block (Candidate 3). No examples, target definitions, few-shot demonstrations, or schema changes are introduced.
- The 12 target names, report text, CLI/runtime, model, output parsing, and all other inference settings must be held fixed if evaluated. Codex CLI model resolution/configuration may be difficult to freeze; resolve and document that before any run without recording secrets.
- The current schema forces every target to true/false. It cannot express abstention, and an absent report mention becomes `false`; this may conflate “not reported” with “not present.” Do not change that schema in this candidate set. Any alternate treatment requires a separate user decision and a revised, re-reviewed candidate set.
- Candidate 3's uncertainty wording changes the interpretation policy. Its numeric meaning, calibration, and relationship to the binary label are not specified here and must be decided before approval. Candidate results cannot establish clinical validity by themselves.
- The repository materials do not provide complete operational clinical definitions for all targets. Do not infer or fabricate such definitions. If target criteria are needed to label consistently, obtain and approve a source-backed definition set before evaluation.
- No labeled non-Gold development/reference set has been verified. Therefore, do not tune these prompts from pilot outputs or labels. Operational success of the prior pilot is not evidence of label quality.
- The prior aggregate screen reported sparse MCL positives (9 of 58); metric choice and treatment of small/constant-target folds materially affect interpretation. This document does not select a primary metric, aggregation rule, tie-break, uncertainty summary, or invalid-fold policy.
- Before any Gold-58 comparison, freeze the approved candidate text, metric set, inner-fold selection/tie-break procedure, and reporting protocol under D004. Only inner training/validation may select a candidate. Outer folds evaluate the complete selection procedure; their outcomes must not guide prompt changes. These nested results are not an independent final test set.

## Approval and execution gate

This document remains **DRAFT / UNFROZEN** until the user reviews the exact text and separately approves a complete evaluation protocol. No prompt has been executed, no labels were generated or inspected, and no API/Codex invocation was made for this draft. Do not run these candidates until candidate text, model/runtime controls, metrics, and inner-selection rules are frozen and all D004 and competition constraints are satisfied.

## Provenance

- Baseline prompt: `src/rsna_knee/labels/subscription.py`.
- Policy: `docs/DECISIONS.md` D004.
- Phase context: `docs/ROADMAP.md`, Phases 2–3.
- Operational pilot context: `experiments/E001_codex_subscription_label_pilot.md` and `experiments/E006_codex_subscription_utf8_batch.md`; both distinguish operational feasibility from label validity.
