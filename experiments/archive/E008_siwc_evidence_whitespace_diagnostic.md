# E008: SIWC Evidence Whitespace Diagnostic

- Date: 2026-10-14
- Status: Completed diagnostic; canary remains incomplete.
- Hypothesis: A literal-evidence rejection may reflect formatting rather than an absent report excerpt.
- Baseline: The user-run C2/Luna canary saved two accepted records and rejected eligible offset2 with generic evidence_not_grounded. The original rejected response was not retained.
- Changed variable: Added only payload-free mismatch classification in commit `5f61b86`; frozen Candidate2, gpt-6-luna, Gold58 exclusions, exact acceptance and checkpoint semantics were unchanged.
- Mechanism: On an exact-substring rejection, compare whitespace, case and canonical Unicode normalization for diagnosis only. Never normalize acceptance or repair model output.
- Validation: Exactly one user-authorized request for eligible offset2 using the same checkpoint/journal and skip2/max1. Check record counts and equality of the existing two records internally, without exposing their contents.
- Result: One attempt started, zero completed, one failed; exit2; evidence_mismatch_kind=whitespace_only. Checkpoint count2; previous_records_unchanged=True. No failed output was saved.
- Runtime observations: The new evidence matched a report substring after whitespace collapsing, but not literally. No additional attempt or full-cohort labeling was launched.
- Offline checks: Targeted8, Responses38 and CLI33 passed. Subscription34/35 and full204/205 passed; the sole failure was the pre-existing synthetic Windows child-timeout cleanup test. Independent privacy/correctness verification and parent Responses38 spot check passed.
- Conclusion: The new response's mismatch is whitespace-only. This does not recover the original response's cause, establish clinical relevance/label correctness, or validate the full14-record canary. Exact acceptance remains unchanged. A future deterministic recovery of a literal source excerpt for whitespace-only discrepancies requires an explicit policy decision; no such repair was performed.
- Safety: No report, evidence, label, study/account identifier, token, or callback content is included in this record. Private outputs remain local and ignored; no remote Git operation occurred.
