# E013 - SIWC bounded resume after earliest-span policy

- **Date:** 2026-10-15
- **Status:** Partial operational result; D012 recovery execution unproven.
- **Parent:** E012
- **Hypothesis:** One bounded resume after D012 can preserve prior checkpoint records and the existing private capture artifact. A completed case without evidence of the whitespace recovery branch does not establish that the updated branch ran.
- **Baseline:** E012 ended with 12 checkpoint records, one failed case, and one remaining case; its captured evidence mapped to two identical contiguous-token spans. D012 subsequently changed only the fresh whitespace-equivalent tie-break to select the earliest source span.
- **Changed variable:** The fresh Responses used the D012 implementation. The prompt/model configuration remained unchanged; fresh provider responses are uncontrolled. Capture opt-in was not enabled so the existing private capture remained untouched.
- **Protocol:** One authorized two-case bounded resume from checkpoint 12. Preserve the prior 12 records and existing capture artifact. No retry, further request, or full-cohort launch.
- **Outcome:** CLI exit 2 in 39.70 seconds; started 2, completed 1, failed 1. Checkpoint advanced from 12 to 13, with the prior 12 records' byte-line hashes unchanged. The existing E012 capture artifact's hash was unchanged. The failure at eligible offset 13 had safe category `case_only`; no new rejected evidence was captured.
- **Interpretation:** The completed case does not prove that D012's whitespace recovery branch executed. `case_only` is a static case-fold substring classification; it does not reveal the actual quote or establish that other aspects of the response are valid. No new evidence text was retained, and no conclusion about quote quality is possible.
- **Conclusion:** The checkpoint now contains 13 of 14 cases, with one failure and none unattempted. The canary remains incomplete; the full cohort was not launched. No retry or policy weakening is authorized by this outcome.

No evidence quote, report, label, identifier, profile detail, credential, callback data, or provider payload is included.
