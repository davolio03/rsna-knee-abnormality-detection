# E010 - SIWC authorized sequential window stopped on validation failure

- **Date:** 2026-10-14
- **Status:** Inconclusive for canary completion.
- **Parent:** E009
- **Hypothesis:** A sequential SIWC window preserves prior checkpoint entries and stops after a validation failure without retrying or issuing later requests.
- **Baseline:** The checkpoint contained three accepted records after E009; those records were to remain unchanged.
- **Changed variable:** No source, prompt, or model change was made. This run exercised the explicitly authorized remaining 11-case sequential window with the existing C2/Luna configuration.
- **Protocol:** One bounded window only; inspect aggregate counts and safe diagnostics. No retry, additional window, or broader cohort was launched.
- **Outcome:** Exit 2 in 35.61 seconds; started 2, completed 1, failed 1, and 9 planned cases were not attempted. The first case completed; the next case at offset 4 failed with safe category `no_literal_match`. Checkpoint count advanced from 3 to 4, and the prior 3 records' byte-line hashes were unchanged. The safe journal reported `stream_completed=true`; refusal or JSON-format detail was not recorded.
- **Interpretation:** The bounded window ended after one completed case and one validation failure, preserving the earlier records. `no_literal_match` is a static diagnostic category, not proof of hallucination; the specific difference is unknown. No label-quality, evidence-relevance, or clinical-validity conclusion follows.
- **Conclusion:** The planned window is incomplete: one case failed and nine were not attempted. No further inference is authorized by this run; any continuation requires separate authorization.

No raw report, evidence, labels, identifiers, credentials, or provider payload is included.
