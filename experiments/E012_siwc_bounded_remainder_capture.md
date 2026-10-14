# E012 - SIWC bounded remainder run with one local capture

- **Date:** 2026-10-14
- **Status:** Partial operational completion; conditional full-cohort condition unmet.
- **Parent:** E011
- **Hypothesis:** One authorized run of the nine remaining canary cases with local rejected-evidence capture enabled can preserve prior checkpoint records and retain evidence for a fresh rejected response within the configured local size bound.
- **Baseline:** E011 left five checkpoint records and nine cases unattempted. Its one-case invocation used capture opt-in but produced no capture artifact; actual capture behavior remained untested.
- **Changed variable:** No code, prompt, or model change. Capture opt-in was enabled for this bounded invocation; the fresh model response remained stochastic. Account, SIWC Responses service, C2 prompt candidate, and Luna model were unchanged.
- **Protocol:** One authorized request window for the nine remaining canary cases, without retry. Continue to full-cohort labeling only if every remaining case completes and prior checkpoint records remain unchanged. Stop on any failure.
- **Outcome:** Exit 2 in 143.52 seconds; started 8, completed 7, failed 1, with one case unattempted. The checkpoint advanced from five to twelve; prior five records were byte-for-byte unchanged. The failed case at eligible offset 12 emitted `whitespace_only`; one fixed local capture was written and verified, containing 143 UTF-8 bytes with the evidence string and eligible offset. The full-cohort condition was not met, and `label-all` was not launched.
- **Interpretation:** A safe in-memory comparison found two normalized substring matches and two identical contiguous-token span matches, so there was no unique source occurrence. The existing unique-span recovery correctly rejects this ambiguity. This comparison does not identify the earlier offset-4 failure or permit broader normalization. Only the failed response was captured; no conclusion applies to other uncaptured responses.
- **Conclusion:** Seven additional cases completed and checkpoint preservation was observed, but the canary remains incomplete: one case failed and one was not attempted. The conditional full-cohort run was not authorized by its unmet success condition. No label-quality, evidence-relevance, or clinical-validity claim follows. Any further inference requires a separate user decision.

No report or evidence quote, label, identifier, profile detail, credential, callback data, or provider payload is included. The captured evidence remains in the fixed local private slot for manual retention management.
