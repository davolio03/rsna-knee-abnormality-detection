# E009 - SIWC third-case bounded confirmation

- **Date:** 2026-10-14
- **Status:** Inconclusive for recovery-path execution.
- **Parent:** E008
- **Hypothesis:** Fresh-only unique whitespace span recovery can preserve exact source evidence and checkpoint state when a whitespace-only mismatch occurs.
- **Baseline:** E008's one retry failed with `whitespace_only`; the checkpoint remained at two records, and the discarded original response remains unclassified.
- **Changed variable:** The approved single third-case invocation used the fresh-only recovery implementation from local commit `0c2ff4b`; C2 prompt and Luna model were unchanged.
- **Protocol:** One separately authorized third-case request only. The run used the existing checkpoint and reported only aggregate counts and a static mismatch category, if present. No additional case or retry was launched.
- **Outcome:** Exit 0; started 1, completed 1, failed 0 in 21.55 seconds. Checkpoint count advanced to three, with the prior two records unchanged. No mismatch diagnostic was emitted.
- **Interpretation:** The bounded request completed, but the result does not show that recovery code executed; the response may already have matched the report literally. Exact substring acceptance and checkpoint preservation remain required. No inference about clinical relevance, label quality, or recovery behavior on a live whitespace mismatch is justified.
- **Conclusion:** Operational completion is confirmed for this single request. Recovery-path behavior remains unverified on live mismatched evidence. Further inference or any broader normalization policy requires separate authorization.

No raw report, evidence, labels, identifiers, profile details, callback data, or provider payload is included.
