# E011 - SIWC offset-four invocation with capture opt-in

- **Date:** 2026-10-14
- **Status:** Operationally completed; capture-path execution unverified.
- **Parent:** E010
- **Hypothesis:** A single authorized fresh request with local rejected-evidence capture enabled may complete without modifying prior checkpoint records; only an actual capture artifact would demonstrate the rejected-evidence persistence path.
- **Baseline:** E010 left four checkpoint records, including the first window case, while the next case at offset 4 failed with `no_literal_match`; the specific text discrepancy remains unknown because the rejected evidence was not retained.
- **Changed variable:** This invocation enabled the existing local capture opt-in. The response itself is a fresh stochastic outcome, so response-level variability was not controlled. The prompt, C2 candidate, and Luna model remained unchanged.
- **Protocol:** One separately authorized invocation for one pending case only, with capture opt-in. Observe aggregate execution, checkpoint preservation, and whether the fixed capture artifact exists. No further request or retry.
- **Outcome:** Exit 0 in 24.47 seconds; started 1, completed 1, failed 0. The checkpoint advanced from four to five records, and the prior four records were byte-for-byte unchanged. No safe mismatch category was emitted, and the fixed capture file was absent.
- **Interpretation:** The case completed, but absence of a capture artifact does not show whether the response matched literally or whether any recovery/capture branch ran. The earlier `no_literal_match` remains unexplained. No inference about the prior failure's cause, evidence relevance, label quality, or clinical validity is supported.
- **Conclusion:** One operational request completed and checkpoint preservation was observed. Actual rejected-evidence capture and recovery-path behavior remain untested; nine of 14 cases remain unattempted. No further inference is authorized by this result.

No raw report, evidence, labels, identifiers, account/profile details, credentials, callback data, or provider payload is included.
