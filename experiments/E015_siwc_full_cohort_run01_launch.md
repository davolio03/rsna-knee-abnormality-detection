# E015 - SIWC full-cohort run01 launch snapshot

- **Date:** 2026-10-15
- **Status:** Launched; terminal outcome pending.
- **Parent:** E014
- **Hypothesis:** One explicitly authorized full-cohort invocation can make sequential checkpointed progress without retrying or importing the canary.
- **Baseline:** The separate 14-case canary had operationally completed across multiple attempts and policy revisions. The new run uses isolated output and checkpoint paths; no canary records are imported or resumed.
- **Changed variable:** Scope expanded from the completed canary to the eligible non-Gold cohort. No source, prompt, or labeling-policy change is recorded for this launch.
- **Protocol:** One `label-all` invocation for 4,349 eligible non-Gold rows. Process sequential windows of 14, checkpoint each completed case, set retries to zero, and stop on the first failure or provider limit. Do not automatically restart or resume an in-flight row.
- **Outcome:** At the first safe post-launch inspection, private job metadata reported the process running, with 3 started, 2 completed, 0 failed, and 2 checkpoint records. This is a launch-time snapshot, not a current-status guarantee; terminal exit status and final counts are not yet known.
- **Runtime note:** A prior-runtime estimate is approximately 29 hours 34 minutes, but duration varies and this is not a completion promise. Sleep or managed-session/runtime termination may interrupt execution; completed checkpoint records are preserved. Consult observed job status before any separately authorized resume.
- **Conclusion:** The full-cohort run has launched and shown initial progress. No terminal result, labeling quality, evidence relevance, or clinical validity is established.

No report text, prediction, identifier, credential, profile detail, callback data, or provider payload is included.
