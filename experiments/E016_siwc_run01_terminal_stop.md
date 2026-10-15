# E016 - SIWC run01 terminal stop after initial launch

- **Date:** 2026-10-15
- **Status:** Terminal partial result; bounded 500-record resume not yet run.
- **Parent:** E015
- **Hypothesis:** The original run01 invocation preserves completed checkpoint records when it stops at the first reported failure.
- **Baseline:** E015 records the initial post-launch snapshot of 2 checkpoint records. The run's eligible universe contains 4,349 non-Gold rows.
- **Changed variable:** No source or labeling-policy change is recorded for this terminal result; it is the outcome of the original unbounded invocation.
- **Protocol:** One full-cohort invocation with no automatic retry; stop at the first reported failure and preserve the checkpoint and journal.
- **Outcome:** CLI exit 2 after 260.41 seconds; 16 started, 15 completed, 1 failed, and 15 checkpoint records. The safe diagnostic category was `case_only`.
- **Interpretation:** `case_only` is a static classification and does not establish the actual evidence difference or root cause. No response text or private checkpoint contents are reproduced.
- **Conclusion:** The original run is terminal at 15 saved records. The later user-authorized 500-total target includes these 15 records, so at most 485 new successes are needed. The bounded implementation is in progress; no capped resume has occurred. No label-quality, evidence-relevance, or clinical-validity claim follows.

No report text, prediction, identifier, credential, profile detail, callback data, or provider payload is included.
