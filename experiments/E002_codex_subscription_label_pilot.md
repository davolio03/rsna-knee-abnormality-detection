# E002 — Codex subscription label pilot continuation

- **Date:** 2026-10-10
- **Status:** Inconclusive; stopped after the first nonzero invocation as required
- **Baseline:** E001 had one earlier invocation with unknown transmission outcome; that candidate was not resent
- **Runner:** Existing bounded runner, using the user's confirmed ChatGPT OAuth sign-in; no API key
- **Output handling:** Local Git-ignored output path only. File existence and ignore status were checked; contents were not read.

## Hypothesis and protocol

The bounded runner can complete a one-record inference invocation for eligible non-Gold training data through the user's ChatGPT-authenticated Codex CLI session. The user authorized up to nine further eligible records after the earlier ambiguous attempt. Invocations were sequential, one record each, with no retry. The first offset with a nonzero result terminates the run. Gold-58 was excluded by the runner's configured exclusion input; no identifiers, report text, labels, prompt contents, or output rows were inspected or retained in this note.

## Observed execution

| Measure | Outcome |
|---|---:|
| New invocations attempted | 2 |
| Runner invocations exiting successfully | 1 |
| Nonzero invocations | 1 |
| Exact sanitized command results | Offset 1: exit 0; offset 2: exit 2 |
| Later authorized invocations skipped | 7 |
| Earlier ambiguous attempt, counted against ceiling | 1; unknown transmission, never resent |
| Total consumed against 10-attempt ceiling | 3 |
| Observed timeout | 0 |
| Transmission status for nonzero invocation | Unknown; do not infer that no request was sent |
| Output file | Present and Git-ignored; contents not inspected |

The successful exit is counted as one completed runner invocation. The offset-2 process returned a nonzero code; stdout and stderr were suppressed, so no raw CLI output or payload was captured. The run stopped immediately, and no later offset was attempted. No timeout was observed; cleanup status for the nonzero attempt was not independently established. No retry was made.

## Conclusion and limitations

**Inconclusive.** One invocation completed and the next returned a nonzero status, so the bounded continuation did not establish repeatable pilot feasibility. Transmission of the nonzero attempt remains unknown. The earlier ambiguous attempt also remains unknown. Output contents and label validity were intentionally not inspected; no quality or ground-truth validation was performed. Generated labels remain provisional and unvalidated. Do not retry either consumed offset or continue this run; any future inference requires fresh explicit authorization and a new bounded plan.
