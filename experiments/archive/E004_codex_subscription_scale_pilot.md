# E004 — Codex subscription scale pilot

- **Date:** 2026-10-10
- **Status:** Inconclusive; one batch invocation exited nonzero; labels unvalidated
- **Parent:** E003
- **Code:** `d9c7aa00d90e0c21a2cdc011de77547eccefa829` (`feat(labels): allow nine-record pilot batch`)
- **Authentication:** User-authorized Codex CLI subscription route; no API key was used

## Protocol

One batch invocation was attempted for the nine-record range using `--skip-records 4 --max-records 9`. Local preflight confirmed that the training CSV and Gold exclusion path existed and that the output path existed and was Git-ignored; no file contents were read. The invocation exited nonzero, so the stop-on-first-error rule prevented the eight remaining records from being attempted. No retry was made. The transmission outcome is unknown; no successful result was confirmed.

No CSV, report, output, payload, study identifier, model response, or label was inspected. No credentials, logs, network, Kaggle, push, or PR were accessed. No labels were validated, and no label-quality or clinical conclusion is supported.

## Observed result

| Measure | Outcome |
|---|---|
| Batch arguments | `--skip-records 4 --max-records 9` |
| Invocation attempts | 1 |
| Process result | Exit code 1; elapsed 0.565 seconds |
| Sanitized diagnostic | `Codex CLI exited with status 1; stdout and stderr details are suppressed for privacy.` |
| Confirmed successful result | None |
| Remaining attempts | 8 not made under stop-on-first-error rule |
| Transmission outcome | Unknown |
| Output | Path exists and is Git-ignored; contents not inspected |
| Labels | None validated; provisional/unvalidated |

## Conclusion and limitations

This attempt does not establish whether the remote request was transmitted or whether any records were processed. The operational outcome is inconclusive; the pilot produced no confirmed successful result, and no output or label content was inspected. Do not retry or infer a cause from the sanitized diagnostic. Any further invocation requires fresh explicit authorization and a safe resolution of the unknown transmission state.
