# E001 — Codex subscription label pilot invocation

- **Date:** 2026-10-10
- **Status:** Inconclusive; invocation error before any valid output
- **Baseline:** No prior report-label inference run
- **Code:** `e0f5d7e` (`fix(labels): name targets in inference prompt`)
- **CLI:** Codex CLI 0.162.1; user confirmed this session used Sign in with ChatGPT. Read-only `codex login status` exited 0 but did not identify the stored authentication method.

## Hypothesis and protocol

The bounded runner can produce valid structured labels for at most 10 non-Gold training studies when invoked through the user's ChatGPT-authenticated Codex CLI session. The preflight loaded the training CSV through the runner's parser to validate rows, then computed only aggregate schema, Gold-ID membership, and target-presence checks: 4,407 training rows, 58 Gold IDs matched, all 58 matching rows had all 12 targets present, 0 partially labeled rows, and 4,349 eligible rows with all targets blank. Report content and target values were not semantically inspected or emitted. Maximum requested records: 10.

Command:

```text
python scripts/label_reports.py --train-csv data/train.csv --gold-exclusions data/private/gold58_study_ids.csv --max-records 10 --subscription-oauth-confirmed --output data/outputs/codex-report-labels.jsonl
```

Output destination is Git-ignored. The output file did not exist after the invocation error.

## Observed result

| Measure | Outcome |
|---|---:|
| Runner invocation attempts | 1 |
| Completed labels | 0 |
| Parsed valid responses | 0 |
| Invocation errors | 1 |
| Error reported | `Codex CLI invocation failed or timed out` |
| Runner wall time | 0.17 s (selection plus invocation; not model latency) |
| Rate-limit outcome | Unknown; no specific rate-limit result reported |
| Per-inference latency | Not measurable; no valid completion |
| Remote request/transmission outcome | Unknown; no retry performed |
| Codex model | Not recorded by the runner |

The runner failed on its first inference attempt. Its sanitized error does not distinguish process-start failure from timeout or determine whether a remote request was accepted; no retry was attempted. No valid response was parsed or persisted.

## Conclusion and limitations

**Inconclusive.** The operational path did not return usable output. No label-quality assessment is possible; there is no independent ground truth comparison, and Gold-58 was excluded from this pilot. Do not infer successful transmission or failure from the generic invocation message. Preserve this failure as evidence and investigate the CLI invocation issue before any further attempt.
