# E003 — Codex subscription diagnostic pilot

- **Date:** 2026-10-10
- **Status:** One authorized runner invocation completed; labels unvalidated
- **Parent:** E002
- **Code:** `8be96961fe414e580b172b6b7e89937650a37b15` (`fix(labels): preserve sanitized Codex exit status`)
- **Authentication:** User-confirmed ChatGPT OAuth subscription through Codex CLI; no API key

## Protocol

One new eligible non-Gold, fully unlabeled training record was authorized. Previously consumed offsets 0–2 were excluded; exactly one invocation used offset 3 and `--max-records 1`. No retry or additional request was made. Gold-58 was excluded by the configured local mask. No report text, identifiers, output rows, labels, prompt content, credentials, or raw CLI output were inspected or retained.

## Observed result

Command:

```text
python scripts/label_reports.py --train-csv data/train.csv --gold-exclusions data/private/gold58_study_ids.csv --max-records 1 --skip-records 3 --subscription-oauth-confirmed --output data/outputs/codex-report-labels.jsonl
```

| Measure | Outcome |
|---|---:|
| New invocations | 1 |
| Runner result | Success; exit 0 |
| Sanitized Codex CLI status | Not reported on success |
| Elapsed time | 16.9 seconds |
| Retry/additional requests | 0 |
| Output file | Present and Git-ignored; contents not inspected |

## Conclusion and limitations

The single authorized Codex OAuth inference invocation completed successfully. This is an operational result only: output contents and label validity were not inspected, labels remain provisional and unvalidated, and no ground-truth comparison was performed. No inference beyond this one request is authorized by this experiment.
