# Scripts

Place focused CLI or automation entry points here when a repeatable task needs one. Keep reusable logic in `src/`, expose inputs/configuration explicitly, and avoid hard-coded machine paths or credentials.

## Synthetic submission smoke test

Run `python scripts/smoke_submission.py` to create and read back an ignored local, synthetic-only `submission.csv` at `outputs/phase0-smoke/submission.csv`. Pass `--output PATH` to choose another path. The command refuses to overwrite an existing file.

The fixture uses three fictitious study identifiers and twelve constant 0.5 scores. It checks the exact ordered competition header, row width, identifier completeness/uniqueness, and finite probabilities in `[0, 1]`. It does not access competition data or the network, and it does not establish model quality, local competition-data compatibility, or Kaggle runtime/submission behavior. Generated outputs are ignored by Git.

## Subscription-backed report-label pilot

`python scripts/label_reports.py --train-csv <local-train.csv> --gold-exclusions <private-one-column-csv> --max-records <1-14> --subscription-oauth-confirmed` labels eligible reports through `codex exec` and resumes from ignored `data/outputs/codex-report-labels.jsonl`. The combined `--skip-records` offset plus `--max-records` batch size must not exceed 14; for example, `--skip-records 5 --max-records 9` selects fresh eligible rows at offsets 5–13 in one command. The exclusion CSV must have exactly one `StudyInstanceUID` column and exactly 58 unique verified IDs. Every row must provide `StudyInstanceUID`, `Report`, and all 12 target columns; any nonblank target excludes that row. The Gold exclusion file is mandatory and must be verified before use.

Every actual run appends fixed-schema attempt-start and outcome records to ignored `data/outputs/codex-pilot-diagnostics.jsonl` (override with `--diagnostics`). The journal stores only numeric process status/errno when available, allowlisted cause codes with fixed explanations, stage, eligible-row offset, elapsed milliseconds, and aggregate started/completed/failed counts. It never stores CLI output, raw errors, report text, identifiers, model text, credentials, or paths. The journal must remain under `data/outputs` and must not alias the label output; unsafe or failed journal writes stop the run before another inference. Known CLI patterns are classified conservatively; unrecognized failures remain `unknown`. The runner stops on the first error and does not retry.

Subprocess text input is explicitly encoded as UTF-8 so Unicode report text is transported consistently across Windows code pages.

**This command sends report text to the Codex service.** The required confirmation flag is an operator attestation, not an authentication check: before a real run, independently verify that Codex CLI is authenticated through ChatGPT subscription OAuth, not API-key billing, and confirm applicable provider/competition terms. This implementation step did not run the command or transmit data. Keep the exclusion file and generated labels local; output is restricted to ignored `data/outputs/`.
