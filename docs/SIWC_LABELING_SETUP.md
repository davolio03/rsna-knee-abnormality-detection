# Local Sign in with ChatGPT Labeling

This guide describes the optional local SIWC pilot. It does not authorize report transfer, establish competition or licensing permission, or claim that any labeling run has occurred. The SIWC `label` command sends eligible report text to OpenAI; use it only after obtaining separate, explicit authorization for that external transfer and the specific run.

## Quick path

Run these commands from the repository root in Windows PowerShell. Installation and authentication are not performed by this guide's implementation task.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-siwc.txt
python scripts\siwc_label_reports.py signin
python scripts\siwc_label_reports.py models --profile-id "<profile-id-from-signin>"
```

Choose a slug printed by `models`. `gpt-6-luna` is the adapter default, not a promise of account access; if it is absent from the selected account's visible catalog, choose a listed model instead. The CLI does not accept or send a `reasoning_effort` option.

## Sign in and select a model

`signin` opens the official browser sign-in and listens on `127.0.0.1:1455/auth/callback` by default. The callback is handled by the same local process, request-line logging is suppressed, and credentials are stored through the operating-system keyring. The printed profile ID is safe to use in later commands; do not copy tokens or authorization callback values into command arguments or environment variables.

To reauthorize an existing profile, pass its profile ID to `signin --profile-id`. For a non-default local port, pass `--callback-port <port>`; the default timeout is 180 seconds and can be changed with `--timeout-seconds <seconds>` (1-600).

`models --profile-id ...` lists only models the account catalog marks visible (`visibility == "list"`). The exact slug in parentheses is the value for `--model`. Catalog availability varies by account and can change. SIWC is a preview integration with limitations; see the official [Sign in with ChatGPT preview limitations](https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations) and [models and inference guide](https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference).

## Label command - not run

The following is the command shape derived from the current CLI. **It is documented for a possible later run only. It was not run here, and using it requires separate explicit user authorization before any report is transferred or inference begins.** The two flags are per-run confirmations, not a substitute for that authorization or for verifying external-service, privacy, competition, and licensing constraints.

```powershell
python scripts\siwc_label_reports.py label --train-csv "<training-csv-path>" --gold-exclusions "<58-study-exclusion-csv-path>" --profile-id "<profile-id>" --model "<visible-model-slug>" --max-records 14 --confirm-siwc-oauth --confirm-report-transfer
```

The `--gold-exclusions` CSV must have the sole header `StudyInstanceUID` and exactly 58 unique IDs. **Gold-58 reports are never labeled or sent through this SIWC path**; its IDs are used only to exclude matching rows. The runner also selects only eligible non-Gold rows with blank target fields, rejects duplicate study keys and invalid/blank reports, and requires an explicit batch size from 1 through 14. `--skip-records` defaults to 0; skip plus batch size may not exceed 14.

By default, labels are written to `data/outputs/siwc-c2-report-labels.jsonl` and the payload-free diagnostics journal to `data/outputs/siwc-c2-pilot-diagnostics.jsonl`. These distinct ignored paths keep Candidate 2 output separate from the earlier generic SIWC pilot artifacts. Explicitly selecting either legacy default (`siwc-report-labels.jsonl` or `siwc-pilot-diagnostics.jsonl`) fails before input files or the saved account profile are accessed; use the two `--output` and `--diagnostics` paths above instead. Existing legacy files are not migrated or overwritten.

The SIWC adapter uses the exact frozen Candidate 2 prompt from the comparison registry; it does not construct a parallel prompt/schema or coerce output. The label JSONL remains a resumable checkpoint and validates cohort membership, but this runner does not generally fingerprint prompt/model settings. When using explicit output overrides or changing the selected model, choose a new ignored path rather than assuming prior output is compatible. Optional output and diagnostics paths must remain under `data/outputs`.

When strict label validation rejects a non-boolean JSON label, safe diagnostics may include only `invalid_label_json_type` with one of `string`, `number`, `null`, `array`, `object`, or `other`. They never include the target name, raw value, report text, or runtime type name.

## Authorization and privacy boundary

- The SIWC path is separate from the existing Codex CLI labeler; it does not reuse Codex CLI credentials or accept API keys/access tokens on the command line.
- Inference requires the selected profile's verified `chatgpt.tokens.use.direct` plan-use grant. The adapter uses the official Responses endpoint with `store=false`, streaming enabled, and accepts results only after `response.completed`.
- The report text is sent to an external service during `label`. The local output and diagnostics paths do not make that transfer private from the provider or authorize it under competition/data rules.
- The CLI's required `--confirm-siwc-oauth` and `--confirm-report-transfer` flags record operator assertions only. Do not run until separate authorization for the specific data, destination/service, and operation is explicit and applicable rules have been checked.
- This setup guide records implementation and intended use only. No login, API request, inference, report/data read, or SIWC labeling run was performed for it.

## Official references

- [Sign in with ChatGPT sign-in flow](https://developers.openai.com/siwc/token-sharing-open-source/sign-in)
- [Models and inference](https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference)
- [Preview limitations](https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations)
