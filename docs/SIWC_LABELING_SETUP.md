# Local Sign in with ChatGPT Labeling

This guide describes the optional local SIWC pilot. It does not authorize report transfer, establish competition or licensing permission, or claim that any labeling run has occurred. The SIWC `label` command sends eligible report text to OpenAI; use it only after obtaining separate, explicit authorization for that external transfer and the specific run.

## Quick path

Run these commands from the repository root in Windows PowerShell. Installation and authentication are not performed by this guide's implementation task.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --use-feature=truststore -r requirements-siwc.txt
python scripts\siwc_label_reports.py signin
python scripts\siwc_label_reports.py models --profile-id "<profile-id-from-signin>"
```

`requirements-siwc.txt` includes `truststore==0.10.5`. Each SIWC command initializes the system certificate store in its own process before creating authentication or Responses clients. Certificate-chain and hostname verification remain enabled. The SSL-module adjustment is process-local; no persistent Python setting or operating-system trust store is changed. If the CLI reports that system certificate support is unavailable, install the SIWC requirements in the active environment and rerun the command.

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

For a fresh canary, override both C2 paths with new, distinct ignored paths that do not already exist. In particular, do not resume or overwrite the earlier 14-record checkpoint; its evidence guard makes it unsafe to reuse.

By default, labels are written to `data/outputs/siwc-c2-report-labels.jsonl` and the payload-free diagnostics journal to `data/outputs/siwc-c2-pilot-diagnostics.jsonl`. These distinct ignored paths keep Candidate 2 output separate from the earlier generic SIWC pilot artifacts. Explicitly selecting either legacy default (`siwc-report-labels.jsonl` or `siwc-pilot-diagnostics.jsonl`) fails before input files or the saved account profile are accessed; use the two `--output` and `--diagnostics` paths above instead. Existing legacy files are not migrated or overwritten.

The SIWC adapter uses the exact frozen Candidate 2 prompt from the comparison registry; it does not construct a parallel prompt/schema or coerce output. The label JSONL remains a resumable checkpoint and validates cohort membership, but this runner does not generally fingerprint prompt/model settings. When using explicit output overrides or changing the selected model, choose a new ignored path rather than assuming prior output is compatible. Optional output and diagnostics paths must remain under `data/outputs`.

For each target, non-empty evidence must occur literally in that row's source report. The runner rejects invalid fresh evidence before saving it and checks cached evidence before skipping completed records; on a mismatch it stops and preserves the existing output file. This is a provenance check only—it does not establish clinical correctness.

When strict label validation rejects a non-boolean JSON label, safe diagnostics may include only `invalid_label_json_type` with one of `string`, `number`, `null`, `array`, `object`, or `other`. They never include the target name, raw value, report text, or runtime type name.

## Full eligible non-Gold cohort

The user has separately authorized this rollout: implement the scalable runner, perform a fresh canary, then label all eligible non-Gold rows. The implementation task itself performs no live provider calls; the canary and full run are separate later operations.

Use `label-all` only for the explicitly authorized run. It validates the exact 58-ID Gold exclusion list, the training-row schema, duplicate study keys, and eligibility (all targets blank and report nonblank), then reports the eligible non-Gold count before constructing the authenticator or beginning inference. The selected model is fixed to the frozen Candidate 2 `gpt-6-luna` configuration. Gold-58 remains excluded.

Choose fresh, distinct output and diagnostics paths under ignored `data/outputs/` that have never been used by a prior pilot. The command refuses existing paths unless `--resume` is explicit, rejects the old generic and C2 canary defaults, and never merges their contents. Each checkpoint record is schema-validated and atomically persisted; the diagnostics journal is payload-free. Processing uses sequential windows of at most 14 eligible rows, with one report per Responses request. Provider failures stop the run. A recorded usage denial blocks `--resume`; do not retry or bypass provider limits.

```powershell
python scripts\siwc_label_reports.py label-all --train-csv "<training-csv-path>" --gold-exclusions "<58-study-exclusion-csv-path>" --profile-id "<profile-id>" --model "gpt-6-luna" --output "data/outputs/siwc-c2-cohort-run01.jsonl" --diagnostics "data/outputs/siwc-c2-cohort-run01-diagnostics.jsonl" --confirm-siwc-oauth --confirm-report-transfer
```

The example paths are names to select only if they do not exist; choose a new unique pair for the fresh run. If the process is interrupted, resume only by repeating the same paths and adding `--resume`:

```powershell
python scripts\siwc_label_reports.py label-all --train-csv "<training-csv-path>" --gold-exclusions "<58-study-exclusion-csv-path>" --profile-id "<profile-id>" --model "gpt-6-luna" --output "data/outputs/siwc-c2-cohort-run01.jsonl" --diagnostics "data/outputs/siwc-c2-cohort-run01-diagnostics.jsonl" --resume --confirm-siwc-oauth --confirm-report-transfer
```

An interruption during an in-flight request leaves that row uncheckpointed. Explicit resume may submit that one unresolved row again if the provider completed the request but the client did not receive its response; the runner never retries automatically. Do not resume after a provider usage denial.

## Authorization and privacy boundary

- The SIWC path is separate from the existing Codex CLI labeler; it does not reuse Codex CLI credentials or accept API keys/access tokens on the command line.
- Inference requires the selected profile's verified `chatgpt.tokens.use.direct` plan-use grant. The adapter uses the official Responses endpoint with `store=false`, streaming enabled, and accepts results only after `response.completed`.
- Rejected non-empty evidence may include a fixed `evidence_mismatch_kind` diagnostic (`whitespace_only`, `case_only`, `unicode_only`, or `no_literal_match`). This is classification only: acceptance remains an exact substring check, and `no_literal_match` does not prove hallucination. The journal and CLI never include report or evidence text.
- For fresh SIWC Responses only, a non-literal evidence mismatch may be recovered when the complete contiguous token sequence matches after applying Python `str.lower()` to each token. Repeated or overlapping matches select the earliest original report span; the stored evidence is the exact source substring from its first through last token, then the unchanged literal guard runs. No word, token, punctuation, accent, or token-order change is permitted; casefold expansion and Unicode normalization are not used. Exact literal matches remain unchanged. Cached records still use strict literal validation and are never rewritten. The separate safe `case_only` diagnostic may be broader than this recovery predicate; it is not proof of recovery eligibility. D013 has only synthetic verification and does not establish a live response or canary result.
- The report text is sent to an external service during `label`. The local output and diagnostics paths do not make that transfer private from the provider or authorize it under competition/data rules.
- The CLI's required `--confirm-siwc-oauth` and `--confirm-report-transfer` flags record operator assertions only. Do not run until separate authorization for the specific data, destination/service, and operation is explicit and applicable rules have been checked.
- This guide documents implementation constraints, not authorization. The single separately authorized E008 retry is summarized in `PROJECT_STATE.md` and `experiments/E008_siwc_evidence_whitespace_diagnostic.md`; no report/evidence/label text or credential is reproduced here.

## Optional rejected-evidence capture

The bounded `label` command supports an explicit `--capture-rejected-evidence` opt-in. It is off by default and unavailable on `label-all`. On the first fresh, schema-valid response rejected by the exact source-evidence guard, the CLI may create the single fixed file `data/private/siwc-rejected-evidence.json`. The JSON object contains only the rejected evidence string and its eligible-row offset; it does not include report text, study identifiers, target names, labels, provider response bodies, or credentials. Cached records, successful responses, and schema/provider failures are not captured.

Capture accepts at most 16 KiB of UTF-8 evidence and never truncates. The destination is created exclusively; an existing file or unsafe path (including a symlink/reparse point or alias) is refused, and there is no arbitrary destination option. Capture-write failures do not replace the original validation error or change the checkpoint. Remove the file manually when no longer needed: it is not automatically expired or encrypted, and Git ignore rules are not a privacy or retention control. Treat its evidence as sensitive local data and do not share it.

```powershell
python scripts\siwc_label_reports.py label --train-csv "<training-csv-path>" --gold-exclusions "<58-study-exclusion-csv-path>" --profile-id "<profile-id>" --model "gpt-6-luna" --max-records 1 --confirm-siwc-oauth --confirm-report-transfer --capture-rejected-evidence
```

The flag affects only optional local capture after a fresh rejection; it does not relax literal evidence acceptance, authorize report transfer, or permit a retry. No live capture has been exercised.

## Official references

- [Sign in with ChatGPT sign-in flow](https://developers.openai.com/siwc/token-sharing-open-source/sign-in)
- [Models and inference](https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference)
- [Preview limitations](https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations)
- [Truststore documentation](https://truststore.readthedocs.io/en/latest/)
