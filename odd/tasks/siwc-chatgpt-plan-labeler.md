# Local ChatGPT Plan Labeling

## Objective

Add a local, official Sign in with ChatGPT OAuth path to the existing Python report-labeling pilot so the user can authenticate with their ChatGPT plan and later run explicitly authorized labeling without an API key or Codex CLI credential reuse.

## Problem and rationale

The current pilot invokes Codex CLI through a subscription adapter. The repository has no Node package manifest, declared Python dependencies, or OAuth/OIDC implementation. Official OpenAI guidance now documents an open-source local Sign in with ChatGPT flow for eligible Responses API inference. Keep this Python project intact, isolate authentication from the existing pilot, and require the official plan-use grant before requests.

## Authorized scope and constraints

- User authorized local repository changes for the official OAuth integration.
- This authorization does not include launching inference, transmitting reports, reading private data, inspecting credentials, or accessing remote accounts.
- Do not use API keys, Codex CLI session tokens, undocumented ChatGPT backend endpoints, or any existing ambient credentials.
- Gold-58 remains excluded from training and ordinary label generation; preserve existing exact exclusion checks.
- Implement the documented local dynamic-registration flow with fresh state, OIDC nonce, PKCE, callback validation, issued client ID persistence, validated identity, and secure token storage/refresh.
- A verified chatgpt.tokens.use.direct grant is mandatory; identity-only sign-in must not enable inference.
- Responses calls must use https://api.openai.com/v1/responses, store=false, stream=true, and consider only response.completed a success.
- Preserve current checkpointing, privacy-safe diagnostics, explicit CLI authorization guard, and user-selected Luna/high model unless the user's account does not expose it.
- Do not alter the uncommitted user edits in docs/DECISIONS.md.
- Strict TDD applies from the user's prior explicit choice for this report-label implementation: RED, then GREEN, then REFACTOR. Source: historical explicit user choice, not project config. Test runner: python -m unittest discover -s tests.
- No project package/dependency manifest exists. Keep the implementation Python-only and add a narrowly scoped, pinned dependency declaration for Authlib (OAuth/OIDC validation), keyring (OS-backed token storage), and the official OpenAI Python SDK (Responses streaming). The Node/npm versions the user checked are not needed for this route.

## Route and delivery

- Route: delegated direct implementation after delegated read-only mapping. Writer trigger: multiple non-trivial Python, CLI, test, dependency, and documentation files.
- Trigger evidence: authentication lifecycle and runner integration span multiple files and require testing across OAuth, persistence, and streaming boundaries.
- Forecast: approximately 350 authored changed lines; generated lockfiles excluded.
- Delivery strategy: ask-on-risk (default). No push or pull request is authorized.
- Work-unit commit is required on the current feature branch before task closure; do not include the pre-existing docs/DECISIONS.md edit.

## Checklist

### SIWC-01: Build the local OAuth profile boundary

- [x] Implement first-time dynamic registration, loopback callback, state/nonce/PKCE verification, token exchange, OIDC identity validation, and token refresh using Authlib.
- [x] Store OAuth secrets through the operating-system keyring; explicitly reject a null or insecure backend and never fall back to plaintext files.
- [x] Require the exact plan-use scope before exposing an inference-capable session.
- [x] Add synthetic unit tests covering callback tampering, declined/missing scope, identity mismatch, invalid tokens, refresh rotation, and storage failure.
- [x] Acceptance: authentication code never logs or returns credentials; no plaintext fallback is allowed.

### SIWC-02: Integrate completed Responses inference

- [ ] Add a Responses adapter using the selected OAuth profile and account-available models.
- [ ] Inject it through the existing run_pilot(..., infer=...) boundary, retaining the existing Codex CLI path and explicit CLI guard behavior.
- [ ] Require store=false, stream=true, and a terminal response.completed; fail on incomplete, failed, or interrupted streams without exposing report text.
- [ ] Add mocked tests for request shape, streaming completion/failure, and no-auth/no-scope rejection.
- [ ] Acceptance: no live inference or report transfer is run during this task.

### SIWC-03: Document local setup and verify

- [ ] Document installation, local sign-in, how the user selects the account's available model, and the exact explicit command needed for a later labeling run.
- [ ] Record the new provider, current status, and remaining external-transfer/data authorization boundary in the appropriate project records without changing Phase 2 status.
- [ ] Run focused and full unit tests; inspect changes for credential/data leakage.
- [ ] Commit the completed work unit using a Conventional Commit message without AI attribution.

## Progress and evidence

- 2026-10-13: User authorized implementation only; no report transfer or inference authorized/performed.
- 2026-10-13: Read-only mapping confirmed the Python runner's injectable inference function, test suite, and historical strict-TDD preference. No tests or data were accessed during mapping.
- 2026-10-13: SIWC-01 implemented in `src/rsna_knee/labels/siwc_auth.py` with isolated synthetic coverage in `tests/test_siwc_auth.py`; pinned Authlib/keyring dependencies are in `requirements-siwc.txt`. Focused tests passed (15 tests). Full suite ran 102 tests: 101 passed; `test_subscription_labels.SubscriptionLabelContractTests.test_windows_timeout_kills_synthetic_child_process` failed because its synthetic child survived timeout cleanup. No OAuth sign-in, external request, report/data read, or inference was run.
- 2026-10-13: SIWC-01 correction: fixed a literal `\n` SyntaxError at the module end and serialized refresh by profile so concurrent callers re-check the stored expiry and only one consumes a rotating refresh token. Added compile and concurrent-refresh regression tests. RED: `python -m unittest discover -s tests -p test_siwc_auth.py` failed at import with the reported SyntaxError. GREEN: the same focused command passed all 17 tests; `python -m py_compile src/rsna_knee/labels/siwc_auth.py` passed. No dependency install, sign-in, network request, inference, report/data read, or credential inspection was performed.
- Official reference: https://developers.openai.com/siwc/token-sharing-open-source/sign-in
- Official inference contract: https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference
- Official integration example: https://developers.openai.com/cookbook/articles/sign-in-with-chatgpt

## Next step

Implement SIWC-02 (the isolated Responses adapter) with strict TDD. Do not launch inference.
