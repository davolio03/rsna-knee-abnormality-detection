# E005 — Windows Codex subprocess UTF-8 transport

- **Date:** 2026-10-10
- **Status:** Transport defect reproduced locally; live pilot outcome remains inconclusive
- **Parent:** E004
- **Scope:** Synthetic local subprocess regression and aggregate-only record of the already observed offset-5 failure

## Hypothesis

On the Windows Python 3.10.5 runtime with UTF-8 mode disabled and a CP1252 preferred encoding, `subprocess.Popen(text=True)` may encode Unicode stdin using the active code page rather than UTF-8, causing a UTF-8-consuming child process to reject otherwise valid text.

## Local synthetic reproduction

A child process read stdin as bytes and decoded it strictly as UTF-8. Before the implementation change, sending only synthetic text containing an em dash, curly apostrophe, and `é` made the child exit with sentinel status 17 (`UnicodeDecodeError`). With explicit `encoding="utf-8"`, the same synthetic subprocess returned status 0. The regression is exercised by the repository's local subprocess test; no Codex CLI, network, report, or dataset was used for this reproduction.

## Pilot evidence and limits

The separately authorized CP-09 batch attempted fresh eligible offset 5 once and stopped on its first nonzero result. The CLI exited 1 after 327 ms; its captured stdout was empty and stderr was 165 bytes. The privacy-safe journal retained cause `unknown`; no stderr contents were inspected or used to relabel that attempt. The runner exited 2 after approximately 420 ms. No successful label output was confirmed. Offsets 6–13 remain unattempted; offsets 0–5 are consumed and must not be resent.

One prior harmless ASCII Codex probe completed with runner exit 0 in 6.296 seconds. This establishes only that the basic ASCII startup path can succeed; it does not prove the exact cause of the offset-5 failure.

An authorized scalar check established only that offset 5 contains non-ASCII text and that the active CP1252 encoder can represent it while producing bytes that are not valid UTF-8. The report text, identifiers, credentials, model output, and local output contents are not included here. Thus, the transport defect is a plausible explanation for the earlier CLI failure, but the exact failure cause is not proven because its raw stderr was intentionally unavailable.

## Conclusion

The local subprocess encoding bug is reproduced and corrected by explicitly selecting UTF-8 for text-mode pipes. The earlier remote invocation remains an unknown failure and is not retroactively classified as an encoding error. No label-quality or clinical conclusion is supported. Continue only with the remaining eight authorized fresh offsets after the parent completes the applicable review disposition; stop on the first failure and do not retry.
