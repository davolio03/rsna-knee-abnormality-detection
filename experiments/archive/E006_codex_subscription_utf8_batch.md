# E006 — UTF-8-corrected Codex subscription continuation

- **Date:** 2026-10-10
- **Status:** Bounded operational feasibility demonstrated; labels remain unvalidated
- **Parent:** E005
- **Scope:** Aggregate-only record of the eight fresh eligible offsets remaining after the CP-09 offset-5 failure

## Hypothesis

After explicitly selecting UTF-8 for Windows text-mode subprocess pipes, the eight still-authorized fresh records can complete sequential Codex subscription requests without a local text-transport failure.

## Baseline and changed variable

The prior authorized CP-09 campaign attempted offset 5 once and received CLI exit 1; the remote transmission outcome remains unknown and that attempt was not resent. E005 separately documents a local synthetic UTF-8 transport defect and its correction, without claiming it caused the historical failure. This continuation changed no inference strategy: it used the corrected UTF-8 subprocess transport for fresh offsets 6–13 only, one request at a time.

## Validation protocol

Run the remaining eight records sequentially with the existing Codex ChatGPT OAuth session, skip 6 and max 8. Attempt each offset once; stop on any failure and do not retry. Record only aggregate status, offsets, and elapsed times. Do not inspect labels or raw outputs.

## Results

All eight requests completed once: started 8, completed 8, failed 0; all Codex CLI statuses and the runner status were 0. Total elapsed time was 179,750 ms. Per-offset durations were: 6 — 18,359 ms; 7 — 33,733 ms; 8 — 16,016 ms; 9 — 23,483 ms; 10 — 14,171 ms; 11 — 26,890 ms; 12 — 24,391 ms; 13 — 22,577 ms.

Together with the earlier offset-5 failure, the authorized nine-new-attempt campaign is complete: one failure with unknown transmission outcome and eight successful requests. The failed attempt remains unresent. The historical CP-08 outcome remains inconclusive and unchanged.

## Conclusion and limits

The eight-record sequential continuation completed successfully on the corrected local transport, supporting bounded operational feasibility for these eight requests. This is not evidence of sustained scaling, label correctness, clinical accuracy, or generalization. No labels were inspected or quality-validated; no raw reports, identifiers, payloads, credentials, or output contents are recorded here. No single API bulk request or parallel execution was used. The campaign used the existing Codex subscription session; no API key, network/Kaggle access, or Gold-58 data use is claimed.

The implementation candidate `63e10136e5715dee1cd5798380ed01cf69b489b0` passed four review lenses with no findings and was approved, acknowledged, and burned. The advisory hardlink issue remains an unfixed low-risk follow-up; this record does not claim otherwise.

## Next step

Before any broader inference, perform separately authorized label-quality validation on a non-Gold development set. A larger controlled feasibility run may be considered separately; no full-scale campaign or label strategy is selected by this result.
