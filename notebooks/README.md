# Notebooks

Use notebooks for focused exploration and visualization when they improve understanding. Keep reusable logic in `src/` and record meaningful conclusions in durable documentation or experiment records. Do not use notebooks to bypass reproducibility, data-safety, or validation rules.

## Phase 0 Kaggle submission smoke check

`phase0_kaggle_submission.py` is the source for the constant-prediction smoke check. It looks only at two explicit competition input roots for `test.csv` and optionally reads `sample_submission.csv`; it does not traverse image files or access reports. It writes `/kaggle/working/submission.csv` with the exact study-level header and 0.5 values for every target, then reads the file back to validate the header, row widths, unique/nonblank identifiers, finite bounded scores, and identifier order.

One authorized constant-prediction smoke submission completed successfully with public score 0.500, providing end-to-end submission/scoring-path evidence only. Private execution metadata is intentionally omitted from this public record. The result is not a model-performance claim. No raw data was downloaded, no training occurred, and no final selection was changed. Do not create a duplicate submission or retry the completed smoke. Remaining Phase 0, licensing, and environment questions still apply. See [the recorded Phase 0 evidence](../docs/COMPETITION.md).
