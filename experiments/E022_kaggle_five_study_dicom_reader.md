# E022 - Private five-study Kaggle DICOM reader pilot

## Outcome

The bounded CPU pilot decoded up to eight geometrically ordered slices from
each of the 30 listed series in the first five manifest studies. The observed
run completed 240 slice decodes with zero decode failures. This verifies a
small real-image reader path in the configured Kaggle environment; it does not
establish full-series preprocessing, training readiness, or model performance.

## Objective and baseline

- **Date:** 2026-10-16, Europe/Madrid.
- **Type:** Bounded raw DICOM reader execution, not a model experiment.
- **Question / hypothesis:** Can the existing guarded reader access the
  competition image layout and decode a bounded real-image sample on Kaggle
  without changing production code or installing dependencies?
- **Baseline:** The local BP-03 synthetic notebook harness decoded one stub
  slice; no real competition pixel decode or Kaggle mount had been verified.
- **Changed variable:** Execution against actual competition DICOM files in
  Kaggle, with `max_studies=5` and `max_slices_per_series=8`; notebook
  configuration paths were set to the private source/manifest and mounted
  competition input. Reader code and dependencies were not changed.
- **Expected mechanism:** A real bounded run would establish image mount,
  header access, and basic pixel-decoder compatibility for the sampled files.

## Protocol and authorization

The user authorized private upload of only the reviewed code and manifest, and
execution of the five-study reader; training was not authorized or performed.
The dataset and notebook remained private; Kaggle's sharing control showed
owner-only access. The notebook used CPU (`accelerator=None`). Internet access
was left at its default enabled setting, so this was **not** an offline-runtime
test. The mounted competition path was
`/kaggle/input/competitions/rsna-knee-abnormality-detection/train_series`; the
reader dataset was mounted at
`/kaggle/input/datasets/davidsulleiro/rsna-knee-five-study-reader-private`.

The uploaded transfer archive was the reviewed 157,730-byte bundle with
SHA-256
`70efbc0bec6a3f0c234bd4e276990d86970f53c86f22bdc829e9dbc25afcc167`;
its source snapshot was commit `1d286a1`. No raw reports, identifiers,
predictions, pixel arrays, or split assignments are included in this record.

## Results

| Measure | Observed |
|---|---:|
| Manifest studies requested | 5 |
| Series requested | 30 |
| Series per study, in manifest order | 5, 5, 5, 10, 5 |
| Header files scanned | 1,067 |
| Slices decoded (maximum 8 per series) | 240 |
| Decode failures | 0 |
| Decoded input bytes | 131,112,146 |
| Decoded pixel bytes | 130,752,512 |
| Maximum pixel shape | 768 x 768 |
| Reader-reported elapsed time | 8.675 s |
| Notebook cell execution time | 9.482 s |

The header count is not the decoded-pixel count: only 240 selected slices were
decoded. The observed successful decodes support the reader's bounded raw
header/geometry and slice-decode path for this sample. They do not show that all
competition series, slices, transfer syntaxes, or images decode successfully.

## Evidence and verification limits

The aggregate came from a parent-observed completed draft-cell output. Kaggle
Quick Version 1 preserved the source but stripped execution outputs, and a
notebook download was not confirmed; the aggregate is separately transcribed
locally from the observed cell output, not recovered from a downloaded
notebook. Quick Version 2 subsequently preserved an explicitly labeled Markdown
transcription of the observed result in notebook source without a rerun. The CPU
draft session was then stopped and the UI confirmed it was off. A fresh verifier
could not access the isolated browser session. Local
independent checks of the reviewed ZIP/import closure, manifest, first-five
series counts, and aggregate consistency passed, but there was **no independent
live Kaggle rerun or live-output verification**.

No dependency versions, codec/transfer-syntax coverage, peak memory, cold-start
behavior, full-series runtime, full-cohort runtime, in-plane position drift,
regular inter-slice spacing, preprocessing policy, training result, or clinical
label validity were measured. Eight slices per series are a bounded reader
sample, not a clinically validated volume or proof of complete geometry.

## Conclusion and next step

The first five manifest studies and up to eight slices from each listed series
were readable through the configured Kaggle mount on CPU, with zero observed
decode failures. This is evidence of bounded access and decoding only. Next,
define and validate preprocessing/geometry policy before any separately
authorized overfit smoke test or training. Do not infer cohort-wide readiness
from this five-study sample.

Private source reference: [Kaggle notebook](https://www.kaggle.com/code/davidsulleiro/notebook9f843b5d00/edit).
