# E023 - Private five-study Kaggle preprocessing pilot

## Outcome

The bounded CPU pilot completed PP-02/PP-01 preprocessing for all 30 series
listed by the first five manifest studies: 1,067 headers scanned and 240
selected slices (eight per series), with 30 successes and zero failures. All
outputs were finite `(8, 1, 256, 256)` tensors within `[0, 1]`. The parent
visually inspected the original 30-panel processed-image/mask montage and saw
no blank panels and recognizable knees across views. This is an observed pilot,
not evidence of raw-image fidelity, anatomical or clinical accuracy, or model
readiness.

## Hypothesis and baseline

- **Date:** 2026-10-16, Europe/Madrid.
- **Type:** Bounded preprocessing pilot; no model training or evaluation.
- **Question / hypothesis:** Can the existing PP-02 DICOM adapter and PP-01
  decoded-array preprocessing complete for every listed series in the same
  five-study Kaggle sample without silent omissions or invalid outputs?
- **Baseline:** E022 observed successful bounded decoding of 240 slices across
  these 30 series, but performed no PP-01 preprocessing. Local synthetic tests
  covered the adapter and preprocessing independently; they did not establish
  actual-image behavior.
- **Changed variable:** The existing PP-02/PP-01 pipeline ran on the authorized
  five-study sample, capped at eight slices per series. No source changes or
  dependency installs were made for this run.
- **Expected mechanism:** Aggregate per-series status, output shape/range/
  finiteness checks, and ID-free processed-image/mask panels would reveal
  failures within the bounded sample.

## Protocol and environment

The parent recovered the existing private Kaggle notebook editor without
security changes, updated its attached input from V1 to V2 (UI-confirmed
success), imported the audited output-free notebook, and used Run All once.
The run was CPU-only (`accelerator=None`); internet remained enabled, so this
was not an offline-runtime check. The editor's earlier failure cause is
unknown. No reports, labels, Gold-58 studies, new cohort, training, or public
publication were involved.

The downloaded report and montage were copied to the ignored local directory
`data/private/kaggle-five-study-preprocessing/observed-run01/`. The JSON is
34,783 bytes with SHA-256
`94c944300cc3d9569fe1c85269c49d75b059211984a4afbf57f90956675b11ae`. Its
aggregate arithmetic and per-series status, shape, range and finiteness were
recomputed locally with Python's standard library. Pillow identified the
montage as an RGB PNG, 3072 x 1410 pixels. No IDs or raw-image artifacts are
included in this record.

## Results

| Measure | Observed |
|---|---:|
| Manifest studies requested | 5 |
| Series requested / succeeded / failed | 30 / 30 / 0 |
| Header files scanned | 1,067 |
| Slices selected | 240 (8 per series) |
| Output shape | `(8, 1, 256, 256)` for all series |
| Output values finite | 30/30 |
| Output range | Within `[0, 1]` for all series |
| Report elapsed time | 16.79 s |
| UI elapsed time | 17.895 s |
| Failure placeholders in montage | 0 |

The 30-panel montage shows each series' middle selected slice as processed
image and validity mask. The parent reviewed the original 3072 x 1410 montage
and observed no blank panels and recognizable knee anatomy across views; this
was a visual sanity check, not raw-image comparison or clinical/anatomical
validation. Series indices 9 and 10 had validity masks with 84.375% valid area
and letterbox margins. The other 28 masks were fully valid all-white validity
fields. These masks encode valid pixels, not segmentation labels.

## Saved notebook and evidence limits

The existing private notebook source was saved as Quick Version 3 with title
“Five-study preprocessing pilot - observed success”; the two previous versions
were preserved. Quick Save confirmed source, not retained remote execution
outputs. No second run was made. A UI output-tree “Could not load more files”
message appeared, but the exact report and montage were later found in the
Downloads directory and copied locally after a transient media-event timeout;
their local bytes were checked. Parent stopped the CPU session and the UI
confirmed it was off.

This report is parent-observed evidence supported by the downloaded ID-free
aggregate, not an independent live rerun or verification of saved remote
outputs. The five-study sample does not establish full-series or full-cohort
coverage, codec/transfer-syntax coverage, offline compatibility, raw-image
fidelity, complete geometric consistency, intensity-policy suitability,
clinical validity, label quality, or training readiness. PP-01 percentile
normalization and pixel-space letterboxing remain provisional.

## Conclusion and next step

The fixed five-study sample passed the recorded preprocessing and bounded
output checks. **Conclusion: inconclusive** for general preprocessing quality:
the positive run supports basic end-to-end behavior on this sample, but does
not validate policy choices or representativeness. Preserve the local output
evidence and decide separately whether to authorize additional preprocessing
validation before any training smoke test.
