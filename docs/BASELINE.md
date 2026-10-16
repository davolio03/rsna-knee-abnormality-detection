# Private 500-study cohort manifest

Build the validated, report-free input manifest before planning a baseline model:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\prepare_training_cohort.py
```

The command joins the ordered first 500 eligible non-Gold studies, the validated
checkpoint labels, series metadata, and the prompt-boundary journal. It writes
only to ignored `data/private/500-study-training-cohort.json`; it refuses to
overwrite an existing file or follow a symlinked destination. Review the local
aggregate output and retain the manifest with its source SHA-256 values.

The manifest stores binary labels, series metadata, prompt era, and a normalized
report hash for later dependency grouping. It never stores report or evidence
text. Gold-58 rows are excluded by study ID before report content is used. This
step does not create folds, train a model, or use Gold-58 for model selection.

## Grouped development folds

After the manifest exists, create the fixed three-fold development allocation:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\prepare_training_folds.py
```

The allocator groups records by the manifest's normalized-report SHA-256 and
uses group-wise OR labels with the repository's iterative multilabel allocator
(seed 42). Study assignments are expanded in manifest order; every fold must
retain positive and negative studies for all 12 targets in both training and
validation. The private artifact binds to the exact manifest SHA-256 and records
the group assignments and aggregate class-count audit. This report-similarity
grouping is a dependency proxy, not proof of patient identity. The step creates
development folds only; it does not train or evaluate a model or use Gold-58.

## Bounded raw DICOM reader benchmark

`notebooks/baseline_dicom_benchmark.py` is a source-only notebook scaffold for
measuring raw header access and pixel decoding on a small local sample. Set
`SOURCE_ROOT`, `PRIVATE_MANIFEST`, and `COMPETITION_IMAGE_ROOT` explicitly in the
runtime. The expected, configurable image layout is
`<image-root>/<study-id>/<series-id>/*.dcm`; the reader does not recursively scan
the corpus. It validates the private manifest, considers the first five studies
in manifest order by default, and evaluates every listed series for those
studies. Slice selection is capped per series and ordered by DICOM patient-space
geometry; missing/non-finite geometry, inconsistent orientation or pixel
spacing, and duplicate projected slice positions are rejected rather than
replaced with an instance-number or filename guess. This scaffold does not check
in-plane position drift or regular inter-slice spacing, and therefore does not
certify complete geometric consistency.

The benchmark lazily requires already-installed `pydicom` and `numpy`, decodes
one slice at a time, enforces file/pixel/time bounds, and emits aggregate counts
only. It rejects symlink/path escapes, UID mismatches, multiframe objects, and
non-monochrome or inconsistent decoded shapes. Optional compressed-pixel decoder
failures are categorized without exposing exception text. It does not install
dependencies, access a network, print IDs or images, normalize pixels, choose an
orientation policy, or train a model. A bounded CPU Kaggle run has now verified
the configured competition mount and decoded 240 slices across 30 series from
the first five manifest studies, with zero observed decode failures. This small
parent-observed pilot does not verify full-series or full-cohort coverage,
offline compatibility, complete geometry, preprocessing, or training
readiness; see [E022](../experiments/E022_kaggle_five_study_dicom_reader.md).

## PP-01 decoded-array preprocessing prototype

`src/rsna_knee/data/preprocessing.py` defines a local-only contract for raw
decoded 2D monochrome arrays. The caller must supply the slices already in
geometric order and provide explicit photometric interpretation, optional
paired rescale slope/intercept, optional stored-value pixel padding metadata,
and optional pixel spacing. Arrays must be finite, real-valued, and same-shaped
within a series; boolean/string intensity metadata, non-finite padding values,
and unrepresentable normalization ranges fail closed. Integer stored pixels and
integer padding tags are limited to ±2^53 so conversion to the prototype's
float64 intensity domain cannot collapse distinct stored values; percentile
arguments must be real numerics, not booleans. The prototype selects at most eight unique slices
with both endpoints, masks explicit padding before rescaling, and rejects
unsupported Modality LUTs and inconsistent per-series shape, spacing, or
photometric metadata. It applies one provisional linear 1st/99th-percentile
range to valid pixels across selected slices within that series, handles
MONOCHROME1 explicitly, then bilinear-resizes and letterboxes to a pixel-space
aspect-preserving square. Output is float32 `(K, 1, 256, 256)` by default,
alongside a boolean validity mask and ID-free JSON-serializable transforms.
Bilinear resizing uses normalized mask-weighted interpolation so tagged padding
does not attenuate neighboring valid intensities.

These choices are engineering defaults, not tuned or clinically validated
preprocessing. Pixel spacing is retained as metadata; no physical resampling,
DICOM file adapter, orientation policy, ROI, augmentation, or training path is
included. `NumPy` and `Pillow` are optional image dependencies listed in
`requirements-images.txt`; sampling can be imported without them. The focused
synthetic suite was executed with bundled Python 3.12.14 (NumPy 2.3.5, Pillow
12.3.0): 25 focused tests passed after strict input validation and padding-edge
interpolation fixes. The project venv explicitly skipped this
optional-dependency suite; the full project suite passed outside the restricted
sandbox (281 run, three skips). Independent verification passed, including numerical edge-case probes. Actual
competition-image preprocessing remains pending.

## PP-02 explicit DICOM series adapter

`src/rsna_knee/data/dicom_preprocessing.py` adds an explicit local adapter for
one caller-selected series directory. It scans direct `.dcm` headers, reuses
the raw reader's patient-space projection geometry, rejects missing or
inconsistent orientation/spacing/layout and duplicate projected positions,
then uniformly selects up to eight geometrically ordered slices before pixel
decode. `max_slices=1` means the first slice in ascending projected position.
Only monochrome single-frame images are accepted. Paired rescale values and
padding-range metadata are converted to finite native numeric values; partial
rescale metadata, orphan padding range limits, and Modality LUT sequences are
rejected rather than guessed. Each decoded array and its header geometry must
agree before the unchanged PP-01 transform runs. Integer padding is checked
before float conversion; rescale/padding metadata must also match the scanned
header before pixel access.

The result contains PP-01 images, validity masks, transform metadata and
ID-free counts/selected projected positions; it does not expose file paths,
instance identifiers, or decoder exception text. Pydicom remains a lazy
optional dependency; the callable `dcmread` seam supports synthetic tests.
Eighteen synthetic adapter tests passed with real NumPy/Pillow PP-01 numerical
processing, and the 25 PP-01 regression tests passed under bundled Python
3.12.14. These tests do not establish pydicom codec compatibility, actual
competition-image behavior, image quality, or training readiness. No image
files were read and no dependencies were installed for this change.

Independent PP-02 verification passed. The full project suite outside the
restricted sandbox passed: 282 tests, zero failures, four skips. The optional
image suites execute in the bundled runtime; they are skipped in the project
venv when NumPy/Pillow are unavailable.

### Bounded five-study preprocessing observation

The separately authorized CPU Kaggle pilot then ran this adapter with PP-01 on
the first five manifest studies and all 30 listed series. It scanned 1,067
headers, selected eight slices per series (240 total), and completed all 30
series without failure. Every output was finite, shaped `(8, 1, 256, 256)`, and
within `[0, 1]`. The parent visually inspected the original 30-series processed
image/mask montage: no blank panels and recognizable knees across views. Series
indices 9 and 10 had validity masks with 84.375% valid area and letterbox
margins; the other 28 masks were fully valid all-white validity fields. These
masks encode valid pixels, not segmentation labels. These are
observations from a bounded pilot, not raw-image comparisons, anatomical or
clinical accuracy, full-series/codec coverage, offline evidence, or training
readiness. See [E023](../experiments/E023_kaggle_five_study_preprocessing_pilot.md).
