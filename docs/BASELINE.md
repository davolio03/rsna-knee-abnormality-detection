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
