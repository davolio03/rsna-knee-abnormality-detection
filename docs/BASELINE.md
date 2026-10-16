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

## Isolated CPU training runtime

The first trainable baseline uses a separate ignored `.venv-training` so the
project's existing `.venv` is not changed. Create it with the repository Python
and install binary wheels only from the official CPU PyTorch index and explicit
PyPI index:

```powershell
.\.venv\Scripts\python.exe -m venv .venv-training
.\.venv-training\Scripts\python.exe -m pip install --isolated --use-feature=truststore --only-binary=:all: --index-url https://download.pytorch.org/whl/cpu torch
.\.venv-training\Scripts\python.exe -m pip install --isolated --use-feature=truststore --only-binary=:all: --index-url https://pypi.org/simple -r requirements-images.txt
```

Do not install `torchvision`, `torchaudio`, or Lightning for this scratch-model
runtime. `--use-feature=truststore` makes pip use Windows' system certificate
store while retaining TLS verification; do not replace it with a trusted-host
or certificate-verification bypass. The verified runtime on 2026-10-16 resolved
to PyTorch `2.14.1+cpu`, NumPy `2.4.6`, Pillow `12.3.0`, and pydicom `3.0.2`.
`pip check` reported no broken requirements. A local CPU autograd smoke check
confirmed `torch.version.cuda is None`, CUDA unavailable, and finite nonzero
gradients. The PP-01, PP-02, and five-study preprocessing test modules passed
(25, 18, and 5 tests respectively). This verifies the environment and existing
synthetic/pilot suites, not model training, full-cohort behavior, or clinical
performance. The existing `.venv` configuration hash remained unchanged.

## Reusable trainable components (provisional)

`src/rsna_knee/data/training_dataset.py` loads the private manifest and fold
artifact, binds folds to the manifest's exact bytes, and recomputes the
repository's deterministic allocation before accepting assignments. It exposes
one requested fold partition and decodes each selected study lazily through
PP-02; it does not cache full-cohort pixels. Each study retains every
manifest-listed series, with a maximum of eight selected slices per series.
Invalid paths, preprocessing failures, and malformed tensors fail with a safe
category rather than silently dropping a study or exposing a path/identifier.
The bounded subset helper is available only on a training partition.

`src/rsna_knee/models/baseline.py` provides a shared three-stage scratch 2D CNN
with mask-zeroed inputs and validity-weighted spatial feature pooling. It then
averages selected slices within each series, series within a study, and emits
12 ordered logits. `src/rsna_knee/validation/metrics.py` computes tie-correct
per-target ROC-AUC using the existing NumPy dependency, without adding
scikit-learn. Missing classes are reported as undefined; macro AUC is undefined
if any target is undefined.

Synthetic CPU forward/backward and metric tests do not demonstrate model
quality, training convergence, real-image suitability, or generalization. No
competition data was loaded by these tests. The runner and single-fold path are
now available for explicit local execution, but no actual cohort fit is claimed
or authorized by this component work.

## Local training runner (TB-03; provisional)

`scripts/train_baseline.py` loads the explicit checked-in configuration at
`configs/baseline_training.json` and is **plan-only unless `--execute` is
provided**. The plan validates the exact manifest/fold binding and partition
counts without decoding images or creating output files. At this earlier
pre-KG-01/02 snapshot, the checked-in image root deliberately named
`data/private/kaggle-input-update-required` and the private Kaggle package was
not yet updated; the later bounded private-package fit is documented below.

After updating those explicit private inputs locally, inspect the plan and
select one of two modes. `overfit` uses only the deterministic first at most 16
training-partition studies (default 8), caps optimization at 40 steps (default
20), and compares initial/final loss on exactly that same subset. It never
constructs or evaluates a validation dataset and produces no AUC. `fold` trains
on the complete selected training partition by default (one epoch), evaluates
the complete fixed validation partition, and writes all 12 per-target AUCs,
macro AUC, missing-class targets, and private per-row labels/probabilities.
An explicit `max_steps` cap in fold mode is reported as partial training, never
as a full-epoch result. Outputs require a new private directory and include
only a state-dict checkpoint, a JSON report, and (fold mode only) JSON
validation predictions indexed by partition row, without study identifiers.

```powershell
$env:PYTHONPATH = "src"
.\.venv-training\Scripts\python.exe scripts\train_baseline.py
# After updating private inputs, inspect the plan above before an explicit fit:
.\.venv-training\Scripts\python.exe scripts\train_baseline.py --execute
```

`notebooks/baseline_training.ipynb` is output-free and keeps
`RUN_TRAINING = False`; opening or executing its default cell only validates
and renders the config. No installation/download cells, hidden checkpoint
loading, or automatic remote upload are included. Runner tests use synthetic
CPU tensors only. The runner now selects CUDA only when explicitly configured
and `torch.cuda.is_available()` is true; it fails with a safe reason code before
output creation otherwise, never falling back to CPU. CUDA deterministic
algorithms, cuDNN deterministic mode, and TF32-off are configured from the
requested seed. CUDA telemetry synchronizes before resetting the allocation
peak and synchronizes again before reading actual peak allocated bytes; CPU
continues to report `peak_cuda_memory_bytes: null`. These mock checks do not
establish real GPU execution or deterministic-kernel coverage.

## Eight-study private Kaggle fit (KG-01/KG-02 verified; KG-03 smoke test complete)

`notebooks/kaggle_eight_study_overfit.ipynb` and
`configs/kaggle_eight_study_overfit.json` define a separate, output-free private
Kaggle trial. The notebook defaults both `RUN_GPU_PROBE = False` and
`RUN_TRAINING = False`; it performs no installation, download, or automatic
upload. The two mounted roots are explicit and were verified from the existing
Kaggle setup: the private dataset package and the competition's `train_series`
root. Manifest and fold files are resolved by bounded matching within the
private input mount and ambiguity fails closed.

The fixed trial is fold 0, seed 42, overfit mode, at most eight training
studies, 40 optimizer steps, five epochs, 256-pixel inputs, at most eight slices
per series, and CUDA only. A synthetic CUDA forward/backward/optimizer probe is
provided separately from the study run. The scratch architecture and PP-01/02
preprocessing are unchanged. The package builder uses an explicit source
allowlist and import closure, includes package initializers and fold-mode
metrics, binds the private manifest/fold bytes to expected SHA-256 values, and
writes a member-hash inventory. Reports, evidence, image files, checkpoints,
outputs, credentials, and bytecode are excluded from the package.

The package at `data/private/kaggle-eight-study-overfit/` is an ignored,
private-transfer artifact. Its inventory records the complete file membership
and hashes; the two data hashes are
`e88c61c22a463737f16951b7c76db03834034aecd1b31dc61b4ce495080b2be2` (manifest)
and `b82b376b0086201e3364b11a0203b62890e628442075a52ea4a9e09ee0e778ac`
(folds). Independent verification passed the focused 78-test suite and legacy full suite
(287 tests run, OK, 7 optional skips), and confirmed all 24 ZIP members against the
allowlist, source bytes, inventory, and exact input hashes; isolated ZIP import
also succeeded. The KG-01/02 work unit is committed locally as `f79c0cb`.

A separate synthetic CUDA forward/backward/optimizer probe passed in the
existing Kaggle environment (PyTorch `2.11.0+cu128`, CUDA `12.8`, Tesla T4):
deterministic algorithms were enabled, gradients were finite, BCE loss was
`0.6535296440124512`, and peak allocation was `18,198,528` bytes. This proves
only that this synthetic GPU path ran; it does not validate the scratch study
model or DICOM pipeline.

The user manually uploaded the private package, which was mounted as Version 3 (24 files).
Before importing the training package, the ignored launch cell used the pinned
embedded local inventory to verify all 24 unpacked member hashes, including the
manifest, folds, config, notebook, and source files. The embedded inventory's
ZIP SHA-256 is local provenance only; remote ZIP archive bytes were not
verified. The launch cell is
`data/private/kaggle-eight-study-overfit/launch_cell.py` (SHA-256
`2843bee0da1988cf9d59814afac780f46301737f55b1a1e3d6de88f0d390efd8`), and its
training flag defaults off.

After enabling the flag for the authorized run, a fresh Kaggle GPU kernel
completed the fixed fold-0/seed-42 overfit configuration: eight training
studies, five epochs, 40 optimizer steps, 256-pixel inputs, and at most eight
slices per series. The same bounded training-subset mean BCE decreased from
`0.6906174942851067` to `0.48356025479733944`; model runtime was
`83.31482125599996` seconds (`83.430137425` seconds wrapper time), and peak
CUDA allocation was `222,188,032` bytes. Runtime: Python `3.13.15`, PyTorch
`2.11.0+cu128`, NumPy `2.1.3`, Linux. Per-step losses included increases, so
the aggregate endpoint decrease does not establish complete memorization. This
run has no validation metrics, predictions, AUC, or Gold-58 results and does
not establish generalization or a strong baseline. The scratch model,
preprocessing, data split, and fold allocation were unchanged. See
[E025](../experiments/E025_kaggle_eight_study_overfit.md).

The ID-free private report is preserved at
`data/private/kaggle-eight-study-overfit/observed-training-run01/report.json`
(2,336 bytes; SHA-256
`a2871673660d462a9e6a0d27a50c4dd3ac5f4f9c81300873a7255bbc063cf0e3`); its
private checkpoint is `checkpoint.pt` (100,087 bytes; SHA-256
`b09f222057dd8fe8c1886ab3c107e8127264b43450019e4e2367cf6eba38518e`). Parent
verification confirmed remote and local-download hashes match. The GPU UI
confirmed the session was stopped. Browser download calls timed out although
the artifacts were saved; no repeat download was made.

Notebook source and results remain a draft. A Save Version attempt was rejected
before execution because notebook publication had not been explicitly
authorized; no save-version operation ran. The prior three dataset versions
were preserved. The parent will request separate consent before private
notebook publication. No additional training or full-fold run is authorized by
this smoke test.

## First full-fold Kaggle baseline prepared; batch not started (2026-10-17)

`notebooks/kaggle_first_fold_baseline.ipynb` is a clean standalone,
output-free private batch notebook; `configs/kaggle_first_fold_baseline.json`
records the matching frozen configuration. The cell defaults `RUN_TRAINING`
off, performs no installation or download, and verifies every unpacked member
against the pinned 24-file inventory before importing the package. The archive
digest remains local provenance only; the remote ZIP bytes are not verified.
The runner's plan-only call must confirm the pinned manifest/fold hashes and
exact 333/167 partition before its only possible fit. CUDA is required, with no
CPU fallback. Safe categorical failures are logged and re-raised so the Kaggle
batch records a failed run rather than falsely completing a cell after an
audit, plan, device, or training error. A successful summary is emitted only
after the runner returns a complete 1,665-step report with the expected fold,
hashes, and 12 finite per-target metrics.

The prepared run is a fresh scratch fit on fold 0 with seed 42, 5 epochs and
1,665 planned steps, validating the complete 167-study fold. It uses the
unchanged model and preprocessing, 256-pixel inputs, at most eight slices per
series, feature width 64, and learning rate 0.001. Output is isolated at
`/kaggle/working/kaggle-first-fold-baseline/run01`. The runner writes a final
checkpoint only; there is no resume or per-epoch checkpoint promise. This is a
single development fold against provisional silver labels, not clinical truth
or a final generalization claim. The preparation and fixed hypothesis are
recorded in [E026](../experiments/E026_kaggle_first_fold_baseline.md).

Local preparation is complete, but no first-fold batch was started here and no
first-fold results exist yet. Parent review and the authorized private Kaggle
Save Version -> Save & Run All action remain pending. Preserve the completed
eight-study evidence in E025 separately.
