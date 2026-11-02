# E050 — Isolated sixteen-slice attention preparation

**Status:** read-only preparation contract only; no cache or training run was produced.

## Question and frozen comparison

Measure the effect of increasing selected slices from eight to sixteen while keeping target-specific attention and the rest of the recipe fixed. This is an attention-8 versus attention-16 comparison, not a repetition of E048's equal-mean versus attention comparison. E048 artifacts and the E049 inference package remain untouched.

| Element | Isolated E050 plan |
|---|---|
| Cohort | Fold 0, study-level grouping retained, Gold-58 excluded |
| Arms | `target_attention_8` versus `target_attention_16`; both freshly initialized from the same original RadImageNet weights with seed 42 |
| Selection | Uniformly sample up to each arm's limit from the full ordered raw series; fewer available sources remain fewer |
| Input/model | 224 pixels, RadImageNet ResNet-50, native normalization, masked spatial pooling |
| Training | Both arms use microbatch 2, Adam, head LR 1e-3, layer4 LR 1e-5, five epochs, and frozen BatchNorm; compare the prespecified epoch-five checkpoints, with no warm start |
| Identity | New cache/source fingerprint and `local16-attention-run01` output; never reinterpret an eight-slice cache |

Sixteen means a new uniform sample from the full ordered series, not “the old eight plus eight.” The 8- and 16-slice arms hold target-specific attention fixed; softmax renormalizes over each arm's own slice set. Model parameter count and microbatch are unchanged; total autograd memory may grow with the larger slice dimension, so constant VRAM and exactly doubled runtime are not claimed.

## Preparation boundary

`build_local16_plan` is read-only: it checks only that the explicit raw-root path is a non-symlink directory and returns the frozen recipe without checking for DICOM files, opening pixels, creating output, training, downloading weights, or falling back to caches. Its optional `cache_max_slices` argument checks only a caller-declared integer; it does not inspect or authenticate cache bytes and does not prove how a cache was built. The contract accepts no cache path. Verification of a genuinely new 16-slice cache remains future work. The legacy masked-soft dataset/cache schema and all existing training/inference pins remain at eight.

The configured `data/train_series` root is absent in this checkout. Therefore no genuine sixteen-slice cache, training run, or runtime estimate was produced. The shared preprocessing adapter now accepts explicit limits through sixteen while retaining its default of eight; its changed source digest means any new cache needs new provenance. The plan does not establish that the root contains DICOM data, that all series contain sixteen usable slices, or that a GPU run is feasible.

## Verification

- RED: the two new explicit-sixteen adapter tests failed because the shared adapter rejected limits above eight; the preparation tests failed to import the not-yet-created contract.
- Correction RED: after readback caught that an attention-versus-mean plan did not isolate the requested slice-count effect, the plan assertion failed on the comparison identifier before the corrected two-attention-arm recipe was added.
- GREEN: synthetic inputs selected sixteen distinct, ordered sources from 24; recovery retained sixteen usable distinct sources after one invalid header; a three-source series remained three slices; implicit default-eight and explicit-eight tensors/masks/metadata matched exactly. The plan contract specifies the controlled attention-8 versus attention-16 arms with matching fresh-init/seed/epoch settings.
- REFACTOR: kept the preparation surface as one read-only recipe builder rather than introducing a speculative cache writer/training runner; the regression suite remained green.
- Focused and regression command: `PYTHONPATH=src data/private/local-cuda-env/Scripts/python.exe -m unittest tests.test_local16_preparation tests.test_dicom_preprocessing tests.test_preprocessing tests.test_preprocessing_pilot tests.test_masked_soft_dataset tests.test_kaggle_preprocessing_canary tests.test_radimagenet_submission_inference tests.test_radimagenet_attention_submission_inference` — **107 tests ran: 106 passed, one existing test skipped**.
- These synthetic tests establish selection and contract behavior only. No local DICOM pixels, cache conversion, model construction, GPU execution, or training were performed.

## Next step

Only after the raw DICOM root is made available and separately authorized: build a fresh, verifiably sixteen-slice cache under a new identity, inspect its provenance, then separately authorize any training. Do not promote, relabel, or overwrite the historical eight-slice cache.

## Local16 cache package preparation — S16-03A

The private CPU-only Kaggle cache package is prepared but **has not been uploaded or run**. Use only the frozen package at `data/private/attention-sixteen-slices/cache-package-run05`; runs 01–04 are superseded preparation attempts. The package contains a default plan notebook, an explicit 8-study train/validation canary, and one explicit full-export notebook per bounded chunk. Each exported chunk is a new-identity `local-masked-tensor-cache16-v1` ZIP, with exact cohort/fold/series-order bindings, eligible decoded-slice counts, per-study tensor/mask checks, and an output-space bound below 20 GB including the reserve. Full-cohort coverage is 4,298 studies / 23,765 series, train 2,850 / validation 1,448, split into eleven chunks capped at 2,300 series. Gold-58 remains excluded.

Frozen artifact hashes: source ZIP `a97991f488db5fe1f6e55fedf9094111e4bb015d43b1bd5548c15ca7a660a6f8`; chunk plan `6aaeae86f8010399ca35c404c5e9e01a4174f6e14f3b8e810ff747a2ec6ea1b4`; preprocessing source fingerprint `dde2663fc16ab53d6cfc6366baf112e21cfd8c4c162967e2076d6db00d0fd7fb`. Verification was structural and synthetic: the ZIP dependency closure imported in isolated Python, the generated plan notebook ran against copied pinned cohort inputs, and the generated canary path wrote an 8-study synthetic cache ZIP with an observed >8-slice histogram. The configured codec installer/decoder and DICOM data were not used by that synthetic test. No real pixels were decoded, no Kaggle notebook was uploaded or executed, and no full cache was produced. The full-cache verifier requires new identity/provenance, exact 16-or-fewer actual eligible slices without padding, source-bound folds and labels, every chunk hash, complete nonoverlapping cohort coverage, and at least one observed series above eight slices. A local no-training attention-only plan CLI is prepared; it blocks until the downloaded cache is independently verified and the pinned original RadImageNet weights are present.

Focused/regression verification: `PYTHONPATH=src data/private/local-cuda-env/Scripts/python.exe -m unittest tests.test_local16_tensor_cache tests.test_local16_cache_chunk_plan tests.test_local16_full_cache tests.test_build_local16_cache_package tests.test_local16_preparation tests.test_local16_attention_plan tests.test_dicom_preprocessing tests.test_preprocessing tests.test_preprocessing_pilot tests.test_masked_soft_dataset tests.test_kaggle_local_masked_cache_export tests.test_local_masked_cache_chunk_plan tests.test_radimagenet_attention_training` — **111 tests passed**. This is not evidence of a real cache or training result.
