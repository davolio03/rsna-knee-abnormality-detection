# E050 — Isolated sixteen-slice attention preparation

**Status:** all eleven sixteen-slice cache exports are verified, and the authorized local five-epoch attention-16 run was launched once on 2026-11-03. It remains in cache-verification preflight; no training progress is yet evidenced. Earlier sections preserve preparation-stage evidence; this dated status supersedes earlier “not started” statements.

## Current launch status — 2026-11-03

The existing opt-in runner was launched exactly once at `2026-11-03T13:56:31.2338503Z` with `python -u scripts/train_local16_attention.py --train` (the runtime records the full `PYTHONPATH`/interpreter invocation). Private runtime evidence at `data/private/attention-sixteen-slices/runtime-run01/status.json`, observed `2026-11-03T14:03:37.7738953Z`, reports `preflight_running`, runner PID 113952 and Python child PID 65616. Its silent full-cache verification is still in progress. At that observation, the runner log was 0 bytes and the fresh `local16-attention-run01` output directory did not exist. Therefore this confirms launch and live preflight only—not model construction, GPU allocation, epoch start, training progress, completion, metrics, or a runtime estimate. Do not launch a duplicate. The fixed five-epoch recipe and verified source/cache/weight pins remain unchanged; checkpoints are expected only after completed epochs.

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

## Actual CPU canary and training entry — 2026-11-02

The private Kaggle canary execution `364872368` succeeded in 293.3 seconds with accelerator None. Its downloaded ZIP is 111,808,206 bytes, SHA-256 `3fae7a44e8d53a6c989b21f011433927c905b45d07d419857f585d652ee6b4c8`. Independent streaming verification checked all nine ZIP members, eight canonical first-training studies, 43 series, 688 actual selected slices, float32 `(16,1,224,224)` image arrays, bool masks, finite [0,1] values, exact labels/folds, and frozen source/plan provenance. Every canary series has sixteen usable cuts. This proves only this canary, not validation or whole-cohort coverage. Private logs and screenshots are retained under `data/private/attention-sixteen-slices/browser-proof`.

Full CPU chunk exports have now been submitted as separate immutable notebook versions: `train-000` execution `364876456`, `train-001` execution `364876786`, and `validation-000` execution `364877211`; the native version history reported all three running. No full chunk has yet been reported complete or downloaded. Eight remaining full chunks have not yet been launched. Do not confuse this with an eleven-chunk complete cache or reuse the canary as a full chunk.

An executable local attention-only trainer is prepared in `scripts/train_local16_attention.py`; default invocation only validates/plans, and `--train` is required for execution. `scripts/prepare_local16_attention.py` forwards to plan-only mode. The isolated runtime reuses the audited `_train_arm`, binds the full verified sixteen-slice cache, original RadImageNet weights, source closure and fresh fingerprint, uses the fixed E050 recipe, saves epoch checkpoints, and accepts only explicit compatible resume. This prepares the sixteen-slice treatment; a fresh paired eight-slice E050 control has not been rerun. Independent runtime checks passed 23 tests in 5.581 seconds; parent five runtime/plan tests passed in 2.933 seconds and confirmed a missing-cache default refusal with no model/training. Tests use synthetic substitutes, not a real GPU training run. No epochs have been launched; actual GPU cost, memory and checkpoint execution remain unmeasured. Roll back the isolated local16 runtime/CLI/plan/tests independently of the frozen private cache package and historical E048/E049 artifacts.

### Five-session CPU export limit
Two further full exports were observed running: train-002 `364878024` and train-003 `364878382`. Kaggle refused the sixth concurrent CPU launch with its explicit five-session cap; that dialog was cancelled without stopping any job or creating a validation-001 execution. The five accepted full jobs remain in progress; train-000 reached 46/416 studies at log time 577.6 seconds. Six chunks remain unlaunched. No full ZIP has yet been declared complete or downloaded. Cache code is committed as `967d31a`, opt-in trainer as `6278905`; neither was pushed.

## Complete cache and verified training preparation — 2026-11-02

All eleven private CPU export versions succeeded and their ZIPs were downloaded and copied without overwrite to `data/private/attention-sixteen-slices/cache-local-run01/archives`. Aggregate archive size is **61,211,348,960 bytes**. Each original and copied SHA-256, ZIP CRC, exact member set and every NPZ hash passed independent verification against canonical study/series-order hashes, labels, masks, fold membership and frozen run05 provenance. All per-chunk proofs are private and deliberately do not claim full-cohort verification on their own.

The actual default `scripts/prepare_local16_attention.py` invocation then ran the full verifier over all eleven archives and produced a plan successfully, with no `--train`. The complete verified cohort is **4,298 studies** (2,850 training, 1,448 validation), **23,765 series**, and **379,631 actual cuts**. No study is missing, extra or shared between the partitions; Gold-58 remains excluded. Actual slice-count distribution: 50 series have 11 cuts, 39 have 12, eight have 13, twelve have 14, 155 have 15, and 23,501 have 16. Shorter series are not padded with duplicate images.

Private evidence:
- `cache-local-run01/verification/full-cache-preparation-proof.json` binds all eleven archive, manifest and per-chunk proof hashes and the complete-cohort readback.
- `cache-local-run01/verification/default-preparation-plan.json` is the exact successful CLI output, SHA-256 `4231d7010454bad87a6d5e7bf572c29efb8b0ddf64f63413d49e63b9c9f0d939`.
- Prepared run fingerprint: `477fde596e9abd13e3a6fec49d743f69c27995824fa3546ad79a0a5fe379ade5`.
- The aggregate proof explicitly records `canonical_full_verifier_receipt_persisted: false`; the real CLI performed whole-cohort verification, and a duplicate scan was not performed merely to publish a second receipt. Future launch verifies the archives again.

Original RadImageNet weights and the nineteen-file runtime source closure passed the plan checks. The actual reader/model path preserves raw [0,1] cache values and applies native `2*x-1` normalization exactly once inside the attention model. The single target-attention runtime uses the same seed, fold, trainability, frozen BatchNorm, learning rates, microbatch two and five-epoch recipe. Parent read back both evidence files and reran five runtime/plan tests successfully in 3.136 seconds. No whole-project test claim is made.

**No model was constructed, no training output directory was created, and no GPU training or epoch checkpoint was run.** GPU memory, sustained runtime and sixteen-slice model quality remain unmeasured. A fresh paired eight-slice E050 control has not been rerun; the existing E048/E049 eight-slice artifacts and caches remain intact. All code/data separation safeguards remain in place, and no GitHub push or new competition submission was performed.
