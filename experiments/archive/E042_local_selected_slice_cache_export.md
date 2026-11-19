# E042 — Bounded local continuation cache export

## Purpose and scope
Prepare a separate private Kaggle notebook to smoke-test a lossless cache for already-selected fold-0 study tensors before considering a full-cohort export. The smoke is CPU-only and decodes at most eight studies from each train/validation partition. It constructs no model, performs no training, changes no preprocessing or slice selection, and does not alter frozen source/package bytes.

## Cache contract
- Preserve the existing fold, manifest, partition, study order, series order, and within-series selected-slice order.
- Store image tensors as float32 in `[0, 1]` and pixel masks as bool. Preserve the 12 float32 soft targets and 12 bool observation masks without quantization.
- Use per-study NPZ files with `allow_pickle=False`; the JSON integrity manifest contains only ordinals, study SHA-256 digests, shapes, sizes, and file hashes. Raw identifiers stay out of filenames and progress logs.
- Write each sample and completion manifest atomically; publish the cache ZIP from a temporary sibling only after close/fsync succeeds. Reject unlisted cache files.
- Keep all cache/checkpoint artifacts private and outside source control. The 8+8 smoke is not the full 4,298-study cache.

## Pinned source and input provenance
The frozen RadImageNet recipe source ZIP contains 29 members and has SHA-256 `48a64d88c9b5cf72f4fdf6d9ce9839cc2cb3b8632b9fbf4fb7f4db1cdc4f95c8`. The current smoke notebook embeds those exact bytes and the current cache helper source.

The prepared-asset pin is derived from the successful frozen full-training notebook at `data/private/resnet50-pretraining-preparation/run02/kaggle_resnet50_radimagenet_full_epoch01.ipynb`. That notebook is pinned by SHA-256 `b7291f13d8b8127cf005a47032f4d70923f52dbf01978f989b7b99c6fdc15d17` and declares inventory SHA-256 `0421de9f96c1b73806d07788728b74deaacdb13bd3e9231175772a9a829773b3`. The builder verifies the authority notebook's full SHA before extracting that pin and rejects a caller-supplied mismatch. It does not weaken the runtime inventory guard.

## Bounded smoke attempt and correction
Private smoke Version 1 / execution `363987617` failed after 29.7 seconds with `prepared_inventory_pin_mismatch`, before DICOM decode and before cache output (observed output count: zero). The notebook used the local run04 preparation inventory SHA `25833614d3b274956cfa5284f7c0502ccc7d63d9d6c78d87cdbcf169e0a40707`; the attached successful DatasetV4 is pinned by the full-training notebook to `0421de9f96c1b73806d07788728b74deaacdb13bd3e9231175772a9a829773b3`. The mismatch was in the builder's expected pin, not evidence that the guard was wrong; no guard was relaxed.

The corrected authorized smoke variant is `data/private/local-radimagenet-epoch02/cache-export-smoke-run03-authorized.ipynb`, SHA-256 `74b0e54a226af50be6b505803972bee859ac2cb53984e35c54ba46f17422136b`. It differs from the metadata-only default only in `STAGE = "smoke_export"` and `RUN_BOUNDED_EXPORT = True`; the eight-studies-per-partition cap remains. The generated payload was checked against the current builder, helper, source ZIP, and 29 member hashes.

The parent launched the authorized CPU-only V2 smoke (execution `363989193`): all 16 requested studies (8 train + 8 validation) completed in 70.16 seconds. The cache was 122,646,262 bytes and its ZIP was 122,648,850 bytes (SHA-256 `ada0d6bfb98aaa619fbb2584610839885848b062053bbc7124032890108783ea`). This first-eight-per-partition sample is deliberately NOT treated as representative for full-cohort size projection.

## Full-cohort chunk plan (not yet exported)
The exact pinned full cohort has 4,298 studies (2,850 train; 1,448 validation) and 23,765 series, with no Gold-58. The private plan `data/private/local-radimagenet-epoch02/cache-chunk-plan.json` has SHA-256 `7f7e5537b1d18b92a1f129bf0bcac5ae21eea9e44068c30c7542a1dd4a8fb0b5`. It partitions only at study boundaries, preserves manifest order within each partition, uses a 4,000-series maximum, and stores selection digests and ordinal ranges rather than raw identifiers.

Worst-case raw selected tensors are bounded at 2,007,040 bytes per series (8 × 224 × 224 × (float32 image + bool mask)). Including cache plus ZIP duplication and a 1 GB reserve, each planned chunk remains below the 20,000,000,000-byte saved-output limit. The seven chunks are train-000 (727 studies / 4,000 series), train-001 (723 / 3,998), train-002 (724 / 3,998), train-003 (676 / 3,737), validation-000 (714 / 3,995), validation-001 (726 / 3,997), and validation-002 (8 / 40). These are conservative upper bounds, not measured sizes. Each notebook also checks current working-output bytes and actual cache/archive sizes before final ZIP publication; exceeding the budget fails rather than truncating or claiming completion.

Seven explicit CPU chunk-export notebooks were generated under `data/private/local-radimagenet-epoch02/cache-chunks-v2/`; the builder can also generate plan-only counterparts, while only the seven `*-authorized.ipynb` files enable export. These current payloads include a runtime check of all working-output bytes before archive publication. The previous `cache-chunks-v1` notebooks were preserved: readback found that they predated that output guard, so they are stale and must not be launched. Current authorized notebook SHA-256 values are:

| Chunk | Notebook SHA-256 |
|---|---|
| train-000 | `eee824551f3f7c210d8beee8e1f1abeca9980853c56055f0ab6b6dcdd85364c5` |
| train-001 | `d19f626c0764fa2f5ae5a450386aadcbf4f7705ad7c8a19625ccbaabcad075e0` |
| train-002 | `36e1a5fa6d4f776541211e5476b8844e6d8ace25ce101e41546bcb73c876546b` |
| train-003 | `1b67bf4cdffb5571fa68857f5b681e95e31828f21bc6198dd73b1a031427e951` |
| validation-000 | `7eb0677dd245c25efe6ca403512b597a741b93bf4f0bf3a98c157159735604b2` |
| validation-001 | `e8331098bf40439990aa105dccfa35fe86c29ee0d4105f8cd79f48d58e2d5c40` |
| validation-002 | `df364de5c39cf80b0053750b74e59c7106df32c5f1ffa4a4ab678cd054e0c453` |

None of the chunk notebooks has been launched. Chunk archive aggregation, whole-cache coverage validation, and checkpoint migration remain out of scope for this unit.

## Verification
Strict TDD observed RED for missing implementation modules, interrupted ZIP publication, unlisted-shard acceptance, missing explicit smoke-variant support, and rejection of the incorrect run04 inventory pin. Corrections are covered by the exact isolated command:

`data/private/local-cuda-env/Scripts/python.exe -m unittest tests.test_local_masked_tensor_cache tests.test_local_masked_cache_chunk_plan tests.test_kaggle_local_masked_cache_export`

Result: **19 tests passed**. Python compilation and `git diff --check` passed. Strict RED/GREEN covered cache atomicity/integrity, rejection of unlisted shards, whole-study chunk coverage and tampering, and safe plan/authorized notebook generation. The current generated notebook bytes are checked against the frozen v2 artifacts. No full-cohort chunk was exported, no local cache was imported, and no checkpoint migration or training was performed.

## Next evidence
Each chunk notebook defaults to metadata-only plan; only its paired authorized notebook enables that specific chunk. It auto-discovers the prepared DatasetV4 input with the verified inventory and resolves the competition mount using the existing layout helper. Parent verification must precede any chunk launch. Kaggle saved notebook outputs are limited to 20 GB; the plan reserves headroom, but actual chunk output and elapsed time still need measurement. No full-cache completion or training is implied.

## Parent local smoke readback
The measured smoke archive was downloaded and its SHA matched the export report. Independent local verification passed the archive CRC/path/symlink/size checks, all sixteen canonical rows, the float32/bool contracts, labels/masks and ordering, with 91 image arrays and 728 selected slices. The frozen helper read and round-trip checks passed when given the correct per-partition root. No archive or checkpoint bytes were modified.

One real cached training study (five series, forty slices, three observed targets) completed a GPU forward/backward/Adam benchmark from the epoch-one model and all twelve optimizer states: 1.935 seconds, 351.2 MiB peak allocated / 392 MiB reserved. FP32 was used with autocast and TF32 disabled. The optimizer advanced from step 2850 to 2851 in discarded process memory only; no new checkpoint was saved and no RNG restoration or epoch-two continuation is claimed. This single-study test establishes execution feasibility, not a full-epoch timing estimate or model-quality result.

## Completed parent export evidence (2026-10-31)
The preparation-only statements above describe the state when this exporter unit was first verified. All seven authorized CPU exports have since completed successfully and their downloaded archives passed independent streaming integrity, canonical label/mask/order/provenance and aggregate coverage checks. Local coverage is exactly 4,298 studies (2,850 train / 1,448 validation), 23,765 series and 190,120 slices, with no overlap, missing/extra studies or Gold-58 intersection. Total archives: 30,545,610,419 bytes. The final 726-study validation chunk ran in 2,931.1 seconds. Private aggregate proof is `data/private/local-radimagenet-epoch02/cache-all-seven-chunks-independent-proof.json`. No images, identifiers or archive bytes belong in this record or Git. This completion establishes data availability only, not checkpoint continuation or an epoch-two result.
