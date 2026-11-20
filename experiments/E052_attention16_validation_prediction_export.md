# E052 — E050 attention-16 validation prediction export

**Status:** completed local fold-0 validation prediction export for all 1,448 studies. The canonical full-cache verifier passed before inference; prediction coverage and aggregate metrics were independently read back. No training, Kaggle operation, Gold-58 use, or remote transfer occurred.

## Result

| Measure | Export | Historical E050 epoch-five report | Difference |
|---|---:|---:|---:|
| Studies | 1,448 | 1,448 | 0 |
| Targets per study | 12 | 12 | 0 |
| Observed soft-label cells | 13,195 | 13,195 | 0 |
| Masked BCE | 0.48055831521017767 | 0.48055831620144307 | -9.912654e-10 |
| Brier score | 0.08974606086836194 | 0.08974606032303414 | +5.453278e-10 |

Both aggregate metrics agree within the documented numeric tolerance of `1e-4`. Per-study logits, probabilities, labels, masks, and identifiers remain private; the experiment record contains aggregate results only.

## Objective and boundary

Export predictions from the already-trained E050 epoch-five target-attention model over its fixed fold-0 local validation cohort. This was an inference-only task: no optimizer, training step, fold change, threshold selection, or competition submission was performed. Gold-58 remained excluded. Missing targets retain their observed mask and are not interpreted as negatives.

This is fixed-label local validation evidence, not AUC, independent clinical accuracy, a hidden-test result, or a causal slice-count comparison. No fresh paired eight-slice E050 control was run.

## Frozen inputs and runtime

| Input or setting | Value |
|---|---|
| Source checkpoint SHA-256 | `631536a30d0da51c0f6872fb45fac15b5bcb0f6185b5a53708eed1ef2be42f21` |
| Model-only export SHA-256 | `95de045ad077691a099bf7426605bfa433eb1a0abcfcd0d321250f43611f67b5` |
| Sanitized report SHA-256 | `14c7c5ceb92fe0c2debfceb699eeba006074f62afcd09042979c331fcdcd78f8` |
| E050 run fingerprint | `477fde596e9abd13e3a6fec49d743f69c27995824fa3546ad79a0a5fe379ade5` |
| Exporter source SHA-256 | `d3f14e5bf22928c8ae5d74d19665e73b672c1e6b71b5310f09a097ae1db0eec8` |
| Validation cohort/cache | Fold 0; `local16-attention-run01`; canonical full-cache verification passed |
| Model recipe | 224px; up to 16 slices per series; microbatch 2; native RadImageNet normalization once; model evaluation mode and inference mode |
| Numeric runtime | Python 3.11.9; PyTorch 2.11.0+cu128; CUDA 12.8; NVIDIA GeForce RTX 3060 Laptop GPU; FP32; TF32 off; float32 matmul precision `highest` |

## Execution and verification

The strict-TDD exporter was committed as source commit `cfefb15` (two files). Its RED checks preceded implementation and the bounded resume-integrity correction.

- `$env:PYTHONPATH='src'; & 'data/private/local-cuda-env/Scripts/python.exe' -m unittest tests.test_local16_validation_export` — 8 passed.
- `$env:PYTHONPATH='src'; & 'data/private/local-cuda-env/Scripts/python.exe' -m unittest tests.test_local16_full_cache tests.test_attention_sixteen_submission` — 10 passed.
- Independent verification — 18 tests passed; parent spot-check — 8 tests passed.
- `$env:PYTHONPATH='src'; & 'data/private/local-cuda-env/Scripts/python.exe' -m py_compile scripts/export_local16_validation.py tests/test_local16_validation_export.py` — passed; `git diff --check` — passed.

The authorized three-study canary completed first with `--run --max-studies 3` and the default private output. Its per-study elapsed times were 1.5404s, 0.4772s, and 0.6622s; total command elapsed was 10.6s including startup and model loading. The full run resumed that private prefix exactly once:

```powershell
$env:PYTHONPATH='src'
& 'data/private/local-cuda-env/Scripts/python.exe' -u scripts/export_local16_validation.py --run --resume
```

It exited 0 and took **1,270.932s** (21m10.932s), including full-cache verification, resumed inference, and metric validation. The full-cache verifier stage was not separately timed. Peak VRAM was not captured; one live GPU-memory snapshot during inference was 396 MiB of 6,144 MiB.

Every output shard was independently read back: contiguous ordinal coverage, unique manifest-bound study hashes and order, 12 targets per row, finite logits/probabilities/labels, probabilities in range, sigmoid consistency, exact FP32 labels and masks, and exact 13,195-cell aggregate mask coverage. Stable logit-space BCE was recomputed from saved logits; no thresholded predictions or AUC were calculated.

## Reproduction and private-output handling

The default CLI is plan-only. For future execution, select a **new, non-existing private output directory**; the exporter refuses existing output unless `--resume` validates its full provenance and contiguous prefix. A canary example is:

```powershell
$env:PYTHONPATH='src'
& 'data/private/local-cuda-env/Scripts/python.exe' -u scripts/export_local16_validation.py `
  --run --max-studies 3 --output-dir 'data/private/local16-validation-export/<new-run-id>'
```

For a new full export, select another new private output path and omit `--max-studies`; the command performs canonical full-cache verification before model inference. Resume an interrupted export only with `--run --resume`, the same private output directory, and exactly matching model, report, cache, exporter-code, and runtime pins. Do not rerun over the completed output at `data/private/local16-validation-export/run01`; it already contains all 1,448 shards. It also contains the private provenance and progress files. The execution log and ID-free aggregate receipt are adjacent at `data/private/local16-validation-export/full-run01.log` and `data/private/local16-validation-export/full-run01-verification.json`.

No generated prediction data, study identifiers, labels, or private shards were added to Git. The model-only export and source checkpoint hashes matched their pins after execution.
