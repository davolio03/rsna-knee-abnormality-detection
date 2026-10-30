# E038 — Masked-soft training cost pilot

## Hypothesis and baseline

A bounded, deterministic sample stratified by the exact number of series per study can provide a useful **cost-only** timing and peak-memory signal for the existing full-cohort masked-soft training path, without changing model selection or the production runner. The baseline remains fold 0, frozen ImageNet1K V1 ResNet-18 with Adam at 0.001, unweighted masked BCE, 256-pixel inputs, at most eight selected slices, seed 42, and the verified E034 non-Gold fold artifacts.

The existing E037 eight-study decode smoke is not a representative full-cohort cost measurement: it completed 8/8 studies, 43 series, and 344 selected slices in 28.31 seconds of decode time (58.3 seconds end-to-end), without constructing a model or training.

## Changed variable and frozen protocol

Only the amount of work measured is bounded: one pilot epoch over deterministic stratified proxies, up to 64 train and 32 validation studies, with up to four selected studies per nonempty series-count stratum and at least one from every stratum. The same actual dataset, DICOM preprocessing, model factory, loss, optimizer and fold membership are used. No labels or outcomes influence sampling. Each sample is the original dataset's index and tensors; only the outer proxy limits which indices the runner visits.

The generated notebook checks the exact corrected run06 source notebook SHA-256 and embeds its original source cell as a string, executing it with a non-main namespace so the PLAN cell does not auto-run and extract the source archive before the pilot's own run. Its second cell performs one bounded pilot only; it does not launch the five-epoch full run. The runner writes to a fresh pilot-only output sibling, not `run01`; failures leave a categorical reason and aggregate progress outside runner cleanup. The packaged source ZIP intentionally adds only the already-existing `models/baseline.py` module required by the package initializer; no model, loss, optimizer, preprocessing, data, or training behavior changed.

## Measurements and predeclared gate

Record sample allocation and aggregate progress after each successfully decoded proxy study, elapsed runner time and synchronized per-study iteration intervals by partition and series-count stratum, weighted single-epoch population timing, elapsed preflight, CUDA peak allocated/reserved/total memory, host peak RSS and available RAM, free working disk, runtime/device provenance, finite masked-loss metrics, and readback of the checkpoint and report. Logs contain aggregate counts only; no study IDs, labels, reports, image pixels, or row-level outputs.

The projection is elapsed-time-only and explicitly **not a confidence interval**: 5 × the stratum-weighted one-epoch estimate × 1.5 multiplier (50% safety margin) + 300 seconds fixed overhead. “Acceptable for full-run review” requires finite timings/metrics, the real frozen model/weight provenance, checkpoint and report readback, peak allocated and reserved CUDA memory below 80% of device capacity, at least 2 GiB host-available RAM before and after, at least 1 GiB free working disk, and the projected five-epoch runtime within eight hours. The 15-minute deadline is soft (not a kill timer): it starts at pilot entry and is checked after preflight and at study boundaries; an overrun makes the gate false. A measured but unacceptable run still emits its aggregate summary and terminates normally. This is only a cost gate for a separate parent decision; quota, Kaggle UI state, and explicit full-run authorization remain parent-owned.

## Current implementation evidence

- `scripts/build_masked_cost_pilot.py` builds a fresh ignored notebook under `data/private/masked-training-cost/run11/` from the pinned run06 notebook; changed input pins and occupied output paths fail closed. Failure records use exact allowlists for exception type, reason code, frame basename and function; unknowns map to fixed safe values.
- V4 trace confirmed the model initialization failure: `models/__init__.py` imports `rsna_knee.models.baseline`, but the run05 source ZIP omitted that module. The package closure scanner skips imports inside functions and adds parent `__init__.py` files without scanning their imports. Added only `rsna_knee.models.baseline` to explicit required sources. run06 source ZIP SHA-256 `5d4c45fdae05890128c00034529b632e8f6a505058fb279963201e3e312337d2` has 27 members; compared to run05, only `rsna_knee/models/baseline.py` is added and every old member is byte-identical. Inventory SHA-256 `a1f2a730d22bb5c3b6011461e7d677a69d3339c7419b68e8ebd049d2c3f9ce17`; source notebook SHA-256 `9b05364b6df266e0fc603cdd3fe0e5cc93352abd2c78bfe0925a7c113e76e296`.
- Run04 and run07 had deterministic pre-upload failures; both remain local-only and neither was uploaded. Run01–run07 drafts are preserved and superseded. run08/V3 and run11/V5 were imported into the existing private notebook.
- Local verification: isolated archive import regression and deterministic source package test plus `tests.test_masked_cost_pilot` — 13 tests passed. Pycompile passed for both builders and test modules. The full package test suite had three unrelated environment errors because local `.venv` has no NumPy and no installed `rsna_knee` package. Isolated model-source import used a test-only Torch stub because Torch is unavailable in `.venv`; it imports packaged `resnet18`/`baseline` without constructing or training a model.
- run08 notebook SHA-256: `be76b9505cb4095909f3b9dd9b34c47fccf6d4b59388a1dad2710a985ea0955b`. Run05 source notebook pin: `c031b53ac96460b52ccee21aaf8d1e623c2c2af69001e6161f2d6950e6cbcb19`. Fresh diagnostic run09 is local only; parent owns any upload. No Kaggle operation or real training was performed by this worker.
- Fresh diagnostic run09 notebook SHA-256: `a2c53626a0c598fe1e785f857a60ceaa44bffa51664619704d4a6caf23ec2496`.
- Privacy-hardened run10 notebook SHA-256: `aabfa70b9a4108b7fda571eed235181e95e0454ecb9d726564d6101d4c7f4ad3`.
- Corrected-source diagnostic run11 notebook SHA-256: `b2cd589d1d5fc0a2ac9c5f26fe9b3194be6bd58dabd01765d64b00f808f0ff5b`; run06 source pin `9b05364b6df266e0fc603cdd3fe0e5cc93352abd2c78bfe0925a7c113e76e296`.
- Parent verified the run06 inventory pin, uploaded only the three changed package upload files to the same private dataset as Version 3, created the explicit dataset version, and imported run11 into the existing private notebook. Dataset proof: `data/private/masked-training-cost/run11/kaggle-dataset-v3-created.png`.

## Parent-observed private pilots and measured V5 result

The parent imported run08 and launched V3 / execution `362991317`, which failed after 43.009 seconds of pilot elapsed time (63.4 seconds notebook total). Its old recorder flattened the error; see preserved logs/JSON under run08. Parent then launched V4 / execution `362993012` from run10; it failed after 63.7 seconds notebook total, with preflight complete (8 studies, 43 series, 344 slices in 21.04 seconds) and no optimization steps. The V4 sanitized trace showed `ModuleNotFoundError` during `model_construction`, with last frame `rsna_knee.models.__init__` line 3. The missing module was confirmed in the exact source ZIP member list: `rsna_knee/models/baseline.py`. This package omission is fixed in run06. Neither attempt produced training-cost metrics or a cost-gate result; no full training was launched.

The isolated packaging regression went RED on run05 (`ModuleNotFoundError` for `rsna_knee.models.baseline`) and GREEN after adding that single missing source member. The isolated import test uses a test-only Torch stub because this local venv lacks Torch; no model is instantiated. The combined focused local checks pass 13 tests; parent separately reports 11 focused tests and one isolated archive import check passed. The broader local package suite was not green because the venv lacks NumPy/Torch and an installed project root. Parent uploaded run11 and executed V5. The user's conditional authorization for the five-epoch run applied only if the predeclared cost gate passed; V5 failed that gate, so no full training is authorized.

## Conclusion and next step

V4 identified a deterministic packaging omission, repaired in run06 and covered by the isolated import regression. Parent completed [V5 / execution `362995623`](https://www.kaggle.com/code/davidsulleiro/rsna-knee-masked-decode-preflight-run01?scriptVersionId=362995623), Version 5 using explicit private dataset Version 3, T4x2, Internet OFF, original environment. Terminal SUCCESS took 298.5 seconds. The bounded epoch processed 37 train and 32 validation studies, and the runner took 252.540980166 seconds; preflight decoded 8 studies / 43 series / 344 slices in 17.076754646 seconds. The weighted population one-epoch estimate was 9,347.548594 seconds (2.5965 hours); raw five-epoch estimate was 12.9827 hours; the predeclared 1.5x safety margin plus 300 seconds yielded 70,406.614456 seconds (19.5574 hours). `time_gate=false` was the sole failed measurement gate; checkpoint/report readback, finite measurements/metrics, fold, memory, host/disk, and deadline gates passed. The report is `measured_gate_failed` and not acceptable for full-run review. Thus the user's conditional authorization for full five-epoch training was not met; no full training was launched. Actual JSON/log/screenshot: `data/private/masked-training-cost/run11/kaggle-version5-measurement.json`, `kaggle-version5-logs.txt`, `kaggle-version5-measured.png`; prelaunch/dataset evidence remains alongside them.

### V5 measurement readback

| Measure | Observed |
|---|---:|
| Terminal notebook runtime | 298.5 s (SUCCESS) |
| Pilot runner / decode preflight | 252.540980166 s / 17.076754646 s |
| Bounded train / validation studies | 37 / 32 |
| Weighted population epoch estimate | 9,347.548594179 s (2.5965 h) |
| Raw five-epoch estimate / gated projection | 12.9827 h / 70,406.614456 s (19.5574 h) |
| CUDA peak allocated / reserved | 130,295,296 B / 157,286,400 B (124.26 / 150 MiB) |
| CUDA total memory | 15,636,037,632 B |
| Host peak RSS | 1,900,416 Linux KiB |
| Host available RAM, before / after | 27,156,537,344 B / 11,929,870,336 B |
| Free working disk, before / after | 20,940,464,128 B / 20,858,167,296 B |
| Finite metrics, checkpoint/report, memory, host/disk, fold, deadline | Passed |
| Runtime / overall acceptability | Failed / `measured_gate_failed` |

The estimate is sampled and elapsed-time-only, not a confidence interval or proof that every cohort member was decoded. No model-quality inference is drawn from pilot metrics. The predeclared time gate alone failed.

## Decision and next step

Do not launch the full five-epoch run: the predeclared runtime gate failed. This cost projection is a sampled elapsed-time heuristic, not a confidence interval or full-cohort decode proof; no model-quality conclusion is drawn from this pilot epoch or its metrics. A future, separate investigation could assess caching frozen-backbone features to avoid repeating DICOM decoding and CNN inference across epochs. That approach is only a recommendation: it is not authorized, implemented, measured, or guaranteed to meet the time gate.
