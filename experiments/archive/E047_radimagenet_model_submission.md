# E047 — Final RadImageNet model submission preparation

## Objective and scope

Prepare a local, source-only Kaggle candidate for the completed five-epoch RadImageNet ResNet-50. The original preparation phase covered packaging and inference-contract work only; the later authorized Kaggle actions are recorded below. No new training run or model comparison was performed. The parent agent retains all Kaggle import, upload, execution, and submission actions.

## Frozen model evidence

The selected artifact is `data/private/local-radimagenet-final-epochs/final-run01/checkpoint-epoch-05.pt` (214,216,115 bytes; SHA-256 `4777e9b5b0e273767444f5d4d7025ab88b394babf56c1c5116b362bc6b0d023a`) with its adjacent `training-report.json` (SHA-256 `ee6a85e8a01ecf904da8fafad70c89a6b101efe4abda6230055625efa6210f8c`). The report records five completed epochs, fold 0, 2,850 training and 1,448 validation studies, 224px inputs, eight slices, `radimagenet_2x_minus_1` normalization, and mean-slice/mean-series-logit pooling. Its native fingerprint is `5c08ab6be89773802ce4f18fa2209a7e85ec88f2703b5caaded9bd940025a920`.

The fingerprint is a historical training-provenance value, not a hash to recompute from the Kaggle Python 3.13 runtime. The original frozen training source archive is independently pinned at SHA-256 `48a64d88c9b5cf72f4fdf6d9ce9839cc2cb3b8632b9fbf4fb7f4db1cdc4f95c8`; packaging verifies that its `models/resnet50.py` bytes exactly equal the frozen module being embedded. The inference adapter and bootstrap are new package source, not members of the native training fingerprint.

The report's final validation masked BCE/Brier are 0.5139817617573874 / 0.10471965642094785 across 13,195 observed soft-label cells. The same report's training-only constant baseline is 0.6161211079852827 / 0.14907895693661122. These support fit relative to that fixed report-label baseline only; they do not establish clinical accuracy, independent label quality, a benefit caused by RadImageNet versus ImageNet, or a leaderboard result. See E046 for the local training history and outstanding control comparison.

## Inference and publication contract

- Verify the exact report SHA and training-report schema, checkpoint SHA/size, five-epoch history, native fingerprint, finite tensors, and strict model-state names/shapes/dtypes before inference. Load with `torch.load(weights_only=True, map_location="cpu")`; never restore optimizer or RNG state.
- Build torchvision ResNet-50 with `weights=None`, apply the exact trained `ResNet50StudyModel` (`arm="radimagenet"`, seed 42, slice microbatch 2), then strictly load the complete trained state. Do not download generic pretrained weights.
- Preserve the old `fold0.py`, successful `masked.py`, old package builder, and frozen training-source archive unchanged. The new dataset adapter reuses the checked metadata ordering and CSV writer but invokes the canonical DICOM preprocessor directly at the trained 224px size (never resize a 256px output afterward). Inputs remain unit-range float32 with boolean masks; the model applies the trained normalization and masked pooling exactly once.
- Keep test features limited to test metadata and DICOMs. Retain only the exact allowlisted missing-series and recoverable preprocessing behavior; model, provenance, invalid-input, and unexpected failures remain fatal and do not publish a CSV.
- Canary cap is three studies, but competition-root `submission.csv` is published only when the run covers every visible test metadata row. The uncapped notebook is separate; neither creates a fabricated full-submission result from a partial canary.
- Retain the tested offline bootstrap: exact source ZIP byte pin, existing offline-codec manifest/wheels, install before importing `pydicom`, and bounded decoder preflight. Expected official Kaggle profile is Internet OFF and GPU; the target runtime may be Python 3.13 and must not impersonate training Python 3.11.9.

## Verification

Strict TDD first observed RED: `tests.test_radimagenet_submission_inference` could not import the not-yet-created `rsna_knee.inference.radimagenet` module. The completed focused command

```powershell
$env:PYTHONPATH='src'; data/private/local-cuda-env/Scripts/python.exe -m unittest tests.test_radimagenet_submission_inference tests.test_radimagenet_submission_package -v
```

passed 12 tests. It includes exact pinned report/checkpoint loading, corrupt hash/schema/history/fingerprint/state rejection, finite strict model state, source archive pins and frozen model-source byte identity, 224px float/mask inputs, target-column order, safe fatal model errors with no CSV, and canary/full publication guard generation. The real final checkpoint's inference model and a separately constructed training architecture produced exactly equal logits (`torch.equal`) on the first SHA-checked real cached 224px validation study. This single cached-study check is an implementation-equivalence check, not evaluation or model-selection evidence. The affected existing suites `tests.test_masked_model_inference`, `tests.test_kaggle_masked_submission_package`, `tests.test_kaggle_submission_package`, `tests.test_submission_failure_reproductions`, and `tests.test_resnet50_recipes` also passed (36 tests). These scoped runs are not a whole-project pass.

The final source archive and notebooks are local generated artifacts under `data/private/radimagenet-model-submission/run02/`; their hashes and file inventory are recorded in its ignored `inventory.json`. The source ZIP SHA-256 is `851e76d1e6813de27f01bd510949e7d09a112aa47eaabe18675b0809c2d07f5a` (22 code members); canary notebook SHA-256 is `ea9706ee9e9acb9d0ca4c7e1f04a3cb5ef921287ade8f2f7762dfd5aca7c0d66`; uncapped notebook SHA-256 is `9a815f31d29d71c9778f65ff9df881d3a7ea8a6fee0b46c0a6a1f542e260bb7a`. Run01 is preserved as an earlier build and is superseded; do not use it. The staged artifacts `checkpoint-epoch-05.pt` (SHA-256 `4777e9b5b0e273767444f5d4d7025ab88b394babf56c1c5116b362bc6b0d023a`) and `training-report.json` (SHA-256 `ee6a85e8a01ecf904da8fafad70c89a6b101efe4abda6230055625efa6210f8c`) were uploaded together in the private dataset under `/kaggle/input/datasets/davidsulleiro/rsna-knee-radimagenet-final-epochs` Version 1. The notebook source archive is embedded and independently available as `radimagenet-inference-source.zip`; it contains code only. The original training-source ZIP and raw data were not uploaded. At the end of the local preparation phase, the notebooks had not yet been imported or executed; the later Kaggle results follow.

## Authorized Kaggle execution and submission status (2026-10-31)

The private dataset `davidsulleiro/rsna-knee-radimagenet-final-epochs` Version 1 was created with the authorized `checkpoint-epoch-05.pt` and `training-report.json` artifacts and their pinned hashes above.

| Candidate | Kaggle execution | Observed outcome |
|---|---:|---|
| Capped canary Version 1 | `364367354` | Succeeded in 60.2 seconds; 3/3 visible studies; zero skipped series/slices. |
| Uncapped notebook Version 2 | `364368425` | Succeeded in 51.6 seconds; actual inference 25.202 seconds; 3/3 visible studies; `canary=false`; 224px, eight slices, microbatch 2; checkpoint/report pins matched; zero skipped series/slices; root `submission.csv` verified. After the deadline, terminal submission result was success with public score **0.825** and private score **0.802**. |

The parent submitted **exactly Version 2, once**, through the Kaggle UI. The post-deadline terminal result was successful (public/private **0.825 / 0.802**). This confirms the submitted inference candidate completed; it does not establish that RadImageNet or any single recipe component caused a score change. No additional training or push occurred. The prior masked-model scores (0.701/0.683) remain context only and are not a controlled causal comparator. Private evidence is retained in `data/private/radimagenet-model-submission/browser-proof/full-v2.log` and the browser proof.

## Next step

The Version 2 result is terminal and recorded above. No further submission is required by this experiment. Keep any future model comparison separate and controlled; do not interpret this score as causal evidence for the pretraining recipe. No whole-project test pass is claimed here.
