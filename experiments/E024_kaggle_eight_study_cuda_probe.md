# E024 - Kaggle synthetic CUDA probe and private package upload blocker

## Outcome

The synthetic CUDA numerical probe passed on the existing Kaggle GPU runtime.
The subsequent private dataset update did **not** complete: the file chooser
timed out before selection or transfer. No package upload, new dataset version,
real-image read, or eight-study fit occurred. The observed evidence validates a
small synthetic CUDA operation path only, not the scratch model, preprocessing,
training behavior, convergence, or generalization.

## Hypothesis and baseline

- **Date:** 2026-10-16, Europe/Madrid.
- **Type:** Synthetic GPU runtime probe and blocked private package transfer; no
  model or real-data training experiment.
- **Question / hypothesis:** Can the selected Kaggle CUDA runtime execute a
  deterministic synthetic forward/backward/optimizer step with finite loss and
  gradients before attempting the bounded study run?
- **Baseline:** Before KG-01, the local runner rejected CUDA; strict-TDD mocked
  tests and a parent-verified package were not hardware evidence.
- **Changed variable:** The parent selected the existing Kaggle Tesla T4 GPU
  runtime and ran the standalone synthetic numerical probe. No source,
  preprocessing, labels, data split, or model architecture was changed.
- **Expected mechanism:** A synthetic tensor model would exercise CUDA
  availability, deterministic-algorithm enforcement, forward/backward, loss,
  optimizer update, synchronization, finite-gradient checks, and actual peak
  allocation without touching private images or labels.

## Protocol and environment

The existing private Kaggle notebook executed a synthetic Conv2d/ReLU/pooling/
linear forward/backward/Adam probe. Environment reported PyTorch
`2.11.0+cu128`, CUDA `12.8`, and Tesla T4 hardware (two devices selected by the
Kaggle accelerator configuration). The notebook used deterministic algorithms;
no package installation, download, DICOM access, report inspection, or label
inspection occurred. The UI later confirmed the GPU session was stopped.

The separately prepared package was independently verified before this
attempt: commit `f79c0cb`; 78 focused tests passed; the legacy suite ran 287
tests successfully with 7 optional skips; all 24 ZIP members matched the explicit allowlist,
source bytes, member inventory, and exact private input hashes; isolated ZIP
import succeeded. This local verification is distinct from the Kaggle runtime
probe.

## Results

| Measure | Observed |
|---|---:|
| CUDA runtime | Available; CUDA 12.8 |
| PyTorch | `2.11.0+cu128` |
| GPU | Tesla T4 (Kaggle configuration selected two devices) |
| Deterministic algorithms | Enabled |
| Probe status | Passed |
| BCE loss | `0.6535296440124512` |
| Gradients | Finite |
| Peak allocated memory | `18,198,528` bytes |
| Real-image read / training | Not run |

These values describe one synthetic probe. They do not establish reproducibility
across sessions/devices, deterministic coverage of the scratch model's complete
operator set, real-image decoder compatibility, or model convergence.

## Private dataset update blocker

The existing private dataset Version 2 New Version dialog cleared its pending
file selection. The chooser event timed out after the `ChooseFile` wait (10
seconds) and the subsequent visible-input wait (20 seconds). No `setFiles`
operation ran, no file transfer occurred, and no new dataset version was
created. No DICOM or study data was opened and the fixed-eight-study training
run did not start.

Browser troubleshooting advised enabling **“Allow access to file URLs”** for
the ChatGPT extension. No browser permission or extension setting was changed.
The GPU session was confirmed stopped. This record describes the observed
blocker; it does not infer a cause beyond the troubleshooting guidance.

## Conclusion and next step

**Conclusion: inconclusive for training readiness.** The synthetic CUDA path
worked on the selected GPU, but the private package could not be uploaded, so
no actual study images were decoded and no fit/loss trajectory/checkpoint/report
exists. The next step is for the user to resolve the file-chooser access issue;
then the parent may resume the already-authorized private package update. The
bounded run remains exactly fold 0, seed 42, at most eight train studies, 40
steps, five epochs, 256 pixels, and at most eight slices per series. Do not
interpret this probe as model or generalization evidence.
