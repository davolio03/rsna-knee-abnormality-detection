# RSNA Knee Abnormality Detection

This research project develops machine-learning workflows to predict 12 abnormalities from complete knee MRI studies for the RSNA Knee Abnormality Detection competition. The modeling unit is a study, not an isolated image:

**Study → DICOM series → ordered slices → pretrained CNN features → target-specific slice and series attention → 12 probabilities**

Some label pipelines retain report-derived soft targets and an observation mask that distinguishes known labels from missing ones. Those are weak labels, not verified expert truth. Image-validity masks used during preprocessing serve a different purpose. Validation groups studies across folds to reduce leakage and supports per-target as well as aggregate evaluation.

## Quick start

1. Read the [competition guide](docs/COMPETITION.md) and [data guide](docs/DATA.md). Obtain competition files from their official source and keep them in local/private storage.
2. Select the Python environment for your intended workload and install this checkout:

   ```powershell
   python -m pip install -e .
   ```

3. Use the workflow steps below to prepare inputs, inspect a plan, train or evaluate, and optionally build a local package.

## Install and inspect commands

The package requires Python 3.10 or later. Core dependencies are empty; optional groups provide image utilities and the optional SIWC report-label workflow:

```powershell
python -m pip install -e ".[images]"
python -m pip install -e ".[siwc]"
```

Run `python -m pip` with the interpreter of the selected environment. The base package and extras do not install PyTorch, torchvision, CUDA, or pretrained weights. Model workflows need a separately configured compatible runtime and locally available verified assets; the model recipes do not download weights automatically. See [baseline setup notes](docs/BASELINE.md).

The grouped entry points below are checkout scripts, not installed console commands. Review their current options with `--help`:

```powershell
python scripts/data/prepare_attention16.py --help
python scripts/training/train_attention16.py --help
python scripts/evaluation/export_attention16_validation.py --help
python scripts/review/prepare_attention16_sample.py --help
python scripts/packaging/build_attention16_submission.py --help
```

Run help from an environment with workflow dependencies available; importing a help command may load training modules. For exact flags, defaults, input paths, and execution behavior, see [STRUCTURE.md](docs/STRUCTURE.md).

## Workflow

1. **Obtain data.** Download or access competition files through the official competition source after reviewing its current terms. There is no automatic competition-data download step.
2. **Prepare labels (optional).** Report-derived soft-label workflows are separate from the image pipeline. The [SIWC setup guide](docs/SIWC_LABELING_SETUP.md) describes the optional external-service workflow.
3. **Prepare study metadata and image caches.** Review the prerequisites in [DATA.md](docs/DATA.md) and [BASELINE.md](docs/BASELINE.md). The data-planning wrapper checks existing manifests, folds, packages, and cache artifacts; it does **not** generate the large attention cache. Cache construction and verification are separate prerequisite steps.
4. **Train and evaluate.** Inspect the plan and resolved inputs before execution. Training starts only with `--train`; validation inference starts only with `--run`. Evaluation can process a bounded canary with `--max-studies` or the full validation cohort when the limit is omitted. The review-sample builder requires `--build`. These commands may read private inputs or write outputs, so check paths and overwrite behavior before running them.
5. **Build a package if needed.** The submission-package wrapper builds a private local package on normal invocation; it has no plan-only mode. Package construction does not upload a dataset or notebook, launch a Kaggle run, or submit predictions.

## Repository map

```text
src/rsna_knee/   installable data, labels, models, training, inference, validation
scripts/         workflow commands and supporting utilities
configs/         versioned workflow configurations
docs/            data, competition, modeling, and project guides
experiments/     experiment registry and durable evidence records
notebooks/       research and competition notebooks
tests/           automated contract and regression checks
data/            local/private inputs and generated artifacts (not versioned data)
```

## Research safeguards

Keep raw data, DICOMs, report text, identifiers, credentials, checkpoints, predictions, and caches out of Git; review files before publication. Check current competition, privacy, and licensing requirements before external processing. This project is research code, not a clinically validated diagnostic system, and does not declare an open-source license. Gold-58 is limited to the predeclared report-label prompt comparison in [decision D004](docs/DECISIONS.md#d004---gold-58-nested-prompt-strategy-evaluation); do not use it for model training or downstream classifier selection without separate authorization. Full safeguards are in [DECISIONS.md](docs/DECISIONS.md) and [COMPETITION.md](docs/COMPETITION.md).

## Further reading

- [Project state and roadmap](docs/PROJECT_STATE.md) · [research direction](docs/ROADMAP.md)
- [Repository structure and workflow inventory](docs/STRUCTURE.md) · [workflow retention guide](docs/WORKFLOW_INVENTORY.md)
- [Modeling and data baseline](docs/BASELINE.md) · [experiment records](experiments/README.md)
- [Competition rules](docs/COMPETITION.md) · [project decisions](docs/DECISIONS.md)
