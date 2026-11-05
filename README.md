# RSNA Knee Abnormality Detection

This repository is a research project focused on predicting 12 abnormalities from complete knee MRI studies for the RSNA Knee Abnormality Detection competition. Its goal is a reproducible, competition-compatible approach whose quality is assessed through leakage-resistant validation and per-target evidence, not leaderboard movement alone. Report-derived labels are a potential research component, subject to verified competition terms, privacy requirements, and project authorization.

## Project organization

The repository is organized around the research lifecycle:

| Path | Purpose |
|---|---|
| `docs/` | Research roadmap, project decisions, competition and data constraints, and durable project guidance. |
| `src/rsna_knee/` | Python package boundaries for data handling, labels, models, training, inference, validation, and shared utilities. |
| `configs/` | Versioned settings for reproducible data, training, and evaluation workflows. |
| `experiments/` | Experiment records and registry, including hypotheses, validation, results, and conclusions. |
| `scripts/` | Focused command-line utilities for project workflows. |
| `tests/` | Automated checks for data, model, validation, and inference contracts. |
| `notebooks/` | Reproducible competition or research notebooks where they add value beyond reusable package code. |

This describes the intended organization of the project; it does not imply that every component or capability is implemented.

## Install and current entry points

The Python package metadata declares Python `>=3.10`. From the repository root, install the package into the environment you intend to use:

```powershell
python -m pip install -e .
```

Optional dependencies are separate workload groups and mirror the existing requirement files:

```powershell
python -m pip install -e ".[images]"
python -m pip install -e ".[siwc]"
```

The base package does not install Torch or CUDA. Installing an extra does not validate or replace the project's separately maintained training environment. Use the Python environment already qualified for the intended workflow; the `images` and `siwc` extras do not declare a Torch version.

The current attention16 checkout wrappers are grouped by purpose. Start with `--help`:

```powershell
python scripts/data/prepare_attention16.py --help
python scripts/training/train_attention16.py --help
python scripts/evaluation/export_attention16_validation.py --help
python scripts/review/prepare_attention16_sample.py --help
python scripts/packaging/build_attention16_submission.py --help
```

These wrapper files are repository entry points, not installed console commands. They delegate to the existing root-level scripts and keep those compatibility entry points intact. See [Repository structure and workflow entry points](docs/STRUCTURE.md) for execution semantics, private-artifact safeguards, and the incremental migration boundary.

## Research and evaluation principles

- Keep studies grouped across leakage-sensitive validation boundaries and document any change to the evaluation design.
- Report metrics for each target alongside aggregate results; retain negative and inconclusive findings.
- Treat leaderboard movement as supplementary evidence, not proof of generalization.
- Record meaningful experiments with their hypothesis, baseline, changed variable, validation protocol, results, and conclusion.
- Gold-58 may compare a short, predeclared set of report-labeling prompt strategies using nested study-level cross-validation under [D004](docs/DECISIONS.md). Freeze candidates, selection protocol, and metrics before outer-fold evaluation; outer folds evaluate the inner-selection procedure and must not guide prompt iteration. This is not a separate independent final test. Before any run, inspect per-target counts and confirm fold feasibility; neither is claimed as checked. Gold-58 must not be used for model training or downstream classifier model selection without separate authorization. Preserve competition licensing, external-service, and data-privacy constraints.

## Data, competition, and licensing

Competition data is not part of this repository. Do not commit raw data, DICOM images, radiology reports, study identifiers, credentials, model checkpoints, generated predictions, or caches. Review every file intended for publication; `.gitignore` is only a safeguard.

Verify current competition rules before relying on them, including constraints on external data, models, tools, report processing, and inference. The competition terms and their application to derived artifacts may require clarification. This repository does not grant rights to competition data or derivatives and does not declare a project license.

The optional local Sign in with ChatGPT labeling workflow is documented in [`docs/SIWC_LABELING_SETUP.md`](docs/SIWC_LABELING_SETUP.md). It sends eligible non-Gold report text to an external service when run; the guide is operational documentation, not authorization or a determination that such transfer complies with competition, privacy, or licensing rules.

For the project's durable constraints and research conventions, see [`docs/DECISIONS.md`](docs/DECISIONS.md), [`docs/COMPETITION.md`](docs/COMPETITION.md), [`docs/DATA.md`](docs/DATA.md), and [`AGENTS.md`](AGENTS.md).
