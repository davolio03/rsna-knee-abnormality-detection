# Repository structure and workflow entry points

This is the current, incremental repository map. It makes the existing `src/rsna_knee` package installable and groups five attention16 checkout commands by purpose; it does **not** claim a completed migration of all scripts, notebooks, experiments, or project history.

## Quick path

1. From the repository root, install the package in the Python environment you intend to use: `python -m pip install -e .`.
2. Install an optional dependency group only when that workflow needs it: `python -m pip install -e ".[images]"` or `python -m pip install -e ".[siwc]"`.
3. Start a grouped command with `--help`. Run data, training, evaluation, review-build, or package-build operations only when their inputs, outputs, and authorization have been checked.

## Current repository map

| Area | Current role | Boundary |
|---|---|---|
| [`src/rsna_knee/`](../src/rsna_knee/) | Installable Python package: data handling, labels, models, training, inference, validation, and shared utilities. | Package discovery is restricted to `rsna_knee` and its subpackages. |
| [`scripts/`](../scripts/) | Existing workflow CLIs and small supporting utilities. | Root-level scripts remain in place as compatibility entry points; the grouped wrappers delegate to them. |
| [`configs/`](../configs/) | Versioned configuration files used by existing workflows. | No config migration or default change is included in this structure step. |
| [`notebooks/`](../notebooks/) | Competition and research notebooks. | Not moved or rewritten; source-closure and notebook pins remain unchanged. |
| [`tests/`](../tests/) | Unit and regression checks for package, data, model, validation, and inference contracts. | Tests are not included in package distributions. |
| [`experiments/`](../experiments/) | Durable experiment plans and records. | Historical evidence remains in place and is not reclassified here. |
| [`docs/`](./) | Project state, roadmap, decisions, constraints, and operating guidance. | Current state and durable policy remain distinct; this guide is structural, not an experiment record. |
| `data/` | Local data and generated artifacts may be present during authorized work. | Data, identifiers, reports, caches, checkpoints, predictions, and local environments are not package contents and must not be published. |

Directory names describe current responsibilities, not a claim that every workflow has a complete or uniform interface. The broader history and roadmap are intentionally outside this incremental change.

## Package and installation boundary

`pyproject.toml` defines the `rsna-knee-abnormality-detection` package from `src/rsna_knee` and declares Python `>=3.10`. Core dependencies are empty. The optional `images` and `siwc` groups mirror [`requirements-images.txt`](../requirements-images.txt) and [`requirements-siwc.txt`](../requirements-siwc.txt), respectively. The package explicitly includes `labels/codex_label_output.schema.json`.

The base install and both extras do not install or select Torch/CUDA. Training still requires the project's separately maintained, validated compute environment; installing the Python package is not evidence that an environment is ready to train or run inference. Use `python -m pip` in the environment chosen for the intended task. This repository metadata does not declare a project license.

The wheel and source distribution are limited to package source, the required schema, and build metadata. They exclude workflow scripts, tests, notebooks, configs, experiments, data, local environments, and generated model/prediction artifacts. The explicit `MANIFEST.in` allowlist and `include-package-data = false` prevent unrelated root files or additional JSON files from becoming package data by default.

The grouped wrappers are checkout conveniences, not installed console commands. Install the package, keep a repository checkout, and invoke a wrapper by its checkout path. A wrapper resolves the checkout root from its own file location and starts the legacy script with the same Python interpreter and the repository root as the child's working directory. Thus, an absolute path to a wrapper also works when the caller starts outside the checkout; relative arguments accepted by the legacy CLI are interpreted from the checkout root.

## Grouped attention16 commands

Each wrapper forwards arguments unchanged to an existing root-level script. The existing scripts remain the implementation and compatibility boundary.

| Purpose | Grouped wrapper | Existing command | Execution behavior |
|---|---|---|---|
| Data planning | [`scripts/data/prepare_attention16.py`](../scripts/data/prepare_attention16.py) | [`scripts/prepare_local16_attention.py`](../scripts/prepare_local16_attention.py) | Delegates to the plan-only path. It rejects `--train` and `--resume`; use the training entry point for an explicitly requested training launch. |
| Training | [`scripts/training/train_attention16.py`](../scripts/training/train_attention16.py) | [`scripts/train_local16_attention.py`](../scripts/train_local16_attention.py) | Default invocation prepares a plan; `--train` launches the five-epoch run. `--resume` is valid only with `--train`. |
| Validation export | [`scripts/evaluation/export_attention16_validation.py`](../scripts/evaluation/export_attention16_validation.py) | [`scripts/export_local16_validation.py`](../scripts/export_local16_validation.py) | Default invocation plans; `--run` explicitly enables cache verification and inference. `--max-studies` accepts a bounded 1–3 study canary; omitting it selects the full validation cohort. `--resume` continues a matching existing output prefix and requires `--run`. |
| Review sample | [`scripts/review/prepare_attention16_sample.py`](../scripts/review/prepare_attention16_sample.py) | [`scripts/prepare_local16_review_sample.py`](../scripts/prepare_local16_review_sample.py) | Default invocation plans; `--build` explicitly builds the private review bundle. The output must satisfy the legacy command's private-output and no-overwrite checks. |
| Submission packaging | [`scripts/packaging/build_attention16_submission.py`](../scripts/packaging/build_attention16_submission.py) | [`scripts/build_kaggle_attention_sixteen_submission_package.py`](../scripts/build_kaggle_attention_sixteen_submission_package.py) | `--help` displays options. A normal invocation immediately builds a private package; there is no plan-only switch. It does not upload or submit anything. |

Use these help-only examples to inspect the current argument lists:

```powershell
python scripts/data/prepare_attention16.py --help
python scripts/training/train_attention16.py --help
python scripts/evaluation/export_attention16_validation.py --help
python scripts/review/prepare_attention16_sample.py --help
python scripts/packaging/build_attention16_submission.py --help
```

The execution forms below document control flags; they were **not** run as part of the structure change:

```powershell
python scripts/training/train_attention16.py --train
python scripts/evaluation/export_attention16_validation.py --run --max-studies 3
python scripts/review/prepare_attention16_sample.py --build
```

These flags can read private local inputs or create private outputs. Before executing any such command, verify the exact inputs, output location, overwrite behavior, active environment, and separate authorization. Do not substitute `--help` output or a successful package install for data, model, training, validation, licensing, or competition-rule verification.

## Compatibility and migration boundary

The five root-level scripts above were left byte-identical in this step. Some are referenced by source-closure checks or hash-pinned artifacts, so the new wrappers add an organizational layer without changing the legacy source bytes, import behavior, flags, or business logic. The wrapper directories are purposeful groupings for current commands; this is not a requirement that every utility be moved under them.

No mass move, placeholder directory creation, notebook edit, experiment-history rewrite, configuration change, or source-closure regeneration is part of this step. Any later migration of a pinned script, notebook, or package member must separately update its references and regression evidence rather than silently changing compatibility assumptions.

## Verification scope

- Six focused packaging tests passed on Python 3.11.9. An offline wheel, source-distribution, and editable-install smoke test used a disposable environment with available build tools; the schema was present, private/workflow files were absent, and an import plus schema-resource read worked from an unrelated working directory without `PYTHONPATH`.
- The focused workflow and Kaggle source-closure command passed 23 tests. The five grouped wrappers' actual `--help` commands also exited successfully from outside a copied checkout with the package installed and `PYTHONPATH` unset.
- These are bounded packaging, command-surface, and source-closure checks—not full unittest discovery, a training or data-preparation run, a real model inference, a Kaggle upload/submission, or a general Python-version compatibility claim.
