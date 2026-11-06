# Workflow inventory and retention guide

This catalog covers exactly the 52 Git-tracked Python workflow sources in the step-2 baseline: 47 scripts and five notebook sources (three `.ipynb`, two `.py`). The machine-readable, row-level inventory is [`WORKFLOW_INVENTORY.csv`](WORKFLOW_INVENTORY.csv). It is a static map, not a command to execute, a migration plan, or a deletion list.

## How to read the labels

- **Role** describes a source's relationship to the currently recommended repository paths: `current` is a recommended checkout workflow; `shared` has an evidenced direct code dependency, including an imported helper or a template read by a builder; `historical` preserves earlier experiment or baseline behavior. `generated` would identify a tracked generated product, not a builder. No source in this 52-file baseline is classified `generated`.
- **Lifecycle** is independent: `current`, `historical`, `optional`, or `research-scoped`. A historical notebook can be `shared` because a checked-in builder still consumes it. `optional` and `research-scoped` constrain interpretation; neither grants execution authorization.
- **Known consumers** lists only statically confirmed tracked Python runtime sources that import, delegate to, embed, or read the inventoried source. The builder reading a notebook as a template is a runtime-path dependency. Tests, docs, and experiment records belong in **evidence**, never in `known_consumers`. A blank consumer cell means no direct consumer was confirmed in the reviewed source scope, not that a source is unused.
- **Evidence** links to a source symbol, notebook cell, or specifically relevant test/doc/experiment record. Static review does not prove a complete dynamic dependency graph, package pin closure, or all consumers.

Role counts: current 12, shared 10, historical 30, generated 0. Lifecycle counts: current 15, historical 34, optional 2, research-scoped 1. These are inventory classifications, not activity or execution counts.

## Currently recommended checkout workflow

The recommended structural entry points are the five attention16 wrappers documented in [`STRUCTURE.md`](STRUCTURE.md): data planning, training, validation export, blinded review-sample preparation, and submission packaging. The separately documented report-labeling CLIs remain optional, not an assumed default workflow. Each wrapper delegates to its listed legacy root script; wrapper-to-target paths are recorded as code consumers, while the guide is evidence, not a consumer. The current local16 preparation, training, validation, review, cache, and packaging sources are also identified by the E050/E053 records in [`PROJECT_STATE.md`](PROJECT_STATE.md). Current classification means recommended repository path, not that inputs exist, an environment is ready, or a run is authorized.

Earlier baseline and Kaggle commands remain documented in [`BASELINE.md`](BASELINE.md), but documentation that describes a historical workflow does not make it current or recommended. They remain historical in this catalog unless they are directly part of the currently recommended structural workflow. Help-only checks, plan mode, data reads, GPU execution, package building, and remote use each retain their existing boundaries.

## Optional and research-scoped workflows

Report labeling through [`scripts/label_reports.py`](../scripts/label_reports.py) and [`scripts/siwc_label_reports.py`](../scripts/siwc_label_reports.py) is optional. The SIWC CLI reuses labeling helpers from the Codex CLI implementation and can transfer eligible report text externally only through the documented confirmations. Read [`SIWC_LABELING_SETUP.md`](SIWC_LABELING_SETUP.md) and project safeguards first. Gold-58 prompt comparison is research-scoped under D004 in [`DECISIONS.md`](DECISIONS.md), not an ordinary training or downstream model-selection workflow.

## Retained history and compatibility

Historical scripts and notebook sources are preserved evidence, not dead code. "Historical" does not mean obsolete, safe to delete, or safe to move. Earlier package builders may still encode source-closure assumptions; generated notebooks/ZIPs are products, while the scripts that build them remain source. Ignored generated outputs are intentionally outside this tracked-source inventory: listing a producer does not inventory or reveal its private outputs.

The historical [`kaggle_first_fold_baseline.ipynb`](../notebooks/kaggle_first_fold_baseline.ipynb) and [`kaggle_eight_study_overfit.ipynb`](../notebooks/kaggle_eight_study_overfit.ipynb) remain `shared` because [`build_kaggle_training_package.py`](../scripts/build_kaggle_training_package.py) reads them as templates. [`baseline_training.ipynb`](../notebooks/baseline_training.ipynb) remains a source-only scaffold, but its test reference is not a production runtime consumer. `phase0_kaggle_submission.py` is historical Phase 0 smoke-submission evidence, distinct from current model workflows. These examples show why role and lifecycle are separate.

## Evidence limits and retention rules

The review covers Git-tracked sources and bounded static imports, path literals, source lists, template reads, and specific repository evidence. It does not prove dynamic imports, complete package pin closure, generated output contents, runtime readiness, or the absence of untracked consumers. The catalog marks uncertain or unconfirmed consumer relationships honestly. File age and absence of a discovered static reference never justify deletion.

No private outputs, reports, IDs, data, image pixels, checkpoints, caches, or local environment contents are cataloged. Generated/ignored output patterns may be described in producer docs, but private artifacts are not rows here. Before any future move/removal, verify source closure, pins, notebooks, tests, and consumers under a separately authorized change; preserve original experiment records.

