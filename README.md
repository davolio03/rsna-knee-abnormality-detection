# RSNA Knee Abnormality Detection

Research project for the RSNA Knee Abnormality Detection competition. The long-term objective is to develop a reproducible system for predicting 12 abnormalities from complete knee MRI studies, with validation designed to measure generalization rather than leaderboard fit. Report-derived labels may be investigated as a separate research component, subject to competition rules, licensing, and privacy constraints.

## Current status

**The project is in Phase 0: competition setup and submission contract.** Phase 0 remains in progress; Phase 1 data exploration is planned but gated on Phase 0 completion and explicit data-access authorization. A constant-prediction smoke submission verified the end-to-end submission/scoring path, but it is not a model result. No trained model, local baseline, or model-performance claim is available.

## Start safely

The repository currently has no package manifest or dependency installation workflow. The local smoke check uses only Python's standard library and synthetic identifiers; it does not need competition data or network access.

```bash
python -m unittest discover -s tests -p test_smoke_submission.py -v
python scripts/smoke_submission.py
```

The script writes a synthetic `submission.csv` under `outputs/phase0-smoke/`, validates its schema and values, and refuses to overwrite an existing file. Git ignores that generated output. These checks validate only the local synthetic CSV contract; they do not validate competition files, a model, or Kaggle execution. See [tests/README.md](tests/README.md) and [scripts/README.md](scripts/README.md).

## Repository map

| Path | Purpose |
|---|---|
| `docs/ROADMAP.md` | Research phases, dependencies, and acceptance criteria. |
| `docs/PROJECT_STATE.md` | Current phase, evidence, blockers, and next action. |
| `docs/DECISIONS.md` | Durable constraints and their rationale. |
| `docs/COMPETITION.md` | Source-linked competition contract and remaining unknowns. |
| `docs/DATA.md` | Published data description and bounded local metadata evidence. |
| `experiments/` | Experiment conventions and registry; results must be recorded without overwriting prior evidence. |
| `src/rsna_knee/` | Reserved package boundaries for data, labels, models, training, inference, validation, and utilities; implementation is not yet present. |
| `scripts/`, `tests/` | Focused local utilities and synthetic contract checks. |
| `configs/`, `notebooks/` | Configuration and research-support areas; review contents and provenance before use or publication. |

## Research and validation principles

- Preserve study-level grouping where needed to prevent leakage; document and justify any validation change.
- Report per-target behavior alongside aggregate metrics, and retain negative or inconclusive experiment results.
- Do not use leaderboard movement alone as evidence that a method generalizes.
- The 58 expert-labeled Gold-58 studies are reserved exclusively for final evaluation. Do not use them for training, prompt optimization, model selection, or tuning. Their count, identities, and label mapping remain unverified.
- Read the [roadmap](docs/ROADMAP.md) and [project guidance](AGENTS.md) before beginning substantial work.

## Data, licensing, and privacy

Competition files are not included. Do not commit raw data, DICOM images, radiology reports, identifiers, credentials, checkpoints, generated predictions, or caches. The `.gitignore` protects common local data and generated-artifact patterns, but it is not a substitute for reviewing every file before publication. Keep any local data in the ignored `data/` and output locations, and follow the project's authorization boundaries before accessing or processing it.

Internal ODD task logs under `odd/tasks/` and machine-specific agent/runtime instructions are intentionally excluded from the public repository.

The competition's MIRA and winner-license language leaves questions about distributing competition-derived artifacts unresolved. This repository does not declare a project license or grant rights to competition data or derivatives. See [competition notes](docs/COMPETITION.md) and [data notes](docs/DATA.md); seek organizer clarification before distributing competition-derived artifacts or using them commercially.

## Next step

Continue the remaining Phase 0 contract and environment work. Phase 1 is not yet authorized by this repository state. For current blockers and the evidence needed to close Phase 0, see [docs/PROJECT_STATE.md](docs/PROJECT_STATE.md).
