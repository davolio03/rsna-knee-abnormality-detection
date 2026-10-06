# Project Contributor Guidance

## Project goal

This repository studies RSNA Knee Abnormality Detection. Optimize for robust generalization and reproducible evidence, not leaderboard movement alone. Follow the current phase and scope recorded in [`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md) and [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Before substantial work

Read the relevant portions of:

1. [`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md) for current status and next steps.
2. [`docs/ROADMAP.md`](docs/ROADMAP.md) for phase goals and dependencies.
3. [`docs/DECISIONS.md`](docs/DECISIONS.md) for durable constraints.
4. [`docs/COMPETITION.md`](docs/COMPETITION.md) and [`docs/DATA.md`](docs/DATA.md) when competition or data constraints matter.
5. Related records in `experiments/` when planning or recording an experiment.

## Research and validation

- State each meaningful experiment's hypothesis, baseline, changed variable, expected mechanism, validation protocol, results, relevant runtime observations, and conclusion. Change one important variable at a time unless studying an interaction.
- Preserve negative and inconclusive results; never overwrite prior evidence or report only favorable outcomes.
- Treat potential leakage as a blocker. Preserve required study-level grouping and document any change to folds or validation methodology.
- Report per-target behavior alongside aggregate metrics. A leaderboard change alone does not establish generalization.

## Gold-58 and data safety

The 58 expert-labeled Gold-58 studies are reserved exclusively for final evaluation. Do not use them for training, prompt optimization, model selection, or tuning. Their count, study identities, and expert-label mapping remain unverified unless checked within an explicitly authorized scope.

Never commit raw competition data, DICOMs, radiology reports, identifiers, credentials, checkpoints, generated predictions, or caches. `.gitignore` is a safeguard, not a replacement for reviewing every publication candidate. Verify competition rules from official sources before relying on them, and do not assume local-only functionality satisfies the Kaggle inference contract.

## Project records

After substantial work, update `docs/PROJECT_STATE.md` when current status changes, `docs/DECISIONS.md` when a durable decision changes, and `experiments/` when recording meaningful experimental evidence. Change `docs/ROADMAP.md` only when the strategic plan changes. Repository artifacts are authoritative; conversational memory is supplementary.
