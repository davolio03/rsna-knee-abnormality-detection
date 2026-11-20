# Contributor Guidance

## Goal

Predict 12 knee abnormalities from complete MRI studies (RSNA Knee Abnormality Detection). Aim for
robust, reproducible generalization rather than leaderboard movement. Current status and next steps
are in [`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md), and the plan is in
[`docs/ROADMAP.md`](docs/ROADMAP.md).

## Before substantial work

Read `docs/PROJECT_STATE.md` and `docs/DECISIONS.md`. When competition or data constraints matter,
also read `docs/COMPETITION.md` and `docs/DATA.md`. Read the related records in `experiments/`
before planning or recording an experiment.

## Experiments

- Measure model changes with the 3-fold cross-validation protocol in
  [D034](docs/DECISIONS.md#d034--cross-validation-protocol-2026-11-17)
  (`scripts/cv_attention16.py`). Report per-target AUC next to the macro AUC.
- Change one important variable at a time. Fix the selected epoch and the metrics before you look
  at the results.
- Record each meaningful experiment in `experiments/EXXX_*.md` and add a row to
  `experiments/registry.csv`. Include the hypothesis, changed variable, protocol, results,
  runtime and conclusion. Keep negative and inconclusive results.
- Treat potential leakage as a blocker. Keep the report-grouped study-level folds.

## Data safety

- **Gold-58 is an expert evaluation set**
  ([D037](docs/DECISIONS.md#d037--gold-58-for-expert-evaluation-and-final-training-2026-11-20)).
  Declare each Gold-58 comparison and its decision rule before scoring, never tune on it, and
  train on it only for final submission models.
- **Never commit** raw competition data, DICOMs, reports, study identifiers, credentials,
  checkpoints, predictions or caches. They belong under the git-ignored `data/`. Review every
  file before committing; `.gitignore` is only a safeguard.
- **Source files are byte-pinned** by SHA-256 inside the Kaggle packages. Keep LF line endings,
  which `.gitattributes` enforces.
- **Respect the competition rules, the RSNA MIRA license and provider terms.** Do not distribute
  competition-derived artifacts until the license scope is clarified.

## Records

When status changes, update `docs/PROJECT_STATE.md`. When a durable decision changes, update
`docs/DECISIONS.md`. Repository files are authoritative. History before 2026-11-17 is at the git
tag `competition-archive-2026-11`.
