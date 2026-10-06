# Scripts

Place focused CLI or automation entry points here when a repeatable task needs one. Keep reusable logic in `src/`, expose inputs/configuration explicitly, and avoid hard-coded machine paths or credentials.

## Synthetic submission smoke test

Run `python scripts/smoke_submission.py` to create and read back an ignored local, synthetic-only `submission.csv` at `outputs/phase0-smoke/submission.csv`. Pass `--output PATH` to choose another path. The command refuses to overwrite an existing file.

The fixture uses three fictitious study identifiers and twelve constant 0.5 scores. It checks the exact ordered competition header, row width, identifier completeness/uniqueness, and finite probabilities in `[0, 1]`. It does not access competition data or the network, and it does not establish model quality, local competition-data compatibility, or Kaggle runtime/submission behavior. Generated outputs are ignored by Git.
