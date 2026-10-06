# Experiment Records

Use `E001`, `E002`, and increasing IDs for meaningful experiments. A run does not need a formal record for every tiny debugging iteration, but do preserve runs that inform a model, label, preprocessing, or validation decision. Never overwrite evidence; retain informative failures and inconclusive results.

## Record at minimum

- ID and date/time; status and parent/baseline experiment.
- Hypothesis and the important variable(s) changed, with expected mechanism.
- Code commit/hash when available; configuration; labels, preprocessing, and validation versions; model; random seeds.
- Validation protocol and sample/fold provenance.
- Macro OOF ROC-AUC and all 12 per-target AUCs when applicable.
- Training and inference time, peak VRAM when measurable, and relevant hardware/runtime details.
- Kaggle leaderboard score only when actually submitted, with submission context.
- Conclusion (`adopt`, `reject`, or `inconclusive`) and concise notes/limitations.

`registry.csv` is a lightweight index, not a replacement for detailed records/configurations. Leave unknown values blank rather than inventing measurements. Do not store reports, patient/study identifiers, raw predictions tied to sensitive data, credentials, or large artifacts in Git.

## Registry

Append one row per meaningful experiment using the existing header. IDs must be unique and historical rows must not be rewritten to improve the narrative.
