# Decisions

These are the durable decisions in force. The complete dated log (D001–D033), including the
label-pipeline engineering decisions, is in [archive/DECISIONS_LOG.md](archive/DECISIONS_LOG.md).
Numbers below match that log where a decision carries over.

## D002 — The repository is the source of truth

Version-controlled documents and experiment records are authoritative. Conversational or
assistant memory is supplementary.

## D003 — Evidence before adoption

A change becomes the new baseline only with recorded experimental evidence: hypothesis, changed
variable, validation protocol, per-target results and conclusion. Negative and inconclusive
results are kept. A leaderboard movement on its own is not evidence of generalization.

## D004 — Gold-58 is reserved

The 58 studies with expert labels on all targets ("Gold-58") are excluded from training,
validation, fold construction and model selection. The only use authorized so far is a
predeclared nested-CV comparison of report-labeling prompts. That comparison was frozen on
2026-10-11 and has not been run. Any other use, such as evaluating the label source or the final
model, requires an explicit new decision. Gold IDs and label values are never stored in Git.

## D005 — Hardware-aware development

Development targets a 6 GB local GPU (RTX 3060 Laptop) and Kaggle T4 notebooks. Record runtime
and memory costs. Do not assume a larger model is better.

## D006 — LLM report processing

The competition host allows commercially hosted LLMs to read training reports and generate
labels, subject to the competition rules, reasonable cost and access, and the provider's terms
([host post](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965)).
Only non-Gold reports may be sent. Reports and generated labels stay in ignored local storage.

## D027 — Soft labels with an observation mask

Training labels are the report-derived probabilities of the public
`stevenleehans/rsna-knee-llm-report-labels` dataset (v6, original `llm_labels_full.csv`).
Exactly 0.5 means "not addressed" and is masked out, never treated as negative. The loss is
binary cross-entropy over observed cells only.

## D028 — Three report-grouped folds

The 4,298 non-Gold studies with at least one observed target are split into 3 folds. The split
groups studies by a normalized-report hash and uses a deterministic iterative multilabel allocator
(seed 42). Every target has positives and negatives in every fold. The fold file is private and
frozen.

## D034 — Cross-validation protocol (2026-11-17)

Model quality is reported as 3-fold cross-validated ROC-AUC. Each fold trains the full recipe
with the selected epoch fixed in advance (no early stopping on validation) and scores its
held-out studies. The local label for each target is the observed soft label binarized at 0.5;
unobserved cells are excluded. Results are reported as macro mean ± std across folds, pooled
out-of-fold macro AUC, and per-target values. This is a proxy for the expert-labelled hidden test
and must be described that way.

## D035 — Lean main branch, preserved history (2026-11-17)

The main branch keeps only the code, records and documents needed by the current pipeline.
Everything superseded is preserved at the tag `competition-archive-2026-11` and under
`experiments/archive/` and `docs/archive/`: label-pipeline micro-decisions, Kaggle audit and
packaging iterations, earlier models and the detailed project log.

## D036 — Fine-tune layer3 and layer4 (2026-11-19)

The reference training recipe now trains backbone layer3 as well as layer4 (BatchNorm frozen,
both at learning rate 1e-5). In E055 it raised the 3-fold macro AUC from 0.861 to 0.864 and
improved every fold, which met the rule fixed before the run. New experiments use
`--trainable layer3` as their baseline and compare against E055. The code default stays
`layer4`, so the E054 commands and fold fingerprints still reproduce the earlier baseline.

## D037 — Gold-58 for expert evaluation and final training (2026-11-20)

Supersedes the restriction in D004. The project no longer builds its own report labels, so the
prompt comparison that D004 reserved Gold-58 for will not run. Gold-58 now has two uses:

1. **Expert evaluation.** Compare candidate models on the 58 expert labels. Each comparison and its
   decision rule are written in the experiment record before scoring. No hyperparameter or epoch
   search on Gold-58. Cross-validation fold models never train on Gold-58, so each fold model and
   their ensemble can be scored on it. With 58 studies the macro AUC has a noise of about ±0.01;
   read small differences with care.
2. **Final training.** Models trained only for a Kaggle submission may include the 58 studies with
   their expert labels. A model trained this way can no longer be evaluated on Gold-58.

Gold IDs and label values are still never stored in Git. External label tables that were built
with Gold-58 (for example, prompts chosen on it) are allowed as training labels; their Gold-58
rows are not used.
