# Private 500-study cohort manifest

Build the validated, report-free input manifest before planning a baseline model:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\prepare_training_cohort.py
```

The command joins the ordered first 500 eligible non-Gold studies, the validated
checkpoint labels, series metadata, and the prompt-boundary journal. It writes
only to ignored `data/private/500-study-training-cohort.json`; it refuses to
overwrite an existing file or follow a symlinked destination. Review the local
aggregate output and retain the manifest with its source SHA-256 values.

The manifest stores binary labels, series metadata, prompt era, and a normalized
report hash for later dependency grouping. It never stores report or evidence
text. Gold-58 rows are excluded by study ID before report content is used. This
step does not create folds, train a model, or use Gold-58 for model selection.

## Grouped development folds

After the manifest exists, create the fixed three-fold development allocation:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\prepare_training_folds.py
```

The allocator groups records by the manifest's normalized-report SHA-256 and
uses group-wise OR labels with the repository's iterative multilabel allocator
(seed 42). Study assignments are expanded in manifest order; every fold must
retain positive and negative studies for all 12 targets in both training and
validation. The private artifact binds to the exact manifest SHA-256 and records
the group assignments and aggregate class-count audit. This report-similarity
grouping is a dependency proxy, not proof of patient identity. The step creates
development folds only; it does not train or evaluate a model or use Gold-58.
