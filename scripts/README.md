# Scripts

Command-line entry points in pipeline order. Run them from the repository root; each one supports
`--help`. They read and write only under the git-ignored `data/` directory.

| Step | Script | Purpose |
|---|---|---|
| Cohort | `prepare_training_cohort.py`, `prepare_training_folds.py` | Build the original 500-study C2 cohort and its folds. The full-cohort prefilter checks them by hash. |
| Labels | `compare_report_label_sources.py` | Compare the public LLM label source with C2 binary labels (E033). |
| Labels | `prepare_full_cohort_prefilter.py` | Build the masked soft-label manifest and the 3 report-grouped folds for all non-Gold studies (E034). |
| Cache | `build_local16_cache_package.py` | Generate the Kaggle CPU notebooks that decode DICOMs into the 16-slice tensor cache. |
| Cache | `local16_cache_chunk_plan.py`, `local16_tensor_cache.py`, `local16_full_cache.py` | Cache chunk planning, export format, verification and dataset reader. |
| Train / CV | `cv_attention16.py` | Train a fold, score its held-out studies and summarize cross-validated AUC (E054). |
| Evaluate | `export_local16_validation.py` | Export fold-0 validation probabilities of the E051 model export (E052). |
| Review | `prepare_local16_review_sample.py` | Build the private blinded disagreement-review sample (E053). |
| Submit | `build_kaggle_attention_sixteen_submission_package.py` | Package the model export and the offline inference notebook for Kaggle (E051). |
| Labeling (experimental) | `siwc_label_reports.py`, `label_reports.py`, `run_gold58_prompt_comparison.py` | The project's own LLM report-labeling clients and the frozen Gold-58 prompt-comparison protocol. They are not used for the current model. See [LABELING.md](../docs/LABELING.md). |

`build_kaggle_submission_package.py` and `local_masked_full_cache.py` are shared helpers that
the scripts above import.
