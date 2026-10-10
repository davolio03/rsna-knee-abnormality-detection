# Data Contract Notes

The official data description was checked on 2026-10-02 via indexed page content. It describes the downloadable schema. A separate authorized, read-only local metadata-only aggregate inspection is recorded below; it is not a full schema, label, or image review.

## Officially described files

| File or path | Described contents |
|---|---|
| `train.csv` | `StudyInstanceUID`, `Report`, and the 12 target columns listed in [COMPETITION.md](COMPETITION.md). |
| `train_series.csv` | `StudyInstanceUID`, `SeriesInstanceUID`, `Fluid_Sensitive`, `Fat_Suppression`, `Anatomical_Plane`. |
| `train_series/<study>/<series>/` | DICOM series files. |
| `test.csv` | `StudyInstanceUID` only; test reports are not described as available. |
| `test_series.csv`, `test_series/<study>/<series>/` | Test series metadata and DICOM files, analogous to training. |
| `sample_submission.csv` | Sample output file; examples are replaced for hidden scoring. |

The page describes approximately 1,300 hidden test studies, 819,640 files, and 569.76 GB. These are published approximations, not local counts. See the [official data page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data?select=test_series); page access was indexed and direct body rendering was incomplete.

## Local metadata-only aggregate EDA (2026-10-02)

This read-only check loaded only the listed metadata columns; it did not materialize report or target values, emit identifiers, or open DICOMs.

| File / field | Observed aggregate |
|---|---|
| `train.csv` | 4,407 parsed rows; 4,407 unique, nonblank `StudyInstanceUID` values; no duplicates. Only the study identifier was loaded for this check. |
| `train_series.csv` | 24,371 rows across 4,407 distinct studies; 24,371 unique `SeriesInstanceUID` values. No missing values or duplicate rows among the five loaded columns; every parsed training study has series metadata. |
| `Fluid_Sensitive` | 14,010 ones; 10,361 zeros. |
| `Fat_Suppression` | 14,010 ones; 10,361 zeros. |
| `Anatomical_Plane` | Sagittal 9,864; Coronal 8,609; Axial 5,898. |

For `train_series.csv`, the five loaded columns were `StudyInstanceUID`, `SeriesInstanceUID`, `Fluid_Sensitive`, `Fat_Suppression`, and `Anatomical_Plane`. Reports and target columns were excluded by `usecols`; no report or label values were materialized or analyzed. No identifiers were emitted and no DICOM was opened. These aggregates are local observations, not full data-quality/schema validation.

The user explicitly reserved the 58 expert-labeled Gold-58 studies exclusively for final evaluation; they must not be used for training, prompt optimization, model selection, or tuning. This 2026-10-02 local metadata-only inspection did not inspect target values or identify Gold-58. See [D004](DECISIONS.md#d004--reserve-gold-58-for-final-evaluation).

## Private Kaggle Gold-58 candidate identification (2026-10-10)

A private Kaggle notebook, `RSNA Knee | Gold-58 ID Mask (Private)`, ran against the official competition input. It read only `StudyInstanceUID` and the presence/absence of each of the 12 target columns in `train.csv`, using per-field converters; it did not materialize target values. It did not read `Report` or image/DICOM data. The successful private Save & Run All produced these aggregates:

| Observation | Count / result |
|---|---:|
| Training rows | 4,407 |
| Rows with all 12 target fields present | 58 |
| Rows with some, but not all, target fields present | 0 |
| Rows with no target fields present | 4,349 |
| Candidate ID uniqueness | Unique |
| Private candidate-ID artifact | Created in `/kaggle/working/gold58_study_ids.csv`; row count 58 |

This is an operational identification of the 58-row all-targets-present candidate set, consistent with the reserved Gold-58 count. Expert-label provenance and identity are not independently verified from labels: no label values were read. Candidate IDs and target values are intentionally absent from repository documentation; the ID artifact remains private on Kaggle. No public notebook, link, dataset, or submission was created, and no data was downloaded locally.

Gold remains reserved exclusively for final evaluation of the label-generation strategy. Do not use these studies for training, prompt/threshold tuning, model selection, or strategy selection. This observation establishes candidate-set counts and a private exclusion artifact only; it does not establish label quality or expert-label provenance.

## Submission-path evidence

One authorized constant-prediction smoke submission completed successfully with public score 0.500. This is end-to-end submission/scoring-path evidence, not model-performance evidence, and it does not verify local downloadable files, training schema, or the full test schema. Private execution metadata is intentionally omitted from this public record. See the bounded status in [COMPETITION.md](COMPETITION.md).

## Handling rules

- Keep raw competition files outside version control; `.gitignore` protects common raw-data and generated-artifact patterns.
- Never put report text, identifiers, images, credentials, or API keys in tracked documentation or experiment records.
- Keep derived artifacts traceable to source data and transformation versions; review privacy and competition constraints before external processing.
- Prefer the smallest authorized sample needed for an inspection question; do not download the full dataset by default.
