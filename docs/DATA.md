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

## Private Kaggle non-Gold metadata inventory (2026-10-10)

A final corrected run in the private Kaggle notebook completed the light CSV metadata inventory with Gold-58 excluded before training-series aggregation. It read only study keys from `train.csv` and `test.csv`, and the documented study/series keys and metadata columns from `train_series.csv` and `test_series.csv`. It did not examine reports, target values, or DICOM/image payloads. Gold IDs remain private and are not reproduced here.

| File / measure | Observed aggregate |
|---|---|
| `train.csv` study keys | 4,407 rows and unique keys; 0 missing or duplicate keys. Exactly 58 matched the private exclusion artifact; 4,349 non-Gold studies remained. |
| `test.csv` study keys | 3 rows and unique keys; 0 missing or duplicate keys; no train/test key overlap. |
| `train_series.csv` after Gold exclusion | 24,035 rows and unique `SeriesInstanceUID`s; no duplicate rows or missing study/series keys; 4,349 studies represented. |
| Series per non-Gold training study | Min / p25 / median / p75 / p90 / p95 / p99 / max: 3 / 5 / 5 / 6 / 7 / 9 / 10 / 14. |
| Non-Gold training series metadata | `Fluid_Sensitive` and `Fat_Suppression`: 1 = 13,818; 0 = 10,217. `Anatomical_Plane`: Sagittal 9,728; Coronal 8,490; Axial 5,817. |
| `test_series.csv` | 15 rows and unique series keys; 0 duplicates or missing study/series keys; 3 studies represented, each with 5 series. `Fluid_Sensitive` and `Fat_Suppression`: 1 = 9; 0 = 6. `Anatomical_Plane`: Sagittal 7, Axial 4, Coronal 4. |

This is a light CSV metadata inventory, not a full corpus, scanner, or site audit. With only three test studies, these data cannot support reliable distribution-alignment or representativeness conclusions. The inventory does not characterize slices, image resolution, reports, label prevalence, image quality, or model performance. See the [Phase 1 ODD tracker](../odd/tasks/phase1-non-gold-metadata-inventory.md) for task evidence and scope.

## Private Kaggle DICOM slice geometry pilot (2026-10-10)

A bounded header-only scan examined the same deterministic five-study non-Gold training sample used for the earlier header pilot: 26 series and 662 counted `.dcm` files. Each header was read with `pydicom.dcmread(..., stop_before_pixels=True, specific_tags=...)`, selecting only geometry and header attributes. `PixelData`, reports, target values, and Gold-58 studies were not accessed; identifiers and raw content are not included here.

| Check | Pilot observation |
|---|---|
| Header reads | 662/662 readable; zero missing series directories, read failures, or multiframe instances. |
| Present in every header | `Rows`, `Columns`, `PixelSpacing`, `ImageOrientationPatient`, `ImagePositionPatient`, `SliceThickness`, and `Modality` (662/662 each). |
| Other spacing metadata | `SpacingBetweenSlices` present in 415/662 headers, with seven distinct observed values; `SliceThickness` had four distinct observed values. |
| Within-series geometry | All 26 series had positions for every readable header, no orientation variation, and no duplicate slice positions at a 0.001 mm tolerance. |
| Position-derived adjacent distances | All 26 series had calculable, approximately regular adjacent distances under the pilot criterion. Across series, min / p25 / median / p75 / p95 / max: 3.2991 / 3.3 / 3.9 / 4.4 / 6.2783 / 7.7436 mm. |

For each series, the scan projected `ImagePositionPatient` onto the slice normal derived from `ImageOrientationPatient`, sorted slices by that coordinate, and computed adjacent center distances. The `.dcm` count is an observed file count, not a certified image or series total. This is a tiny, deterministic five-study pilot; its distributions and regularity checks are not full-corpus estimates or evidence of population-level generalization. See the [Phase 1 geometry tracker](../odd/tasks/phase1-dicom-slice-geometry.md) for bounded task evidence.

## Submission-path evidence

One authorized constant-prediction smoke submission completed successfully with public score 0.500. This is end-to-end submission/scoring-path evidence, not model-performance evidence, and it does not verify local downloadable files, training schema, or the full test schema. Private execution metadata is intentionally omitted from this public record. See the bounded status in [COMPETITION.md](COMPETITION.md).

## Handling rules

- Keep raw competition files outside version control; `.gitignore` protects common raw-data and generated-artifact patterns.
- Never put report text, identifiers, images, credentials, or API keys in tracked documentation or experiment records.
- Keep derived artifacts traceable to source data and transformation versions; review privacy and competition constraints before external processing.
- Prefer the smallest authorized sample needed for an inspection question; do not download the full dataset by default.
