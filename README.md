# RSNA Knee Abnormality Detection

Study-level multi-label classification of **12 knee abnormalities from complete MRI studies**
(ACL, MCL, medial/lateral meniscus, medial/lateral/patellofemoral OA, effusion, synovitis,
Baker's cyst, bone contusion, fracture), built for the
[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
Kaggle competition. The competition metric is the mean ROC-AUC over the 12 targets.

```text
study ─► DICOM series ─► 16 ordered slices / series ─► RadImageNet ResNet-50 features
      ─► target-specific slice attention ─► target-specific series attention ─► 12 logits
```

## Results

### Cross-validation (3-fold, study-grouped)

<!-- CV_RESULTS:START -->
**Macro ROC-AUC 0.864 ± 0.003** over 3 report-grouped folds (pooled out-of-fold 0.863, 4,298
studies), with layer3 and layer4 fine-tuned ([E055](experiments/E055_attention16_layer3_finetuning.md)).
Per fold: 0.867 / 0.862 / 0.863. The layer4-only baseline (E054) scored 0.861 ± 0.003.

| Target | AUC (mean ± std) | | Target | AUC (mean ± std) |
|---|---:|---|---|---:|
| ACL | 0.832 ± 0.012 | | PF OA | 0.843 ± 0.007 |
| MCL | 0.810 ± 0.016 | | Effusion | 0.844 ± 0.002 |
| Medial Meniscus | 0.870 ± 0.012 | | Synovitis | 0.914 ± 0.027 |
| Lateral Meniscus | 0.827 ± 0.020 | | Baker's | 0.901 ± 0.012 |
| Medial OA | 0.885 ± 0.002 | | Contusion | 0.882 ± 0.009 |
| Lateral OA | 0.863 ± 0.005 | | Fracture | 0.900 ± 0.010 |

Per-fold and pooled tables: [`docs/RESULTS.md`](docs/RESULTS.md).
<!-- CV_RESULTS:END -->

Local AUC is measured against the **report-derived LLM labels** (observed cells only, binarized
at 0.5), not against expert labels, so it is a noisy proxy for the hidden-test metric.

### Kaggle leaderboard (late submissions)

Every model submission below was scored **after** the competition deadline (2026-10-22). None of
them is on the official leaderboard.

| Model | Experiment | Public | Private |
|---|---|---:|---:|
| ResNet-18, binary labels, 500 studies | [E032](experiments/archive/E032_hidden_submission_candidate.md) | 0.642 | 0.620 |
| ResNet-18, masked soft labels, full cohort | [E040](experiments/archive/E040_masked_model_evaluation_submission.md) | 0.701 | 0.683 |
| RadImageNet ResNet-50, 8 slices, mean pooling | [E047](experiments/archive/E047_radimagenet_model_submission.md) | 0.825 | 0.802 |
| + target-specific attention MIL | [E049](experiments/archive/E049_attention_model_submission.md) | 0.869 | 0.848 |
| + 16 slices per series | [E051](experiments/E051_attention16_model_submission.md) | 0.885 | 0.862 |
| + `layer3` fine-tuning | [E056](experiments/E056_attention16_layer3_submission.md) | 0.891 | 0.868 |
| + 3-fold ensemble | [E057](experiments/E057_attention16_layer3_ensemble_submission.md) | 0.902 | 0.877 |
| + 7th-place training labels | [E058](experiments/E058_attention16_seventh_place_labels.md) | 0.932 | 0.919 |
| + 4 final all-data models, 7-member ensemble (**current submission**) | [E059](experiments/E059_attention16_final_all_data.md) | **0.934** | **0.923** |

Consecutive rows change more than one variable and use one validation fold, so they show the path
the project took. They are not controlled ablations.

## Method

| Stage | Choice |
|---|---|
| **Labels** | Since [E058](experiments/E058_attention16_seventh_place_labels.md), the 7th-place image-and-report labels [`songqizhou/rsna-knee-llm-labels`](https://www.kaggle.com/datasets/songqizhou/rsna-knee-llm-labels) (`8_final_label.csv`, CC BY-NC-SA 4.0, by songqizhou; [D038](docs/DECISIONS.md#d038--train-on-the-7th-place-labels-2026-11-21)). Earlier models: report-derived probabilities from the public CC0 dataset [`stevenleehans/rsna-knee-llm-report-labels`](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels) (v6, `llm_labels_full.csv`). A value of exactly 0.5 means "not mentioned" and is masked out. 4,298 non-Gold studies keep at least one observed target. |
| **Splits** | 3 folds grouped by normalized-report hash and balanced with an iterative multilabel allocator (seed 42). The 58 expert-labelled "Gold" studies are excluded from all training and model selection. |
| **Preprocessing** | All series per study; up to 16 uniformly sampled ordered slices per series; 224 × 224; intensities scaled to [0, 1]; pixel validity masks. |
| **Model** | RadImageNet ResNet-50 backbone (`layer3` and `layer4` trainable, BatchNorm frozen). Each target has its own attention over slices, then over series. |
| **Objective** | Masked soft-label binary cross-entropy over observed cells only. |
| **Training** | Adam (head 1e-3, backbone 1e-5), 5 epochs, one study per step, seed 42. The final epoch is fixed in advance, with no early stopping on validation. About 95 min/epoch on an RTX 3060 Laptop (6 GB), or 70 min with `layer4` only. |
| **Inference** | Offline Kaggle notebook: pinned model export and a code-only source archive. It runs in about 30 s on the visible test studies (T4 × 2). |

## Repository layout

```text
src/rsna_knee/
  data/         DICOM decoding, slice selection, cohort manifests, cached datasets
  labels/       report-label tooling (soft-label prefilter, label-source comparison,
                experimental LLM labeling clients; see docs/LABELING.md)
  models/       ResNet-50 recipes and target-attention MIL head
  training/     masked soft-label training loop and k-fold cross-validation
  inference/    offline Kaggle inference adapter
  validation/   grouped multilabel folds and AUC metrics
scripts/        command-line entry points (see scripts/README.md)
tests/          unit and regression tests (synthetic data only)
docs/           competition contract, data notes, decisions, results
experiments/    experiment records (current) and experiments/archive/ (history)
data/           local, git-ignored inputs, caches, checkpoints and predictions
```

## Reproducing

All competition data, caches, checkpoints and predictions live under the git-ignored `data/`
directory. Nothing in Git contains images, reports or study identifiers.

```powershell
# 1. Environment (Python 3.11). Use the CUDA wheels matching your driver.
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[images]"

# 2. Labels and folds (needs train.csv, train_series.csv and the public label CSV)
python scripts/prepare_full_cohort_prefilter.py --help

# 3. 16-slice tensor cache (built as offline Kaggle CPU notebooks, then downloaded)
python scripts/build_local16_cache_package.py --help

# 4. Cross-validation: train each fold, score its held-out studies, aggregate
python scripts/cv_attention16.py train   --fold 1
python scripts/cv_attention16.py predict --fold 1
python scripts/cv_attention16.py summarize --folds 0 1 2

# 5. Kaggle submission package (model export + offline inference notebook)
python scripts/build_kaggle_attention_sixteen_submission_package.py --help
```

Run the tests with:

```powershell
$env:PYTHONPATH = "src;."; python -m unittest discover -s tests
```

## Limitations

- **The training and validation labels come from an LLM, not from experts.** Their generation
  model and prompt are undisclosed, and local AUC inherits their noise.
- **Each configuration was trained with one seed.** The main reported number is the 3-fold
  cross-validation estimate.
- **Earlier comparisons are not controlled ablations.** They used a single fold or several
  changes at once.
- **This is research code, not a clinically validated diagnostic system.**

## Data, privacy and licensing

Competition data are covered by the competition rules and the
[RSNA MIRA license](http://rsna.org/mira-license). The scope of that license over derived
labels, features and weights has not been clarified with the organizers. Do not redistribute
competition-derived artifacts or use them commercially until it is. See
[`docs/COMPETITION.md`](docs/COMPETITION.md) and [`docs/DATA.md`](docs/DATA.md).

## Project records

- [`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md): current status and next steps
- [`docs/RESULTS.md`](docs/RESULTS.md): cross-validation results per fold and target
- [`docs/DECISIONS.md`](docs/DECISIONS.md): durable decisions
- [`experiments/`](experiments/): experiment records

The full research history is preserved at the git tag `competition-archive-2026-11`. It includes
the label-pipeline engineering, superseded models, Kaggle packaging iterations and the detailed
decision log.
