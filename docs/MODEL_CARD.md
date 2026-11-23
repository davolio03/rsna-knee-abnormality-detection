# Model Card — Attention16 knee MRI ensemble (E059)

## Summary

| | |
|---|---|
| **Task** | Study-level multi-label classification of 12 knee abnormalities from a complete MRI study |
| **Targets** | ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's cyst, Bone contusion, Fracture |
| **Output** | One probability per target (sigmoid of the mean logit of the ensemble members) |
| **Version** | 1.0.0, final submission [E059](../experiments/E059_attention16_final_all_data.md) |
| **Score** | Kaggle late submission, mean ROC-AUC over 12 targets: public 0.934, private 0.923 |
| **Status** | Research prototype. Not a medical device and not clinically validated. |

## Architecture

- **Input:** every series of the study; up to 16 uniformly sampled, ordered slices per series,
  resized to 224 × 224, intensities scaled to [0, 1], with pixel validity masks.
- **Backbone:** RadImageNet ResNet-50, shared across slices. `layer3` and `layer4` are fine-tuned;
  earlier stages and all BatchNorm statistics are frozen.
- **Head:** a linear classifier gives 12 logits per slice. For each target, a slice-attention
  softmax pools slice logits into series logits, and a series-attention softmax pools series
  logits into study logits. The attention heads start at zero, so training begins from mean
  pooling.
- **Ensemble:** mean logits of 7 members: four models trained on all labelled studies (seeds
  42–45) and the three cross-validation fold models of E058.

Code: [`src/rsna_knee/models/attention_mil.py`](../src/rsna_knee/models/attention_mil.py).

## Training data

| Source | Studies | Targets |
|---|---:|---|
| Training cohort | 4,298 | 7th-place image-and-report LLM labels (soft, 0–1) |
| Training studies outside the cohort | 51 | 7th-place labels |
| Gold-58 | 58 | expert labels from the competition `train.csv` |

The 7th-place labels are [`songqizhou/rsna-knee-llm-labels`](https://www.kaggle.com/datasets/songqizhou/rsna-knee-llm-labels)
(`8_final_label.csv`, CC BY-NC-SA 4.0). They mix an LLM reading of the image and the report with
out-of-fold image-model predictions; they are not expert labels. The fold models (E058) did not
see Gold-58; the four final models did.

**Objective:** masked soft-label binary cross-entropy. **Optimizer:** Adam, head 1e-3, backbone
1e-5, 5 epochs, one study per step; the final epoch is fixed in advance.

## Evaluation

| Evaluation | Data | Result |
|---|---|---|
| Kaggle hidden test (late) | competition test set | public 0.934 / private 0.923 |
| Expert labels (E058 fold ensemble) | Gold-58, 58 studies, never trained on | macro AUC 0.897 |
| 3-fold CV (E055 recipe, report labels) | 4,298 studies | macro AUC 0.864 ± 0.003 |

Per-target results and the evaluation protocol are in [`RESULTS.md`](RESULTS.md). Every
comparison and its decision rule was declared before scoring.

## Limitations

- **Labels are noisy.** Training labels come from LLMs, not radiologists. Expert-labelled data is
  limited to 58 studies, so per-target expert AUCs are noisy.
- **Single source of data.** All studies come from one competition dataset. Performance on other
  scanners, protocols, sites or populations is unknown.
- **Late submissions.** Kaggle scores were obtained after the deadline; they are not on the
  official leaderboard.
- **Not for clinical use.** The model has not been validated prospectively, has no calibration
  analysis and must not inform diagnosis or treatment.

## Licensing and availability

The code is MIT-licensed. Model weights are **not** distributed: they derive from competition data
under the [RSNA MIRA license](http://rsna.org/mira-license), whose scope over trained weights has
not been clarified, and from labels licensed CC BY-NC-SA 4.0 (non-commercial, share-alike).
