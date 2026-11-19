# Results

## Current recipe: layer3 + layer4 fine-tuning (E055)

The E054 recipe with backbone layer3 trainable as well as layer4
([D036](DECISIONS.md#d036--fine-tune-layer3-and-layer4-2026-11-19)). Details:
[E055](../experiments/E055_attention16_layer3_finetuning.md).

**Macro ROC-AUC: 0.864 ± 0.003** (mean ± std over 3 folds) · pooled out-of-fold: 0.863 · 4,298 studies

| Target | Fold 0 | Fold 1 | Fold 2 | Mean ± std | Pooled OOF | Positives / n |
|---|---:|---:|---:|---:|---:|---:|
| ACL | 0.826 | 0.823 | 0.846 | 0.832 ± 0.012 | 0.831 | 954 / 3,987 |
| MCL | 0.824 | 0.812 | 0.793 | 0.810 ± 0.016 | 0.804 | 679 / 3,912 |
| Medial Meniscus | 0.878 | 0.877 | 0.856 | 0.870 ± 0.012 | 0.870 | 2,330 / 4,108 |
| Lateral Meniscus | 0.805 | 0.845 | 0.831 | 0.827 ± 0.020 | 0.827 | 1,116 / 3,917 |
| Medial OA | 0.884 | 0.884 | 0.887 | 0.885 ± 0.002 | 0.883 | 1,648 / 3,469 |
| Lateral OA | 0.857 | 0.865 | 0.868 | 0.863 ± 0.005 | 0.861 | 1,259 / 3,280 |
| PF OA | 0.850 | 0.842 | 0.836 | 0.843 ± 0.007 | 0.842 | 2,022 / 3,600 |
| Effusion | 0.846 | 0.844 | 0.841 | 0.844 ± 0.002 | 0.843 | 2,606 / 3,919 |
| Synovitis | 0.945 | 0.895 | 0.901 | 0.914 ± 0.027 | 0.917 | 551 / 701 |
| Baker's | 0.893 | 0.896 | 0.915 | 0.901 ± 0.012 | 0.901 | 1,075 / 2,256 |
| Contusion | 0.892 | 0.875 | 0.879 | 0.882 ± 0.009 | 0.881 | 1,080 / 3,304 |
| Fracture | 0.904 | 0.889 | 0.907 | 0.900 ± 0.010 | 0.900 | 315 / 2,482 |
| **Macro** | 0.867 | 0.862 | 0.863 | **0.864 ± 0.003** | 0.863 | |

Compared with E054, the mean macro AUC rises by 0.0034 and every fold improves (+0.003 to +0.004).
The gain is about one fold standard deviation, so treat it as small but consistent. The weakest
targets improve (ACL +0.006, MCL +0.005, Lateral Meniscus +0.005), and Baker's cyst gains most
(+0.015).

## Cross-validated baseline (E054)

Model: RadImageNet ResNet-50 with target-specific slice and series attention, 16 slices at 224 px,
5 epochs, epoch 5 chosen in advance. Protocol:
[D034](DECISIONS.md#d034--cross-validation-protocol-2026-11-17). Details:
[E054](../experiments/E054_attention16_cross_validation.md).

**Macro ROC-AUC: 0.861 ± 0.003** (mean ± std over 3 folds) · pooled out-of-fold: 0.860 · 4,298 studies

| Target | Fold 0 | Fold 1 | Fold 2 | Mean ± std | Pooled OOF | Positives / n |
|---|---:|---:|---:|---:|---:|---:|
| ACL | 0.816 | 0.819 | 0.841 | 0.825 ± 0.013 | 0.825 | 954 / 3,987 |
| MCL | 0.813 | 0.805 | 0.796 | 0.805 ± 0.008 | 0.800 | 679 / 3,912 |
| Medial Meniscus | 0.876 | 0.873 | 0.854 | 0.868 ± 0.012 | 0.867 | 2,330 / 4,108 |
| Lateral Meniscus | 0.801 | 0.841 | 0.826 | 0.823 ± 0.020 | 0.822 | 1,116 / 3,917 |
| Medial OA | 0.883 | 0.885 | 0.881 | 0.883 ± 0.002 | 0.880 | 1,648 / 3,469 |
| Lateral OA | 0.855 | 0.858 | 0.869 | 0.861 ± 0.007 | 0.859 | 1,259 / 3,280 |
| PF OA | 0.848 | 0.841 | 0.831 | 0.840 ± 0.009 | 0.839 | 2,022 / 3,600 |
| Effusion | 0.845 | 0.836 | 0.834 | 0.839 ± 0.006 | 0.837 | 2,606 / 3,919 |
| Synovitis | 0.949 | 0.897 | 0.904 | 0.917 ± 0.028 | 0.920 | 551 / 701 |
| Baker's | 0.884 | 0.875 | 0.902 | 0.887 ± 0.014 | 0.886 | 1,075 / 2,256 |
| Contusion | 0.894 | 0.874 | 0.879 | 0.883 ± 0.010 | 0.882 | 1,080 / 3,304 |
| Fracture | 0.908 | 0.889 | 0.908 | 0.901 ± 0.011 | 0.902 | 315 / 2,482 |
| **Macro** | 0.864 | 0.858 | 0.860 | **0.861 ± 0.003** | 0.860 | |

Labels are the observed public LLM report labels, binarized at 0.5; unobserved cells are
excluded, so `n` differs per target. This is a proxy for the hidden-test metric, not an estimate
against expert labels. Gold-58 is excluded.

**Reading the numbers**

- The macro AUC barely moves between folds (0.858–0.864). A change to the model needs a clear
  margin over ±0.003 before it can count as an improvement, and ideally a gain on every fold.
- Per-target spread is larger. Synovitis (n = 701) and Lateral Meniscus vary by 0.02–0.03
  between folds; compare those targets with care.
- The weakest targets are MCL (0.80), Lateral Meniscus (0.82) and ACL (0.83) in E054.
- For reference, the same recipe scored 0.885 public / 0.862 private on the Kaggle leaderboard
  (E051, late submission). The local proxy and the private score are close, but they use
  different labels and studies.

## Hardware

The recipe is effectively deterministic across GPUs: retraining folds 0 and 1 on Kaggle T4s
reproduced the local macro AUC to four decimals (maximum probability difference 0.001), so folds
from either machine can be combined. On a T4 x2 session, two folds train in parallel in about
9.8 hours.

Local folds were trained on an RTX 3060 Laptop (6 GB), PyTorch 2.11.0+cu128, FP32 with TF32 off.
About 75 minutes per epoch; fold 2 took 6.3 hours for 5 epochs. Scoring a fold takes 11–25
minutes.
