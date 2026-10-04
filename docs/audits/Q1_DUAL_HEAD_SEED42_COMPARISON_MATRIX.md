# Dual-head seed-42 comparison matrix

Date: 2026-10-04. This matrix separates matched system-level timing endpoints
from task-mismatched Shape diagnostics. It is not a promotion claim.

## ECG Anchor Estimation / waveform-derived event comparison

All entries use `q1_timing_screen_v2` prominence reference endpoints and the
same held-out `loso_v2` subjects. Baselines obtain events by delineating their
continuous waveform; dual-head REM-Net uses its frozen structured probability
read-out. Thus this is a system-level endpoint comparison, not a claim that the
read-out algorithms are identical. MedAE entries are medians over folds. The
dual-head MedAE excludes fold 7 because it has zero matches; its failure remains
visible in coverage and F1.

| system (seed 42) | Q | R | S | T | macro MedAE | mean coverage | mean F1@100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dual-head REM-Net | 35 | 35 | 37.5 | **27.5** | 33.75 | 0.565 | 0.627 |
| Chen-TCN | 40 | **30** | **30** | 30 | **32.50** | 0.858 | **0.863** |
| radarODE-MTL adapted continuous | **30** | 35 | 45 | 50 | 40.00 | **0.870** | 0.849 |
| WaveGRU-Net adapted | 40 | 40 | 40 | 50 | 42.50 | 0.659 | 0.636 |
| Attention-DCGAN adapted | 50 | 45 | 45 | 50 | 47.50 | 0.797 | 0.724 |
| Cross-Modal Conformer adapted | 35 | 40 | 40 | 45 | 40.00 | 0.820 | 0.800 |

The four-event head has competitive conditional localization and the best T
MedAE, but its coverage/reliability is substantially below Chen-TCN,
radarODE-MTL and Conformer. It cannot be called the best event estimator.

## ECG Shape Recovery compatibility boundary

| output/training formulation | seed-42 fold-macro PCC | interpretation |
|---|---:|---|
| Dual-head REM-Net direct 200-point head | 0.886 | directly supervised known-cycle Shape task |
| radar-free training-fold template | 0.881 | mandatory floor; no radar input |
| Chen-TCN continuous output, GT-cycle crop | 0.345 | diagnostic read-out from a model trained for 1600-point continuous output |
| REM-Net continuous output, GT-cycle crop | 0.321 | diagnostic read-out from a model trained for 1600-point continuous output |
| radarODE adapted continuous output, GT-cycle crop | 0.308 | diagnostic only |
| Cross-Modal Conformer continuous output, GT-cycle crop | 0.308 | diagnostic only |
| WaveGRU-Net continuous output, GT-cycle crop | 0.221 | diagnostic only |
| Attention-DCGAN continuous output, GT-cycle crop | 0.140 | diagnostic only |

The 0.886 versus 0.14--0.35 contrast cannot be described as model superiority:
the proposed model is directly trained on the normalised cycle while these
baselines are trained on the harder continuous target. More importantly, the
proposed model is not significantly above the 0.881 radar-free template
(5/11 wins, paired Wilcoxon p=0.8984).

## Required next evidence

1. Prospectively freeze a multi-objective validation checkpoint rule that does
   not select an immature event head solely for a transient Shape-PCC maximum.
2. Use a matched direct-Shape training target for at least Chen-TCN,
   radarODE-MTL and Attention-DCGAN; keep the five continuous rows separately.
3. Require radar-derived skill above the training-fold template before spending
   two additional seeds or extending the four-event model to CR_RVS.
