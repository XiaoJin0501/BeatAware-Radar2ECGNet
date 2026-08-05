# Next Experiments for the Q1 Paper

This file defines the next formal experiments. Do not start additional probe
runs unless they are explicitly temporary and archived afterward.

## Immediate Priority

The next formal task is:

> Implement radarODE-style long-term ECG assembly and evaluate radarODE-MTL and
> REM-AlignNet-MTL under the same protocol.

This is required because the current strongest REM-AlignNet-MTL result is a
Shape-200 morphology result, while the original radarODE-MTL paper reports
final long-term ECG reconstruction after combining ECG_shape, Anchor, and PPI.

## Formal Experiment Tags To Use Next

| Tag | Dataset | Purpose |
|---|---|---|
| `q1_mmecg_radarode_mtl_radarode_long` | MMECG public | Re-evaluate adapted radarODE-MTL with long-term assembly |
| `q1_mmecg_remalign_mtl_radarode_long_oracle` | MMECG public | Upper-bound REM assembly using GT anchors/PPI |
| `q1_mmecg_remalign_mtl_radarode_long_pred` | MMECG public | Realistic REM assembly using predicted events/PPI |
| `q1_crrvs_remalign_mtl_5fold_long_oracle` | CR_RVS / Nature | External upper-bound long-style assembly |
| `q1_crrvs_remalign_mtl_5fold_long_pred` | CR_RVS / Nature | External realistic long-style assembly |

## Required Evaluations

For each formal long-style experiment, save:

```text
results_long/global_summary_long.json
results_long/segment_metrics_long.csv
results_long/trial_metrics_long.csv
results_long/predictions_long.npz
```

Metrics:

```text
PCC
RMSE
MAE
R2
HR error
R-peak timing error
MDR
```

Figures:

```text
long_ecg_waveform_examples.png
pcc_cdf_long.png
rmse_cdf_long.png
rpeak_error_cdf.png
statewise_long_metrics.png
```

## Do Not Run Yet

Do not start more Shape-200-only architecture probes until the long-term
assembly gap is clarified. The current Shape-200 evidence is already strong
enough for supervisor reporting.

## Current Supervisor Snapshot

Use this report for quick progress updates:

```text
experiments_mmecg/q1_supervisor_shape200_snapshot/SUPERVISOR_SHAPE200_SNAPSHOT.md
```

Main current values:

```text
MMECG REM-AlignNet-MTL Shape-200 PCC: 0.8898 +/- 0.0538
MMECG adapted radarODE-MTL Shape-200 PCC: 0.7283 +/- 0.1328
CR_RVS REM-AlignNet-MTL Shape-200 PCC: 0.8865 +/- 0.0228
CR_RVS CW-STFT + TCN Shape-200 PCC: 0.5503 +/- 0.0746
```
