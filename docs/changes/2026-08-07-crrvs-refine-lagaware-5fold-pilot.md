# CR_RVS Lag-Aware Refine 5-Fold Pilot

## Why

The fold-01 lag-aware refine pilot was the first REM variant to beat CW-STFT+TCN
on both full-window PCC and RR timing while preserving Shape-200 morphology.
This experiment tested whether that result generalizes across all CR_RVS folds.

## Experiment

Experiment ID:

`q1_crrvs_remalign_refine_lagaware_5fold_pilot`

Configuration:

- Dataset: CR_RVS / Nature
- Protocol: 5-fold subject-independent CV
- Input: CW-STFT
- Model: `rem_align_mtl_refine`
- `narrow_bandpass=false`
- `beta_peak=0.3`
- `use_shape200_loss=true`
- `lambda_shape200=0.2`
- `shape200_pcc_weight=0.3`
- `use_lag_aware_loss=true`
- `lag_max_ms=80`
- `lambda_lag_pcc=0.05`
- `lambda_lag_l1=0.01`
- `lambda_zero_pcc=0.05`
- `lambda_lag_penalty=0.02`

## Results

5-fold summary:

| Method | Shape-200 PCC | Shape-200 RMSE | Full-window PCC | Full-window RMSE | RR Error | Avg F1 150 ms |
|---|---:|---:|---:|---:|---:|---:|
| CW-STFT + REM-AlignNet-MTL | 0.8865 | 0.1038 | 0.3475 | 0.1575 | 27.17 ms | 0.5800 |
| CW-STFT + REM-AlignNet-MTL-Refine-Lag | 0.8862 | 0.1002 | 0.5001 | 0.1407 | 14.65 ms | 0.8777 |
| CW-STFT + TCN | 0.5503 | 0.1737 | 0.4379 | 0.1522 | 17.69 ms | 0.7487 |

Report artifacts:

`experiments_mmecg/q1_crrvs_full_window_report/`

## Interpretation

This is a strong CR_RVS external-validation pilot:

- The lag-aware refine model keeps the original REM Shape-200 morphology
  advantage.
- It also turns the previously weak full-window branch into a competitive
  continuous ECG diagnostic output.
- Compared with CW-STFT+TCN, it improves full-window PCC and RR timing on the
  5-fold average.

## Decision

Candidate for final CR_RVS external validation, but still `publishable=false`
until metric audit and final protocol promotion are completed.

## Next Agent Task

Run a metrics audit for CR_RVS aggregation and then mirror the same
lag-aware/refine configuration on MMECG to test whether the continuous-branch
gain also transfers to the main MMECG dataset.
