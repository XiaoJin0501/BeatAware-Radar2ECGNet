# CR_RVS Lag-Aware Refine Fold-01 Pilot

## Why

The full-balanced refine pilot improved REM-AlignNet-MTL full-window
reconstruction on CR_RVS fold 01, but it still trailed CW-STFT+TCN in
full-window PCC and RR error. This pilot added a constrained lag-aware
full-window loss to test whether the remaining gap is primarily a dynamic
alignment problem.

## Experiment

Experiment ID:

`q1_crrvs_remalign_refine_lagaware_fold01_pilot`

Configuration:

- Dataset: CR_RVS / Nature
- Fold: 1 of 5 subject-independent CV
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

Fold-01 comparison:

| Method | Shape-200 PCC | Full-window PCC | Full-window RMSE | RR Error | Avg F1 150 ms |
|---|---:|---:|---:|---:|---:|
| CW-STFT + REM-AlignNet-MTL | 0.8752 | 0.4555 | 0.1513 | 27.5 ms | 0.6143 |
| REM-Refine full-balanced | 0.8842 | 0.5455 | 0.1446 | 21.7 ms | 0.7459 |
| REM-Refine lag-aware | 0.8775 | 0.6121 | 0.1382 | 12.1 ms | 0.9067 |
| CW-STFT + TCN | 0.6243 | 0.5651 | 0.1425 | 14.1 ms | 0.8302 |

Report artifacts:

`experiments_mmecg/q1_crrvs_refine_lagaware_fold01_report/`

## Interpretation

This is the first CR_RVS fold-01 REM variant that beats CW-STFT+TCN on
full-window PCC and RR timing while preserving strong Shape-200 morphology.
The result supports the hypothesis that dynamic alignment, not representation
alone, is a key missing piece in the continuous decoder.

## Decision

Strong-positive pilot. Do not promote directly to final, but this configuration
should now be expanded to CR_RVS 5-fold validation.

## Next Agent Task

Register and run `q1_crrvs_remalign_refine_lagaware_5fold_pilot` with the same
configuration across folds 1-5, then regenerate the CR_RVS external validation
report with both Shape-200 and full-window diagnostics.
