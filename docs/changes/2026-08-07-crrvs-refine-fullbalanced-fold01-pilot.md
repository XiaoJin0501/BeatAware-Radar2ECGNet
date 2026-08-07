# CR_RVS Full-Balanced Refine Fold-01 Pilot

## Why

CR_RVS Shape-200 results show strong morphology reconstruction for
REM-AlignNet-MTL, but full-window diagnostic evaluation showed that the dense
continuous decoder was weaker than CW-STFT+TCN. This pilot tested whether the
existing morphology-guided `REMAlignMTLRefineNet` can improve the full-window
branch without sacrificing Shape-200 morphology.

## Experiment

Experiment ID:

`q1_crrvs_remalign_refine_fullbalanced_fold01_pilot`

Configuration:

- Dataset: CR_RVS / Nature
- Fold: 1 of 5 subject-independent CV
- Input: CW-STFT
- Model: `rem_align_mtl_refine`
- Epochs: 50
- `narrow_bandpass=false`
- `beta_peak=0.3`
- `use_shape200_loss=true`
- `lambda_shape200=0.2`
- `shape200_pcc_weight=0.3`

An initial launch with default MMECG values was stopped immediately because
preflight logs showed `beta_peak=1.0` and `narrow_bandpass=true`. The partial
output directory was cleaned before the controlled run.

## Results

Fold-01 comparison:

| Method | Shape-200 PCC | Full-window PCC | Full-window RMSE | RR Error |
|---|---:|---:|---:|---:|
| CW-STFT + REM-AlignNet-MTL | 0.8752 | 0.4555 | 0.1513 | 27.5 ms |
| CW-STFT + REM-AlignNet-MTL-Refine | 0.8842 | 0.5455 | 0.1446 | 21.7 ms |
| CW-STFT + TCN | 0.6243 | 0.5651 | 0.1425 | 14.1 ms |

Report artifacts:

`experiments_mmecg/q1_crrvs_refine_fullbalanced_fold01_report/`

## Interpretation

This is a weak-positive pilot. The refine branch substantially improves the
REM full-window decoder and preserves the Shape-200 morphology advantage.
However, it does not yet beat CW-STFT+TCN on full-window PCC or RR timing.

## Decision

Do not promote to 5-fold final yet.

The next controlled experiment should either:

1. add explicit lag-aware full-window loss while keeping the refine branch, or
2. distill/initialize the full decoder from the stronger CW-STFT+TCN full-window
   behavior while preserving REM's Shape-200 morphology head.
