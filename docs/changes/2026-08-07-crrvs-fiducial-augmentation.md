# CR_RVS Fiducial Augmentation for Timing Diagnostics

## Why

The current CR_RVS external validation results are strong for Shape-200
waveform morphology, but the first H5 preprocessing pipeline stored only
R-peaks and left Q/S/T arrays empty. This prevented fair full-window timing
diagnostics such as QRS and QT errors.

## What Changed

- Added `scripts/augment_crrvs_fiducials.py`.
- The script delineates each full 8-second ECG window using stored R-peaks and
  NeuroKit2 DWT delineation.
- It writes full-window `q_indices`, `s_indices`, `tpeak_indices`, and
  `delineation_valid` back to MMECG-compatible CR_RVS H5 files.
- It generates an audit report under
  `experiments_mmecg/q1_crrvs_fiducial_audit/`.

## Data Action Performed

Command:

```bash
/home/qhh2237/miniconda3/envs/cyberbrain/bin/python scripts/augment_crrvs_fiducials.py \
  --in_place \
  --folds 1 2 3 4 5 \
  --splits test
```

Result:

- CR_RVS test folds 1-5 were augmented in-place at
  `/home/qhh2237/Datasets/CR_RVS/processed/q1_shape_kfold`.
- Mean valid delineation rate: `1.000`.
- Each test segment now has full-window Q/S/T labels available for diagnostics.

## Scientific Meaning

This enables local full-window timing diagnostics on CR_RVS. These labels are
ECG-derived ground truth and must not be used as radar inputs, test-time model
selection signals, or evidence of radar-side oracle access.

## Remaining Risk

The timing diagnostics are still a local re-evaluation protocol. They should
not be described as exactly identical to VMD-CBNeT's original evaluation unless
the original split, preprocessing, and interval definitions are fully matched.

## Next Agent Task

Run CR_RVS full-window evaluation for REM-AlignNet-MTL and CW-STFT+TCN using
the augmented test labels, then summarize waveform and timing metrics beside
the existing Shape-200 external-validation table.
