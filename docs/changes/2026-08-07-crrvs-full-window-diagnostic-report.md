# CR_RVS Full-Window Diagnostic Report

## Why

After augmenting CR_RVS test H5 files with ECG-derived Q/S/T fiducials, we ran
full-window evaluation to understand whether the current CR_RVS evidence also
supports 8-second continuous ECG claims.

## What Changed

- Added `scripts/make_crrvs_full_window_report.py`.
- The script summarizes:
  - CR_RVS Shape-200 morphology results;
  - CR_RVS full-window diagnostic results;
  - reported-only VMD-CBNeT Nature/CR_RVS reference numbers.
- It writes report artifacts to
  `experiments_mmecg/q1_crrvs_full_window_report/`.

## Commands

```bash
/home/qhh2237/miniconda3/envs/cyberbrain/bin/python scripts/test_mmecg.py \
  --exp_tag q1_crrvs_remalign_mtl_5fold --fold_idx 1

for fold in 2 3 4 5; do
  /home/qhh2237/miniconda3/envs/cyberbrain/bin/python scripts/test_mmecg.py \
    --exp_tag q1_crrvs_remalign_mtl_5fold --fold_idx "$fold"
done

for fold in 1 2 3 4 5; do
  /home/qhh2237/miniconda3/envs/cyberbrain/bin/python scripts/test_mmecg.py \
    --exp_tag q1_crrvs_wsst_tcn_5fold --fold_idx "$fold"
done

/home/qhh2237/miniconda3/envs/cyberbrain/bin/python scripts/make_crrvs_full_window_report.py
```

## Main Evidence

Shape-200 morphology on CR_RVS:

- CW-STFT + REM-AlignNet-MTL: PCC `0.8865`, RMSE `0.1038`.
- CW-STFT + TCN: PCC `0.5503`, RMSE `0.1737`.
- Reported VMD-CBNeT Nature/CR_RVS: PCC `0.856`, RMSE `0.112`.

Full-window diagnostic on CR_RVS:

- CW-STFT + REM-AlignNet-MTL: PCC `0.3475`, RR error `27.17 ms`.
- CW-STFT + TCN: PCC `0.4379`, RR error `17.69 ms`.

## Scientific Interpretation

The current proposed model has strong external morphology reconstruction via
the Shape-200 head. However, the full-window decoder is not yet the strongest
continuous ECG branch. For the Q1 paper, CR_RVS Shape-200 morphology should be
the main external-validation evidence, while full-window results should remain
diagnostic until the continuous branch is improved.

## Next Agent Task

Design the next experiment around a shape-guided continuous decoder or
full-window loss schedule, using CW-STFT + TCN as the full-window baseline and
REM-AlignNet-MTL Shape-200 as the morphology anchor.
