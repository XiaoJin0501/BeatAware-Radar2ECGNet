# 2026-08-07 MMECG Long Assembly Pilot

## Why

The current Shape-200 results are strong but cannot be described as final long-term continuous ECG reconstruction. A radarODE-style comparison requires preserving record metadata and evaluating assembled ECG in record coordinates.

## What Changed

- Added `scripts/evaluate_mmecg_long_assembly.py`.
- Added `experiments/specs/q1_mmecg_long_assembly_fold01_pilot.json`.
- Registered `q1_mmecg_long_assembly_fold01_pilot` in `experiments_mmecg/EXPERIMENT_REGISTRY.csv`.

## Scientific Meaning

Fold-01 pilot evidence separates three questions:

- REM-AlignNet-MTL full-window continuous decoding: record-level PCC `0.394`, coverage `1.00`.
- REM-AlignNet-MTL morphology head under oracle cycle placement: record-level PCC `0.937`, coverage `0.44`.
- Adapted radarODE-MTL ECG_shape under oracle cycle placement: record-level PCC `0.871`, coverage `0.44`.
- Adapted radarODE-MTL ECG_shape under predicted Anchor/PPI placement: record-level PCC `0.151`, coverage `0.37`.
- Adapted radarODE-MTL ECG_shape under center-selected Anchor/PPI placement: record-level PCC `0.113`, coverage `0.44`.

This suggests the REM morphology representation is strong, but the current continuous decoder/assembly path is the bottleneck. The oracle-cycle protocols are diagnostics and must not be promoted as final deployable long ECG claims.

The radarODE Anchor/PPI audit found median Anchor nearest-R error `42.5 ms`
and median PPI length error `20.0 ms`, yet both predicted assembly variants
remain poor. Therefore the current adapted predicted Anchor/PPI long assembly
is treated as a failed diagnostic, not a paper-comparison result.

## Commands

```bash
/home/qhh2237/miniconda3/envs/cyberbrain/bin/python -m py_compile scripts/evaluate_mmecg_long_assembly.py
/home/qhh2237/miniconda3/envs/cyberbrain/bin/python scripts/evaluate_mmecg_long_assembly.py --exp_tag q1_remalign_mtl_shapefirst_loso --fold_idx 1 --protocol rem_full_ola --batch_size 32
/home/qhh2237/miniconda3/envs/cyberbrain/bin/python scripts/evaluate_mmecg_long_assembly.py --exp_tag mmecg_radarode_mtl_ega_strict_loso --fold_idx 1 --protocol radarode_shape_oracle_cycle_ola --batch_size 32
/home/qhh2237/miniconda3/envs/cyberbrain/bin/python scripts/evaluate_mmecg_long_assembly.py --exp_tag q1_remalign_mtl_shapefirst_loso --fold_idx 1 --protocol rem_shape_oracle_cycle_ola --batch_size 32 --output_dir experiments_mmecg/q1_remalign_mtl_shapefirst_loso/fold_01/results_long_shape_oracle
/home/qhh2237/miniconda3/envs/cyberbrain/bin/python scripts/evaluate_mmecg_long_assembly.py --exp_tag mmecg_radarode_mtl_ega_strict_loso --fold_idx 1 --protocol radarode_pred_anchor_ppi_cycle_ola --batch_size 32 --output_dir experiments_mmecg/mmecg_radarode_mtl_ega_strict_loso/fold_01/results_long_pred_anchor
```

## Outputs

- `experiments_mmecg/q1_mmecg_long_assembly_fold01_pilot/summary_long_fold01.csv`
- `experiments_mmecg/q1_mmecg_long_assembly_fold01_pilot/long_assembly_fold01_bar.png`
- `experiments_mmecg/q1_mmecg_long_assembly_fold01_pilot/long_assembly_waveform_fold01_examples.png`
- Per-method `results_long/global_summary_long.json` files under the corresponding fold directories.

The intermediate `predictions_long.npz` files were deleted after figure/table
generation because they are reproducible from the saved checkpoints and H5
metadata. Lightweight CSV/JSON/PNG evidence was retained for reporting.

## Next Agent Task

Refocus the next model experiment on improving REM-AlignNet-MTL full-window
continuous decoding/assembly. Keep radarODE in the main comparison as
Shape-200/oracle-cycle morphology unless a faithful official long ECG assembly
implementation becomes available.
