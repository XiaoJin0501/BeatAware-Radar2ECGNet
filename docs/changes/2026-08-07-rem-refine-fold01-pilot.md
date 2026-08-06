# 2026-08-07 REM Refine Fold-01 Pilot

## Why

The long-assembly pilot showed a large gap between REM-AlignNet-MTL's strong
Shape-200 morphology head and weak full-window continuous ECG output. This
pilot tested the existing `REMAlignMTLRefineNet` as a minimal morphology-guided
residual refinement path.

## Protocol

- Dataset: MMECG fold-01 strict LOSO.
- Model: `rem_align_mtl_refine`.
- Input: WSST, `test_wsst.h5`.
- Training: 30 epochs, no target-subject calibration.
- Loss settings: `beta_peak=0.3`, `use_shape200_loss=true`,
  `lambda_shape200=0.8`, `shape200_pcc_weight=0.5`.
- Commit before train: `71de136`.

## Result

| Method | Protocol | PCC | RMSE | Coverage |
|---|---:|---:|---:|---:|
| REM-AlignNet-MTL | full-window OLA | 0.394 | 0.198 | 1.00 |
| REM-AlignNet-MTL | shape oracle-cycle OLA | 0.937 | 0.129 | 0.44 |
| REM-AlignNet-MTL-Refine | full-window OLA | 0.420 | 0.194 | 1.00 |
| REM-AlignNet-MTL-Refine | shape-head | 0.938 | 0.134 | n/a |

## Interpretation

The refine path is weak-positive: it slightly improves full-window OLA while
preserving morphology, but the improvement is too small to promote to 11-fold
final experiments. The next architecture should explicitly assemble or inject
cycle-level morphology into the continuous waveform rather than relying on a
small gated residual refinement.

## Cleanup

Deleted reproducible prediction NPZ files, redundant checkpoints, and the
TensorBoard event file. Retained:

- `best.pt`
- `best_shape200.pt`
- `config.json`
- `train.log`
- metric CSV/JSON files
- `experiments_mmecg/q1_remalign_mtl_refine_fold01_pilot/report/rem_refine_fold01_summary.csv`
- `experiments_mmecg/q1_remalign_mtl_refine_fold01_pilot/report/rem_refine_fold01_pcc.png`

## Next Agent Task

Design and test a cycle-assembled continuous decoder: use the strong Shape-200
head plus predicted/GT-for-training event positions to reconstruct full-window
ECG, then evaluate with full-window OLA and Shape-200 metrics on fold-01.
