# Script Registry

Use this file as the entry point before launching experiments.

## Status Labels

| Status | Meaning |
|---|---|
| active | Main entry point for current Q1 work |
| frozen | Reproduced or adapted baseline; do not modify unless fixing protocol bugs |
| report | Generates tables/figures from existing registered experiments |
| legacy-compatible | Kept for backward compatibility, not the current Q1 main path |

## Core Training and Evaluation

| Script | Status | Purpose |
|---|---|---|
| `train_mmecg.py` | active | Main MMECG/CR_RVS-compatible trainer for BeatAware, REM, and simple baselines |
| `test_mmecg.py` | active | Full 8-s continuous ECG evaluator |
| `evaluate_mmecg_shape200.py` | active | Unified Shape-200 evaluator for continuous-output baselines and radarODE-MTL |
| `evaluate_remalign_shape_head.py` | active | Direct Shape-200 head evaluator for REM-AlignNet-MTL |
| `eval_tta_pilot.py` | pilot | Registered Fold-2 zero-label scalar-shift TTA falsification test |

## radarODE-MTL Baseline

| Script | Status | Purpose |
|---|---|---|
| `precompute_wsst4s_mmecg.py` | frozen | Build radarODE-style 4-s WSST caches for MMECG |
| `train_radarode_mtl_ega_mmecg.py` | frozen | Adapted radarODE-MTL/EGA training on MMECG strict LOSO |
| `run_radarode_ega_strict_loso_all.sh` | frozen | Wrapper for all radarODE-MTL strict LOSO folds |

## CR_RVS External Validation

| Script | Status | Purpose |
|---|---|---|
| `build_crrvs_wsst_shape_h5.py` | active | Build MMECG-compatible H5 files from raw CR_RVS `.mat` data |
| `run_crrvs_external_5fold.sh` | active | Run CR_RVS 5-fold external validation |
| `make_crrvs_external_report.py` | report | Generate CR_RVS external validation report |

## Report Generation

| Script | Status | Purpose |
|---|---|---|
| `make_supervisor_shape200_snapshot.py` | report | English supervisor-facing MMECG + CR_RVS Shape-200 snapshot |
| `make_q1_paper_style_report.py` | report | MMECG paper-style Shape-200 figures/tables |
| `make_q1_teacher_report.py` | report | Focused REM-AlignNet-MTL vs radarODE-MTL report |
| `summarize_shape200_results.py` | report | Aggregate Shape-200 fold summaries |

## Published Baseline Adaptations

| Script | Status | Purpose |
|---|---|---|
| `train_airecg_mmecg_strict_loso_v2.py` | frozen | Adapted AirECG baseline |
| `train_rssrnet_mmecg_strict_loso.py` | frozen | Adapted RSSRNet baseline |

Historical ICONIP ablation runners were removed because they created
unregistered `*_ablation` experiment directories. If an ablation is needed
again, add it to `experiments_mmecg/EXPERIMENT_REGISTRY.csv` first and use a
clear `q1_mmecg_*` experiment tag.

Generic CNN/LSTM/BiLSTM/Conformer result directories are retained as frozen
outputs for reporting. They are not current training entry points in
`train_mmecg.py`.

## Deleted Probe Scripts

Temporary probe runners were deleted during cleanup. Formal entry points must
be listed in this registry before they are used for paper-facing results. The
cleanup record is maintained in `../docs/DELETED_EXPERIMENTS_LOG.md`.

The completed 60-second paired-ECG calibration evaluator is archived at
`archive/legacy_scripts/audits/eval_60s_calibration_loso.py`. It supports the
Discussion audit only and is not an active optimization or model-selection
entry point.
