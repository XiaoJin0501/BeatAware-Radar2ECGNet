# Q1 60-second priming calibration audit

## Scope and protocol

This is a **paired-ECG post-hoc calibration track**, not strict zero-shot LOSO. The frozen seed-42 REM-Net predictions are reused; no network weight is trained or selected.

For each held-out subject, calibration uses every 8-second window from the first chronological record whose source interval ends at or before 60 seconds. Windows crossing the 60-second boundary are discarded. Evaluation uses windows starting at or after 60 seconds plus all later records. Therefore calibration and evaluation contain no overlapping source samples despite the 2-second window stride.

The calibration set selects one integer lag from -150 to +150 ms in 5-ms steps by mean per-window PCC, then estimates one scale factor `std(ECG) / std(shifted prediction)`. Both parameters are frozen before evaluation. Ground-truth ECG is used only for calibration and final scoring.

## Results

| Fold | Calibration / evaluation windows | Lag (ms) | Scale | PCC zero | PCC 60 s | Delta PCC | R2 zero | R2 60 s | NRMSE zero | NRMSE 60 s | F1@100 zero | F1@100 60 s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 27 / 140 | +10 | 2.001 | 0.349 | 0.409 | +0.061 | -0.054 | -0.788 | 2.436 | 3.171 | 0.791 | 0.796 |
| 2 | 27 / 140 | +55 | 2.440 | 0.037 | 0.408 | +0.371 | -0.153 | -0.266 | 2.384 | 2.493 | 0.900 | 0.926 |
| 3 | 27 / 990 | -50 | 1.725 | 0.030 | 0.306 | +0.276 | -0.272 | -0.368 | 2.528 | 2.563 | 0.728 | 0.869 |
| 4 | 27 / 480 | +15 | 2.046 | 0.442 | 0.374 | -0.068 | 0.152 | -0.773 | 2.462 | 3.539 | 0.883 | 0.879 |
| 5 | 27 / 225 | +10 | 1.528 | 0.606 | 0.513 | -0.093 | 0.353 | -0.100 | 1.382 | 1.814 | 0.954 | 0.948 |
| 6 | 27 / 480 | +5 | 2.156 | 0.393 | 0.422 | +0.029 | 0.132 | -0.295 | 1.944 | 2.367 | 0.790 | 0.794 |
| 7 | 27 / 735 | +35 | 6.103 | 0.215 | 0.175 | -0.039 | 0.009 | -2.174 | 2.225 | 3.951 | 0.005 | 0.031 |
| 8 | 27 / 565 | -20 | 3.245 | 0.257 | 0.274 | +0.017 | 0.044 | -1.124 | 2.025 | 2.994 | 0.514 | 0.512 |
| 9 | 27 / 735 | -15 | 1.632 | 0.499 | 0.439 | -0.060 | 0.240 | -0.003 | 1.612 | 1.844 | 0.907 | 0.895 |
| 10 | 27 / 1925 | +95 | 1.583 | 0.221 | 0.039 | -0.183 | -0.187 | -1.215 | 2.359 | 3.293 | 0.929 | 0.666 |
| 11 | 27 / 990 | -60 | 1.299 | 0.311 | 0.236 | -0.075 | -0.059 | -0.387 | 2.295 | 2.662 | 0.911 | 0.861 |
| **Fold macro mean** | — | — | — | **0.305** | **0.327** | **+0.021** | **0.018** | **-0.681** | **2.150** | **2.790** | **0.756** | **0.743** |
| **Fold macro SD** | — | — | — | **0.171** | **0.131** | — | **0.181** | **0.606** | **0.355** | **0.642** | **0.266** | **0.256** |

Machine-readable output: `experiments_mmecg/q1_remnet_60s_calibration_summary.json`.

## Decision

The calibration rescues the two timing-dominated failures, folds 2 and 3, by +0.371 and +0.276 PCC. However, only 5/11 folds improve; fold 10 loses 0.183 PCC. The fold-macro PCC gain is only +0.021, while R2, NRMSE, and F1 all deteriorate on average. The large fold-7 scale (6.103) is an additional warning that a first-record variance estimate is not stable across the subject's later data.

Consequently, a single lag and unconstrained standard-deviation scale estimated from the first 60 seconds **fails as a general 11-subject calibration method**. It may be reported as a bounded diagnostic showing that paired timing calibration can rescue particular subjects, but it is not promoted as the main method or a general clinical solution. No duration, lag range, scaling rule, or fold-specific fallback will be tuned on these test outcomes.
