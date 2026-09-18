# Continuous Reconstruction Failure Diagnostic

Post-hoc, GT-assisted analysis of existing formal predictions only; no model was trained. PCC is computed per native 8-second window. Oracle lag scans ±100 ms at 200 Hz and is diagnostic, not deployable performance. Positive lag means `rho(GT[t], prediction[t+lag])`. Representative windows are deterministically selected as median PCC, minimum PCC, maximum PCC, and maximum lag gain.

| Fold | n | PCC@0 mean | PCC@0 median | negative PCC | median std ratio | near-flat (<0.1) | oracle NCC mean | lag gain | median best lag | ±100 ms boundary | polarity-flip preferred |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 170 | 0.021 | 0.013 | 43.5% | 0.406 | 0.0% | 0.457 | +0.437 | +55 ms | 0.0% | 0.0% |
| 3 | 1020 | 0.030 | -0.034 | 62.1% | 0.558 | 0.0% | 0.537 | +0.507 | -60 ms | 11.9% | 0.2% |
| 5 | 255 | 0.600 | 0.607 | 0.0% | 0.718 | 0.0% | 0.671 | +0.071 | -10 ms | 0.0% | 0.0% |
| 9 | 765 | 0.504 | 0.516 | 0.0% | 0.542 | 0.0% | 0.684 | +0.179 | +15 ms | 0.0% | 0.0% |

## Fold-level fixed-lag check

This check applies one GT-selected lag to every window in a fold. It distinguishes
a nearly static subject-level offset from a separate oracle shift for every
window. It remains diagnostic-only because the test ECG selects the lag.

| Fold | PCC@0 | Best single fixed lag | PCC after fixed lag | Gain |
|---:|---:|---:|---:|---:|
| 2 | 0.021 | +55 ms | 0.416 | +0.395 |
| 3 | 0.030 | -60 ms | 0.339 | +0.309 |
| 5 | 0.600 | -10 ms | 0.627 | +0.027 |
| 9 | 0.504 | +10 ms | 0.516 | +0.012 |

## Interpretation

- Fold 2 is primarily a stable timing-offset failure: its per-window optimal
  lags have a 10th/50th/90th percentile of +35/+55/+70 ms, and one +55 ms shift
  recovers PCC from 0.021 to 0.416.
- Fold 3 is also strongly timing-confounded, but less purely static. One -60 ms
  shift recovers PCC to 0.339, while per-window optimal lags span
  -100/-60/+15 ms at the 10th/50th/90th percentiles; 11.9% hit the registered
  +/-100 ms boundary.
- Neither failed fold is amplitude-flat: median prediction/GT standard-deviation
  ratios are 0.406 and 0.558, with no window below 0.1. Neither is explained by
  polarity inversion: a negative oracle correlation is preferred in only 0.0%
  and 0.2% of windows.
- Folds 5 and 9 already have near-zero fold-level offsets (-10 and +10 ms), and
  a fixed correction adds only +0.027 and +0.012 PCC. Their advantage is not an
  artifact of applying an oracle shift.
- Timing correction does not fully restore Fold 2/3 to Fold 5/9 quality. The
  remaining gap is visible as attenuated QRS amplitude and imperfect local
  P-QRS-T morphology, so the result supports a combined timing-plus-morphology
  failure rather than an alignment-only explanation.

## Figures

- `fold_02_gt_vs_prediction.png`
- `fold_03_gt_vs_prediction.png`
- `fold_05_gt_vs_prediction.png`
- `fold_09_gt_vs_prediction.png`
