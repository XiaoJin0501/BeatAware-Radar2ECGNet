# Timing-Identifiability Gate — Preflight

## 1. The 187 s figure was a per-run total, not per-epoch

`(187s)` in the aborted log was printed by `t0 = time.time()` set **before**
`run(arm, fold, ...)`, which trains up to 80 epochs with early stopping. At the measured
~7 s/epoch, 187 s is ~27 epochs — exactly patience-20 early stopping after a best near
epoch 7. The log printed one unlabelled timing per run, which reads as per-epoch. Fixed:
per-epoch timing is now logged explicitly and the summary says `RUN TOTAL`.

## 2. Inputs were already RAM-preloaded; measured epochs are in band

`DS.__init__` already did `np.asarray(h[...][:])` for both arms, so there was no lazy
HDF5 read in the training loop. Measured, full code path:

| fold | arm | preload | train | val-fwd | peak_stats | **epoch** |
|---|---|---|---|---|---|---|
| 1 | T0b | 4.8 s | 6.3 | 0.6 | 0.0 | **7.0 s** |
| 1 | T1 | 1.6 s | 2.4 | 0.1 | 0.0 | **2.5 s** |
| 5 | T0b | 6.3 s | 6.1 | 1.8 | 0.0 | **7.9 s** |
| 5 | T1 | 2.1 s | 2.4 | 0.2 | 0.0 | **2.6 s** |
| 9 | T0b | 9.2 s | 5.0 | 7.1 | 0.2 | **12.2 s** |
| 9 | T1 | 3.1 s | 1.9 | 0.6 | 0.2 | **2.7 s** |

Worst case 12.2 s/epoch, inside the 10–20 s target. No preload change was needed and
none was made, rather than claiming a fix for a problem that did not exist.

`cudnn.deterministic=True` costs T1 5.1 → 23.2 ms/batch but leaves the epoch at ~2.7 s.
Determinism is kept; it is scientifically required and affordable here.

## 3. Smoke can no longer contaminate the formal directory

The 2-epoch harness check had written `results.json` into the formal tree. Moved to
`q1_timing_identifiability_gate_smoke/` with a README. In code, output now routes to
`_smoke` whenever `--smoke` is passed, `--epochs` is below the budget, **or fewer than
6 runs completed**, and partial files are named `results_partial_NofM_Xep.json`. The
formal `results.json` can only appear when all 6 runs finish at the full budget.

## 4. Raw `rcg` is signed time-domain, phase intact — verified

| check | value | reading |
|---|---|---|
| fraction of samples negative | **0.4993** | an `abs`/magnitude signal would be 0.0 |
| per-window mean | −0.00000 | zero-mean (z-scored), not rectified |
| zero-crossing rate, dominant bin | **8.9 /s** | an envelope would be ~0 |
| skewness, dominant bin | −4.93 | envelopes are strongly *positively* skewed |

`rcg` is a genuine signed oscillatory time series at 200 Hz. It has not been
abs-ed, enveloped or magnitude-transformed. This is exactly the property WSST lacks:
`precompute_wsst8s_mmecg.py` computes `np.abs(tx)`, discarding phase. The T1 premise
holds.

## 5. Unchanged

Scientific configuration and the 80/20 budget are untouched: 80 epochs, patience 20,
AdamW lr 1e-4 wd 1e-4, batch 16, seed 42+fold, loso_v2 folds 1/5/9, train-only
supervision, validation ECG opened solely to score, R-peak heatmap only.
