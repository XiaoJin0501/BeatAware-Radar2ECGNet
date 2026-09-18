# Q1 REM-Net Continuous 11-Fold Metric Audit

Experiment: `q1_remnet_continuous_strict_loso_seed42`
Date: 2026-09-18. Scope: metric definitions, units, aggregation, NaN handling,
and the reporting rules that govern how this run may enter manuscript tables.

This audit is **reporting-layer only**. No evaluation or training code was
changed, and no frozen artifact was recomputed. The run finished on 2026-09-18
at 03:41 with all 11 folds in stage `COMPLETE`.

## 1. Protocol verification

| Check | Result |
|---|---|
| Split root | `loso_v2` (the repaired split; `loso` is refused by `_set_split_root`) |
| Fold states | 11/11 `COMPLETE`, each with `test_metrics.json` + `test_predictions.npz` |
| Test access | `test_opened_after_selection = true` in all 11 folds; test split constructed only in stage `TEST_EVALUATING` |
| Checkpoint rule | `checkpoint_selected_by = "validation composite only"`; Shape-200 excluded (`compute_shape200=False` on test) |
| Seed | `effective_seed = 42 + fold` (`scripts/train_continuous_ecg_pilot.py:63`) |
| Budget | 100 epochs / patience 15; every fold early-stopped (28-79 epochs run; best epoch 13-64) |
| Scoring space | raw stored ECG units — `evaluate()` inverts the per-sample transform before metrics |

### Split provenance (SHA256 prefixes)

| Fold | val | test | Fold | val | test |
|---|---|---|---|---|---|
| 1 | `acdd39f8` | `a354e77f` | 7 | `6772b41a` | `6f3df0e7` |
| 2 | `a7137894` | `acdd39f8` | 8 | `02e86baa` | `6772b41a` |
| 3 | `255300e8` | `a7137894` | 9 | `e8afcc53` | `02e86baa` |
| 4 | `ef67a85b` | `255300e8` | 10 | `e1584a23` | `e8afcc53` |
| 5 | `af44ba76` | `ef67a85b` | 11 | `a354e77f` | `e1584a23` |
| 6 | `6f3df0e7` | `af44ba76` | | | |

All 11 validation files are distinct and all 11 test files are distinct; the
assignment is a clean rotation in which each subject serves as validation
exactly once and as test exactly once. This is the defect class that invalidated
the old `loso` aggregates (`docs/MMECG_SPLIT_PROVENANCE_AUDIT.md`); it does not
recur here.

## 2. Metric definitions as implemented

Source: `scripts/train_continuous_ecg_pilot.py` (`waveform_metrics`,
`smoothing_diagnostics`, `region_errors`, `timing_metrics`, `evaluate`).

| Metric | Definition | Unit | Aggregation inside a fold |
|---|---|---|---|
| `pcc` | Pearson r per 1600-sample window | dimensionless | window-macro (`nanmean`) |
| `rmse` | RMSE per window | **z-score** (see 2.1) | window-macro |
| `nrmse` | window RMSE / **IQR of that window's ground truth** | dimensionless | window-macro |
| `mae` | MAE per window | z-score | window-macro |
| `r2` | `1 - SSE/SS_tot` per window, `SS_tot` = that window's GT variance | dimensionless | window-macro |
| `amplitude_variance_ratio` | `var(pred)/var(gt)` per window | dimensionless | window-macro |
| `above_10hz_energy_ratio` | (pred >10 Hz energy fraction) / (GT >10 Hz energy fraction), per window | dimensionless | window-macro |
| `f1_{50,100,150}ms` | F1 of **waveform-derived** R peaks (`detect_rpeaks` on the predicted waveform) | dimensionless | TP/FP/FN **pooled across windows** (micro) |
| `event_f1_{50,100,150}ms` | F1 of **event-head** peaks (`peaks_from_heatmap`) | dimensionless | pooled across windows (micro) |
| `rr_error_ms` | \|mean predicted RR - mean GT RR\| per window, windows with >=2 peaks in both | ms | mean over qualifying windows |
| `hr_error_bpm` | same windows, converted to bpm | bpm | mean over qualifying windows |
| `rr_coverage` | qualifying windows / all windows | fraction | per fold |
| `qrs_region_mae` | \|err\| over R+-50 ms | z-score | **beat-pooled** across all beats in the fold |
| `stt_region_mae` | \|err\| over R+80..R+400 ms | z-score | beat-pooled |

Tolerances are converted at `FS = 200 Hz` (50 ms = 10 samples).

### 2.1 Units

The stored MMECG ECG is z-scored per record (measured on
`loso_v2/fold_07/test.h5`: mean 0.000, std 1.000). Therefore `rmse`, `mae`,
`qrs_region_mae` and `stt_region_mae` are in **standard-deviation units of the
subject's own ECG, not millivolts**. Manuscript tables must say so. The
practical reading of `rmse = 0.979` is that the residual is about as large as
the ECG's own standard deviation, which is consistent with `r2 ~ 0`.

### 2.2 Mixed aggregation

Three aggregation styles coexist and must be labeled per metric in every table:
window-macro (waveform and smoothing metrics), window-pooled micro (all F1),
and beat-pooled (region MAE). Across folds, the summary is fold-macro.

### 2.3 Standard-deviation convention

`test_fold_macro_summary.json` uses **population** std (`statistics.pstdev`).
Sample std over 11 folds is slightly larger (PCC 0.1733 vs 0.1817; R² 0.1809 vs
0.1897). The frozen artifact is kept as is; tables must state "population std
across 11 folds".

### 2.4 NaN handling

Zero NaN values occurred: all 19 metrics are finite in all 11 folds.

### 2.5 Validation composite is partially degenerate (disclosure)

`composite = 0.40*PCC + 0.25*(1 - min(NRMSE,1)) + 0.20*max(R2,0) + 0.15*F1@100`
(`scripts/train_continuous_ecg_pilot.py:300`). Because NRMSE > 1 in every
observed epoch, the NRMSE term is identically 0, and `max(R2,0)` is 0 whenever
R² is negative. Verified on fold 11's selected epoch:
`0.40*0.5344 + 0.20*0.216 + 0.15*0.979 = 0.4037`, exactly the recorded best
score. Selection was therefore driven by PCC and waveform F1@100. The rule was
frozen before the run and is **not** changed retroactively; it must be described
accurately in the manuscript's methods.

### 2.6 Two distinct timing readouts

`f1_*ms` (peaks detected on the reconstructed waveform) and `event_f1_*ms`
(peaks from the event head) are different readouts and must never be reported
under one name. Their fold-macro gap here is 0.066 at 100 ms, and fold 7 shows a
0.405 gap.

## 3. Per-fold test results

| Fold | PCC | RMSE | NRMSE | MAE | R² | var ratio | >10 Hz ratio | F1@50 | F1@100 | F1@150 | evF1@100 | RR (ms) | HR (bpm) | cov | QRS MAE | ST/T MAE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.385 | 1.000 | 2.408 | 0.509 | -0.011 | 0.300 | 0.122 | 0.552 | 0.766 | 0.768 | 0.974 | 19.3 | 1.48 | 1.000 | 1.943 | 0.490 |
| 2 | 0.021 | 1.079 | 2.409 | 0.582 | -0.167 | 0.168 | 0.072 | 0.453 | 0.900 | 0.926 | 0.858 | 48.4 | 4.35 | 1.000 | 2.002 | 0.462 |
| 3 | 0.030 | 1.123 | 2.528 | 0.665 | -0.273 | 0.309 | 0.298 | 0.294 | 0.734 | 0.884 | 0.689 | 34.3 | 3.28 | 0.999 | 1.914 | 0.551 |
| 4 | 0.427 | 0.912 | 2.477 | 0.466 | 0.140 | 0.408 | 0.119 | 0.739 | 0.861 | 0.875 | 0.910 | 49.7 | 3.63 | 1.000 | 1.820 | 0.413 |
| 5 | 0.600 | 0.805 | 1.360 | 0.511 | 0.345 | 0.513 | **1.702** | 0.907 | 0.948 | 0.952 | 0.949 | 19.5 | 1.78 | 1.000 | 1.163 | 0.646 |
| 6 | 0.393 | 0.926 | 1.945 | 0.544 | 0.133 | 0.249 | 0.144 | 0.659 | 0.780 | 0.800 | 0.820 | 81.5 | 4.98 | 1.000 | 1.684 | 0.696 |
| 7 | 0.215 | 0.995 | 2.216 | 0.559 | 0.009 | 0.032 | 0.010 | **0.003** | **0.005** | **0.007** | 0.410 | **1962.9** | **33.04** | **0.004** | 1.612 | 0.744 |
| 8 | 0.256 | 0.970 | 2.034 | 0.559 | 0.043 | 0.160 | 0.038 | 0.409 | 0.510 | 0.668 | 0.583 | 116.0 | 7.74 | 1.000 | 1.681 | 0.642 |
| 9 | 0.504 | 0.863 | 1.613 | 0.530 | 0.246 | 0.294 | 0.095 | 0.726 | 0.910 | 0.916 | 0.923 | 30.1 | 2.07 | 1.000 | 1.345 | 0.788 |
| 10 | 0.217 | 1.075 | 2.347 | 0.616 | -0.191 | 0.509 | 0.246 | 0.729 | 0.925 | 0.943 | 0.941 | 18.5 | 1.74 | 1.000 | 1.950 | 0.617 |
| 11 | 0.304 | 1.016 | 2.320 | 0.611 | -0.072 | 0.489 | 0.130 | 0.612 | 0.911 | 0.938 | 0.916 | 27.8 | 2.88 | 1.000 | 1.711 | 0.667 |

Source: `experiments_mmecg/q1_remnet_continuous_strict_loso_seed42/test_fold_metrics.csv`.

## 4. Reporting rules fixed by this audit

### R1 — RR/HR coverage gating (binding)

Measured `rr_coverage`: ten folds at 0.999-1.000, fold 7 at **0.0039** (4 of
1020 windows yielded >=2 waveform-detected R peaks).

- **Rule:** only folds with `rr_coverage >= 0.80` enter RR/HR aggregation.
  Excluded folds must be listed individually and counted as failure cases.
- **Threshold sensitivity:** thresholds 0.5, 0.8 and 0.9 all produce the same
  10/11 partition, so the conclusion does not depend on the cut point. The rule
  was written after seeing the coverage distribution; this is disclosed here and
  the insensitivity is the reason it is acceptable.
- **Gated values (10 folds):** RR **44.5 ± 30.1 ms**, HR **3.39 ± 1.83 bpm**.
- **Retired values:** the ungated 218.9 ± 552.3 ms and 6.09 ± 8.70 bpm must not
  appear in any table; `test_fold_macro_summary.json` retains them because it is
  a frozen artifact, and any table drawing from it must apply R1 first.

### R2 — High-frequency ratio reported as median (binding)

`above_10hz_energy_ratio` is a ratio of ratios and is heavy-tailed: fold 5 is
1.702 while the other ten folds fall in 0.010-0.298.

- **Rule:** report **median 0.122, IQR [0.083, 0.195]**, together with the
  per-fold values and the maximum (1.702, fold 5).
- The mean ± std form (0.270 ± 0.460) must not appear on its own.

### R3 — Unit, aggregation and std labels (binding)

Every continuous-benchmark table states: z-score units for RMSE/MAE/region MAE;
the aggregation style per metric (§2.2); "population std across 11 folds";
and the waveform-vs-event-head distinction for F1.

## 5. Headline numbers under these rules

| Readout | Fold-macro (11 folds unless noted) |
|---|---|
| Continuous PCC@0 | 0.3048 ± 0.1733 (median 0.3043; min 0.021, fold 2) |
| R² | 0.0184 ± 0.1809 (median 0.0087; **negative in 5/11 folds**: 1, 2, 3, 10, 11) |
| RMSE | 0.979 ± 0.092 z-units |
| NRMSE | 2.151 ± 0.360 |
| Amplitude variance ratio | 0.3119 ± 0.1496 |
| >10 Hz energy ratio | median 0.122, IQR [0.083, 0.195] (R2) |
| Waveform F1@50/100/150 ms | 0.553 ± 0.240 / 0.750 ± 0.265 / 0.789 ± 0.261 |
| Event-head F1@50/100/150 ms | 0.614 ± 0.200 / 0.816 ± 0.172 / 0.854 ± 0.147 |
| RR / HR error | 44.5 ± 30.1 ms / 3.39 ± 1.83 bpm (10 folds, R1) |
| QRS-region MAE | 1.711 ± 0.251 z-units |
| ST/T-region MAE | 0.611 ± 0.113 z-units |

### Permitted interpretation

- R² is **centred near zero across folds**, not reliably positive: the mean is
  0.018 with population std 0.181 and five negative folds. Do not write that
  amplitude collapse is resolved.
- The amplitude variance ratio of 0.31 shows the prediction retains roughly a
  third of the ground-truth variance, i.e. **substantial residual regression
  toward the mean**. This is the mechanism evidence for the toned-down wording.
- Fold 7 is a **waveform-level R-peak detection failure** (F1@50 = 0.003,
  coverage 0.4%) even though its event head still reaches F1@100 = 0.410 and its
  PCC (0.215) is not the worst. It is a per-subject failure case, not an
  aggregate to average away.
- Seed 42 alone. No headline claim before the multi-seed confirmation required by
  `docs/METRIC_CONTRACT.md`, and no baseline exists yet under this protocol.

## 6. Open items

1. Multi-seed confirmation (seeds 123 / 2024) under identical rules.
2. Same-protocol continuous baselines on `loso_v2` (radarODE-MTL, AirECG,
   WSST-TCN); the old `loso` baseline aggregates remain BLOCKED.
3. Fold 7 diagnosis: whether the waveform readout failure is subject-specific
   morphology, amplitude collapse (its variance ratio 0.032 is the lowest), or a
   detector threshold artifact. Diagnostic only; it must not motivate retuning
   this frozen run.
