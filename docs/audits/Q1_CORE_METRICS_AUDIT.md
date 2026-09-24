# Q1 Core Metrics Audit (radarODE-MTL / mmJEPA-ECG alignment)

Date: 2026-09-25. Evaluator: `scripts/evaluate_core_metrics.py` (read-only over frozen
test predictions; nothing trained or selected; no experiment directory created).
Outputs: `docs/audits/Q1_CORE_METRICS_fold.csv`, `docs/audits/Q1_CORE_METRICS_scenario.csv`.
Manuscript: Table I (`tab:benchmark`), Fig. 3 (`fig:robustness`), Fig. 4 (`fig:bottleneck`).

## Definitions (from the papers, not paraphrased from memory)

| Metric | Definition | Source in the paper |
|---|---|---|
| Shape PCC / Shape R^2 | single centre cycle cut at reference R peaks, 200 samples, min-max normalised | radarODE-MTL Sec. IV-B: "RMSE, PCC and R^2 for the recovered single-cycle ECG pieces" |
| Long-term R^2 | full 8-s window, stored units (frozen `test_metrics.json`) | radarODE-MTL Fig. 7, long-term recovery |
| MDR | reference R peaks without a predicted anchor within +/-150 ms, greedy one-to-one | radarODE-MTL: deviations < 150 ms are not "missed detected" |
| Timing Error | mean absolute offset of matched anchors (ms) | radarODE-MTL "absolute Timing Error"; mmJEPA "event-based timing errors" |
| Cycle Length Error | abs(mean predicted RR - mean reference RR) per window, folds with coverage >= 0.80 | radarODE-MTL "absolute PPI Error"; mmJEPA "interval-based R-R" |

## Verification against frozen results

1. Shape PCC reproduces `q1_tier1_readout_from_continuous_seed42/tier1_fold_metrics.csv`: max |diff| = 0.
2. Template floor reproduces `tier1_summary.json`: 0.8806 = 0.8806.
3. Waveform-derived cycle length reproduces the frozen `rr_error_ms`: max |diff| = 0.000 ms over 198 rows.
4. Fig. 4(d) jitter sweep at sigma = 0 reproduces the frozen oracle-timing bound R^2 = 0.730; the
   template at REM-Net's anchor head reproduces the frozen T1 value R^2 = -0.711.

## Findings that bind the manuscript

1. **REM-Net leads on two metrics only**: long-term R^2 (0.041; vs radarODE-MTL p = 0.032, 9/11)
   and anchor-head timing error (38.8 ms; vs radarODE-MTL p = 0.042, 9/11). Neither is significant
   against Chen-TCN.
2. **radarODE-MTL leads on MDR (11.2 %) and cycle length (45.5 ms).**
3. **The anchor head's cycle length is 115.5 ms**, against 49.0 ms from REM-Net's waveform peaks.
   The head is precise (precision 0.957) but conservative (0.84x the reference anchor count); a
   missed beat merges two RR intervals and inflates the window mean. The frozen Introduction's
   sentence "Cycle length is therefore estimated from the event stream" is therefore true as a
   design statement and must be reported with this number. The earlier "lowest in six of seven
   scenarios" result is waveform-derived.
4. **Shape R^2 is negative for every trained system** (-1.4 to -2.0 on MMECG) against a template
   floor of 0.532; no system reaches the floor on shape PCC either.
5. **R^2 crosses zero at an injected jitter of ~30 ms**; the best trained system's R^2 corresponds
   to an equivalent jitter of ~28 ms, while REM-Net's anchor-head offset spread is 52 ms (pooled).
6. **At +/-50 ms every system misses ~40 % of beats, and ~70 % at +/-20 ms**; no architecture
   separates from the others at clinical timing precision.

## Convention note

Table I follows the METRIC_CONTRACT headline convention (fold-macro per seed, mean +/- std over
seeds): REM-Net waveform cycle length 49.0 ms. `Q1_MULTISEED_BENCHMARK_AUDIT.md` reports 48.8 ms
under seed-averaged-then-fold-macro aggregation. Both are correct for their convention; the
manuscript uses 49.0 ms in Table I.

Paired tests in Table I are uncorrected (20 tests against REM-Net's anchor head).
