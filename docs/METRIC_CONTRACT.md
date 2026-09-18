# Metric Contract

This document defines how metrics must be interpreted and reported.

## General Rules

- Metric names in manuscript tables must map to code functions and manifest
  fields.
- Report whether metrics are computed per segment, per subject, per fold, or
  after long-signal concatenation.
- Report whether aggregation is macro, micro/pooled, subject-macro, or
  fold-macro.
- NaN handling must be explicit.
- Standard deviation and confidence intervals must state their unit of
  resampling.

## Waveform Metrics

| Metric | Required Definition |
|---|---|
| PCC | State whether computed per window then averaged, per subject, per fold, or on assembled long ECG |
| RMSE | State normalization space: raw, z-score, min-max, or normalized ECG |
| MAE | Same normalization rule as RMSE |
| R2 | State sample unit and aggregation method |

## Event Metrics

| Metric | Required Definition |
|---|---|
| R-F1 | Must state tolerance window in ms and sampling rate |
| RR error | Must state unit, matching rule, and aggregation |
| QRS/QT/QTc error | Must state delineation method and missing-value handling |
| QMR | Must state quality threshold and denominator |

## Checkpoint Selection

Checkpoint selection must use validation-only criteria. Test metrics must never
choose checkpoints.

## Current Limitation

Existing `compute_waveform_metrics()` style code often computes per-sample or
per-window metrics before higher-level aggregation. This does not automatically
equal subject-level, fold-level, or manuscript-level aggregation. Formal
experiments must include a manifest that records the aggregation contract.

## Locked Aggregation (Q1 tables)

- Primary aggregation is **fold-macro**: compute per-fold metrics, then report
  **mean ± std across folds**. State the resampling unit explicitly.
- Key conclusions (proposed model headline numbers, ablation) require **≥3 seeds**;
  report mean ± std across seeds.
- Checkpoint selection is **validation-best only**; test metrics never select
  checkpoints or models.
- NaN handling: metrics use `nanmean`; report the count of NaN/failed segments.

## RQ1 Multi-Metric Requirement

RQ1 conclusions must not be based on PCC alone. The minimum reportable metric set is:

- **Morphology:** PCC, RMSE, MAE, R², and where available region-level fidelity
  around QRS/ST-T/non-QRS morphology.
- **Event/timing:** R-F1 at 150/100/50 ms, RR-interval error, HR error, and
  Q/QRS/S/T fiducial timing only when delineation coverage is audited.
- **Monitoring reliability:** QMR, matched-beat coverage, and failure rate.
- **Engineering:** parameters and latency; FLOPs if the profiler is stable.

The earlier BeatAware evaluation used PCC/RMSE plus QMR, R-F1, and fiducial timing;
RQ1 keeps that principle but explicitly separates Shape-200 morphology from
full-window continuous ECG.

## Ablation Requirement

No manuscript sentence may claim that a component, representation, or loss term is
responsible for an improvement unless a controlled ablation supports it. A valid
ablation must keep data split, preprocessing, training budget, checkpoint rule, and
evaluation script fixed, changing only the declared independent variable. The
current gate is defined by
`experiments/specs/q1_continuous_ecg_loss_ablation.json` and the master plan.

## Continuous ECG and timing-tolerance metrics

- **Deployment endpoint — strict blind full-window:** compute per-window
  PCC@0/RMSE/MAE/R² on the native `1 x 1600` output, then aggregate by the
  registered fold/subject contract. No ECG-derived shift is allowed.
- **Timing diagnostic — Max-Lag NCC:** after the locked filter/standardization
  pipeline, scan only the registered ±20/±50/±70/±100 ms tolerances on fixed
  common support. Report the selected-lag distribution and boundary-hit rate.
  Since GT ECG selects the lag, these values are upper-bound diagnostics, not
  deployable reconstruction scores.

The former long-OLA/Route C assembly is withdrawn from the active metric
contract. Do not invoke its archived evaluator as a required endpoint.

## Continuous benchmark reporting rules

Binding from 2026-09-18. Derived from
`docs/audits/Q1_REMNET_CONTINUOUS_11FOLD_METRIC_AUDIT.md`, which audited
`q1_remnet_continuous_strict_loso_seed42`. These rules govern reporting; they do
not alter the frozen evaluator.

### Definitions previously missing from this contract

| Metric | Definition |
|---|---|
| `nrmse` | window RMSE divided by the **IQR of that window's ground truth**, then averaged over windows. Not range- or mean-normalized. |
| `amplitude_variance_ratio` | `var(pred)/var(gt)` per window, averaged. Below 1 means amplitude compression / regression toward the mean. |
| `above_10hz_energy_ratio` | (predicted >10 Hz energy fraction) / (ground-truth >10 Hz energy fraction), per window, averaged. A ratio of ratios, therefore heavy-tailed. |
| `rr_coverage` | fraction of windows in which both the prediction and the ground truth yield >=2 detected R peaks, i.e. the denominator behind RR/HR. |
| `f1_*ms` vs `event_f1_*ms` | `f1_*ms` uses peaks detected **on the reconstructed waveform**; `event_f1_*ms` uses peaks from the **event head**. They are different readouts and must never be merged, renamed into one another, or reported without saying which is which. |

### Units

The stored MMECG ECG is z-scored per record. `rmse`, `mae`, `qrs_region_mae` and
`stt_region_mae` are therefore in standard-deviation units of the subject's own
ECG, **not millivolts**. Every table must say so.

### Aggregation labels

Continuous-benchmark metrics do not share one aggregation style: waveform and
smoothing metrics are window-macro, all F1 values are pooled across windows
(micro), and region MAE is pooled across beats. Each table entry states its
style, and the across-fold line states `population std across N folds` whenever
the frozen summary artifact is the source.

### Coverage gating for rate metrics

RR and HR errors are aggregated only over folds with `rr_coverage >= 0.80`.
Excluded folds are listed individually and reported as failure cases, never
silently dropped and never averaged in. A gated aggregate must state how many
folds it covers.

### Heavy-tailed ratio metrics

`above_10hz_energy_ratio` is reported as median with IQR, plus per-fold values
and the maximum. Its mean ± std may not appear alone.

### Composite disclosure

When a validation composite contains clipped terms that are inactive in the
observed operating range, the manuscript methods section must say which terms
were effectively inactive. A frozen composite is never edited after a run to
make the description tidier.

## Shape-200 vs Continuous

Shape-200 (beat-normalized, amplitude-normalized morphology) is an **easier**
target than continuous ECG and must be reported as a separate track. A large
Shape-200↔continuous gap is itself a reported quantity (motivates the alignment
thesis), not a defect to hide.

## Closed delay-warp route

The learned `δ_sys`/`δ(t)` warp was tested and did not close the continuous gap.
It is not an active metric family or a paper contribution. Historical values may
appear only when explicitly labeled as negative evidence.
