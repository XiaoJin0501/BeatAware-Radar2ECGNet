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
evaluation script fixed, changing only the declared independent variable. See
`docs/RQ1_ABLATION_AND_METRIC_PLAN.md`.

## Continuous ECG Metric (dual)

The continuous target is reported with two complementary definitions:

- **Primary — full-window**: per-window PCC/RMSE/MAE/R² on the `1 x 1600` output
  (`compute_waveform_metrics_protocol`), then fold-macro averaged.
- **Secondary — within-record long-OLA**: overlap-add assembly within each record
  (`scripts/evaluate_mmecg_long_assembly.py --protocol rem_full_ola`), reporting
  PCC and coverage.

Report both; do not present long-OLA as the single success criterion, and do not
claim faithful radarODE-style long-term assembly (see PAPER_CLAIM_LEDGER F2).

## Shape-200 vs Continuous

Shape-200 (beat-normalized, amplitude-normalized morphology) is an **easier**
target than continuous ECG and must be reported as a separate track. A large
Shape-200↔continuous gap is itself a reported quantity (motivates the alignment
thesis), not a defect to hide.

## Delay Analysis Metrics (C3, analysis-only)

For the learned electromechanical delay `δ_sys` (see `docs/ALIGNMENT_DESIGN.md`):

- **δ_sys vs pseudo-GT lag**: correlation between learned `δ_sys` and the
  radar-mechanical-envelope↔GT-R-peak cross-correlation lag. Pseudo-GT is
  **validation-only**, never used in training or inference.
- **δ_sys vs HR**: trend (expect PEP to shorten with heart rate).
- **δ_sys across states**: distribution shift across CR_RVS resting/Valsalva/apnea.

These are diagnostic/interpretability metrics, not primary reconstruction scores.
