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
