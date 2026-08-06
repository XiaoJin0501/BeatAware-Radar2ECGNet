# ADR-0002: Long-Term ECG Assembly Protocol

Date: 2026-08-07

## Status

Pilot implemented; final protocol not locked.

## Context

radarODE-MTL reports final ECG reconstruction through a pipeline that combines
shape prediction, anchor/beat placement, and interval structure. A fair Q1
comparison must evaluate REM-AlignNet-MTL and radarODE-MTL under the same
long-term assembly logic.

## Decision

Implement a radarODE-style long-term ECG assembly evaluator before promoting
any continuous ECG claim to final.

The first MMECG fold-01 pilot supports three explicitly separated evaluation
families:

- full-window overlap-add for models that directly output `1 x 1600`;
- oracle cycle placement for morphology heads, used only as a diagnostic upper
  bound;
- predicted Anchor/PPI placement for radarODE-style deployable assembly, still
  under source/paper audit.

## Required Variants

- oracle-anchor upper bound;
- predicted-anchor final inference;
- failure and confidence reporting.

## Current Pilot Evidence

- REM-AlignNet-MTL full-window OLA: record PCC 0.394, coverage 1.00.
- REM-AlignNet-MTL shape oracle-cycle OLA: record PCC 0.937, coverage 0.44.
- radarODE-MTL shape oracle-cycle OLA: record PCC 0.871, coverage 0.44.
- radarODE-MTL predicted Anchor/PPI OLA: record PCC 0.151, coverage 0.37.
- radarODE-MTL center Anchor/PPI OLA: record PCC 0.113, coverage 0.44.

Anchor/PPI audit on the same fold found median Anchor nearest-R error 42.5 ms
and median PPI length error 20.0 ms. Despite reasonable median subtask errors,
the predicted assembly variants remain poor, likely because the public source
does not expose the official long ECG assembly logic and the adapted Anchor
target is a multi-peak mask while ECG_shape is a center-cycle target.

These values are pilot evidence only and must not be promoted to final tables.
Predicted Anchor/PPI variants are failed diagnostics under the current adapted
protocol.

## Open Questions

- Whether an official or more faithful radarODE long ECG assembly can be
  recovered from paper details beyond the released source.
- Whether manuscript PCC is computed on assembled trials, subjects, folds, or
  pooled segments.
