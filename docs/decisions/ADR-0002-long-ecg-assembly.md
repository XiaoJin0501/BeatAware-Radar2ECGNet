# ADR-0002: Long-Term ECG Assembly Protocol

Date: 2026-08-07

## Status

Proposed.

## Context

radarODE-MTL reports final ECG reconstruction through a pipeline that combines
shape prediction, anchor/beat placement, and interval structure. A fair Q1
comparison must evaluate REM-AlignNet-MTL and radarODE-MTL under the same
long-term assembly logic.

## Decision

Implement a radarODE-style long-term ECG assembly evaluator before promoting
any continuous ECG claim to final.

## Required Variants

- oracle-anchor upper bound;
- predicted-anchor final inference;
- failure and confidence reporting.

## Open Questions

- How to handle missing anchors?
- How to merge overlapping windows?
- Whether manuscript PCC is computed on assembled trials, subjects, folds, or
  pooled segments.
