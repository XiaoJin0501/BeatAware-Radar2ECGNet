# ADR-0001: Shape-200 vs Continuous ECG Evidence

Date: 2026-08-07

## Status

Accepted.

## Context

Current REM-AlignNet-MTL results are strongest in Shape-200 beat morphology
space. These results are useful but are not equivalent to long-term continuous
ECG reconstruction.

## Decision

Shape-200 is treated as morphology evidence and auxiliary supervision only.
Final manuscript claims about continuous ECG reliability require a locked
continuous or long-term evaluation protocol. Shape-200 can still support the
Q1 model as morphology evidence if it is labeled correctly.

## Consequences

- Shape-200 tables can appear as morphology analysis or pilot evidence.
- Shape-200 results must not be labeled final long-term continuous ECG results.
- The next formal task is to lock the final evaluation protocol. radarODE-style
  long ECG assembly is required if comparing against radarODE-MTL's final
  long-term result.
