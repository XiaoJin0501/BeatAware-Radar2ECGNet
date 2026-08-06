# Current Project State

Last updated: 2026-08-07
Main commit: e3d368b
Active branch: q1/radarode-long-ecg-evaluator
Active GitHub issue: none

## Current Research Objective

Serve RQ1: develop and validate a new radar-to-ECG reconstruction model
(`REM-AlignNet-MTL` unless renamed) that performs strongly on both MMECG and
CR_RVS and compares favorably against published/reproduced baselines.

Immediate technical objective: build a paper-first benchmark matrix for MMECG
and CR_RVS. Compare each published/reproduced method under its closest fair
output formulation instead of forcing every method into full-window `1 x 1600`
ECG.

## Verified Evidence

- MMECG Shape-200 results: paper-relevant pilot evidence; final promotion
  requires protocol/aggregation lock and literature source labeling.
- CR_RVS Shape-200 results: paper-relevant pilot evidence; final promotion
  requires protocol/aggregation lock and literature source labeling.
- MMECG fold-01 long-assembly pilot exists:
  - REM-AlignNet-MTL full-window OLA: record PCC 0.394, coverage 1.00.
  - REM-AlignNet-MTL shape oracle-cycle OLA: record PCC 0.937, coverage 0.44.
  - radarODE-MTL shape oracle-cycle OLA: record PCC 0.871, coverage 0.44.
  - radarODE-MTL predicted Anchor/PPI OLA: record PCC 0.151, coverage 0.37.
  - radarODE-MTL center Anchor/PPI OLA: record PCC 0.113, coverage 0.44.
- MMECG fold-01 radarODE Anchor/PPI audit exists:
  - Anchor median nearest-R error: 42.5 ms.
  - PPI median length error: 20.0 ms.
- MMECG fold-01 REM refine pilot exists:
  - REM-AlignNet-MTL-Refine full-window OLA: record PCC 0.420, coverage 1.00.
  - REM-AlignNet-MTL-Refine shape-head PCC: 0.938.
  - Improvement over REM full-window OLA is only +0.026 PCC.
- Current registry explicitly marks all retained results as `publishable=false`.

## Current Blockers

- Long-ECG evaluator is implemented for pilot use, but not yet locked for
  final manuscript tables.
- radarODE predicted Anchor/PPI assembly variants are currently failed
  diagnostics. They should not be used as paper-comparison results unless the
  original long ECG assembly details can be reproduced more faithfully.
- REMAlignMTLRefineNet is weak-positive for full-window output but does not
  affect the main paper-first benchmark strategy unless it improves a
  paper-relevant table.
- Metric aggregation contract is not fully locked for manuscript tables.
- Literature comparison ledger is not yet complete: each cited method needs
  dataset, split, output target, metrics, and comparability status.
- No machine-validated experiment spec/manifest workflow yet.
- Current Shape-200 evidence is strong but not yet enough by itself to support
  the full Q1 paper claim.

## Next Three Approved Tasks

1. Create the literature comparison ledger for MMECG and CR_RVS papers.
2. Lock the MMECG and CR_RVS morphology/fiducial tables as the primary Q1
   evidence track.
3. Keep full-window `1 x 1600` as a secondary continuous-monitoring table, not
   the single success criterion.

## Do Not Do

- Do not launch additional architecture probes unless they target a
  paper-relevant comparison table.
- Do not promote pilot results.
- Do not describe Shape-200 results as long-term continuous ECG evidence.
- Do not start final training without an experiment spec and preflight.
- Do not expand the project into unrelated RQ2/RQ3-style side quests.
