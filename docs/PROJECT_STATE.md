# Current Project State

Last updated: 2026-08-07
Main commit: e3d368b
Active branch: q1/radarode-long-ecg-evaluator
Active GitHub issue: none

## Current Research Objective

Serve RQ1: develop and validate a new radar-to-ECG reconstruction model
(`REM-AlignNet-MTL` unless renamed) that performs strongly on both MMECG and
CR_RVS and compares favorably against published/reproduced baselines.

Immediate technical objective: audit and refine the fair long-term ECG
assembly route against radarODE-MTL, especially the predicted Anchor/PPI
assembly needed for non-oracle comparison.

## Verified Evidence

- MMECG Shape-200 results: pilot only.
- CR_RVS Shape-200 results: pilot only.
- MMECG fold-01 long-assembly pilot exists:
  - REM-AlignNet-MTL full-window OLA: record PCC 0.394, coverage 1.00.
  - REM-AlignNet-MTL shape oracle-cycle OLA: record PCC 0.937, coverage 0.44.
  - radarODE-MTL shape oracle-cycle OLA: record PCC 0.871, coverage 0.44.
  - radarODE-MTL predicted Anchor/PPI OLA: record PCC 0.151, coverage 0.37.
- Current registry explicitly marks all retained results as `publishable=false`.

## Current Blockers

- Long-ECG evaluator is implemented for pilot use, but not yet locked for
  final manuscript tables.
- radarODE predicted Anchor/PPI assembly has a first approximation, but it
  must be audited against the original paper/source before full 11-fold runs.
- Metric aggregation contract is not fully locked for manuscript tables.
- No machine-validated experiment spec/manifest workflow yet.
- Current Shape-200 evidence is strong but not yet enough by itself to support
  the full Q1 paper claim.

## Next Three Approved Tasks

1. Audit radarODE-MTL Anchor/PPI coordinates and cycle placement against the
   original source/paper.
2. Refine or reject `radarode_pred_anchor_ppi_cycle_ola` as the fair
   non-oracle radarODE long-assembly protocol.
3. After protocol lock, run all 11 MMECG folds and then mirror the accepted
   protocol on CR_RVS.

## Do Not Do

- Do not launch additional Shape-200 architecture probes.
- Do not promote pilot results.
- Do not describe Shape-200 results as long-term continuous ECG evidence.
- Do not start final training without an experiment spec and preflight.
- Do not expand the project into unrelated RQ2/RQ3-style side quests.
