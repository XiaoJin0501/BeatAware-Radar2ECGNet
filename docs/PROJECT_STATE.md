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
  - radarODE-MTL center Anchor/PPI OLA: record PCC 0.113, coverage 0.44.
- MMECG fold-01 radarODE Anchor/PPI audit exists:
  - Anchor median nearest-R error: 42.5 ms.
  - PPI median length error: 20.0 ms.
- Current registry explicitly marks all retained results as `publishable=false`.

## Current Blockers

- Long-ECG evaluator is implemented for pilot use, but not yet locked for
  final manuscript tables.
- radarODE predicted Anchor/PPI assembly variants are currently failed
  diagnostics. They should not be used as paper-comparison results unless the
  original long ECG assembly details can be reproduced more faithfully.
- Metric aggregation contract is not fully locked for manuscript tables.
- No machine-validated experiment spec/manifest workflow yet.
- Current Shape-200 evidence is strong but not yet enough by itself to support
  the full Q1 paper claim.

## Next Three Approved Tasks

1. Refocus model work on improving REM-AlignNet-MTL full-window continuous
   decoding/assembly, because the morphology head is strong but full-window
   OLA is weak.
2. Keep radarODE comparison scoped to Shape-200/oracle-cycle morphology unless
   a faithful official long ECG assembly implementation becomes available.
3. After a REM continuous-decoder fix passes fold-01 pilot, run all 11 MMECG
   folds and then mirror the accepted protocol on CR_RVS.

## Do Not Do

- Do not launch additional Shape-200 architecture probes.
- Do not promote pilot results.
- Do not describe Shape-200 results as long-term continuous ECG evidence.
- Do not start final training without an experiment spec and preflight.
- Do not expand the project into unrelated RQ2/RQ3-style side quests.
