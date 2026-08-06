# Current Project State

Last updated: 2026-08-07
Main commit: e3d368b
Active branch: q1/agent-research-governance
Active GitHub issue: none

## Current Research Objective

Implement and validate a radarODE-style long-term ECG assembly evaluator so
radarODE-MTL and REM-AlignNet-MTL can be compared under the same continuous ECG
protocol.

## Verified Evidence

- MMECG Shape-200 results: pilot only.
- CR_RVS Shape-200 results: pilot only.
- Current registry explicitly marks all retained results as `publishable=false`.

## Current Blockers

- No unified long-ECG assembly evaluator.
- No formal predicted-anchor protocol.
- Metric aggregation contract is not fully locked for manuscript tables.
- No machine-validated experiment spec/manifest workflow yet.

## Next Three Approved Tasks

1. Define the long-term ECG assembly protocol and manifest schema.
2. Implement the long-assembly evaluator.
3. Run synthetic validation and one-fold MMECG validation before full runs.

## Do Not Do

- Do not launch additional Shape-200 architecture probes.
- Do not promote pilot results.
- Do not describe Shape-200 results as long-term continuous ECG evidence.
- Do not start final training without an experiment spec and preflight.
