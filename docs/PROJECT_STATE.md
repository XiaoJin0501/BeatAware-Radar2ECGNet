# Current Project State

Last updated: 2026-08-07
Main commit: e3d368b
Active branch: q1/agent-research-governance
Active GitHub issue: none

## Current Research Objective

Serve RQ1: develop and validate a new radar-to-ECG reconstruction model
(`REM-AlignNet-MTL` unless renamed) that performs strongly on both MMECG and
CR_RVS and compares favorably against published/reproduced baselines.

Immediate technical objective: define a fair evaluation route against
radarODE-MTL, including long-term ECG assembly if the manuscript compares
against radarODE-MTL's final long-ECG results.

## Verified Evidence

- MMECG Shape-200 results: pilot only.
- CR_RVS Shape-200 results: pilot only.
- Current registry explicitly marks all retained results as `publishable=false`.

## Current Blockers

- No unified long-ECG assembly evaluator.
- No formal predicted-anchor protocol.
- Metric aggregation contract is not fully locked for manuscript tables.
- No machine-validated experiment spec/manifest workflow yet.
- Current Shape-200 evidence is strong but not yet enough by itself to support
  the full Q1 paper claim.

## Next Three Approved Tasks

1. Lock the exact final evaluation protocols for the Q1 paper: Shape-200,
   8-second continuous ECG, long-term assembled ECG, or a combination.
2. Implement the long-assembly evaluator if required for fair radarODE-MTL
   comparison.
3. Run synthetic validation and one-fold MMECG validation before full runs.

## Do Not Do

- Do not launch additional Shape-200 architecture probes.
- Do not promote pilot results.
- Do not describe Shape-200 results as long-term continuous ECG evidence.
- Do not start final training without an experiment spec and preflight.
- Do not expand the project into unrelated RQ2/RQ3-style side quests.
