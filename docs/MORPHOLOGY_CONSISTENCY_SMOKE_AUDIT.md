# Dual-dataset consistency engineering audit

Date: 2026-09-08. Tier: smoke, publishable=false.
Implementation commit: 5f379f5d6e725af8cfb1121ade4575d87c8692bb.
No held-out validation or test samples were opened by the smoke script.

## Checks and correction

The legacy cycle helper substitutes [700,900] when annotations do not define
a real R-to-R cycle. The new consistency objective previously could not identify
that fallback. Added a cycle_valid flag to the WSST loader and required it for
consistency training. Historical target construction remains unchanged for the
baseline. Seven tests pass, including fallback exclusion, amplitude sensitivity,
detached-head gradients and actual REM-Net forwards/backwards for 1/50 channels.

## Protocol

Fold 01 training data only, first sample from each of eight sorted training
subjects per dataset. All arms have identical initialization and fixed batch,
seed 42, AdamW 1e-4, weight decay 1e-4, gradient clipping at 1.0. Existing full
waveform/event loss and Shape-200 loss remain enabled. B0 has no consistency;
B1 uses detached head at weight 0.05; B2 uses direct reference at weight 0.05.
Measure in eval mode before and after a predeclared step count. No checkpoint
selection. One run used 80 updates; a separately registered follow-up used 800.
These are optimization checks, not converged population-level comparisons.

## Results at 800 updates on the repeated training batch

| Dataset | Arm | Waveform PCC | Waveform RMSE | Waveform MAE | Waveform R2 |
|---|---|---:|---:|---:|---:|
| MMECG | B0 | 0.93569 | 0.04973 | 0.01207 | 0.86547 |
| MMECG | B1 | 0.93814 | 0.04856 | 0.01224 | 0.87210 |
| MMECG | B2 | 0.93633 | 0.04906 | 0.01050 | 0.86887 |
| CR_RVS | B0 | 0.76645 | 0.07602 | 0.02089 | 0.50346 |
| CR_RVS | B1 | 0.79519 | 0.07117 | 0.02042 | 0.56198 |
| CR_RVS | B2 | 0.75795 | 0.08034 | 0.02214 | 0.44889 |

All six arms have finite losses/gradients and decreasing total loss. All selected
cycles have valid annotation coverage (8/8 per dataset). The parameter counts
are unchanged across arms: MMECG 1,788,452; CR_RVS 1,681,828. These differ because
input channel counts differ. This does not measure inference latency.

At 80 updates, Shape-200 PCC was already approximately 0.99 while amplitude
errors were high. At 800 updates, MMECG B0 full-waveform PCC reaches 0.936 and
its shape RMSE falls to 0.0282. Early poor fitting alone was not sufficient
evidence to replace the backbone. CR_RVS remains harder to fit in this check.

B1 does not dominate every metric: its MMECG MAE is slightly worse than B0,
and B2 has the lowest MMECG MAE. No superiority or causal generalization claim
can be drawn from repeated training batches or one random seed.

## Artifacts

- experiments_mmecg/q1_morphology_consistency_smoke_20260908/report.json
- experiments_mmecg/q1_morphology_consistency_overfit_smoke_20260908/report.json
- scripts/smoke_morphology_consistency.py
- Matching specifications in experiments/specs and rows in EXPERIMENT_REGISTRY.csv.

Reports record code hashes, source paths/stat metadata, selected indices and
subjects, every optimization step, before/after metrics and parameter counts.
The source metadata are not a cryptographic fingerprint of the full H5 files.

## Decision

Engineering checks pass; candidate efficacy is unresolved. Next perform a
proper train/validation comparison on both datasets with locked checkpoint
criteria and the B2 control, including waveform-derived timing and coverage.
Do not promote this smoke into a manuscript table. Mathematical scope and
counterexamples are documented in MORPHOLOGY_CONSISTENCY_THEORY.md.
