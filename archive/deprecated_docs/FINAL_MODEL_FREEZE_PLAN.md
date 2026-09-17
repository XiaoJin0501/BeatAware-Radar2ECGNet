# Final Model Freeze Plan

This document defines the controlled path from the current REM-Net evidence to
a frozen supervised RQ1 model. It is intentionally narrow: no SSL, no new
research question, and no test-driven hyperparameter search.

## Objective

Freeze one supervised REM-Net configuration for formal MMECG and CR_RVS
experiments.

The target model is **REM-Net**: representation encoder + event auxiliary head
+ morphology multi-task decoder. The event-conditioned alignment branch is
disabled unless a pre-registered ablation reverses the current G5 result.

## Current Incumbent

- Architecture: `rem_align_mtl` with `use_alignment=false`.
- MMECG: 8 s WSST, `narrow_bandpass=true`, `target_norm=minmax`.
- CR_RVS: CW-compatible single-channel representation, `narrow_bandpass=false`,
  `target_norm=minmax`.
- Loss: `use_shape200_loss=true`, `lambda_shape200=0.8`,
  `shape200_pcc_weight=0.5`, `beta_peak=0.3`.
- Multi-seed evidence: MMECG Shape-200 PCC `0.885 +/- 0.002`; CR_RVS
  Shape-200 PCC `0.886 +/- 0.001`.

## Freeze Logic

1. **F0 controlled tuning** uses development folds only.
   - MMECG dev folds: 1, 5, 9.
   - CR_RVS dev folds: 1, 3, 5.
   - Primary selection signal: validation/checkpoint morphology behavior.
   - Test-fold shape-head summaries are diagnostic only during F0.
2. **F1 candidate confirmation** reruns the top candidate(s) with multiple seeds
   on the same dev folds.
3. **F2 model freeze** records exactly one configuration, commit, and dataset
   recipe.
4. **F3 formal final runs** train/evaluate all folds on both datasets with the
   frozen configuration and pre-declared seeds. These runs produce the final
   paper numbers and figures.

No test metric from F3 may change the frozen model.

## F0 Search Space

The search is deliberately small and supervised-only.

| Candidate | beta_peak | lambda_shape200 | shape200_pcc_weight | baseline_hidden | Purpose |
|---|---:|---:|---:|---:|---|
| incumbent | 0.3 | 0.8 | 0.5 | 128 | Current best recipe |
| shape12 | 0.3 | 1.2 | 0.5 | 128 | Stronger morphology loss |
| pcc08 | 0.3 | 0.8 | 0.8 | 128 | Stronger correlation term |
| peak01 | 0.1 | 0.8 | 0.5 | 128 | Reduce event supervision pressure |
| peak06 | 0.6 | 0.8 | 0.5 | 128 | Increase event supervision pressure |
| hidden192 | 0.3 | 0.8 | 0.5 | 192 | Capacity check |

Fixed controls:

- `model_arch=rem_align_mtl`
- `use_alignment=false`
- `use_shape200_loss=true`
- `target_norm=minmax`
- `wsst_suffix=_wsst.h5`
- `lr=1e-4`
- `weight_decay=1e-4`
- `epochs=35` for F0 only
- `early_stop_patience=100`
- `seed=42` for F0

## Go / No-Go

Promote a candidate to F1 only if it beats or matches the incumbent on dev-fold
mean PCC without worsening hard-fold behavior. A gain smaller than about 0.003
is treated as noise unless it consistently improves both datasets or improves
region/fiducial diagnostics.

If no candidate clears this bar, freeze the incumbent and move to F3.

## Reporting Rules

- F0/F1 are `pilot`, `publishable=false`.
- F3 is the only candidate for `final`, `publishable=true`.
- All rows must appear in `experiments_mmecg/EXPERIMENT_REGISTRY.csv`.
- All formal claims must map back to `docs/PAPER_CLAIM_LEDGER.md`.

