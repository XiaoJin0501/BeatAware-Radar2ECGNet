# Q1 Journal Master Plan

This document is the single authoritative roadmap for the IEEE Q1 journal
paper. When an older roadmap, experiment note, or agent recap conflicts with
this file, this file governs. Scientific definitions remain subject to
`RESEARCH_CONTRACT.md`, `DATA_PROTOCOL.md`, and `METRIC_CONTRACT.md`.

## Paper objective and model identity

The sole objective is a rigorous Q1 paper on radar-to-ECG reconstruction. The
model is **REM-Net: Representation–Event–Morphology Network**, with explicit
event–morphology factorization:

- **Representation:** a shared radar representation of cardiac mechanical
  activity;
- **Event:** cardiac-event timing and cycle structure;
- **Morphology:** beat-local P–QRS–T morphology recovery under waveform and
  spectral constraints.

`REM-AlignNet` is a deprecated working name. Do not create a new model name or
change the core idea merely to accommodate an unfavorable experiment.

## Factorized three-tier benchmark

### Tier 1 — Beat-normalized morphology (Shape-200)

Purpose: measure the morphology recoverable when the cardiac cycle is known and
mapped to a normalized 200-sample grid. This is a morphology upper-bound task,
not continuous deployment.

Required metrics: beat-wise PCC, RMSE, and R², reported under an explicit
normalization and fold/subject aggregation contract.

### Tier 2 — Strict blind continuous stream

Purpose: measure real blind deployment under repaired strict LOSO, without ECG
priors at inference. The primary output is the full 8-second, 1600-sample ECG.

Required metrics: continuous PCC@0, R-F1 with stated tolerance, R-peak MAE, and
IBI MAE. Checkpoint selection is validation-only; test ECG cannot influence
normalization, alignment, or model selection.

### Tier 3 — Temporal-tolerance diagnostics

Purpose: quantify sensitivity of continuous reconstruction to residual timing.
Max-Lag NCC uses ground truth to choose the lag and is therefore a diagnostic
upper bound, never deployable performance.

Required readouts: Max-Lag NCC at ±20, ±50, ±70, and ±100 ms, accompanied by
the selected-lag distribution and boundary-hit rate.

## Frozen Step-1 evidence

On repaired `loso_v2` folds 1/5/9 with frozen B1 checkpoints:

| Readout | Fold-macro value |
|---|---:|
| Continuous PCC@0 | 0.223 |
| Max-Lag NCC @ ±20 ms | 0.366 |
| Max-Lag NCC @ ±50 ms | 0.563 |
| Max-Lag NCC @ ±70 ms | 0.624 |
| Max-Lag NCC @ ±100 ms | 0.635 |

The curve bends and begins to saturate around 50–70 ms. This time scale is
compatible with electromechanical-delay magnitudes discussed as EMAT, but the
experiment does **not** identify EMAT: lag is selected using ground-truth ECG
per window and may also contain model error, preprocessing delay, and
cross-subject generalization error. The defensible result is a large
timing-sensitive diagnostic ceiling.

Artifacts live in
`experiments_mmecg/q1_temporal_tolerance_curve_audit/` and are registered in
`EXPERIMENT_REGISTRY.csv`.

## Step 2 — loss gate completed

The `q1_loss_gate` completed all nine formal runs on 2026-09-17. It compared on
folds 1/5/9, with all other variables fixed:

- L1: Huber + event BCE + PCC loss;
- L2: Huber + event BCE + first-derivative L1;
- L3: Huber + event BCE + MR-STFT using windows/FFTs 128, 256, 512.

The fold-macro L0/L3 comparison is PCC 0.2231→0.2935, R² -0.1356→+0.0121,
NRMSE 2.2719→2.1319, QRS MAE 1.8622→1.8092, and ST/T MAE 0.5720→0.5294.
However, waveform F1@100 falls 0.9027→0.8688, and the registered smoothing
ratios do not consistently move toward 1. L3 is therefore the waveform winner
but not a clean pass of the original joint multi-metric promotion rule. L4 is
not authorized, and Step 2 is not being expanded directly to 11 folds.

The complete nine-run matrix and exact interpretation are frozen in
`docs/audits/Q1_LOSS_GATING_AUDIT.md`.

**Locked development decision:** MR-STFT is the waveform core loss for the
next controlled model experiment. This locks the best observed waveform
objective; it does not erase the measured F1 trade-off or convert the pilot
into manuscript evidence.

## Step 2.5 — shared-gradient audit completed

A read-only audit measured gradients of MR-STFT and event BCE on the parameters
actually shared by B1 (`wsst_stem + decoder.tcn`) across every validation batch.
Global cosine means were -0.137/+0.276/+0.088 on folds 1/5/9. Only fold 1 was
significantly negative; folds 5 and 9 were significantly positive, and no
predeclared module reached mean cosine ≤ -0.30 on any fold. The preregistered
conflict rule failed. PCGrad and branch-gradient isolation are therefore not
authorized from this hypothesis. Full evidence:
`docs/audits/Q1_SHARED_GRADIENT_CONFLICT_AUDIT.md`.

## Step 3 — native-RCG resolution gate completed and closed

The capacity-matched B2a-wide/B2b gate completed all six validation runs on
2026-09-17. B2b reduced PCC on 3/3 folds (fold-macro 0.1221 to 0.0863), made R2
more negative (-0.0577 to -0.0661), and worsened QRS-region MAE on 3/3 folds.
It substantially increased waveform F1@100 (0.2013 to 0.5487), but RR/HR error
worsened in the macro aggregate and the high-frequency ratio was unstable,
including a 2.383 overshoot on fold 5. Thus the waveform and mechanism gates
failed even though event safety passed.

Per the preregistered rule, the B2b native-RCG direct-fusion route is
**closed**. Do not tune another fusion variant or scale B2b to 11 folds. This
failure bounds the tested direct-fusion formulation; it does not establish
that native radar contains no useful information. The immutable six-run table
and decision are in `docs/audits/Q1_NATIVE_RCG_GATE_AUDIT.md`.

## Official continuous model and next benchmark

The official continuous candidate is now frozen as **WSST-driven REM-Net
(continuous B1) with the L3 MR-STFT waveform core**. “Official” means the
architecture, loss, normalization, checkpoint rule, and evaluation contract
are frozen for the next benchmark; it does not claim that the model has
already achieved publishable performance.

The next authorized experiment is
`q1_remnet_continuous_strict_loso_seed42`: all 11 repaired `loso_v2` folds,
validation-only checkpoint selection, followed by one-time held-out test
evaluation. Primary reporting is fold-macro across 11 test subjects and keeps
PCC, R2, NRMSE, regional error, smoothing diagnostics, and event/timing metrics
together. Shape-200 remains auxiliary and excluded from selection. Seed 42 is
a complete-fold final candidate; any headline model claim still requires the
multi-seed confirmation mandated by `METRIC_CONTRACT.md`.

**Current launch blocker:** the base `loso_v2` H5 files are complete, but the
full-window `_wsst8s.h5` companions currently exist only for train/validation
on folds 1/5/9; no test companion exists. The legacy per-fold duplicated-cache
layout would require roughly 50–60 GiB while only about 37 GiB is free. The
runner must fail closed until either a numerically verified deduplicated cache
is available or storage is increased. Four-second WSST is prohibited as a
fallback because it recreates the known 50%-unsupported-input confound.

## Non-negotiable reporting rules

- Never present Shape-200 as continuous ECG reconstruction.
- Never present Max-Lag NCC as achievable radar-only performance.
- Never mix reported literature numbers with matched reproductions without
  labeling the protocol difference.
- Never use test subjects to select a model, hyperparameter, affine map, lag,
  or checkpoint.
- Do not fabricate expected numbers or retrofit a story to a favorable result.
