# Continuous ECG Task Contract (RQ1)

Status: **binding** from 2026-09-09. Supersedes any Shape-200-centred task definition.

## 1. Primary objective

Reconstruct the **complete, continuous ECG waveform** from radar under strict
subject-independent evaluation, on MMECG (77 GHz FMCW) and CR_RVS (24 GHz CW).

Shape-200 is **auxiliary morphology analysis only**. It may not define the primary
result, drive checkpoint selection, or carry the paper's headline claim.

## 2. Task definition

- **Input:** a fixed-length radar window.
- **Output:** the ECG over the *same* time span at the *same* sampling rate
  (MMECG: 1600 samples at 200 Hz for an 8 s window).
- **Primary ground truth:** raw ECG, or raw ECG under a **global robust scaler whose
  statistics are fitted on training folds only**.

## 3. Prohibited

- Per-window or per-cycle min-max normalization as the **primary** GT. This is the
  defect that produced retracted claim D1: it discards cycle duration, true amplitude,
  baseline variation, and inter-beat continuity, and it makes correlation
  structurally blind to amplitude error.
- Checkpoint selection on Shape-200 PCC.
- Any hyperparameter, threshold, or architectural choice informed by test data.
- Hard peak-assembly or per-cycle template copying as the mechanism producing the
  primary continuous output.

## 4. Checkpoint selection

A **full-waveform validation composite**, fixed in writing before training and never
adjusted after seeing test data. Components:

- waveform PCC (against the primary GT of §2)
- NRMSE and MAE
- R^2
- R-peak F1 or QRS-region error

The exact combination rule must be written into the experiment spec before the run.

## 5. Protocol

- Strict LOSO on both datasets; no subject in more than one test fold.
- All hyperparameters determined on train/validation only.
- Final test executed **once**, after model and evaluator are frozen.

## 6. Reporting

Both datasets, every arm: PCC; RMSE, NRMSE, MAE; R^2; R-peak F1; RR and HR error;
QRS and ST/T region error; per-fold mean ± std; paired significance; parameter count
and inference time; and full waveform plots at quantitatively selected
high/median/hard operating points.

Results from different datasets, different splits, or reported-only literature go in
**separate context blocks**, never in one ranking column.

## 7. Known input-representation constraint (measured 2026-09-09)

The WSST tensor the current model consumes is `(50, 71, 120)` for an 8 s window:
**120 time bins, i.e. 15 Hz effective, against a 200 Hz target.**

- one WSST time bin = **66.7 ms**
- a QRS complex (~80 ms) is **1.2 time bins** wide
- the waveform path therefore performs a **13.3x** temporal upsampling from an input
  that cannot resolve the structure it is asked to produce

This is an information ceiling, not a training failure, and it bounds any sub-50 ms
timing claim *a priori*. The raw `rcg` tensor `(50, 1600)` at native 200 Hz is present
in the same H5 files and is **unused by the waveform path**. Any new continuous
architecture must address this before module-level tuning is meaningful.
