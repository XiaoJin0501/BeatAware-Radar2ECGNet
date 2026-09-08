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

The WSST tensor the current model consumes is `(50, 71, 120)`, and it spans only the
**centre 4 s** of the 8 s window (`crop = center_4s_samples_400_1200`).

- 120 bins over 4.0 s = **30 Hz effective**, one bin = **33.3 ms**
- a ~80 ms QRS spans **~2.4 bins** — a **severe temporal-resolution bottleneck**
  relative to a 200 Hz target, not a demonstration that QRS recovery is impossible
- **50 % of every output window has no radar input support**: the ECG target covers
  1600 samples while the radar covers samples 400-1200 only

The support asymmetry is the more serious of the two and is the first-order suspect for
the continuous-reconstruction failure. See `docs/RCG_PROVENANCE_AUDIT.md`.
