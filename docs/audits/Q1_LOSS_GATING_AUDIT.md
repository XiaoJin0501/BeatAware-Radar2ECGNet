# Q1 Continuous-ECG Loss Gate Audit

**Experiment:** `q1_continuous_ecg_loss_ablation`
**Completed:** 2026-09-17 18:55 NZST
**Scope:** repaired `loso_v2` validation folds 1/5/9; B1 architecture fixed;
80 epochs maximum with patience 20; paired effective seeds; validation-best
checkpoint selection. These are pilot/development results, not final-test or
publishable 11-fold evidence.

## Registered comparison

- **L0 (reused baseline):** Huber + 0.20 event BCE.
- **L1:** L0 + 0.30 × (1 − PCC).
- **L2:** L0 + 0.10 × first-derivative L1.
- **L3:** L0 + 0.50 × MR-STFT at FFT/window sizes 128/256/512.

All nine newly trained runs completed without OOM or an incomplete fold. The
matrix below transcribes each run's `best.json`; no run or metric was omitted.
PCC, NRMSE, R², region MAE, variance ratio, and spectral ratio are continuous
full-window waveform readouts. F1 is waveform-derived R-peak F1 at 100 ms.

## Nine-run result matrix

| Arm | Fold | Best epoch | Stopped | PCC ↑ | NRMSE ↓ | R² ↑ | F1@100 ↑ | QRS MAE ↓ | ST/T MAE ↓ | Var(pred)/Var(gt) →1 | >10 Hz ratio →1 | Composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L1 | 1 | 2 | 22 | -0.0054 | 2.3528 | -0.1123 | 0.8986 | 1.9647 | 0.4516 | 0.0839 | 0.0111 | 0.1326 |
| L1 | 5 | 4 | 24 | 0.4914 | 1.8367 | 0.2216 | 0.8029 | 1.5876 | 0.6262 | 0.3421 | 0.2257 | 0.3613 |
| L1 | 9 | 19 | 39 | 0.2245 | 2.4579 | -0.3406 | 0.9523 | 2.0103 | 0.6216 | 0.7328 | 0.4510 | 0.2326 |
| L2 | 1 | 4 | 24 | -0.0644 | 2.6086 | -0.3692 | 0.9206 | 2.0591 | 0.5228 | 0.2886 | 0.2072 | 0.1123 |
| L2 | 5 | 22 | 42 | 0.5211 | 1.8313 | 0.2146 | 0.8750 | 1.5861 | 0.5820 | 0.5995 | 0.9295 | 0.3826 |
| L2 | 9 | 19 | 39 | 0.2098 | 2.4614 | -0.3464 | 0.9519 | 2.0137 | 0.6213 | 0.7015 | 0.5107 | 0.2267 |
| L3 | 1 | 21 | 41 | 0.1175 | 2.3128 | -0.0737 | 0.8440 | 1.9570 | 0.3944 | 0.1475 | 0.0308 | 0.1736 |
| L3 | 5 | 25 | 45 | 0.5134 | 1.7843 | 0.2642 | 0.8331 | 1.5612 | 0.6047 | 0.3122 | 0.1942 | 0.3832 |
| L3 | 9 | 55 | 75 | 0.2494 | 2.2987 | -0.1543 | 0.9293 | 1.9093 | 0.5891 | 0.5227 | 0.3311 | 0.2392 |

## Fold-macro summary

| Arm | PCC ↑ | NRMSE ↓ | R² ↑ | F1@100 ↑ | QRS MAE ↓ | ST/T MAE ↓ | Variance ratio →1 | >10 Hz ratio →1 | Composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L0 reused | 0.2231 | 2.2719 | -0.1356 | **0.9027** | 1.8622 | 0.5720 | 0.5437 | 0.5033 | 0.2391 |
| L1 | 0.2368 | 2.2158 | -0.0771 | 0.8846 | 1.8542 | 0.5665 | 0.3863 | 0.2293 | 0.2422 |
| L2 | 0.2222 | 2.3005 | -0.1670 | **0.9159** | 1.8863 | 0.5753 | 0.5299 | 0.5491 | 0.2405 |
| L3 MR-STFT | **0.2935** | **2.1319** | **0.0121** | 0.8688 | **1.8092** | **0.5294** | 0.3275 | 0.1854 | **0.2653** |

The L0 smoothing ratios are the frozen-checkpoint diagnostic values registered
before the gate (folds 1/5/9: variance 0.521/0.290/0.820; >10-Hz energy
0.565/0.234/0.711), rather than fields retroactively added to the older
`best.json` files.

## Decision and trade-off

L3 is the clear **waveform-metric winner**. Relative to L0, its fold-macro PCC
rises from 0.2231 to 0.2935 (+0.0704), R² rises from -0.1356 to +0.0121, NRMSE
falls by 0.1399, and both region MAEs improve. R² improves on all three folds,
and QRS-region MAE improves on all three folds.

This gain is not a clean gate pass. F1@100 falls from 0.9027 to 0.8688 overall:
fold 1 falls 0.9210→0.8440 and fold 5 falls 0.8665→0.8331, while fold 9 rises
0.9207→0.9293. The direct smoothing diagnostics also do not move consistently
toward 1; their L3 fold-macro values are lower than L0. Therefore the evidence
supports “MR-STFT improves waveform reconstruction under the current joint
training objective,” but does **not** support “MR-STFT fixes smoothing” or “L3
improves waveform and event performance jointly.”

Under the original preregistered promotion rule, no arm passes because L3
degrades waveform-derived F1 and the smoothing ratios do not consistently move
toward 1. L4 remains unauthorized, and these pilot results are not expanded
directly into an 11-fold claim.

For the next controlled development stage, MR-STFT is locked as the waveform
core loss because it provides the strongest and most consistent waveform
signal. The next single variable is **branch-level gradient isolation**: test
whether preventing morphology-loss gradients from damaging the event branch can
retain L3's waveform gains while recovering event F1. This is a new controlled
gate, not a retrospective declaration that Step 2 passed its original rule.

## Traceability

- Specification: `experiments/specs/q1_continuous_ecg_loss_ablation.json`
- New-run artifacts: `experiments_mmecg/q1_continuous_ecg_loss_ablation/`
- Queue log: `experiments_mmecg/q1_continuous_ecg_loss_ablation/queue.log`
- Reused L0 artifacts:
  `archive/legacy_experiments/pilots_and_gates/q1_continuous_ecg_b1_lososv2_baseline/`
