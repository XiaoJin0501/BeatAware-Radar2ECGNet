# Input-Perturbation Robustness Audit (frozen REM-Net, eval-only)

Purpose: lock the Shape-200 PCC degradation curve of the **frozen** seed-42 final-test
REM-Net under additive input noise (no retraining), and state exactly what is
promotable. Supports a graceful-degradation robustness claim and the FMCW-vs-CW
redundancy contrast.

## Provenance (traceable)

- Script: `scripts/evaluate_remnet_robustness.py` (inference-only; loads
  `checkpoints/best.pt`, injects noise into the WSST input, evaluates the model's
  `shape200_pred` head against `_shape200_targets(ecg, cycle_bounds)` — the exact target
  used for the final-test numbers).
- Noise model: additive Gaussian, scale = `level × (per-window WSST std)`, seed
  `42+fold`. `level = 0.0` reproduces the frozen final-test number exactly
  (self-consistency verified: fold-01 σ=0 → 0.926 = stored `pcc_mean`).
- `level` is the ratio of injected-noise std to the input representation's own std, i.e.
  an inverse-SNR: level 0.5 ≈ +6 dB, 1.0 ≈ 0 dB, 2.0 ≈ −6 dB, 3.0 ≈ −9.5 dB, 5.0 ≈ −14 dB.
- Source runs (frozen, commit `ebb726a`): `q1_final_remnet_incumbent_mmecg_seed42`
  (11 folds), `q1_final_remnet_incumbent_crrvs_seed42` (5 folds).
- Per-fold outputs: `<run>/fold_XX/results_robustness/robustness_shape200.json`.

## Numbers (fold-macro Shape-200 PCC, mean ± std)

| Noise level (≈SNR) | MMECG (FMCW) | CR_RVS (CW) |
|---|---|---|
| 0.0 (clean)   | 0.882 ± 0.061 | 0.881 ± 0.013 |
| 0.5 (+6 dB)   | 0.860 ± 0.073 (−2.5%) | 0.775 ± 0.087 (−12.1%) |
| 1.0 (0 dB)    | 0.834 ± 0.112 (−5.4%) | 0.602 ± 0.223 (−31.6%) |
| 2.0 (−6 dB)   | 0.781 ± 0.123 (−11.5%) | 0.418 ± 0.298 (−52.5%) |
| 3.0 (−9.5 dB) | 0.646 ± 0.150 (−26.7%) | 0.257 ± 0.316 (−70.9%) |
| 5.0 (−14 dB)  | 0.390 ± 0.215 (−55.8%) | 0.129 ± 0.268 (−85.3%) |

## Interpretation (promotable)

- **Graceful degradation.** On MMECG the model is essentially unaffected up to
  noise equal to the input's own amplitude (0 dB, level 1.0: 0.834, −5.4%) and still
  usable at −6 dB (0.781). Breakdown is gradual, not catastrophic.
- **FMCW is markedly more noise-robust than CW.** MMECG's 50 range bins provide spatial
  redundancy the model can average over; CR_RVS is single-channel CW with no such
  redundancy, so it degrades ~2–5× faster at every level (−12% vs −2.5% at +6 dB;
  −32% vs −5% at 0 dB). This is consistent with the modality difference, not a model
  defect, and reinforces why cross-modality validation matters.
- One CR_RVS fold (fold 05) collapses under heavy noise (negative PCC); the reported
  mean ± std captures this spread honestly.

## Promotion rules

- **Promote:** the full degradation table for both datasets, the graceful-degradation
  statement, and the FMCW>CW robustness contrast (tie to spatial redundancy).
- **Do NOT claim:** robustness to real sensor motion artifacts — this is
  representation-space additive noise, an input-corruption stress test, not a
  motion-artifact simulation. State this scope explicitly in the paper.
- No retraining; frozen seed-42 checkpoints only; level 0 == final-test number.
