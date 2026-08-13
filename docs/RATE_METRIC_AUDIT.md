# Rate / Beat-Localization Metric Audit (frozen REM-Net, eval-only)

Purpose: lock the R-peak F1 / RR / HR numbers computed on the **frozen** seed-42
final-test REM-Net (no retraining), state exactly which are promotable, and tie every
number to a registered run directory. These metrics **support the localization-precision
discovery** (contribution C4/D1); they are not a headline strength claim.

## Provenance (traceable)

- Script: `scripts/evaluate_remnet_rate_metrics.py` (inference-only; loads
  `checkpoints/best.pt`, runs the model, reads the **event head R-mask** `peak_preds[0]`).
- Source runs (frozen, commit `ebb726a`):
  - MMECG: `experiments_mmecg/q1_final_remnet_incumbent_mmecg_seed42/fold_{01..11}`
  - CR_RVS: `experiments_mmecg/q1_final_remnet_incumbent_crrvs_seed42/fold_{01..05}`
- Per-fold outputs written to `<run>/fold_XX/results_rate/rate_metrics.json`.
- Ground truth: `test.h5[rpeak_indices]` (sample space, fs=200 Hz, 1600-sample window).
- Peak extraction: `peaks_from_heatmap` (rel. thr `max(0.3, 0.5·ymax)`, min_dist 40=200 ms);
  matching: greedy one-to-one within tolerance (`match_counts`). Event-head R-mask is at
  the native 1600-sample resolution, so predicted peaks and GT `rpeak_indices` share space.
- Both configs supervise the R-mask identically (`beta_peak=0.3`), so cross-dataset
  differences reflect the model, not a missing training target.

## Numbers (fold-macro, mean ± std)

### MMECG (FMCW, 11-fold strict LOSO) — PROMOTABLE

| Metric | Value |
|---|---|
| R-peak F1 @ 50 ms  | 0.544 ± 0.153 |
| R-peak F1 @ 100 ms | 0.749 ± 0.127 |
| R-peak F1 @ 150 ms | 0.806 ± 0.083 |
| RR-interval error (matched beats) | 30.4 ± 5.7 ms |
| HR error (matched beats) | 0.32 ± 0.06 bpm |

RR/HR coverage = **92.8%** of test windows have ≥2 matched beats → representative.
Per-fold F1@150 range 0.637 (fold 08) – 0.952 (fold 01).

**Key discovery signal:** F1 degrades monotonically and sharply as tolerance tightens
(0.806 → 0.749 → **0.544** across 150 → 100 → 50 ms). The event head finds *where a beat
is* coarsely, but fine (<50 ms) anchor precision is where it breaks — exactly the
bottleneck that blocks continuous assembly. HR is near-perfect (0.32 bpm) because
zero-mean RR jitter averages out over the window; this is a rate metric, not a
localization-precision metric, and must not be presented as evidence of precise anchoring.

### CR_RVS (CW, 5-fold subject-independent) — LOCALIZATION FAILURE (promotable as such)

| Metric | Value | Promotable? |
|---|---|---|
| R-peak F1 @ 50 ms  | 0.084 ± 0.017 | yes (failure evidence) |
| R-peak F1 @ 100 ms | 0.099 ± 0.018 | yes |
| R-peak F1 @ 150 ms | 0.102 ± 0.019 | yes |
| RR-interval error | 45.5 ± 7.4 ms | **NO — biased** |
| HR error | 0.05 ± 0.01 bpm | **NO — biased** |

RR/HR coverage = **4.2%** of windows only (236 / 5605). These are computed on the rare
"easy" windows where ≥2 beats happened to match; they are **not representative** and must
NOT be reported as CR_RVS rate performance.

**Interpretation (strongest form of the discovery):** on CW radar the event head localizes
essentially one dominant beat per window and misses the rest (F1@150 ≈ 0.10), yet the same
frozen model reconstructs beat **morphology** at Shape-200 PCC **0.881** — as well as on
FMCW. Morphology transfers across radar modality; **beat-anchor localization does not**, and
is far worse on CW. This dissociation (high morphology, failed localization) is the central
evidence for C4/D1: the barrier to continuous ECG is cross-subject anchor localization
precision, not morphology.

## Promotion rules

- **Promote:** MMECG R-F1@{50,100,150} + RR/HR (coverage stated); CR_RVS R-F1@{50,100,150}
  **as localization-failure evidence** with morphology contrast.
- **Do NOT promote:** CR_RVS RR/HR (4.2% biased subset). If mentioned at all, state
  explicitly they are conditional on the <5% of windows with matched beats and therefore
  uninformative.
- These are event-head (predicted-anchor) numbers; keep consistent with the route-C
  oracle-vs-predicted anchor evidence (oracle OLA 0.93–0.98 vs predicted-anchor collapse).
- No retraining; frozen seed-42 checkpoints only.
