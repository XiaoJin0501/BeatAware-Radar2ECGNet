# Anchor-Guided Morphology Assembly — Design (Route C)

Status: **DEMOTED to discovery / mechanism / future-direction (2026-08-10).** It is
NOT the Q1 main model. 3-fold LOSO validation showed anchor detection recall
generalizes across subjects but anchor **precision does not** (fold_02/03 MAE
40–55 ms vs fold_01 16–20 ms; continuous OLA ~0 on 2/3 folds even after recall is
restored via threshold tuning). The Q1 main line reverts to the robust
morphology + external-validation track (see `docs/Q1_EXPERIMENT_MATRIX.md`,
`docs/PAPER_CLAIM_LEDGER.md` M1–M4); route C contributes the discovery D1–D2.

### Route-C evidence chain (fold_01 → 3-fold, all committed)
- Oracle-cycle OLA 0.93–0.98, morphology ≈0.96 → gap is localization, not shape (D1).

> **RETRACTED 2026-09-09 (audit C):** this oracle-cycle OLA figure is not raw
> continuous-ECG evidence — scored against per-cycle min-max normalized
> `_shape200_targets`, not the original ECG, at ~44% coverage; reproduces the
> Shape-200 score to within 0.002. See `docs/PAPER_CLAIM_LEDGER.md` (D1 / D1').

- fold_01 PoC: fusion (WSST detect + rcg timing) F1@50 0.82, MAE 16 ms, OLA 0.51.
- 3-fold: mean F1@50 0.56, MAE 38 ms, OLA 0.17; folds 02/03 collapse (OLA ~0.02).
- Diagnosis: recall collapse was largely a threshold artifact (model fires at ~88%
  of beats), BUT threshold-tuned re-eval did NOT recover OLA → **precision
  generalization** is the true bottleneck (D2). fold_01 was an easy subject
  (HR 66, all normal-breathing, high confidence).

The design below is retained as the mechanism/future-direction blueprint.

## 0. Paper contributions (restructured)

- **Discovery.** The radar-to-ECG continuous gap comes mainly from **beat-anchor
  localization failure**, not from ECG morphology-modeling failure. Evidenced by a
  strong oracle upper bound: morphology high (~0.96), oracle-anchor assembly high
  (0.93–0.96), predicted-anchor assembly low (0.23–0.37).
- **Model.** An **anchor-guided morphology assembly framework**: predict beat
  anchors → generate/retrieve/reconstruct beat morphology → assemble ECG on the
  continuous time axis (differentiably or semi-differentiably).
- **Evidence.** Quantify the upper-bound gap with **predicted-anchor vs
  oracle-anchor** assembly; show continuous metrics rise as anchor quality
  improves. This is more convincing than a delay-field story because the oracle
  evidence is so strong.

## 1. Finding that motivates this route

fold_01 diagnostic (`q1_align_ceval_pred_anchor_fold01`):

| assembly | continuous PCC |
|---|---|
| dense full-window (floor) | 0.38–0.40 |
| predicted-anchor cycle | **0.23–0.37** |
| oracle cycle (morphology into GT beat positions) | **0.93–0.96** |

Morphology is essentially solved (shape-head PCC ≈0.96). With correct beat
positions, continuous ECG is near-perfect. Predicted anchors are worse than the
dense floor. **The bottleneck is beat localization, not morphology and not a
time-varying electromechanical delay.**

## 2. Revised thesis

> Radar-to-ECG continuous fidelity is gated by accurate **beat localization**
> (rhythm/anchor prediction), not by beat morphology. We reconstruct continuous
> ECG by (1) a strong beat-localization head predicting precise anchor positions
> and local rhythm from radar, and (2) assembling the learned morphology at those
> positions. The electromechanical delay is demoted to a small per-beat
> correction / interpretability side-analysis.

## 3. Components

1. **Beat-localization (anchor) head** — the new core.
   - Output: a sharp per-sample anchor likelihood over the window (1600 or a
     downsampled grid), plus extraction to discrete anchor positions.
   - Training target: narrow Gaussian heatmap around GT R-peaks (σ ≈ 20 ms).
   - Loss: focal / weighted BCE (or MSE) on the heatmap + optional soft-argmax
     position regression for sub-sample precision. Dedicated weight `lambda_anchor`
     (the current broad R-mask uses `beta_peak=0.3` and is NOT optimized for
     localization — that is why predicted anchors are poor).
2. **Local rhythm / PPI** — start parameter-free (cycle length = interval between
   consecutive predicted anchors); optionally predict per-anchor PPI later.
3. **Morphology head** — reuse the existing shape head (~0.96). Start with a
   per-window template; per-beat morphology is a later ablation.
4. **Assembly** — place the morphology at each predicted anchor, cycle length from
   local inter-anchor interval, overlap-add. Extend the evaluator from center-cycle
   only (~44% coverage) to **all beats in the window** (coverage → ~1.0).

## 3b. Implementation staging (PRINCIPLE: anchor localization first, standalone)

Do NOT mix the anchor head with the waveform decoder yet. Build and prove beat
localization as an **independent task** first:

```
Radar / WSST  ->  Anchor Heatmap Head  ->  beat anchor positions
```

**Stage 1 (this minimal loop) — standalone anchor localization.** No waveform
decoder, no shape head. Gate metrics:

- Anchor MAE (ms) — median predicted-anchor to nearest GT R.
- Anchor F1 @ 50 / 100 / 150 ms.
- RR error (ms) from predicted anchors.
- Predicted-anchor OLA PCC and Oracle-anchor OLA PCC (assembly uses a given/
  reused morphology so the ONLY变量 is anchor quality).

**Stage-1 goal**: not to beat radarODE. Prove that as anchor F1 rises from ~0.3
toward 0.7–0.8, continuous (predicted-anchor OLA) PCC rises substantially toward
the oracle ceiling. If anchor F1 cannot improve, stacking a waveform decoder is
pointless — stop and rethink the localization signal/representation.

**Stage 2 (only if Stage 1 passes)**: integrate morphology (learned shape head)
and the assembly into an end-to-end / semi-differentiable model.

## 4. Key design decisions

- **Localization representation**: narrow-Gaussian heatmap + peak extraction, with
  soft-argmax refinement for ms precision. (vs pure per-sample classification.)
- **Target σ**: ≈20 ms (narrow → precise; not so narrow it fails to train).
- **Loss weight**: `lambda_anchor` high enough to actually optimize positions;
  this is a new, separate term from the QRS BCE `beta_peak`.
- **Coverage**: assemble every predicted cycle, not just the center one, so the
  continuous metric reflects the whole record.
- Keep the dense full-window decoder as a parallel output / fallback and for the
  floor comparison.

## 5. Metrics

- **Primary**: continuous long-OLA PCC via predicted-anchor assembly
  (`rem_pred_anchor_cycle_ola`, extended to all beats). Target: climb from
  0.23–0.37 toward the oracle ceiling 0.9+.
- **Localization**: anchor error (ms, median nearest-GT-R) and detection
  F1@50/100/150 ms.
- **Morphology**: shape-head PCC (must stay ≈0.95).
- Always report oracle-cycle as the labeled upper bound.

## 6. Ablation (route-C)

| Variant | Change | Expectation |
|---|---|---|
| L0 | current broad R-mask → predicted-anchor assembly | 0.23–0.37 (baseline) |
| L1 | + dedicated localization head (heatmap + focal) | anchor error ↓, assembly PCC ↑ |
| L2 | + soft-argmax sub-sample refinement | anchor error ↓ further |
| L3 | + all-beat assembly (coverage ↑) | continuous PCC ↑, coverage ~1.0 |
| L4 (opt) | + per-beat morphology | morphology robustness |

## 7. Go/no-go (fold_01, before scaling)

Predicted-anchor assembly PCC must climb **clearly above the dense floor (~0.40)**
toward oracle, and anchor error must drop (target median < ~30–40 ms). If the
localization head cannot beat the floor, reconsider the route.

## 8. Retained interpretability (from the old thesis)

The residual electromechanical delay (predicted anchor vs GT R offset) can still be
reported as a small per-beat analysis — a side-figure, not the main claim.

## 9. Files to touch (implementation)

- `src/models/remalign/rem_align_net.py`: add the localization head (new/extended model).
- `src/losses/losses.py`: heatmap/focal localization loss.
- `scripts/train_mmecg.py`: build the Gaussian R-heatmap target + wire `lambda_anchor`.
- `scripts/evaluate_mmecg_long_assembly.py`: extend `rem_pred_anchor_cycle_ola` to
  all-beat assembly + report anchor-error metric.
