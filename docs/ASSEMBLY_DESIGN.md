# Structured Rhythm-Localized Assembly — Design (Route C)

This is the design contract for the pivoted approach after the alignment/delay
thesis was falsified on MMECG fold_01 (see `docs/ALIGNMENT_DESIGN.md` §7b).
Concrete code lives in `src/models/remalign/`, `src/losses/`, and the assembly
evaluator `scripts/evaluate_mmecg_long_assembly.py`.

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
