# Anchor-Conditioned Phase Resampling — Stage 0

2026-10-27. Author-supplied staged task. **STOP after Stage0**; no training,
validation model comparison or test-result access was performed in this stage.
Stage1's six proposed trainings are not launched by this implementation.

## Corrected architecture boundary

`R–R Phase Alignment` returns ONLY native sample coordinates `t_i [B,200]`.
`Differentiable Phase Resampling` separately accepts `(H_full, P_det, t_i)` and
produces `H_beat [B,128,200]`, `P_beat [B,4,200]`. Concatenation occurs AFTER
resampling, at the input of the existing two-convolution morphology decoder.

```text
H[128,400] ── original event head ── logits[4,1600]
     │                                  ├─ ordered decoder → soft refinement → sg(R pair)
     │                                  └─ sigmoid → sg(P)
     │                                                        │
     │       sg(R pair) → Phase Alignment → t_i ────────────────┤
     └─ upsample1600, align_corners=True → Phase Resampling ────┤
                                                H_beat, P_beat
                                                       │
                                     concatenate → shape decoder → shape[1,200]
```

Neither `t_i` nor detached event probabilities receives shape-loss gradients.
Shared encoder still receives gradients from the morphology branch. The event
head still receives its own unchanged BCE gradients. Original event-path
upsampling remains `align_corners=False`; only the new shape-path full-grid
upsampling uses `True`. `phase_pd`'s additional window-coordinate features use
the original `400→200, align_corners=False` interpolation.

### Exclusive right endpoint — verified, not a changed target

Existing `shape_target()` takes `cycle=ECG[a:b]`, then resamples with
`linspace(0,len(cycle)-1,200)`. Therefore:

`t_i = a + (i/199)*(b-a-1)`.

`b` remains the next R boundary, exclusive in slicing. Naively using
`a+(i/199)*(b-a)` includes an additional sample and does NOT invert the existing
target. No historical target/evaluator has been changed.

Sources reused exactly via AST compilation in `frozen_bridge.py`:

* `/tmp/rem_dual_head_formal/src/models/continuous/ce_net.py`: stem and TCN.
* `src/models/continuous/dual_head_rem.py`: original heads and interp model.
* `src/data/mmecg_dataset.py`: central-cycle bounds.
* `scripts/run_dual_head_interaction.py`: target, ordered decoder, reference
  delineation wrapper, loss.
* `src/utils/clinical_endpoints.py`: reference delineation.

File hashes are in `stage0/results.json`. No complete training module is
imported or run; no continuous-model predictions are loaded.

## T1 — all validation windows of two folds/dataset

User zero-based fold0/1 means physical `fold_01/fold_02`; fold files untouched.
Float32 ECG/grid sampling, CPU; no selection/skipping of windows.

| Dataset | Physical fold | Windows | Failures | Minimum PCC | Maximum absolute difference |
|---|---:|---:|---:|---:|---:|
| MMECG | 1 | 170 | 0 | 0.999999762 | 2.29478e-5 |
| MMECG | 2 | 1,020 | 0 | 0.999999762 | 2.22325e-5 |
| CR_RVS | 1 | 195 | 0 | 0.999999762 | 2.72989e-5 |
| CR_RVS | 2 | 194 | 0 | 0.999999881 | 3.59416e-5 |

All **1,579 / 1,579** pass fixed PCC≥0.999 and max-abs<1e-3 thresholds.
Every window's bounds/PCC/error/pass is in `stage0/identity_windows.csv`.

## T2/T3 — nine unit tests pass

* Reference-grid identity in float32 and float64.
* Grid/anchor and detached probability gradient blocking.
* `phase`, `phase_p`, `phase_pd` shape/device/dtype checks and actual stem/TCN
  gradients; event head has no morphology-only gradient and DOES receive BCE.
* `interp` strict state-dict compatibility and bitwise equality to incumbent
  outputs (random initialized controls, no checkpoint/test result reading).
* Matched/valid-unmatched/missing-anchor fallback; same predicted anchors under
  different GT status labels; out-of-window fallback uses border sampling and
  remains finite/scored, not discarded.
* Schedule 0/15/30/45 epoch boundaries; reference fallback only while training;
  eval outputs invariant to supplied reference status labels; training oracle
  explicitly rejected.

CPU float32/float64 verified; GPU and mixed precision are not claimed tested.

| Variant | Decoder input channels | Total parameters | Shape-head parameters |
|---|---:|---:|---:|
| interp | 128 | 1,438,789 | 41,409 |
| phase | 128 | 1,438,789 | 41,409 |
| phase_p | 132 | 1,440,069 | 42,689 |
| phase_pd | 260 | 1,481,029 | 83,649 |

Totals above use the MMECG 50-channel input. Shared backbone/event head are
unchanged. CR_RVS retains the original one-channel input adapter.

## Existing recipe checked for the next stage (not run)

* AdamW lr1e-4, weight decay1e-4; no LR scheduler; batch16; maximum100 epochs;
  patience15; dropout0.1; no input augmentation in these original runners.
* Same shape loss L1+0.5(1−PCC), plus0.3 BCE; Gaussian sigma4samples=20ms.
* Original seed implementation is `42+physical_fold`, not literal42 for every
  fold. Preserve it when matching the seed42 incumbent runs.
* Original MMECG REM-Net `drop_last=False`; CR_RVS `drop_last=True`. Do not
  replace these with a supposedly uniform recipe.
* Both selected incumbent runners use validation harmonic mean of arithmetic
  Shape-PCC and independent-event macro F1@100ms, then Shape-PCC and −RMSE.
  Source: MMECG `run_dual_head_balanced_full_loso.py` sets
  `CHECKPOINT_RULE='shape_event_hmean'`; CR_RVS inherits baseline `rank()`.
* Requested subject Fisher-z reporting must remain separate from this original
  checkpoint selection; do not silently change the incumbent selector.

## Implementation caveats kept explicit

The requested prediction rule is nearest-centre R **and the next R**. Existing
reference target instead prefers the R–R pair **containing** the centre. Those
can be different beats when the nearest R is after the centre. The implementation
preserves BOTH specified rules; it does not use GT to repair predicted selection.
This is a possible cause of later valid-unmatched status, to be measured at
Stage1, not corrected after looking at test results.

Fallback may sample beyond the 1600-point window. This uses explicit border
padding and records `outside_grid_fraction`; no GT replacement at eval, no
clamping RR, no exclusion of fallback windows. Training uses reference anchors
only as prescribed by the schedule/invalid-pair fallback.

The input remains one representative cycle per window, NOT a continuous ECG
output. Nothing here makes a clinical-boundary claim or changes old results.

## Reproduction / artifacts

`python -B experiments/phase_resampling_20261027/stage0.py`

`python -B -m pytest -q tests/test_anchor_conditioned_phase.py`

`stage0/results.json`, `stage0/identity_windows.csv`, `stage0/unit_tests.txt`.
Configuration is `spec.json`. Stage2 remains confirmation-locked.
