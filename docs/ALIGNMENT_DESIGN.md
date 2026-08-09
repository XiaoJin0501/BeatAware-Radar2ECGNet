# Electromechanical Alignment Module — Design

This document specifies the design of the alignment module that is the central
methodological contribution of the Q1 paper. It is a design contract; concrete
code lives in `src/models/remalign/rem_align_net.py`.

## 1. Motivation (central thesis)

Radar measures cardiac **mechanical** wall motion; ECG records **electrical**
activity. Between them there is an **electromechanical delay** `δ(t)`
(physiologically related to the pre-ejection period, PEP) that **varies with
physiological state** (heart rate, contractility, respiration / Valsalva /
apnea). Existing reconstructors implicitly assume fixed alignment, so they can
reconstruct beat-normalized morphology (Shape-200 PCC is high) but not
time-aligned continuous ECG (full-window PCC is low).

This module explicitly models and **self-emergently learns** `δ(t)` to close the
morphology↔continuous gap. The learned delay is designed to be **interpretable
and verifiable** (see §6), turning a module into a scientific finding.

## 2. Formulation

```
δ(t) = δ_sys(t) + r(t)
```

- `δ_sys(t)` — **systematic, interpretable, low-DoF** delay. A smooth spline
  with `K ≈ 5` knots (≈ one per 1.6 s over the 8 s window), whose knot values are
  predicted from a global latent context and may be modulated by instantaneous
  heart rate `HR(t)`. Defined in **milliseconds**. Carries the PEP story.
- `r(t)` — **small constrained residual** for local jitter:
  `r(t) = ε · tanh(conv([h, event]))`, with TV + sparsity regularization,
  `|r| ≤ ε`.

Application: `δ` (ms) is converted to a latent-step offset and applied as a
smooth warp of the temporal latent `h` via `grid_sample` (ms semantics for
interpretability, latent warp for training stability). The module returns
`δ_sys` and `r` **separately** so `δ_sys` is a reportable scientific quantity.

## 3. Inputs (inference-safe)

- `HR(t)`: derived from the **predicted** R-mask of the event head (e.g.
  sliding-window autocorrelation of the R-mask → differentiable rate proxy).
  No ground-truth peaks at inference.
- Global latent context `ctx = pool(h)` for the spline knots.
- Physiological **state is NOT an input**; state-dependence of `δ_sys` is shown
  post-hoc (stronger evidence for C3).

## 4. Identifiability & honesty

Because the decoder can also shift signals, `δ` risks being cancelled/absorbed
and losing meaning. Three constraints keep `δ_sys` honest:

1. **Anchor-to-zero** regularization (reuse lag-aware `zero_pcc` / `λ_zero`):
   `δ_sys` deviates from 0 only when it earns waveform PCC.
2. **Low degrees of freedom** of `δ_sys` is itself a strong prior.
3. **Residual budget `ε`** keeps `r` small, forcing systematic alignment onto
   the interpretable `δ_sys`.

Supervision policy: **`δ` is fully self-emergent** (no delay supervision). The
pseudo-ground-truth delay (§6) is used **for validation only**, never in
training or inference — this yields the strongest narrative ("the model
discovers a PEP-like delay without being told") and avoids any leakage debate.

## 5. Division of labor with the lag-aware loss

- **Alignment module** removes the *systematic* time-varying delay; `δ_sys` is
  the product.
- **Lag-aware loss** (`src/losses/losses.py::LagAwareWaveformLoss`) supervises
  the waveform tolerant to *residual* small lag and provides the anchor
  regularization. Once alignment is on, the lag window is shrunk to the `ε`
  scale so the two mechanisms do not do the same job.

Ablation grid (must show separability):

| Variant | δ_sys | r | lag-aware | Expectation |
|---|---|---|---|---|
| B0 | – | – | – | continuous weak (floor; WSST-TCN) |
| B1 | ✓ | – | – | continuous ↑, δ interpretable |
| B2 | – | – | ✓ | continuous ↑ but no interpretable δ |
| B3 | ✓ | ✓ | – | better |
| B4 | ✓ | ✓ | ✓ | best |

Success = B1 and B2 each help independently and B4 is best. The minimal
closed-loop (see `docs/Q1_EXPERIMENT_MATRIX.md`) runs B0/B1/B3/B4 on fold_01.

## 6. Interpretability / validation hooks (C3, analysis-only)

- **Pseudo-GT delay**: per beat, cross-correlation lag between a radar
  mechanical-beat envelope and the GT R-peak. Used **only** to validate that
  learned `δ_sys` correlates with a measured delay.
- **δ_sys vs HR**: PEP shortens with heart rate.
- **δ_sys across states** (CR_RVS resting / Valsalva / apnea): autonomic
  perturbation → delay drift. Strongest mechanism evidence.

## 7. Default hyper-parameters (tune at implementation, not hard constraints)

- `K ≈ 5` spline knots; `δ_sys` range ≈ `±150 ms` (covers PEP + modality offset).
- residual budget `ε ≈ 20–40 ms`; TV + L1 sparsity on `r`.
- once alignment on, lag window ≈ `ε` scale.
- HR proxy: differentiable sliding-window rate from the predicted R-mask.

## 8. Component toggles (for the ablation)

The module exposes independent switches so a single backbone realizes B0–B4:

- `use_delta_sys` (δ_sys on/off), `use_delta_residual` (r on/off),
  `use_lag_aware_loss` (already in config).

This keeps the ablation a controlled single-variable progression on one
architecture rather than swapping models.
