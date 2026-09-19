# Q1 Template-Timing Audit: the Continuous Gap Is Timing

Date: 2026-09-19. Experiment: `q1_template_timing_diagnostics_seed42`.
Spec (readings declared before the run):
`experiments/specs/q1_template_timing_diagnostics_seed42.json`.
Evaluator: `scripts/evaluate_template_timing.py`.
Outputs: `experiments_mmecg/q1_template_timing_diagnostics_seed42/`.

## Design

A single beat template per fold -- the mean R-to-R cycle of the **training** fold,
in the stored per-record z-scored ECG units -- is written into each test window at
a given set of R-peak times and scored with the frozen evaluator functions
(`waveform_metrics`, `smoothing_diagnostics`, `timing_metrics`, `region_errors`),
exactly as the models were. The template carries no radar information and no
test-subject morphology. Only the timing source changes:

| Variant | Timing source | Status |
|---|---|---|
| T0 | ground-truth R peaks | oracle upper bound, never achievable |
| T1 | REM-Net event-head peaks | radar-only |
| T2 | peaks detected on REM-Net's waveform | radar-only |
| T3 | peaks detected on Chen-TCN's waveform | radar-only |

## Results (MMECG `loso_v2`, 11 folds, fold-macro, population std)

| Variant | PCC | R² | NRMSE | F1@100 | QRS MAE (z) | Var. ratio |
|---|---:|---:|---:|---:|---:|---:|
| REM-Net continuous output | 0.305 ± 0.173 | +0.018 ± 0.181 | 2.151 | 0.750 | 1.711 | 0.312 |
| **T0 template @ GT peaks** | **0.856 ± 0.079** | **+0.730 ± 0.141** | **1.052** | 0.999 | **0.514** | 0.772 |
| T1 template @ REM-Net event head | 0.203 ± 0.147 | −0.711 ± 0.623 | 2.710 | 0.820 | 1.897 | 0.954 |
| T2 template @ REM-Net waveform peaks | 0.186 ± 0.139 | −0.477 ± 0.239 | 2.588 | 0.751 | 1.879 | 0.779 |
| T3 template @ Chen-TCN waveform peaks | 0.197 ± 0.144 | −0.485 ± 0.282 | 2.610 | 0.820 | 1.915 | 0.826 |

Paired Wilcoxon against REM-Net's own output (11 folds):

| Variant | ΔPCC | p | ΔR² | p |
|---|---:|---:|---:|---:|
| T0 | +0.551 | 0.001 (11/11) | +0.712 | 0.001 (11/11) |
| T1 | −0.102 | 0.005 (2/11 higher) | −0.730 | 0.001 |
| T2 | −0.119 | 0.002 (1/11) | −0.495 | 0.001 |
| T3 | −0.108 | 0.014 (2/11) | −0.504 | 0.001 |

Windows in which the timing source found no peak: T1 108, T2 702, T3 0, T0 0.

## Readings (as pre-declared)

1. **Timing is the dominant bottleneck.** With exact beat timing, a fixed training
   template that knows nothing about the test subject's morphology reaches PCC
   0.856 and explains 73 % of the variance (R² 0.730), against 0.305 and 0.018 for
   REM-Net. Every model in the benchmark is closer to zero than to this bound.
2. **REM-Net's waveform beats a template placed at its own beats** (T1, T2 lower on
   PCC and R², p ≤ 0.005). Read together with (1), the reason is mechanistic rather
   than morphological: radar-derived beat times are good enough to count beats
   (F1@100 0.75–0.82) but not to place them, and a sharp beat placed tens of
   milliseconds off is penalised twice -- once where it is, once where it is not.
   A shrunken, smoothed prediction is the lower-loss hedge under that uncertainty.

## Mechanism this closes

The three continuous findings are one phenomenon:

- every family sits at **R² ≈ 0** (matched benchmark);
- every family **shrinks amplitude** (variance ratio 0.29–0.54; REM-Net 0.31);
- a template with **exact timing reaches R² 0.73**, with radar timing **R² < 0**.

Under subject-independent radar-derived timing, beat placement error dominates;
regression toward the mean is the loss-optimal response to it, which is why
MSE-, Huber- and spectral-loss models alike converge to smoothed, low-variance
outputs. Improving continuous reconstruction therefore requires better beat
**timing**, not richer beat **morphology** -- morphology beyond a training-fold
template is not the limiting factor. This is consistent with the Tier-3
temporal-tolerance curve (Max-Lag NCC rising from 0.22 at 0 ms to 0.56 at ±50 ms).

## Scope

- T0 uses ground-truth timing and is an upper bound only.
- The template is one fixed shape per fold; a subject-adaptive template could only
  raise T0, strengthening reading 1.
- MMECG only; the CR_RVS counterpart belongs to workstream W3.
- Seed 42 models only; W2 extends the model side to three seeds.
