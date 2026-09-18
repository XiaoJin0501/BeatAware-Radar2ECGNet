# Matched-Protocol Continuous Baselines

Status: implementation complete 2026-09-18; benchmark not yet run.
Experiment: `q1_matched_baselines_continuous_strict_loso_seed42`
Spec: `experiments/specs/q1_matched_baselines_continuous_strict_loso_seed42.json`

This document defines the fair-comparison lane for Table 1. It exists because
the old `loso` baseline aggregates are BLOCKED: across 11 folds that split had
only two distinct validation subjects, so its cross-fold aggregate carries
model-selection bias (`docs/MMECG_SPLIT_PROVENANCE_AUDIT.md`). Nothing from that
split may enter a matched table.

## What is held identical

Every baseline shares the REM-Net run's protocol, not a re-implementation of it.
`scripts/train_continuous_baseline.py` **imports** the following from the frozen
`scripts/train_continuous_ecg_pilot.py`, so one implementation serves both sides:

| Held identical | Source |
|---|---|
| Fold loader and split files | `Fold`, `loso_v2`, same per-fold SHA256 |
| Target space and inverse | `fit_scaler` (train-fold median/IQR), scored in raw ECG units |
| Metric functions | `evaluate` -> waveform / smoothing / timing / region metrics |
| Checkpoint rule | validation composite `0.40 PCC + 0.25 (1-min(NRMSE,1)) + 0.20 max(R2,0) + 0.15 F1@100` |
| Budget | 100 epochs, patience 15, AdamW lr 1e-4, weight decay 1e-4, batch 16 |
| Seed | `42 + fold` |
| Test access | `TRAINING -> CHECKPOINT_FROZEN -> TEST_EVALUATING -> COMPLETE`, opened once |
| Aggregation | fold-macro, population std, plus the binding rules in `docs/METRIC_CONTRACT.md` |

Three things differ per baseline, by design: its architecture, the input
modality its paper requires, and its own published objective.

## The five baselines

| Key | Paper | Family | Input | Objective |
|---|---|---|---|---|
| `mmecg_tcn` | Chen et al., IEEE TMC 2024 | spatio-temporal causal TCN (the MMECG dataset paper's own model) | rcg (50, 1600) | MSE |
| `radarode_mtl` | Zhang et al., IEEE TIM 2025 | ODE-embedded multitask; official DCNResNet + squeeze backbone | WSST (50, 71, 240) | MSE |
| `wavegru_net` | Xu et al., Signal Processing 2025 | discrete-wavelet temporal (sym4 MODWT + CNN + Bi-GRU) | rcg (50, 1600) | MSE |
| `attention_dcgan` | Jin et al., IEEE Sensors J. 2025 | generative adversarial (ConvTranspose1d + self-attention) | rcg (50, 1600) | adversarial + L1 |
| `cross_modal_conformer` | Liu et al., IEEE Sensors J. 2024 | lightweight self-attention (conv subsampling + 2 Conformer blocks) | rcg (50, 1600) | waveform + spectrogram MSE + deep feature |

Sources: `reference_papers/MMECG_baselines model/`. Implementations:
`src/models/baselines/continuous/`.

## Adaptations

Every deviation lives in `ADAPTATIONS` in
`src/models/baselines/continuous/__init__.py`, is copied into the spec, and is
written into each fold's `config.json`. All five rows are **common-protocol
(adapted) reproductions** under `docs/BASELINE_REPRODUCTION_POLICY.md` and must
carry that label in the manuscript. The load-bearing ones:

- **radarODE-MTL**: the official backbone is imported verbatim from the authors'
  repository, but ECG_shape(200) / PPI(260) / anchor(800) cannot emit a
  continuous stream without the long-term assembly this project could not
  faithfully reproduce (ledger F2). An adapted continuous decoder, built from
  the authors' own blocks, emits 1x1600. EGA multitask alignment is therefore
  not exercised. Label the row `adapted (continuous head)`.
- **Attention-DCGAN**: the paper feeds 200-sample fragments centred on ECG R
  peaks. That segmentation is a ground-truth oracle here, so the generator maps
  whole 8 s windows. The published objective is purely adversarial, which does
  not converge for paired reconstruction, so an L1 term with weight 10 is added
  and reported as a deviation.
- **WaveGRU-Net**: MODWT is a fixed a-trous sym4 filter bank (5 levels, 6 bands)
  running inside the model; `tests/test_continuous_baselines.py` checks it
  band-for-band against `pywt.swt` (correlation > 0.999 up to the filter delay).
  A 1x1 layer mixes 50 bins x 6 bands down to 32 channels before the published
  CNN widths.
- **Chen 2024**: autoregressive 256-way mu-law softmax decoding is replaced by a
  parallel continuous head; the CNN-Transformer encoder and the 9-block causal
  dilated TCN are kept.
- **Cross-modal Conformer**: the ECG-to-ECG autoencoder is pretrained per fold on
  TRAINING-fold ECG only, preserving the deep-feature loss inside the protocol.

## Event metrics are not applicable

No baseline has an event head. The adapter supplies a constant -10 logit so the
shared evaluator runs unchanged, which means `event_f1_*` is **structurally
undefined** for these rows. Leave those cells blank and state why; never report
them as a score of zero, and never compare them with REM-Net's event head.

## Running the benchmark

```bash
# registered preflight: spec status, clean worktree, commit + branch
python scripts/run_baseline_continuous_strict_loso.py --preflight_only

# full benchmark: 5 baselines x 11 folds, one-time test access per fold
python scripts/run_baseline_continuous_strict_loso.py --device cuda:0

# one baseline only
python scripts/run_baseline_continuous_strict_loso.py --baselines wavegru_net
```

Validation-only trials (no test access) belong in the `_smoke` tree:

```bash
python scripts/train_continuous_baseline.py --baseline wavegru_net --fold 1 \
    --smoke --smoke_epochs 100 --smoke_limit 999999
```

Measured cost on the RTX 4080 SUPER, fold 1 (1665 training windows), seconds per
epoch: wavegru_net 2, cross_modal_conformer 2, attention_dcgan 4, mmecg_tcn 13,
radarode_mtl 31. At the observed early-stopping range the full 55-run benchmark
is roughly seven hours.

## Reporting rules

Table 1 rows must state, for every baseline: `adapted` lane, the input modality,
the objective, parameter count, and the blank `event_f1_*` cells with their
reason. The metric rules of `docs/METRIC_CONTRACT.md` ("Continuous benchmark
reporting rules") apply unchanged: RR/HR gated at `rr_coverage >= 0.80`, the
>10 Hz ratio as a median with IQR, z-score units, per-metric aggregation labels,
and population std across folds.

Seed 42 only. A sentence of the form "REM-Net outperforms X" additionally
requires the multi-seed confirmation mandated by the metric contract.
