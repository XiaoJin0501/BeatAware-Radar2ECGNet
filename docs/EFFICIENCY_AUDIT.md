# Efficiency Audit (params / latency)

Purpose: lock the model-size and latency numbers for the T5 efficiency table with a
reproducible source. Parameter counts are exact (`sum(p.numel())`); latency is measured
on the project GPU.

**Scope (binding).** Every row here profiles a **Shape-200** model: the beat-normalised
morphology track. The continuous-track networks benchmarked in Table 1 of
`docs/audits/Q1_MULTISEED_BENCHMARK_AUDIT.md` are different networks with different
counts — REM-Net continuous B1 is 1.44 M, not the 1.788 M below, and the radarODE-MTL
comparator there is the 41.6 M four-task EGA model, not the 26.77 M backbone-only
variant below. The two tables may never be mixed in one sentence or one manuscript
table, and a parameter count must always name the track it belongs to.

## Setup

- GPU: NVIDIA GeForce RTX 4080 SUPER; input = single 8-s window WSST tensor
  `(1, 50, 71, 120)`; `torch.no_grad()`, `model.eval()`, 10 warmup + 200 timed iters.

## Numbers (traceable)

| Model | Params | Latency (ms/window) | Shape-200 PCC | Source |
|---|---|---|---|---|
| **REM-Net (ours)** | **1.788 M** | **0.98** (bs=1); 0.40 (bs=32 amortized) | 0.882 | `build_mmecg_model` on incumbent config |
| WSST-TCN (baseline) | 1.474 M | — | 0.273 | `build_mmecg_model(model_arch=wsst_tcn)` |
| radarODE-MTL (morphology comparator) | 26.77 M | — | 0.728 | `train_radarode_mtl_ega_mmecg.py` (DCNResNet+Squeeze backbone 23.25 M + ShapeDecoder200 3.52 M) |
| ICONIP BeatAware (prior conf.) | 4.847 M | — | 0.671 † | `BeatAwareRadar2ECGNet()` default args |

† BeatAware 0.671 is a calibration-prior (Block B), not a strict-LOSO comparable number.

## Interpretation (promotable)

- REM-Net attains the **highest** morphology PCC with the **second-smallest** parameter
  count — ~**15× smaller** than the radarODE-MTL morphology comparator (1.79 M vs
  26.77 M) while scoring higher (0.882 vs 0.728), and ~2.7× smaller than the ICONIP
  BeatAware prior. Its inductive bias comes from output structure and auxiliary
  supervision, not parameter count.
- Sub-millisecond single-window latency (0.98 ms) supports real-time/embedded use.

## Promotion rules

- **Promote:** parameter counts (exact) and REM-Net latency (state GPU + bs).
- Latency measured only for REM-Net (the deployment-relevant model); do not report
  cross-model latency, since the baselines use different input pipelines and a naive
  comparison would be misleading. FLOPs omitted (no profiler installed); params + latency
  are sufficient and exact.
