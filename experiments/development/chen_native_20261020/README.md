# Chen native categorical/autoregressive preparation

**Implementation + unit tests + CPU smoke ONLY. No formal training authorised.**
This is a paper-guided reimplementation with disclosed assumptions, not official
code and not a claim of exact replication. Do not reuse the historical dual-head
Chen adaptation's weights or numbers for this model.

## Paper audit

Chen et al., *Contactless Electrocardiogram Monitoring With Millimeter Wave
Radar*, IEEE TMC23(1),270–285(2024), DOI
https://doi.org/10.1109/TMC.2022.3214721 . Local PDF:
`reference_papers/MMECG_baselines model/Chen et al. - 2024 - Contactless Electrocardiogram Monitoring With Millimeter Wave Radar.pdf`.

- §3.3, pp.277–278, equation13: conditional distribution of x_t given previous
  ECG samples and radar-derived h_t. Training can parallelise known past labels;
  it does NOT justify using GT ECG during inference.
- §3.3, p.278, equation14: μ-law companding, μ255,256categorical values and CE;
  this is not continuous numerical regression. Author explicitly selected it.
- §4.2.3, p.279:640samples at200Hz, training-data stride30; temporal encoded
  shape50×32×80; fused feature4×640;9TCN stacks, claimed RF512; training128output
  points per640context; single-step autoregressive inference.
- Original optimiser is Adam1e−3,batch64. The author's requested matched recipe
  intentionally replaces that with AdamW1e−4,batch16,patience15,max100,
  drop_last=True; epochs1–10 consume no patience. It is not the original recipe.

No public original training implementation was supplied locally. Known paper
ambiguities are not filled in as if they were verified original facts:

| Detail | This implementation / explicit assumption |
|---|---|
| Four pool-by2 blocks vs stated640→80 | Four two-convolution blocks, pooling only after first3, to match stated80points |
| Four upsampling stages vs80→640 | Three doubling transposed-convolution stages, matching stated output length |
| Unspecified temporal widths | 8,16,32,32, shared across50native radar signals |
| Q/K/V dimension64 | Transformer model width64,4heads,FF128,3blocks, projected back to32 |
| Spatial expansion to50×4×time | Learned32→4 channel projection broadcast over time, enabling arbitrary length; not a literal fixed640-time dense map |
| TCN kernel/hidden width not fully specified | Nine causal residual single-convolution k2 blocks, width32,dilations1…256; exact RF512 |
| History encoding/initial value |256-class embedding plus separate zero BOS; no GT prefix |
| Emission rule not specified | Deterministic argmax, no sampling or averaging selected after looking at results |
| Target normalisation details incomplete | Existing A train-only median/IQR plus one train-only maximum absolute scale; validation cannot fit scale |

The old `MMECGBaseline` is NOT silently reused: it has a parallel scalar head,
no autoregressive ECG feedback, bin-index rather than measuredXYZ embedding,
and different receptive-field arithmetic.

## Model and training

Native time-domain50-channel RCG plus corresponding measured `posXYZ`, no WSST
or added event/Shape heads. CNN spatial-temporal encoder →4-channel conditional
features → causal categorical TCN. Real training windows640points, stride30,
from training subjects only. Teacher-forcing input is `[BOS, q_0,…,q_638]`.
CE is evaluated on indices512…639 only. No current/future target is provided
to its own prediction. Cached single-step decoder is tested against parallel
decoder on identical histories, not accepted merely because shapes match.

ECG reference is the existing A **raw-record-derived filtered** signal, not old
per-window z-score caches. First transform with A training fold median/IQR;
fit a further scalar max|z| on unique training records to place labels in[-1,1].
Encode μ-law classes; inverse μ-law and multiply by that scalar **before** all
waveform metrics. No prediction/reference per-window amplitude alignment,
validation scale fitting, or units in mV. Out-of-range labels would saturate;
the fitted scale and its scope are retained in the smoke manifest.

## 3.2s →8s adaptation

Train at640points. At inference the fully convolutional temporal encoder and
time-broadcast spatial gate process all1600radar samples, once. The decoder
then generates1600ECG classes one by one, feeding only its own previous class
back into the next step. Cached state is mathematically checked against full
causal convolution. No GT events, ECG, amplitudes or teacher-forced prefix are
accepted by the generation API.

This extends conditioning length; it is **not** a claim that the paper natively
used8s or that its exact640-point sliding encoder was reproduced. Offline
encoder access to8s radar is distinct from causal generation of ECG history.
The native rate remains200Hz: there is no640→1600waveform interpolation.

For records, use exactly approved_layout:8s inputs,1024stride, retain central
[288,1312), end-window uncovered suffix only, deduplicate by half-open support,
last288points excluded. State resets to zero/BOS at each8s input, not each beat.
The decoder RF512 exceeds the288discard:224early retained points can still
have startup effects. **Keep them in scoring**; no extra favourable exclusions.
The stitch/GT-identity tests include tail overlap and exact native indices.

## Frozen evaluation

`bridge.py` imports the actual immutable source from
`BeatAware-Radar2ECGNet-continuous-screen-v2/.../event_conditioned_screen_20261019`,
checks module locations and the running v2 freeze. It never rewrites the source,
freeze, outputs, registry or training service of that worktree.

- M1 with matched and scored coverage; M2/M3; record-optimal PCC±150ms;
  strict native beat PCC and window RMSE: exact frozen functions/aggregation.
- Q/R/S/T MAE and R-F1: NeuroKit2 prominence on the continuous **waveform**,
  native150ms one-to-one matching; missing events/coverage explicit. Same
  detected R is passed to M1. These are not historical event-head MAEs.
- Checkpoint: exact frozen v2 harmonic(max(0,M1)×coverage,waveform R-F1),
  same warmup/patience function and earliest tie. No new0.8coverage floor.
- Validation uses eval/no_grad and restores mode and all RNG states.

## This turn's commands

Run with one CPU thread and low CPU/I/O priority, with CUDA hidden:

```sh
nice -n 15 ionice -c 3 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -B -m pytest experiments/development/chen_native_20261020/test_native.py -q
nice -n 15 ionice -c 3 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -B experiments/development/chen_native_20261020/runner.py --mode smoke
```

The CLI has ONLY `--mode smoke`; there is no formal-training option. Smoke:
fold1,16actual640-point train windows,batch16,2epochs(one batch each); one real
8s validation excerpt generates all1600points autoregressively each epoch.
This is NOT a whole-record quality evaluation and its values are NOT performance
evidence. Full record inference/evaluation entry points exist and have alignment
tests. A random-initialisation CPU check also ran one actual35505point validation
record through generation/stitch/scoring:34929retained points,34complete windows.
This is not trained-model quality evidence. Trained-model evaluation and formal
pilot training await approval.
No weights are saved, so there are none to reuse or delete.

New outputs are exclusively
`BeatAware-Radar2ECGNet/experiments/development/chen_native_20261020/`.
