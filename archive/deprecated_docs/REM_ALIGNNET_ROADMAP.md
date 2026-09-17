# REM-AlignNet Roadmap for the Q1 Journal Extension

> Historical planning note. The current source-of-truth project structure is
> `docs/PROJECT_STRUCTURE_Q1.md`, and the next formal experiment plan is
> `docs/NEXT_EXPERIMENTS_Q1.md`. Some early probe/report paths below may have
> been deleted during cleanup; see `docs/DELETED_EXPERIMENTS_LOG.md`.

## 1. Purpose

This roadmap defines the implementation and experiment path for the Q1 journal extension. The new main model is tentatively named:

```text
REM-AlignNet: Rhythm-Event-Morphology Alignment Network
```

The model should not be presented as "BeatAware plus more modules." The journal story is:

```text
Reliable radar-to-ECG reconstruction requires:
  1. cardiac-informative radar representation,
  2. rhythm/event/morphology-aware intermediate structure,
  3. event-conditioned electromechanical alignment,
  4. continuous ECG output with shape-level morphology supervision.
```

Main output remains:

```text
Continuous ECG: [B, 1, 1600]
```

Shape-200 is an evaluation and auxiliary supervision protocol, not the main output task.

## 2. Current Project Audit

### 2.1 Keep As Core Assets

These files/directories should remain active in the Q1 project:

| Path | Role |
|---|---|
| `configs/mmecg_config.py` | MMECG config base; add Q1 flags conservatively. |
| `src/data/mmecg_dataset.py` | Existing MMECG 8 s H5 loader and LOSO splits. |
| `src/models/BeatAwareNet/radar2ecgnet.py` | ICONIP BeatAware model; keep as baseline and module source. |
| `src/models/modules/fmcw_encoder.py` | KI/FMCW range encoder; keep for raw/KI baselines. |
| `src/models/modules/peak_module.py` | Prior CP/PAM implementation; useful reference for event layer. |
| `src/losses/losses.py` | Existing waveform/STFT/event loss utilities. |
| `scripts/train_mmecg.py` | Main continuous-output MMECG training loop. |
| `scripts/test_mmecg.py` | Full-window monitoring evaluation. |
| `scripts/evaluate_mmecg_shape200.py` | Shape-200 morphology evaluation. |
| `scripts/summarize_shape200_results.py` | Fold/global shape-200 result summarizer. |
| `scripts/precompute_wsst4s_mmecg.py` | Existing radarODE-style WSST cache generator. |
| `experiments_mmecg/mmecg_radarode_mtl_ega_strict_loso/` | Frozen reproduced radarODE-MTL baseline. |
| `experiments_mmecg/q1_paper_style_report/` | Current paper-style Shape-200 report outputs. |

### 2.2 Archive Or De-Emphasize

These are useful historical materials but should not drive the REM-AlignNet story:

| Path pattern | Recommendation | Reason |
|---|---|---|
| `experiments_mmecg/*fewshot*` | Archive / do not use in main tables. | The current claim is not few-shot adaptation. |
| `experiments_mmecg/*calib*` | Archive / use only if explicitly marked. | Calibration violates strict subject-independent narrative. |
| `experiments_mmecg/mmecg_airecg_strict_loso_v4/` | Keep as external generative baseline, not central. | Diffusion is heavier and not the proposed route. |
| `experiments_mmecg/mmecg_rssrnet_strict_loso_v2/` | Keep as reproduced/implemented baseline if needed. | Useful comparison but not the mechanism story. |
| `experiments_mmecg/mmecg_*_strict_loso_baseline/` | Keep as historical baseline pool. | Do not let many generic backbones dilute the paper. |
| `ICONIP2026/` | Keep for ICONIP continuity. | Figures are legacy evidence, not new contribution. |
| deleted probe/report directories | See `docs/DELETED_EXPERIMENTS_LOG.md`. | Duplicate verification outputs and probes were removed during cleanup. |
| `__pycache__/`, `*.pyc` | Safe to delete anytime. | Generated Python cache files. |

Current retained organization before major experiments:

```text
experiments_mmecg/
  EXPERIMENT_REGISTRY.csv
  q1_paper_style_report/
  q1_crrvs_external_report_5fold/
  q1_supervisor_shape200_snapshot/
  q1_remalign_mtl_shapefirst_loso/
  q1_crrvs_remalign_mtl_5fold/
  mmecg_radarode_mtl_ega_strict_loso/
```

Avoid reusing older tags for new experiments.

### 2.3 Cleanup Candidates

Deleted cleanup candidates are recorded in `docs/DELETED_EXPERIMENTS_LOG.md`.
For future work, generated cache files are safe to delete anytime:

```text
**/__pycache__/
*.pyc
```

The large experiment folders should be moved or compressed only after the new Q1 tables are stable.

## 3. Literature Use Boundary

### 3.1 Core Motivation Papers

Use these to motivate REM-AlignNet:

| Paper | Use |
|---|---|
| radarODE-MTL | Supports task decomposition: ECG shape, PPI/rhythm, anchor. Use as frozen strong pipeline baseline. |
| Kong et al. 2026 SSM | Supports heartbeat-as-event and dynamics-aware reconstruction. |
| Cao et al. fiducial detection | Supports explicit event/fiducial structure from radar. |
| Radar2ECG semantic distillation | Supports multi-scale bottleneck, feature alignment, and morphology transfer. |
| mmJEPA-ECG | Supports physiology-oriented representation learning; use carefully, not as a domain-generalization claim. |
| Finding Order in Chaos | Supports rhythm/morphology/interference disentanglement. |
| SS-S2SA | Supports segmentation/partitioned reconstruction and acceleration/time-frequency representation. |

### 3.2 Baseline Or Background Papers

Use as baselines or literature context, not as direct architecture drivers:

| Paper type | Use |
|---|---|
| RSSRNet / CNN / BiLSTM / WaveGRU / LifWavNet | Generic reconstruction baselines. |
| AirECG / ECGDiff / conditional diffusion | Mention for high-fidelity generation, but avoid making diffusion the main claim. |
| GAN-based ECG generation | Background; note mode-collapse/instability risk. |
| Arrhythmia classification papers | Do not turn this paper into an arrhythmia classification paper. |
| Cross-posture/domain adaptation papers | Useful context; do not claim full domain generalization. |

### 3.3 Contradiction Risk

Avoid statements that conflict with the new thesis:

- Do not claim raw range-bin end-to-end learning is inherently superior.
- Do not claim radarODE-MTL is only a stronger backbone.
- Do not claim the paper solves domain generalization.
- Do not make diffusion or arrhythmia classification the center of the journal extension.
- Do not overemphasize QT unless valid delineation quality is confirmed.

## 4. Final Architecture Direction

### 4.1 Concept

REM-AlignNet decomposes radar-to-ECG reconstruction into:

```text
Radar representation
  -> multi-scale cardiac representation encoder
  -> rhythm/event/morphology intermediate structure
  -> event-conditioned electromechanical alignment
  -> continuous ECG decoder
  -> QRS-gated morphology refinement
```

### 4.2 Inputs

Primary MMECG inputs:

| Representation | Shape | Use |
|---|---:|---|
| Raw FMCW range-time | `[B, 50, 1600]` | Raw/KI baselines and ICONIP continuity. |
| KI-compressed range-time | `[B, C_ki, 1600]` | Representation ablation. |
| STFT/SST/WSST representation | e.g. `[B, 50, F, T]` or compressed `[B, 1, F, T]` | Main REM-AlignNet route. |

Primary CR_RVS inputs:

| Representation | Shape | Use |
|---|---:|---|
| CW phase/displacement | `[B, 1, 1600]` | Simple baseline. |
| CW STFT | `[B, 1, F, T]` | External validation route. |

### 4.3 Outputs

Main output:

```text
ecg_pred: [B, 1, 1600]
```

Auxiliary outputs:

```text
r_mask:      [B, 1, 1600]
qrs_mask:    [B, 1, 1600]
phase_sincos [B, 2, 1600]   # optional after MVP
delay:       [B, 1, 400]    # latent-resolution delay
shape_pred:  [B, 1, 200]    # optional explicit auxiliary head
```

### 4.4 MVP Architecture

The first implementation should be intentionally modest:

```text
STFT/SST input
  -> 2D CNN representation encoder
  -> temporal latent h: [B, C, 400]
  -> event head: R mask + QRS mask
  -> delay head: smooth low-resolution delay
  -> differentiable latent alignment
  -> TCN decoder
  -> ECG [B, 1, 1600]
```

Do not implement full rhythm/morphology/nuisance disentanglement in the first version.

### 4.5 Full Architecture After MVP

After the MVP shows signal:

```text
shared temporal latent h
  -> rhythm projection z_r
  -> morphology projection z_m
  -> nuisance projection z_n
  -> event map E(t)
  -> event-conditioned alignment
  -> dual-path ECG decoder
       - coarse continuous ECG path
       - QRS-gated residual morphology refinement
```

Minimal disentanglement regularizers:

```text
L_dis = |corr(z_r, z_n)| + |corr(z_m, z_n)|
```

Only add this if the event/alignment MVP is stable.

## 5. Loss Design

### 5.1 MVP Loss

```text
L = L_full + lambda_event L_event + lambda_shape L_shape200 + lambda_delay L_delay
```

Where:

| Loss | Purpose |
|---|---|
| `L_full` | Full-window ECG reconstruction: L1 + multi-resolution STFT. |
| `L_event` | R-peak mask BCE + QRS-region BCE. |
| `L_shape200` | Center-cycle morphology supervision, matching the radarODE-MTL comparison space. |
| `L_delay` | Smoothness and magnitude regularization for delay field. |

### 5.2 Later Optional Losses

Only after MVP is stable:

| Loss | Use |
|---|---|
| phase consistency | If soft cardiac phase improves rhythm. |
| decorrelation/disentanglement | If nuisance features visibly hurt robustness. |
| QRS residual sparsity | If refinement creates artifacts outside QRS. |

Avoid making loss design the main contribution.

## 6. Implementation Phases

### Phase 0: Project Hygiene

Goal: prevent history from contaminating the Q1 experiment trail.

Tasks:

1. Keep `docs/Q1_JOURNAL_EXPERIMENT_PLAN.md` as high-level plan.
2. Use this roadmap as the implementation contract.
3. Do not delete old experiment outputs yet.
4. Remove generated `__pycache__` and duplicate verify summaries if desired.
5. Use new `q1_remalign_*` experiment tags only.

Done criteria:

- Roadmap exists.
- All new tags follow `q1_remalign_*`.
- Frozen radarODE-MTL summary remains available.

### Phase 1: Representation Sanity Benchmark

Goal: answer whether the new architecture is worth building.

Experiments:

| Tag | Model | Input | Output |
|---|---|---|---|
| `q1_remalign_raw_tcn_loso` | Raw-TCN | raw range-bin | `[1,1600]` |
| `q1_remalign_ki16_tcn_loso` | KI16-TCN | KI range representation | `[1,1600]` |
| `q1_remalign_stft_tcn_loso` | STFT/SST-TCN | time-frequency | `[1,1600]` |
| frozen radarODE-MTL | original pipeline | WSST4s | `[1,200]` |

Evaluation:

- Shape-200 morphology: PCC/RMSE/MAE/R2.
- Full-window only for continuous-output models.

Stop/go logic:

- If STFT/SST-TCN is not better than Raw-TCN, revisit representation preprocessing before building REM-AlignNet.
- If STFT/SST-TCN approaches radarODE-MTL in shape-200, prioritize lightweight REM-AlignNet.
- If radarODE-MTL is still far stronger, emphasize task formulation and shape supervision.

Implementation tasks:

1. Raw/KI TCN path is already available through `simple_temporal.py`.
2. Add STFT/SST loader for continuous-output models.
3. Add model factory path for `stft_tcn`.
4. Add shape-200 summary commands for all tags.

### Phase 2: REM-AlignNet MVP

Goal: test event structure and dynamic alignment without overbuilding.

Variants:

| Tag | Variant |
|---|---|
| `q1_remalign_stft_tcn_loso` | representation-only baseline |
| `q1_remalign_event_loso` | + R/QRS event head |
| `q1_remalign_event_pa_loso` | + R/QRS event + old PA |
| `q1_remalign_align_loso` | + R/QRS event + dynamic alignment |

MVP modules to create:

```text
src/models/remalign/
  __init__.py
  rem_align_net.py
  spec_encoder.py
  event_head.py
  dynamic_alignment.py
  qrs_refinement.py
```

Training additions:

- `model_arch=remalign`
- representation type flag: `raw`, `stft`, `sst`, `wsst`
- event loss weights
- shape200 loss weight
- delay regularization weights

Done criteria:

- Single fold forward/backward passes.
- Fold 5 smoke run completes.
- Shape-200 and full-window outputs are generated.

### Phase 3: QRS-Gated Morphology Refinement

Goal: improve QRS sharpness without diffusion.

Architecture:

```text
coarse_ecg = decoder(h_aligned)
qrs_residual = small residual head(h_aligned)
ecg = coarse_ecg + qrs_mask * qrs_residual
```

Regularization:

- residual magnitude penalty
- residual outside-QRS penalty

Only keep this module if:

- QRS timing or morphology improves,
- full-window waveform does not become noisy,
- shape-200 morphology improves or stays stable.

### Phase 4: Full MMECG LOSO

Goal: produce main MMECG tables.

Main comparison:

| Model | Shape-200 | Full-window | Efficiency |
|---|---:|---:|---:|
| Raw-TCN | yes | yes | yes |
| KI16-TCN | yes | yes | yes |
| ICONIP BeatAware | yes | yes | yes |
| STFT/SST-TCN | yes | yes | yes |
| radarODE-MTL | yes | limited | yes |
| REM-AlignNet | yes | yes | yes |

Main tables:

1. Shape-200 morphology benchmark.
2. Full-window monitoring benchmark.
3. Minimal component ablation.
4. Efficiency comparison.

### Phase 5: CR_RVS External Validation

Goal: external validity under CW radar representation.

Preprocessing starts from:

```text
/home/qhh2237/Datasets/CR_RVS/CR_RVS/GDNXXXX/*.mat
```

Build new outputs under:

```text
/home/qhh2237/Datasets/CR_RVS/processed_remalign/
```

Do not reuse old residual scripts unless they are audited.

Initial pipeline:

```text
radar_i / radar_q
  -> I/Q correction or atan2 demodulation
  -> phase unwrap
  -> bandpass / second derivative candidate
  -> downsample to 200 Hz
  -> 8 s windows
  -> STFT
  -> tfm_ecg2 target
  -> R/QRS labels
```

Initial experiments:

| Model | Purpose |
|---|---|
| CW-STFT-TCN | external representation baseline |
| CW-STFT-REM-AlignNet | external validation of proposed principle |

Optional:

- Resting -> Valsalva
- Resting -> Apnea
- Resting+Valsalva -> Apnea

Do not make cross-dataset zero-shot/few-shot a main paper claim.

## 7. Metrics

### Shape-200 Morphology

Use for radarODE-MTL comparison:

- PCC
- RMSE
- MAE
- R2

Do not use shape-200 peak/QT metrics as main clinical evidence.

### Full-Window Monitoring

Use for continuous-output models:

- PCC
- RMSE
- MAE
- R-F1
- R timing error
- RR error
- QMR
- QRS duration error
- fiducial timing error
- QT error only on valid delineation subset

### Efficiency

Report:

- parameters
- FLOPs
- model inference time
- preprocessing time
- total pipeline time
- GPU memory

Always separate model-only time from representation-preprocessing time.

## 8. First Concrete Commands

Run raw/KI sanity baselines:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cyberbrain
cd /home/qhh2237/Projects/BeatAware-Radar2ECGNet

python scripts/train_mmecg.py \
  --exp_tag q1_remalign_raw_tcn_loso \
  --protocol loso \
  --fold_idx 5 \
  --model_arch tcn \
  --epochs 30 \
  --narrow_bandpass true \
  --balance_by subject

python scripts/train_mmecg.py \
  --exp_tag q1_remalign_ki16_tcn_loso \
  --protocol loso \
  --fold_idx 5 \
  --model_arch ki16_tcn \
  --epochs 30 \
  --narrow_bandpass true \
  --balance_by subject
```

Evaluate smoke runs:

```bash
python scripts/evaluate_mmecg_shape200.py \
  --model_type continuous \
  --exp_tag q1_remalign_raw_tcn_loso \
  --fold_idx 5

python scripts/evaluate_mmecg_shape200.py \
  --model_type continuous \
  --exp_tag q1_remalign_ki16_tcn_loso \
  --fold_idx 5
```

After STFT/SST loader is implemented, run the same fold-5 smoke test for `q1_remalign_stft_tcn_loso`.

WSST/STFT continuous baseline path is now available through:

```bash
python scripts/train_mmecg.py \
  --exp_tag q1_remalign_wsst_tcn_fold05_probe \
  --protocol loso \
  --fold_idx 5 \
  --model_arch wsst_tcn \
  --epochs 30 \
  --early_stop_patience 100 \
  --narrow_bandpass true \
  --balance_by subject

python scripts/evaluate_mmecg_shape200.py \
  --model_type continuous \
  --exp_tag q1_remalign_wsst_tcn_fold05_probe \
  --fold_idx 5 \
  --device cuda:0
```

The `wsst_tcn` path uses cached companion files such as:

```text
/home/qhh2237/Datasets/MMECG/processed/loso/fold_05/train_wsst.h5
/home/qhh2237/Datasets/MMECG/processed/loso/fold_05/val_wsst.h5
/home/qhh2237/Datasets/MMECG/processed/loso/fold_05/test_wsst.h5
```

For intentional radarODE-style center-4s input, pass:

```bash
--wsst_suffix _wsst4s.h5
```

## 9. Paper Claim After Roadmap

The strongest final claim should be:

```text
This work advances radar-to-ECG reconstruction from raw-input black-box regression to a representation-aware, rhythm-event-morphology structured, and alignment-guided continuous ECG reconstruction framework. The proposed model outputs full 8-second ECG waveforms while remaining comparable to radarODE-MTL in the shape-200 morphology space.
```

Boundary:

```text
We do not claim full domain generalization. We claim external validity of the reconstruction principle under heterogeneous radar representations.
```

## 10. Immediate Next Development Task

The next code task should be:

```text
Implement STFT/SST continuous-output dataset path and STFT-TCN baseline.
```

This is the fastest way to test whether the REM-AlignNet direction is worth completing.

## 11. Probe Results And Updated Decision

### 11.1 Representation Probe

Fold 1 and fold 5 shape-200 probes show that WSST representation is much stronger than raw range-bin input, but representation-only decoding is not enough to match radarODE-MTL:

| Fold | Model | PCC | RMSE | MAE | R2 |
|---:|---|---:|---:|---:|---:|
| 1 | Raw-TCN | 0.1682 | 0.3438 | 0.2669 | -3.1146 |
| 1 | WSST-TCN | 0.3624 | 0.2761 | 0.1968 | -1.6809 |
| 1 | radarODE-MTL | 0.8849 | 0.1383 | 0.1207 | 0.3486 |
| 5 | Raw-TCN | 0.1183 | 0.3152 | 0.2454 | -1.2844 |
| 5 | KI16-TCN | 0.1179 | 0.3249 | 0.2587 | -1.4084 |
| 5 | WSST-TCN | 0.4473 | 0.2834 | 0.2241 | -0.8753 |
| 5 | radarODE-MTL | 0.4903 | 0.2176 | 0.1541 | -0.0538 |

Interpretation:

- Representation quality is a major bottleneck.
- KI16 alone does not solve the raw-input limitation on fold 5.
- radarODE-MTL is not strong only because of WSST; its shape task, anchor/PPI decomposition, and MTL optimization likely matter.

### 11.2 REM-Event Probe

The first REM-Event MVP adds an R/QRS event head and event-conditioned decoder, but no shape200 auxiliary loss or dynamic alignment.

| Fold | Model | PCC | RMSE | MAE | R2 |
|---:|---|---:|---:|---:|---:|
| 1 | WSST-TCN | 0.3624 | 0.2761 | 0.1968 | -1.6809 |
| 1 | REM-Event | 0.3564 | 0.2865 | 0.2055 | -1.8126 |
| 5 | WSST-TCN | 0.4473 | 0.2834 | 0.2241 | -0.8753 |
| 5 | REM-Event | 0.4324 | 0.2410 | 0.1808 | -0.3517 |

Interpretation:

- Event supervision improves full-window R/F1 signals during validation, but does not reliably improve shape-200 PCC.
- On fold 5, REM-Event improves RMSE/MAE/R2 compared with WSST-TCN, suggesting better amplitude/error stability.
- On fold 1, REM-Event slightly hurts shape morphology. The event loss may be too strong, or the model lacks explicit shape-level morphology supervision.

The immediate next ablation should be:

| Variant | Purpose |
|---|---|
| WSST-TCN | representation-only baseline |
| REM-Event | event-only MVP |
| REM-Event + lower event loss | check whether event BCE is over-weighted |
| REM-Event + shape200 auxiliary loss | check whether morphology supervision closes the radarODE gap |

### 11.3 Automated Probe Result

Command:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cyberbrain
cd /home/qhh2237/Projects/BeatAware-Radar2ECGNet

FOLDS="1 5" EPOCHS=30 DEVICE="cuda:0" \
  bash scripts/run_remalign_shape_probe.sh
```

Result:

| Fold | Model | PCC | RMSE | MAE | R2 |
|---:|---|---:|---:|---:|---:|
| 1 | Raw-TCN | 0.1682 | 0.3438 | 0.2669 | -3.1146 |
| 1 | WSST-TCN | 0.3624 | 0.2761 | 0.1968 | -1.6809 |
| 1 | REM-Event | 0.3564 | 0.2865 | 0.2055 | -1.8126 |
| 1 | REM-Event-lowpeak | 0.3909 | 0.2793 | 0.2041 | -1.7305 |
| 1 | REM-Event-shape200 | 0.3972 | 0.2368 | 0.1589 | -1.0823 |
| 1 | radarODE-MTL | 0.8849 | 0.1383 | 0.1207 | 0.3486 |
| 5 | Raw-TCN | 0.1183 | 0.3152 | 0.2454 | -1.2844 |
| 5 | WSST-TCN | 0.4473 | 0.2834 | 0.2241 | -0.8753 |
| 5 | REM-Event | 0.4324 | 0.2410 | 0.1808 | -0.3517 |
| 5 | REM-Event-lowpeak | 0.5235 | 0.2237 | 0.1658 | -0.1763 |
| 5 | REM-Event-shape200 | 0.4887 | 0.2403 | 0.1796 | -0.3487 |
| 5 | radarODE-MTL | 0.4903 | 0.2176 | 0.1541 | -0.0538 |

Decision:

- `REM-Event-lowpeak` is the current main candidate for full LOSO.
- Lowering `beta_peak` from 1.0 to 0.3 is consistently helpful, so the original event loss was over-weighted.
- Shape-200 auxiliary loss improves fold 1 and reduces RMSE/MAE, but it does not beat lowpeak on fold 5 PCC. Keep it as an ablation, not the default main model.
- Dynamic alignment should be added after the lowpeak full LOSO run, not before. This keeps the next claim clean: first show event structure works, then show alignment gives additional timing benefit.

## 12. Historical Automation

This section records an earlier probe automation plan. For current formal runs,
use `scripts/README.md`, `experiments_mmecg/EXPERIMENT_REGISTRY.csv`, and
`docs/NEXT_EXPERIMENTS_Q1.md`.

The old Q1 probe entry point was:

```bash
scripts/run_q1_remalign_pipeline.sh
```

Stages:

| Stage | Command | Purpose |
|---|---|---|
| probe | `bash scripts/run_q1_remalign_pipeline.sh` | Run the two-fold low-cost REM-Event probe. |
| summarize | `STAGE=summarize bash scripts/run_q1_remalign_pipeline.sh` | Rebuild CSV/JSON summaries from existing outputs. |
| full | `STAGE=full FOLDS="1 2 3 4 5 6 7 8 9 10 11" EPOCHS=120 bash scripts/run_q1_remalign_pipeline.sh` | Run Raw-TCN, WSST-TCN, and REM-Event-lowpeak LOSO. |
| all | `STAGE=all FOLDS="1 2 3 4 5 6 7 8 9 10 11" EPOCHS=120 bash scripts/run_q1_remalign_pipeline.sh` | Probe plus full LOSO. |

Key outputs:

```text
experiments_mmecg/q1_paper_style_report/summary_all_methods.csv
experiments_mmecg/q1_crrvs_external_report_5fold/
experiments_mmecg/q1_supervisor_shape200_snapshot/
```

The earlier probe summary paths have been replaced by the current report
directories documented in `experiments_mmecg/README.md`.

Recommended next run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cyberbrain
cd /home/qhh2237/Projects/BeatAware-Radar2ECGNet

STAGE=full \
FOLDS="1 2 3 4 5 6 7 8 9 10 11" \
EPOCHS=120 \
DEVICE="cuda:0" \
bash scripts/run_q1_remalign_pipeline.sh
```
