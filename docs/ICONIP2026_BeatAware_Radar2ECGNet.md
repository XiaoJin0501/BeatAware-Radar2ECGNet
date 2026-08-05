# ICONIP2026 BeatAware-Radar2ECGNet

This document is the single current project note for the ICONIP2026 paper:

`BeatAware-Radar2ECGNet: Physics-Constrained Continuous ECG Reconstruction from Radar`

The authoritative manuscript is:

`ICONIP2026/ICONIP2026_BeatAware_Radar2ECGNet.pdf`

## Current Scope

The project targets continuous ECG waveform reconstruction from the public MMECG FMCW radar dataset under strict subject-independent evaluation.

The current paper does not use few-shot subject calibration, Mamba/SSM backbones, or diffusion as the main result. The final model is a lightweight TCN-based BeatAware model with:

- KI: kinematic inversion from 50 FMCW range bins to compact radar motion channels.
- CP: cardiac prior conditioning through auxiliary QRS/P/T peak prediction and TFiLM modulation.
- TCN: lightweight temporal convolutional backbone.
- PA: physical alignment through EMD-style latent temporal alignment.
- Decoder: regression decoder for 8-second ECG waveform reconstruction.

## Dataset And Protocol

Dataset:

- MMECG public dataset.
- 11 healthy subjects.
- 77 GHz TI AWR1843 FMCW radar.
- 50 range-bin radar input.
- ECG and radar sampled at 200 Hz.
- Four conditions: NB, IB, PE, SP.

Window format:

- Radar input: `(B, 50, 1600)`.
- ECG target: `(B, 1, 1600)`.
- Each segment is 8 seconds.

Evaluation protocol:

- 11-fold leave-one-subject-out.
- One subject is held out for testing.
- One remaining subject is used for validation.
- All other subjects are used for training.
- No target-subject calibration is used for the main paper protocol.

## Main Model

The paper's full model is:

```text
FMCW radar (B,50,1600)
  -> KI / FMCWRangeEncoder, KI16
  -> CP / PeakAuxiliaryModule + TFiLM
  -> multi-scale temporal encoder
  -> lightweight TCN fusion backbone
  -> PA / EMDAlignLayer
  -> ConvTranspose ECG decoder
  -> reconstructed ECG (B,1,1600)
```

Current code defaults are aligned with the paper:

- `model_arch = "beataware"`
- `backbone_type = "tcn"`
- `fmcw_out_channels = 16`
- `use_pam = True`
- `use_emd = True`
- `use_diffusion = False`

Important files:

- `configs/mmecg_config.py`
- `src/models/model_factory.py`
- `src/models/BeatAwareNet/radar2ecgnet.py`
- `src/models/modules/fmcw_encoder.py`
- `src/models/modules/peak_module.py`
- `src/models/modules/emd_align.py` if present, otherwise the EMD layer is defined inside the current model module.
- `src/models/modules/tfilm.py`
- `src/losses/losses.py`
- `src/utils/metrics.py`

## Paper Results To Match

Main result reported in the manuscript:

```text
PCC  = 0.772
RMSE = 0.096
QMR  = 99.9%
R-F1 = 0.946
Q-Q  = 15.2 ms
R-R  = 8.20 ms
S-S  = 12.7 ms
T-T  = 11.5 ms
```

Compared with the TCN-only baseline:

```text
TCN only: PCC 0.683, RMSE 0.111
Full:     PCC 0.772, RMSE 0.096
```

## Ablation Table

The paper's Table 2 uses strict LOSO variants:

```text
TCN only
KI8 + TCN
KI16 + TCN
KI16 + TCN + CP
KI16 + TCN + PA
Full Model = KI16 + TCN + CP + PA
```

The historical ablation runner has been removed during Q1 cleanup because it
created unregistered experiment directories. Recreate these ablations only
after adding explicit rows to `experiments_mmecg/EXPERIMENT_REGISTRY.csv`.

## Baselines

The paper compares against:

```text
GRU
BiLSTM
Transformer
U-Net1D
TCN
WaveGRU
AirECG
RSSRNet
radarODE
Ours
```

Current retained external-baseline scripts:

- `scripts/train_airecg_mmecg_strict_loso_v2.py`
- `scripts/train_rssrnet_mmecg_strict_loso.py`
- `scripts/train_radarode_mtl_ega_mmecg.py`
- `scripts/precompute_wsst4s_mmecg.py`
- `scripts/run_radarode_ega_strict_loso_all.sh`
- `scripts/evaluate_mmecg_shape200.py`

The simple baseline implementation file `src/models/baselines/simple_temporal.py` is not currently present. Therefore `model_arch` values such as `gru`, `bilstm`, `transformer`, `unet1d`, `tcn`, and `wavegru` are not runnable through `train_mmecg.py` until that file is restored or reimplemented.

## Main Commands

Train one LOSO fold:

```bash
python scripts/train_mmecg.py \
  --exp_tag mmecg_beataware_strict_loso_full \
  --protocol loso \
  --fold_idx 1 \
  --model_arch beataware \
  --backbone_type tcn \
  --fmcw_out_channels 16 \
  --use_pam true \
  --use_emd true \
  --epochs 150 \
  --early_stop_patience 20
```

Train all LOSO folds:

```bash
python scripts/train_mmecg.py \
  --exp_tag mmecg_beataware_strict_loso_full \
  --protocol loso \
  --fold_idx -1 \
  --model_arch beataware \
  --backbone_type tcn \
  --fmcw_out_channels 16 \
  --use_pam true \
  --use_emd true \
  --epochs 150 \
  --early_stop_patience 20
```

Evaluate all LOSO folds:

```bash
python scripts/test_mmecg.py \
  --exp_tag mmecg_beataware_strict_loso_full \
  --protocol loso \
  --fold_idx -1
```

Historical ablation commands were removed during Q1 cleanup. Use
`scripts/README.md` and the experiment registry before launching new ablations.

Run radarODE strict LOSO:

```bash
bash scripts/run_radarode_ega_strict_loso_all.sh
```

## What Was Removed From The Current Project Narrative

The following are no longer part of the current ICONIP2026 documentation:

- Mamba or SSM backbone claims.
- Conformer as the main paper backbone.
- Few-shot subject calibration as the main protocol.
- `loso_calib` as the reported paper result.
- Diffusion decoder as the main paper result.
- Old roadmap, diagnostic, and audit narratives.
- Schellenberger/CW dual-dataset framing.

## Cleanup Rule

Use this document and the PDF as the current source of truth. If a file, script, figure, or experiment label refers to `fewshot40v10`, `calib40v10`, Mamba, SSM, Conformer-main, or diffusion-main, treat it as historical unless it is explicitly needed for backward compatibility.
