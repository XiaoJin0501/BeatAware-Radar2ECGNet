# Q1 Journal Extension Experiment Plan

## Core Positioning

This journal extension advances BeatAware-Radar2ECGNet from a single-dataset raw-input reconstruction model into a representation-aware, event-structured, and alignment-guided framework for continuous radar-to-ECG reconstruction.

The central claim is not that a new backbone alone solves radar-to-ECG reconstruction. Reliable reconstruction requires:

1. cardiac-informative radar representations,
2. explicit cardiac event structure, and
3. electromechanical alignment between radar-measured mechanical motion and ECG-recorded electrical activity.

The ICONIP version already showed that KI, CP, and PA improve continuous ECG reconstruction and beat-level timing from raw FMCW range-bin inputs on MMECG. The journal version explains why stronger pipelines such as radarODE-MTL perform well, then proposes a more systematic framework that keeps continuous ECG reconstruction as the primary task.

## Contributions

### Contribution 1: Representation-Controlled Analysis

We provide a representation-controlled analysis of radar-to-ECG reconstruction. Under the same MMECG 11-fold LOSO split, we compare raw range-bin inputs, KI-enhanced range representations, STFT/SST representations, ICONIP BeatAware, and the reproduced radarODE-MTL original pipeline.

This contribution treats radarODE-MTL as a strong pipeline-level reference, not merely as a stronger neural backbone. Its performance reflects representation, task formulation, and optimization. The goal is to show that reconstruction performance depends strongly on whether the radar representation exposes cardiac-informative mechanical patterns.

### Contribution 2: Representation-Aware Event-Structured Continuous ECG Reconstruction

We propose PhysioStruct-Radar2ECG, a representation-aware event-structured framework for continuous ECG reconstruction. The framework uses STFT/SST or other extensible radar representations, extracts cardiac motion latent features, and explicitly predicts R-peak and QRS-region structures to guide ECG waveform reconstruction.

The event structure layer upgrades the ICONIP beat-aware conditioning idea into an interpretable intermediate physiological structure. The model is not framed as black-box sequence regression from arbitrary radar inputs.

### Contribution 3: Event-Conditioned Electromechanical Alignment

We introduce event-conditioned electromechanical alignment for continuous ECG timing recovery. Unlike a fixed temporal shift or the ICONIP PA temporal filter, the journal model predicts a dynamic delay field conditioned on R/QRS event features.

This directly targets the radar-to-ECG electromechanical gap: radar measures mechanical chest motion, whereas ECG records cardiac electrical activity. Their temporal relationship can be subject-dependent, beat-dependent, and non-fixed.

### Contribution 4: External Validation on Heterogeneous Radar Data

We validate the reconstruction principle on both MMECG 77 GHz FMCW radar and the Schellenberger/CR_RVS 24 GHz CW radar dataset. CR_RVS preprocessing will be rebuilt from the original `.mat` files rather than relying on earlier residual scripts.

The claim is external validity under heterogeneous radar representations, not full cross-dataset domain generalization. Cross-dataset zero-shot and few-shot adaptation are not main experiments.

## Evaluation Protocols

### Shape-200 Morphology Protocol

This protocol compares single-cycle ECG morphology in a 200-point normalized cardiac-cycle space. It is the fair comparison space for radarODE-MTL because the reproduced radarODE-MTL pipeline outputs `ECG_shape` with shape `(1, 200)`.

Primary metrics:

- PCC
- RMSE
- MAE
- R2

Shape-200 peak and interval metrics are treated as exploratory only. They are not equivalent to full-window clinical monitoring metrics.

### Full-Window Monitoring Protocol

This protocol evaluates continuous ECG reconstruction over 8-second windows with length 1600 at 200 Hz. It is the primary protocol for BeatAware, PhysioStruct-Radar2ECG, and other continuous-output models.

Primary metrics:

- PCC, RMSE, MAE, R2
- R-peak F1
- R timing error
- RR interval error
- QMR
- QRS duration error
- fiducial timing error

QT interval error is reported only on valid delineation subsets and should not be the central claim.

## Main Experiments

### Experiment 1: MMECG Representation Benchmark

Purpose: support Contribution 1.

Protocol:

- MMECG 11-fold LOSO
- Shape-200 morphology evaluation
- radarODE-MTL results are frozen and reused from the reproduced run

Models and pipelines:

| Group | Input | Model | Purpose |
|---|---|---|---|
| Raw-TCN | raw range-bin | TCN | raw baseline |
| Raw+KI-TCN | KI/compressed range | TCN | KI representation effect |
| ICONIP BeatAware | raw+KI | KI+CP+PA+TCN | ICONIP baseline |
| STFT/SST-TCN | time-frequency | TCN | representation bottleneck test |
| radarODE-MTL | WSST | original pipeline | strong pipeline reference |

Expected interpretation:

- If STFT/SST-TCN outperforms Raw-TCN, representation is a major bottleneck.
- If ICONIP BeatAware improves over raw baselines, physiology-aware structure helps under raw inputs.
- If radarODE-MTL remains strongest, it is interpreted as a strong representation/task/optimization pipeline, not simply a backbone comparison.

### Experiment 2: MMECG Main Model Comparison

Purpose: support Contributions 2 and 3.

Models:

- Raw-TCN
- ICONIP BeatAware
- STFT/SST-TCN
- radarODE-MTL original pipeline
- Proposed PhysioStruct-Radar2ECG

Reports:

- Shape-200 morphology table for fair comparison with radarODE-MTL.
- Full-window monitoring table for continuous ECG timing and beat-level metrics.
- Efficiency table.

### Experiment 3: Minimal Physiological Structure Ablation

Purpose: support Contributions 2 and 3 without expanding into a large module-ablation paper.

Variants:

| Variant | Purpose |
|---|---|
| STFT/SST-TCN | representation-only baseline |
| + R/QRS Event Structure | event structure contribution |
| + R/QRS Event Structure + old PA | connection to ICONIP PA |
| + R/QRS Event Structure + Dynamic Alignment | full proposed model |

Key metrics:

- PCC, RMSE, MAE
- R-F1
- R timing error
- RR error
- QMR
- QRS duration error

Recommended figures:

- R timing error CDF
- before/after alignment waveform examples
- predicted delay distribution

### Experiment 4: CR_RVS External Validation

Purpose: support Contribution 4.

Preprocessing starts from:

```text
/home/qhh2237/Datasets/CR_RVS/CR_RVS/GDNXXXX/*.mat
```

Pipeline:

```text
radar_i / radar_q
  -> I/Q correction or arctangent demodulation
  -> phase unwrap
  -> bandpass filtering
  -> downsample to 200 Hz
  -> 8-second windows
  -> STFT representation
  -> tfm_ecg2 ECG target and R/QRS labels
```

Initial models:

- CW-STFT-TCN
- CW-STFT-Proposed

Initial protocols:

- subject-independent validation on Resting/Valsalva/Apnea
- cross-state validation: Resting -> Valsalva, Resting -> Apnea
- optional: Resting+Valsalva -> Apnea

### Experiment 5: Efficiency Analysis

Purpose: demonstrate practical advantages and clarify pipeline cost.

Metrics:

- parameter count
- FLOPs
- model inference time
- representation preprocessing time
- total pipeline time
- GPU memory

Efficiency should be reported separately for model-only and full-pipeline settings, especially when comparing raw-input models with STFT/SST or radarODE-style preprocessing.

## De-Scoped From Main Paper

The following are not main experiments unless the core results are already stable:

- broad robustness test matrix,
- cross-dataset zero-shot evaluation,
- few-shot target adaptation,
- full P/Q/S/T event-branch ablation,
- large loss ablation,
- exhaustive representation-model cross product.

At most, one small robustness test such as temporal masking or noise injection can be moved to supplementary material.

## Immediate Implementation Steps

1. Freeze and summarize reproduced radarODE-MTL shape-200 results.
2. Build a unified MMECG shape-200 benchmark summary path.
3. Add missing Raw-TCN, Raw+KI-TCN, and STFT/SST-TCN entries under the same LOSO and shape-200 protocol.
4. Implement the minimal PhysioStruct-Radar2ECG model.
5. Add dynamic alignment only after the event-structured model is stable.
6. Rebuild CR_RVS preprocessing from raw `.mat` files.

## First Benchmark Commands

Use the project environment:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cyberbrain
cd /home/qhh2237/Projects/BeatAware-Radar2ECGNet
```

Summarize the reproduced radarODE-MTL baseline:

```bash
python scripts/summarize_shape200_results.py \
  --exp_tag mmecg_radarode_mtl_ega_strict_loso
```

Run the first continuous-output raw/KI baselines:

```bash
python scripts/train_mmecg.py \
  --exp_tag q1_raw_tcn_loso \
  --protocol loso \
  --fold_idx -1 \
  --model_arch tcn \
  --epochs 150 \
  --narrow_bandpass true \
  --balance_by subject

python scripts/train_mmecg.py \
  --exp_tag q1_ki16_tcn_loso \
  --protocol loso \
  --fold_idx -1 \
  --model_arch ki16_tcn \
  --epochs 150 \
  --narrow_bandpass true \
  --balance_by subject
```

Evaluate each trained fold in shape-200 space:

```bash
for f in $(seq 1 11); do
  python scripts/evaluate_mmecg_shape200.py \
    --model_type continuous \
    --exp_tag q1_raw_tcn_loso \
    --fold_idx "$f"

  python scripts/evaluate_mmecg_shape200.py \
    --model_type continuous \
    --exp_tag q1_ki16_tcn_loso \
    --fold_idx "$f"
done

python scripts/summarize_shape200_results.py --exp_tag q1_raw_tcn_loso
python scripts/summarize_shape200_results.py --exp_tag q1_ki16_tcn_loso
```

The `continuous` evaluator path loads the saved `model_arch` from each run's
`config.json`, so it will construct the correct TCN/KI-TCN/REM model through
`build_mmecg_model`.

The STFT/SST-TCN benchmark still requires a dataset/loader path that returns
time-frequency inputs under the same LOSO and target protocol. This is the next
implementation item after the raw/KI baselines are runnable.
