# BeatAware-Radar2ECGNet: Subject-Independent Contactless ECG Reconstruction from FMCW Radar with Beat-Aware and Electromechanical Delay Modeling

## 1. Core Story

This paper should not be framed only as "a new neural network for radar-to-ECG reconstruction." The stronger Q1-level story is:

> Contactless ECG reconstruction from mmWave radar remains fragile under strict subject-independent evaluation. Existing studies often report promising results under samplewise or less stringent protocols, but strict LOSO evaluation reveals substantial cross-subject domain shift, subject imbalance, and electromechanical delay. BeatAware-Radar2ECGNet addresses these challenges through range-aware kinematic encoding, beat-aware auxiliary guidance, and learnable electromechanical delay alignment.

The narrative should turn difficult LOSO results into motivation rather than weakness. The paper can argue that robust contactless ECG requires evaluation beyond average waveform similarity, including subject-level robustness, scene-level stability, fiducial timing, and clinical interval consistency.

## 2. Tentative Contributions

1. We formulate radar-to-ECG reconstruction under a strict subject-independent LOSO protocol on the MMECG FMCW radar dataset, exposing the domain shift hidden by samplewise evaluation.

2. We propose BeatAware-Radar2ECGNet, a radar-to-ECG architecture with three physiologically motivated components:
   - KI: range-aware FMCW kinematic encoding for weak cardiac motion extraction.
   - PA: learnable electromechanical delay alignment for radar-ECG temporal mismatch.
   - CP: beat-aware peak auxiliary guidance for morphology and fiducial preservation.

3. We introduce a multi-level evaluation protocol covering waveform reconstruction, peak timing, qualified monitoring rate, fiducial detection, and clinical interval errors.

4. We analyze both subject-level and scene-level robustness across NB, IB, SP, and PE states, showing where contactless ECG reconstruction succeeds and where it fails.

## 3. Problem Definition

Input:

```text
RCG window: X_radar in R^{50 x 1600}
```

Output:

```text
ECG window: y_ecg in R^{1 x 1600}
```

Sampling rate:

```text
200 Hz
```

Window duration:

```text
8 seconds
```

Main protocol:

```text
Leave-one-subject-out, 11 folds
```

Important framing:

The target is not only heart-rate tracking. The target is morphology-aware ECG reconstruction suitable for downstream fiducial and interval analysis.

## 4. Dataset Story

MMECG contains 11 usable subjects and 91 records across four physiological states:

```text
NB: normal breathing
IB: irregular breathing
SP: sleep posture / special posture
PE: physical exercise or post-exercise
```

Key dataset challenge:

```text
Subject imbalance is severe.
S1 and S2 each have only 2 records.
S29 has 23 records.
```

This motivates subject-balanced sampling during training. The paper should clearly state that subject-balanced sampling does not change labels or test data; it only prevents large-subject dominance in the training distribution.

## 5. Method Outline

### 5.1 FMCW Range-Aware Kinematic Encoder

Goal:

Extract cardiac motion from 50 FMCW range bins while suppressing irrelevant range components and residual clutter.

Main idea:

```text
50 range bins -> temporal filtering -> range attention -> 3-channel kinematic representation
```

Paper angle:

Radar ECG information is not uniformly distributed across range bins. A learnable range encoder is more appropriate than naive averaging or single-bin selection.

### 5.2 Peak Auxiliary Module

Goal:

Guide reconstruction with beat-level structure rather than relying only on waveform L1/STFT losses.

Outputs:

```text
QRS/P/T masks
rhythm vector
TFiLM modulation
```

Paper angle:

ECG morphology is sparse and rhythm-dependent. Auxiliary peak modeling encourages the encoder to preserve clinically meaningful events.

### 5.3 Electromechanical Delay Alignment

Goal:

Compensate for the temporal mismatch between electrical ECG activity and mechanically sensed cardiac motion.

Implementation:

```text
Learnable depthwise FIR alignment layer
```

Paper angle:

Radar measures mechanical consequences of cardiac activity, not electrical activation directly. A learnable delay layer gives the network a physiologically meaningful mechanism for alignment.

### 5.4 Optional Diffusion Decoder

Goal:

Improve high-frequency morphology and realistic waveform generation.

Positioning:

The diffusion decoder should be framed as an advanced decoder compatible with the BeatAware encoder, not necessarily the only main contribution.

## 6. Experimental Plan

### 6.1 Protocol Comparison Table

Purpose:

Clarify why published numbers are not always directly comparable.

Suggested columns:

| Work | Dataset | Radar Type | Task | Split Protocol | Metrics | Directly Comparable? |
|---|---|---|---|---|---|---|
| Cao et al. | TBD | mmWave | Fiducial detection | TBD | F1, timing error | Contextual |
| AirECG | TBD | mmWave | ECG reconstruction | TBD | PCC, RMSE | Contextual |
| radarODE-MTL | TBD | Radar | Multi-task ECG/fiducial | TBD | Timing, waveform | Contextual |
| Ours | MMECG | 77GHz FMCW | ECG reconstruction | Strict LOSO | PCC, RMSE, R2, QMR, F1, intervals | Main |

Use "contextual" when hardware, dataset, or split differs.

### 6.2 Main Result Table

Suggested columns:

| Method | Sampling | PCC ↑ | RMSE ↓ | R2 ↑ | R Error ms ↓ | RR Error ms ↓ | QMR ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| Regression baseline | class-balanced | TBD | TBD | TBD | TBD | TBD | TBD |
| Regression baseline | subject-balanced | TBD | TBD | TBD | TBD | TBD | TBD |
| BeatAware-Regression | subject-balanced | TBD | TBD | TBD | TBD | TBD | TBD |
| BeatAware-Diffusion | subject-balanced | TBD | TBD | TBD | TBD | TBD | TBD |

Notes:

The class-balanced run can be used as a diagnostic baseline. The subject-balanced run is likely the fairer main setting for strict LOSO.

### 6.3 Ablation Table

Suggested columns:

| Variant | KI | EMD/PA | PAM/CP | PCC ↑ | RMSE ↓ | R Error ms ↓ | QMR ↑ |
|---|:---:|:---:|:---:|---:|---:|---:|---:|
| Base encoder | - | - | - | TBD | TBD | TBD | TBD |
| + KI | ✓ | - | - | TBD | TBD | TBD | TBD |
| + KI + EMD | ✓ | ✓ | - | TBD | TBD | TBD | TBD |
| + KI + PAM | ✓ | - | ✓ | TBD | TBD | TBD | TBD |
| Full BeatAware | ✓ | ✓ | ✓ | TBD | TBD | TBD | TBD |

Important:

If a module improves peak timing but not PCC, that is still scientifically meaningful. The analysis should report both waveform and fiducial metrics.

### 6.4 Scene-Level Robustness Table

Suggested columns:

| Scene | N Segments | PCC ↑ | RMSE ↓ | R2 ↑ | R Error ms ↓ | QMR ↑ |
|---|---:|---:|---:|---:|---:|---:|
| NB | TBD | TBD | TBD | TBD | TBD | TBD |
| IB | TBD | TBD | TBD | TBD | TBD | TBD |
| SP | TBD | TBD | TBD | TBD | TBD | TBD |
| PE | TBD | TBD | TBD | TBD | TBD | TBD |

Interpretation:

Scene-level results should be secondary to subject-level LOSO results, because scene distribution is not balanced across subjects.

## 7. Visualization Plan

### Figure 1: Overall Pipeline

Show:

```text
FMCW range-time input -> KI encoder -> PAM/TFiLM -> GroupMamba/Conformer -> EMD alignment -> ECG decoder
```

Purpose:

Communicate the physiological logic of the architecture.

### Figure 2: Protocol Gap

Compare samplewise vs LOSO performance.

Possible plot:

```text
bar plot: samplewise PCC vs LOSO PCC
bar plot: samplewise QMR vs LOSO QMR
```

Purpose:

Show that strict subject-independent evaluation is harder and more clinically relevant.

### Figure 3: Subject-Wise LOSO Performance

Plot each test subject separately.

Metrics:

```text
PCC
RMSE
R-peak timing error
QMR
```

Purpose:

Show cross-subject robustness and identify difficult subjects.

### Figure 4: Subject x Scene Heatmap

Rows:

```text
test subjects
```

Columns:

```text
NB, IB, SP, PE
```

Color:

```text
PCC or RMSE
```

Purpose:

Disentangle subject difficulty from physiological scene difficulty. Missing cells should be shown explicitly rather than hidden.

### Figure 5: Representative Waveforms

Show three categories:

```text
good case
median case
failure case
```

For each:

```text
GT ECG
predicted ECG
detected/GT R peaks
```

Purpose:

Demonstrate morphology reconstruction honestly.

### Figure 6: Clinical Interval Agreement

Options:

```text
Bland-Altman plot for RR/QRS/QT/QTc
scatter plot predicted vs GT intervals
```

Purpose:

Support clinical relevance beyond waveform PCC.

## 8. How to Compare with Prior Papers

The paper should use two comparison categories:

### Direct Comparisons

Only use when:

```text
same dataset
same or reproducible split
same input modality
same task
same metric definition
```

### Contextual Comparisons

Use when published works differ in:

```text
dataset
sensor hardware
subject split
window length
metric definition
task objective
```

Suggested wording:

> Since existing radar-to-ECG studies vary substantially in radar hardware, subject split, and evaluation protocol, we report direct comparisons where possible and use published results as contextual references when protocols differ.

This prevents unfair claims while still demonstrating awareness of the literature.

## 9. Expected Claims

Strong and defensible claims:

1. Strict LOSO exposes a substantial generalization gap in contactless ECG reconstruction.

2. Subject-balanced training mitigates subject imbalance and improves robustness to underrepresented subjects.

3. Beat-aware and electromechanical-delay-aware modeling improves physiologically meaningful reconstruction, especially peak timing and qualified monitoring.

4. Multi-level evaluation gives a more complete view than PCC alone.

Claims to avoid unless results clearly support them:

```text
"State-of-the-art on all metrics"
"Clinical-grade ECG replacement"
"Works robustly for all subjects and all scenes"
```

## 10. Current Running Experiments

### Existing Baseline

Experiment:

```text
experiments_mmecg/mmecg_reg_loso
```

Configuration:

```text
use_diffusion=False
balance_by=class
narrow_bandpass=False
```

Status:

Fold 1-4 partially/completely available. This is useful as a class-balanced baseline but should not be the final main setting.

### Subject-Balanced Minimal Control

Experiment:

```text
experiments_mmecg/mmecg_reg_loso_subject
```

Configuration:

```text
use_diffusion=False
balance_by=subject
narrow_bandpass=False
```

Purpose:

Isolate whether subject imbalance is a major cause of weak LOSO performance.

Decision rule:

If fold 01 improves substantially over the class-balanced fold 01, prioritize full LOSO subject-balanced experiments.

## 11. Immediate Next Steps

1. Finish `mmecg_reg_loso_subject/fold_01`.

2. Test fold 01 with:

```bash
python scripts/test_mmecg.py --exp_tag mmecg_reg_loso_subject --fold_idx 1
```

3. Compare against:

```text
experiments_mmecg/mmecg_reg_loso/fold_01/results/global_summary.json
```

4. If subject-balanced improves, run full LOSO:

```bash
python scripts/train_mmecg.py \
  --exp_tag mmecg_reg_loso_subject \
  --protocol loso --fold_idx -1 --epochs 150 \
  --use_diffusion false \
  --balance_by subject \
  --narrow_bandpass false
```

5. Generate subject-wise and scene-wise summary plots from `segment_metrics.csv`.

## 12. Paper Skeleton

### Abstract

Briefly state the problem, strict LOSO challenge, BeatAware components, and multi-level evaluation.

### Introduction

Motivate contactless ECG reconstruction and explain why strict subject-independent evaluation matters.

### Related Work

Organize by:

```text
radar cardiography
radar-to-ECG reconstruction
fiducial detection from radar
physiology-aware deep learning
diffusion models for biosignals
```

### Dataset and Protocol

Describe MMECG, four scenes, preprocessing, strict LOSO split, and subject imbalance.

### Method

Describe KI, PAM/CP, EMD/PA, backbone, and decoder.

### Evaluation Metrics

Describe waveform, peak timing, QMR, fiducial F1, and clinical interval errors.

### Results

Include main results, protocol gap, subject-wise analysis, scene-wise analysis, and ablations.

### Discussion

Discuss why LOSO is hard, where the model fails, and why multi-level evaluation matters.

### Limitations

Mention:

```text
limited number of subjects
subject-scene imbalance
normalized ECG rather than absolute mV calibration
need for external validation on additional radar datasets
```

### Conclusion

Summarize the framework and the importance of strict subject-independent evaluation for contactless ECG.

