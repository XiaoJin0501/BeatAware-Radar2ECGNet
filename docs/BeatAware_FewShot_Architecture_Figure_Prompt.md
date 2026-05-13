# BeatAware-Radar2ECGNet: Architecture, Experiment Overview, and Figure Prompts

This document is a figure-planning brief for drawing the model architecture and the experimental story of the MMECG radar-to-ECG project. It is intended for another AI assistant, illustrator, or diagramming tool.

## 1. Paper-Level Story

### Working Title

**BeatAware-Radar2ECGNet: Few-Shot Subject Calibration for Contactless ECG Reconstruction from FMCW Radar**

### Core Claim

Contactless radar-to-ECG reconstruction is severely limited by subject-specific radar-to-cardiac transfer. Strict zero-shot LOSO evaluation exposes this difficulty, while a small amount of labeled target-subject calibration data can dramatically restore ECG morphology, R-peak timing, and interval-level clinical metrics.

The paper should not frame the calibration result as strict LOSO. It should be described honestly as:

- **LOSO + few-shot supervised subject calibration**
- **subject personalization**
- **limited target-subject calibration**

### Main Evaluation Protocols

1. **Strict LOSO zero-shot**
   - One subject is completely held out for testing.
   - No target-subject windows are used during training or validation.
   - This protocol measures true cross-subject generalization.

2. **Few-shot subject calibration**
   - For each held-out subject, a fixed number of labeled windows are used for personalization.
   - Current main protocol:
     - 40 target-subject segments for calibration training
     - 10 target-subject segments for target validation / model selection
     - all remaining target-subject segments for final testing
   - Each segment is 8 seconds at 200 Hz.
   - 40-shot calibration = 40 x 8 s = 320 s = 5.33 min.
   - 10-shot validation = 10 x 8 s = 80 s = 1.33 min.

3. **Ablation and shot-efficiency experiments**
   - Suggested shot curve: 5 / 10 / 20 / 40 calibration segments.
   - Suggested module ablation under 40-shot:
     - full model
     - no EMD alignment
     - no cardiac priors
     - no FMCW range encoder attention, if easy to implement.

## 2. Input and Output

### Input

The model receives an 8-second FMCW radar range-time matrix:

```text
X_radar: B x 50 x 1600
```

Where:

- `B` = batch size
- `50` = range bins
- `1600` = samples, 8 s at 200 Hz

The radar signal is an RCG-like range-bin time series. It contains chest wall motion and nuisance motion across multiple range bins.

### Output

The model reconstructs a normalized single-lead ECG waveform:

```text
y_hat: B x 1 x 1600
```

The output is in `[0, 1]` because the current main model uses per-window ECG min-max normalization and a sigmoid regression head.

## 3. Overall Architecture

### High-Level Pipeline

```text
FMCW radar range-time matrix
  B x 50 x 1600
        |
        v
[Module 1] Kinematic Inversion / FMCW Range Encoder
  learnable temporal filtering + range-bin attention
  B x 50 x 1600 -> B x 3 x 1600
        |
        +-----------------------------+
        |                             |
        v                             v
[Module 2A] Multi-scale Conv Encoder  [Module 2B] Cardiac Prior Branch
  k = 3,5,7,9, stride = 4              PeakAuxiliaryModule
  B x 3 x 1600 -> B x 256 x 400        predicts QRS/P/T masks + rhythm vector
        |                             |
        |<--------- TFiLM modulation--+
        v
Conformer Fusion Block
  local convolution + global temporal attention
  B x 256 x 400 -> B x 256 x 400
        |
        v
[Module 3] EMD Physical Alignment
  depthwise FIR temporal alignment
  max delay = 20 samples = 100 ms
  B x 256 x 400 -> B x 256 x 400
        |
        v
Regression Decoder
  ConvTranspose1d x 2 + Conv1d + Sigmoid
  B x 256 x 400 -> B x 1 x 1600
        |
        v
Reconstructed ECG
```

### Current Main Model

The current main experiment uses the slim regression model:

- `use_diffusion = false`
- `use_pam = true`
- `use_emd = true`
- `fmcw_selector = se`
- `n_range_bins = 50`
- model parameters: approximately **1.24M**

## 4. Three Core Innovation Modules

For the architecture figure, use three highlighted colored blocks. Recommended labels:

1. **Kinematic Inversion (KI): Range-Aware Radar Front-End**
2. **Cardiac Priors (CP): Beat-Aware Temporal Modulation**
3. **Physical Alignment (PA): Electromechanical Delay Compensation**

These are architecture-level contributions. The few-shot calibration protocol is an experiment/protocol-level contribution and should be drawn separately or as an outer training/evaluation loop.

### Module 1: Kinematic Inversion (KI)

**Purpose:** Convert noisy 50-bin FMCW radar observations into a compact motion representation suitable for ECG reconstruction.

**Input:** `B x 50 x 1600`

**Operations:**

1. Depthwise temporal convolution per range bin
   - kernel size = 61
   - initialized as identity / Dirac delta
   - preserves each range bin initially
2. BatchNorm + GELU
   - GELU is used instead of ReLU because chest-wall motion has positive and negative phases.
3. SE range-bin attention
   - learns channel-wise weights across the 50 range bins.
4. 1 x 1 projection
   - compresses 50 range bins to 3 learned motion channels.

**Output:** `B x 3 x 1600`

**Figure metaphor:** show 50 thin radar lines entering an attention gate, then compressing into 3 thicker motion traces.

### Module 2: Cardiac Priors (CP)

**Purpose:** Give the waveform decoder explicit beat-level and rhythm-level guidance so ECG morphology is not learned as an unconstrained free-form signal.

This module has two subparts.

#### CP-1: Peak Auxiliary Module

**Input:** `B x 3 x 1600`

**Operations:**

- multi-scale 1D convolutions with kernels 7, 15, 31
- temporal sequence modeling block
- three auxiliary heads:
  - QRS mask
  - P-wave mask
  - T-wave mask
- rhythm vector from temporal pooling

**Outputs:**

```text
QRS mask:    B x 1 x 1600
P mask:      B x 1 x 1600
T mask:      B x 1 x 1600
rhythm vec:  B x 96
```

Current training mainly uses QRS supervision because QRS labels are more reliable than P/T delineations.

#### CP-2: TFiLM Modulation

The rhythm vector is transformed into FiLM parameters:

```text
gamma, beta: B x 256
```

These parameters modulate the four branches of the multi-scale encoder:

```text
feature' = feature * (1 + gamma) + beta
```

**Figure metaphor:** draw QRS/P/T predictions as small side outputs, then route the rhythm vector into a FiLM controller that injects gamma/beta into the main encoder.

### Module 3: Physical Alignment (PA)

**Purpose:** Compensate electromechanical delay between ECG electrical activity and radar-observed mechanical chest-wall motion.

**Input:** `B x 256 x 400`

**Operation:**

- depthwise Conv1d / FIR filter
- kernel size = 41
- max delay = 20 feature samples
- covers approximately ±100 ms at 200 Hz
- initialized as identity, then learns temporal shifts from data

**Output:** `B x 256 x 400`

**Physical meaning:**

ECG electrical activation precedes the mechanical chest-wall displacement seen by radar. This layer gives each latent channel a small learnable temporal alignment filter.

**Figure metaphor:** show two slightly shifted waveforms labeled "electrical ECG" and "mechanical radar motion"; then show an alignment block pulling the mechanical feature earlier in time.

## 5. Few-Shot Calibration Experiment Overview

### Strict LOSO Baseline

For each fold:

```text
source subjects -> train
validation subject -> validation
held-out target subject -> test
```

No target-subject data is used in training.

This setting is scientifically important because it exposes the true zero-shot cross-subject difficulty.

### Few-Shot Subject Calibration

For each fold:

```text
source subjects
  +
40 labeled target-subject segments
        |
        v
train personalized model

10 labeled target-subject segments
        |
        v
target-subject validation / early stopping

remaining target-subject segments
        |
        v
final test
```

This protocol should be visually separate from the model block diagram. It is a training/evaluation protocol diagram.

### Example: Fold 01

Fold 01 held-out subject is S1.

Current 40/10 fixed-count split:

```text
calibration train: 40 segments = 5.33 min
calibration val:   10 segments = 1.33 min
test:              120 segments = 16.00 min
```

Earlier 40/10/50 ratio split gave strong fold_01 evidence:

```text
Strict LOSO fold_01 test:
  PCC        0.2488
  RMSE       0.1891
  R2        -0.2943
  R error   33.4 ms
  RR error  20.7 ms
  F1@150ms  0.6491

LOSO + supervised calibration fold_01 test:
  PCC        0.9162
  RMSE       0.0779
  R2         0.7748
  R error    5.1 ms
  RR error   5.9 ms
  F1@150ms  0.8507
```

These numbers should be marked as **fold_01 pilot evidence**, not full 11-fold final results.

The full fixed-count 40-shot/10-val 11-fold experiment is currently named:

```text
experiments_mmecg/mmecg_reg_fewshot40v10_slim
```

## 6. Recommended Paper Figures

### Figure 1: Task and Evaluation Protocol Overview

Purpose: explain the clinical task and why strict LOSO vs few-shot calibration matters.

Panels:

1. Contactless radar observes chest-wall motion.
2. Model reconstructs single-lead ECG.
3. Strict LOSO protocol.
4. Few-shot calibration protocol.

Key visual contrast:

- strict LOSO: target subject completely unseen
- few-shot calibration: target subject contributes 40 labeled calibration windows + 10 validation windows

### Figure 2: BeatAware-Radar2ECGNet Architecture

Purpose: show the full model pipeline.

Main blocks:

1. Input radar `B x 50 x 1600`
2. KI range encoder `50 -> 3 channels`
3. Cardiac prior branch:
   - QRS/P/T auxiliary masks
   - rhythm vector
   - TFiLM gamma/beta
4. Multi-scale conv encoder
5. Conformer fusion
6. PA EMD alignment
7. Regression decoder
8. reconstructed ECG `B x 1 x 1600`

Use three colors to highlight the innovations:

- KI: blue
- CP: green
- PA: orange

Keep the Conformer and decoder in neutral gray.

### Figure 3: Few-Shot Calibration Split

Purpose: make the protocol defensible.

Draw one held-out subject timeline of windows:

```text
S_target recording windows:
[40 calibration train] [10 calibration validation] [remaining final test]
```

Also show source subjects feeding into training.

Caption idea:

> In each LOSO fold, 40 labeled windows from the held-out subject are used for supervised personalization, 10 target windows are used for early stopping, and the remaining target windows are reserved for final testing.

### Figure 4: Performance Comparison

Purpose: show that calibration improves not only PCC but also timing and interval metrics.

Recommended plots:

- bar plot: strict LOSO vs few-shot calibration
  - PCC
  - RMSE
  - R2
  - R-peak error
  - RR interval error
  - F1@150ms
- segment-level box/violin plot of PCC
- per-subject PCC heatmap or bar plot across 11 folds

### Figure 5: Representative Waveforms

Purpose: convince readers visually.

For selected segments:

- black = ground-truth ECG
- gray dashed = strict LOSO prediction
- red/blue = few-shot calibrated prediction
- vertical faint lines = GT R peaks
- optional dots = predicted R peaks

Pick cases that demonstrate:

1. strict LOSO fails but calibration succeeds
2. medium-quality case
3. difficult residual failure case

### Figure 6: Shot-Efficiency Curve

Purpose: support the "few-shot" claim.

X-axis:

```text
0-shot, 5-shot, 10-shot, 20-shot, 40-shot
```

Y-axis:

- primary: PCC
- secondary or separate panel: R-peak error / RMSE

Expected interpretation:

The curve should show whether useful calibration emerges from seconds-level, 1-minute-level, or 5-minute-level labeled data.

## 7. Detailed Prompt for Another AI to Draw the Main Architecture Figure

Use the following prompt almost verbatim.

```text
Create a clean scientific architecture diagram for a biomedical signal processing paper.

Title: BeatAware-Radar2ECGNet for Few-Shot Contactless ECG Reconstruction

Canvas: wide horizontal diagram, white background, publication style, vector-like, no decorative gradients.

Show the full data flow from left to right:

1. Input block:
   "FMCW radar range-time matrix"
   shape: B x 50 x 1600
   draw as a stack of 50 thin time-series channels or a small heatmap.

2. Highlighted Module 1 in blue:
   label: "KI: Kinematic Inversion / Range Encoder"
   inside the block show:
   - depthwise temporal Conv1D, k=61
   - SE range-bin attention
   - 1x1 projection
   output shape: B x 3 x 1600
   Add annotation: "learns informative chest-motion range combinations".

3. Split the B x 3 x 1600 output into two branches:

   Main branch:
   - Multi-scale Conv Encoder
   - four Conv1D branches with kernels 3, 5, 7, 9 and stride 4
   - output shape B x 256 x 400

   Side branch, highlighted in green:
   label: "CP: Cardiac Priors"
   show:
   - Peak Auxiliary Module
   - QRS mask, P mask, T mask as three small waveform masks
   - rhythm vector B x 96
   - TFiLM controller producing gamma and beta, B x 256
   Draw arrows from gamma/beta into the Multi-scale Conv Encoder.
   Annotation: "beat-level supervision and rhythm-conditioned feature modulation".

4. After the Multi-scale Conv Encoder, add:
   "Conformer Fusion Block"
   shape: B x 256 x 400
   show small icons for local convolution and temporal self-attention.

5. Highlighted Module 3 in orange:
   label: "PA: Physical Alignment / EMD Layer"
   inside the block show:
   - depthwise FIR temporal filter
   - kernel size 41
   - max delay ±100 ms
   Add a tiny inset showing radar mechanical motion delayed relative to ECG electrical activity.
   Annotation: "compensates electromechanical delay".

6. Decoder:
   label: "Regression Decoder"
   show:
   - ConvTranspose1D 256->128, 400->800
   - ConvTranspose1D 128->64, 800->1600
   - Conv1D + Sigmoid

7. Output block:
   "Reconstructed ECG"
   shape: B x 1 x 1600
   draw a clean ECG waveform.

Use colors:
   KI = blue
   CP = green
   PA = orange
   backbone/decoder = neutral gray
   ECG output = red

Add a bottom strip labeled "Training losses":
   - waveform reconstruction loss
   - QRS auxiliary peak loss
   - validation metrics: PCC, RMSE, R-peak error, F1@150ms, interval error

Do not include diffusion decoder or Mamba. Do not include domain adversarial training in this figure.
Keep all labels compact and readable.
```

## 8. Detailed Prompt for Protocol Figure

```text
Create a protocol diagram comparing strict LOSO and few-shot subject calibration for a radar-to-ECG reconstruction study.

Use two stacked panels.

Panel A: Strict LOSO zero-shot
Show source subjects S2-S11 feeding into training.
Show target subject S1 completely held out.
Then show model tested on all S1 windows.
Add label: "No target-subject data used for training or validation".

Panel B: Few-shot subject calibration
Show the same source subjects feeding into training.
Show target subject S1 split into:
   - 40 labeled calibration windows for training
   - 10 labeled target windows for validation / early stopping
   - remaining target windows for final test
Use a horizontal timeline of 8-second windows.
Mark:
   40-shot train = 5.33 min
   10-shot val = 1.33 min
   final test = remaining windows

Make it clear that this is supervised personalization, not strict zero-shot LOSO.
Use publication-style colors:
   source training = gray/blue
   calibration train = green
   target validation = yellow
   final test = red or purple
```

## 9. Detailed Prompt for Results Figure

```text
Create a biomedical results figure showing that few-shot subject calibration improves contactless ECG reconstruction.

Use a 2x2 layout.

Panel A: Metric bars
Compare "Strict LOSO" vs "Few-shot calibration".
Metrics: PCC, RMSE, R2, R-peak error, RR interval error, F1@150ms.
Use upward arrows for metrics where higher is better and downward arrows where lower is better.

Panel B: Segment-level PCC distribution
Use box plots or violin plots.
Two groups: Strict LOSO and Few-shot calibration.

Panel C: Representative waveform overlay
Plot ground-truth ECG in black, strict LOSO prediction in gray dashed, calibrated prediction in red.
Show 8 seconds on the x-axis.
Add faint vertical lines at GT R peaks.

Panel D: Per-subject performance
Bar plot or heatmap for 11 LOSO folds.
X-axis: held-out subject.
Y-axis: PCC or R-peak error.
Use mean ± std if available.

Important annotation:
"Few-shot calibration uses 40 target-subject segments for training and 10 for validation; remaining target segments are held out for testing."

Do not imply that the calibration result is strict zero-shot LOSO.
```

## 10. Current Experiment Names and Files

Main fixed-count few-shot experiment:

```text
experiments_mmecg/mmecg_reg_fewshot40v10_slim
```

Important scripts:

```text
scripts/train_mmecg.py
scripts/test_mmecg.py
scripts/plot_mmecg_calibration_comparison.py
```

Important architecture file:

```text
src/models/BeatAwareNet/radar2ecgnet.py
```

Important data loader:

```text
src/data/mmecg_dataset.py
```

Existing pilot visualization:

```text
experiments_mmecg/calibration_figures/fold_01
```

## 11. Suggested Wording for Contributions

### Contribution 1: Honest Evaluation

We establish a strict subject-independent evaluation protocol for FMCW radar-to-ECG reconstruction and show that zero-shot LOSO performance is substantially lower than subject-overlapping evaluations, revealing the true cross-subject domain shift.

### Contribution 2: BeatAware-Radar2ECGNet

We propose a compact 1D architecture that combines range-aware kinematic inversion, beat-aware cardiac priors, and electromechanical-delay alignment to reconstruct ECG waveforms from FMCW radar range-time signals.

### Contribution 3: Few-Shot Subject Calibration

We introduce and evaluate a fixed-count subject calibration protocol, where only 40 labeled target-subject windows and 10 validation windows are used for supervised personalization, with the remaining target windows held out for final testing.

### Contribution 4: Multi-Level Clinical Evaluation

We report waveform-level, beat-level, fiducial-level, and interval-level metrics, including PCC, RMSE, R-peak timing error, RR interval error, F1@150ms, QMR, and clinical interval errors.

## 12. Reviewer-Safe Language

Use:

- "few-shot supervised subject calibration"
- "subject personalization"
- "target-subject calibration"
- "strict LOSO zero-shot baseline"
- "calibration-mode evaluation"

Avoid:

- calling calibration results "strict LOSO"
- implying target subject was unseen after calibration
- calling 40 segments "5-shot"
- overclaiming that EMD alone caused the calibration gain without ablation

