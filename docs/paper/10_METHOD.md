# Method (draft prose)

Grounded in `src/models/remalign/rem_align_net.py`, `src/losses/losses.py`,
`scripts/train_mmecg.py`. Numbers are the frozen final configuration; the alignment
module is described as ablated/removed (G5). Do not add claims beyond
`docs/PAPER_CLAIM_LEDGER.md`.

## 3. Method

### 3.1 Overview

REM-Net follows a morphology-first design: it maps a cardiac-informative radar
time-frequency representation to a beat-normalized ECG morphology through three
components — a lightweight time-frequency encoder, an auxiliary cardiac-event
branch, and a multi-task morphology decoder. The network is trained under full
supervision with paired radar and reference ECG; it does not attempt to solve
continuous long-window ECG assembly, which we instead analyze separately
(Section on the continuous gap). An earlier event-conditioned temporal-alignment
module was evaluated and removed after controlled experiments showed it did not
improve morphology (Section Ablation); the final model is deliberately simple.

### 3.2 Radar representation

Both datasets are converted to a synchrosqueezed time-frequency representation
(WSST/STFT) over 8-second, 200-Hz windows (1600 samples). For the 77-GHz FMCW
MMECG data, the 50 range bins yield a tensor of shape `[50, F, T]`
(`F=71`, `T=120`); for the 24-GHz CW CR_RVS data, the complex I/Q stream is
demodulated to an unwrapped phase displacement and transformed to a single-channel
time-frequency map `[1, F, T]`. Using a shared time-frequency front end lets the
same architecture operate on two radar modalities with only the input-channel count
changed.

### 3.3 Time-frequency encoder

The encoder (`WSSTEncoder`) applies parallel multi-scale 2-D convolutions
(kernels 3 and 5) with GELU activations and batch normalization, fuses them,
pools over frequency, and projects to a temporal latent sequence
`h ∈ R^{C×T_l}` with `C=128` hidden channels and `T_l=400` latent steps. This
front end is intentionally light; representation richness comes from the
time-frequency input rather than a deep backbone.

### 3.4 Cardiac-event auxiliary branch

An event head (`EventStructureHead`) predicts, from `h`, two per-sample masks at
the output resolution: an R-peak likelihood and a broader QRS-region mask. These
are supervised auxiliarily with a Gaussian soft-label target derived from the
reference R-peaks. The event branch injects cardiac-structure information back into
the shared latent (through a learned event projection) and provides a
localization signal that we later use to analyze continuous assembly. Only QRS/R
events are supervised; P/T labels are treated as unreliable and masked.

### 3.5 Morphology multi-task decoder

The event-conditioned latent is refined by a dilated temporal convolutional stack
(dilations 1–16) and decoded by two heads: a full-window decoder producing the
1600-sample waveform, and an explicit 200-point morphology head producing the
beat-normalized center-cycle morphology (Shape-200). The Shape-200 head is the
paper's primary reported output; the full-window decoder is retained for
completeness and for the continuous-gap analysis. Multi-task decoding lets the
morphology head specialize on beat shape while sharing cardiac-event structure.

### 3.6 Removed alignment module (design decision)

An earlier variant inserted an event-conditioned temporal alignment module between
the encoder and the decoder, intended to correct a hypothesized time-varying
electromechanical delay. A controlled ablation (with vs without alignment, matched
recipe) showed no morphology benefit (Δ Shape-200 PCC within ±0.005, sign varying
across folds). We therefore remove it from the final model. We report this as part
of the research process: the paper's contribution is a morphology-first supervised
model, not an alignment mechanism.

### 3.7 Training objective

The loss combines waveform, spectral, event, and morphology terms with fixed
weights (no adaptive uncertainty weighting):

```
L = L1(pred, ecg) + α · MultiResSTFT(pred, ecg)
      + β · BCE(qrs_pred, qrs_gt)
      + λ · [ L1(shape_pred, shape_gt) + w_pcc · (1 − PCC(shape_pred, shape_gt)) ]
```

with the frozen final configuration α=0.05, β=0.6, λ=0.8, w_pcc=0.5. The Shape-200
target is the reference center cardiac cycle resampled to 200 points and min–max
normalized per beat.

### 3.8 Optimization and checkpoint selection

Models are trained per fold with AdamW (lr 1e-4, weight decay 1e-4), batch size 16,
up to 150 epochs with early stopping (patience 30) and subject-balanced sampling.
The reported checkpoint is selected on a validation criterion only; no test-fold
information is used for model or checkpoint selection. REM-Net has ≈1.79M
parameters (MMECG) and runs at ≈1.1 ms per 8-second window on a single GPU,
markedly smaller than the ICONIP BeatAware model (≈4.9M).
