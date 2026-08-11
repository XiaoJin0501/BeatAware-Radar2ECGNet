# Datasets, Protocol, and Evaluation (draft prose)

Grounded in `docs/DATA_PROTOCOL.md`, `docs/T1_MMECG_MORPHOLOGY_LOCKED.md`,
`docs/T3_CRRVS_EXTERNAL_AUDIT.md`, `docs/METRIC_CONTRACT.md`. Subject-independence
of both protocols is verified.

## 4. Datasets and Evaluation Protocol

### 4.1 Datasets

**MMECG (77-GHz FMCW).** The public MMECG radar dataset provides 50 range-bin FMCW
radar with time-aligned reference ECG. We use 8-second windows at 200 Hz (1600
samples) and evaluate under an 11-fold strict leave-one-subject-out (LOSO)
protocol: in each fold, all windows from the held-out subject form the test set and
never contribute to training, normalization, range-bin selection, or checkpoint
selection.

**CR_RVS (24-GHz CW).** The public CR_RVS / Nature 24-GHz continuous-wave radar
dataset provides I/Q radar with reference Lead-II ECG across 30 subjects and
several physiological states (normal breathing, irregular breathing, sustained
breath-hold / Valsalva, post-exercise). We derive CW phase displacement from I/Q,
resample to 200 Hz, and use the same 8-second windowing. Evaluation uses 5-fold
subject-independent cross-validation; we verified there is no subject overlap
between train and test in any fold and no subject appears in two test folds (30
subjects partitioned 7/7/7/5/4).

### 4.2 Output formulation: beat morphology vs continuous ECG

We deliberately separate two output formulations. **Shape-200** is the
beat-normalized center cardiac cycle resampled to 200 points and min–max
normalized; it isolates beat morphology from absolute timing and amplitude and is
our primary reported output. **Full-window** ECG is the continuous 1600-sample
waveform; we use it only for the continuous-gap analysis, never as the main claim.
This distinction is essential: a method may reconstruct Shape-200 morphology
accurately yet fail to place beats on the correct continuous time axis.

### 4.3 Metrics and aggregation

Morphology is scored by Pearson correlation (PCC), RMSE, MAE, and R² between
predicted and reference Shape-200 curves. To move beyond a single global
correlation, we additionally report region-wise morphology fidelity over fixed
phase regions of the beat (QRS-edge, ST/T, diastolic). All metrics are computed per
test segment, averaged within a fold, then aggregated as fold-macro mean ± standard
deviation across folds. Headline results additionally report the across-seed mean ±
std over three random seeds. Checkpoints are selected on validation criteria only.

### 4.4 Comparison tiers (fairness)

To avoid mixing incompatible protocols, methods are grouped into explicit tiers:
(i) **proposed** (REM-Net, strict subject-independent); (ii) **adapted** baselines
re-implemented and run under our strict-LOSO protocol (radarODE-MTL evaluated in its
native ECG_shape morphology space, RSSRNet, AirECG); (iii) **representation /
ablation** controls (WSST-TCN, REM-Event); (iv) **prior-conference with calibration**
(ICONIP BeatAware, which uses target-subject calibration and is therefore reported
separately, not in the strict-LOSO block); and (v) **reported-only** literature
(VMD-CBNeT, CNN+LSTM on CR_RVS), quoted from their papers under different protocols
with explicit caveats. Numbers from different datasets or private splits are never
placed in a same-protocol comparison.
