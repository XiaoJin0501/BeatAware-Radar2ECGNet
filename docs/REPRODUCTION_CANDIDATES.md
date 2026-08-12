# Reproduction Candidates for the Q1 Comparison

Purpose: identify which published radar-to-ECG models are worth reproducing under
our controlled protocols. This is not a training launch plan. Each candidate must
receive an experiment spec, registry row, and pre-train commit before any run.
See `docs/BASELINE_REPRODUCTION_POLICY.md` for the two-lane rule:
original-protocol reproduction is allowed, but only common-protocol reproduction
can enter the fair main tables.

## Priority A: High Value

### LifWavNet

- Paper/source: Kundu et al., "LifWavNet: Lifting Wavelet-based Network for
  Non-contact ECG Reconstruction from Radar."
- Dataset overlap: CR_RVS (same public Schellenberger dataset).
- Reported protocol: 1024-sample continuous ECG chunks; train/test chunks, not our
  5-fold split.
- Reported numbers: CR_RVS rho 0.926, MRE 0.823, MAEHR 0.63 bpm, MAERMSSD
  13.67 ms.
- Why reproduce: strongest same-public-CR_RVS continuous-waveform candidate and
  conceptually close to our time/frequency morphology framing.
- Fair reproduction target: our CR_RVS 5-fold subject-independent split, same
  radar preprocessing as local CW-STFT/phase pipeline, report both continuous
  1024/8-s metrics and Shape-200 morphology if possible.
- Risk: architecture details may require reimplementation; their train/test split
  is not our split, so reproduced numbers may differ.

### MultiResLinkNet

- Paper/source: Chowdhury et al., "ECG waveform generation from radar signals: A
  deep learning perspective."
- Dataset overlap: CR_RVS/Schellenberger.
- Reported protocol: 5-fold subject-wise CV over Resting/Valsalva/Apnea,
  1024-sample normalized ECG segments.
- Reported numbers: temporal/spectral correlation metrics, e.g., Resting temporal
  CC 66.095 +/- 19.33 and spectral CC 82.439 +/- 18.42; Valsalva temporal CC
  60.136 +/- 21.92; Apnea temporal CC 55.331 +/- 18.95.
- Why reproduce: simple 1D segmentation baseline, same public CR_RVS, plausible
  reviewer expectation.
- Fair reproduction target: our CR_RVS 5-fold subject-independent split with
  same metrics as our REM-Net plus their temporal/spectral CC if easy.
- Risk: their metric scale is percent-like and not identical to our PCC.

## Priority B: Useful but Risky

### Attention-DCGAN

- Paper/source: Jin et al., "Reconstructing ECG-Like Signals From
  Millimeter-Wave Radar Echoes."
- Dataset overlap: MMECG public.
- Reported protocol: 2000 training samples and 100 test samples; no strict LOSO
  found in the audit.
- Reported numbers: PCC 0.9853, RMSE 0.0329 mV.
- Why reproduce: same public MMECG and very high reported number.
- Fair reproduction target: strict 11-fold LOSO only. If the architecture cannot
  be made subject-independent, keep as reported-only Table D context.
- Risk: sample-level protocol may be far easier than LOSO; reproducing it under
  strict LOSO could be a time sink.

### Radar2ECG

- Paper/source: Li et al., "Radar2ECG: Multi-Scale Bottleneck Fusion and
  Cross-modal Semantic Distillation..."
- Dataset overlap: Schellenberger/CR_RVS source data, but resting subset and
  radar heart-sound features.
- Reported protocol: 8:2 sample split on resting state.
- Reported numbers: RMSE 0.0237, MAE 0.0134, COS 0.987, R2 0.879.
- Why reproduce: strong architecture idea and same source dataset.
- Fair reproduction target: only after matching their radar heart-sound extraction
  recipe; otherwise keep as Table D context.
- Risk: task is radar heart-sound-to-ECG, not identical to our CW-STFT morphology
  pipeline.

## Priority C: Related Work Only

- SS-S2SA: custom FMCW dataset; useful for segmentation/attention discussion.
- Cross-modal autoencoder: custom 60 GHz FMCW dataset; useful for event-guided
  autoencoder discussion.
- RSSRNet/AirECG official numbers: private/custom datasets; compare only through
  our adapted reproductions, never through their reported numbers.

## Recommended Next Reproduction Step

If we need one extra reproduced baseline for the Q1 draft, start with
**MultiResLinkNet on CR_RVS** because it is simple and fast. If time allows and
code details are manageable, reproduce **LifWavNet** next. Do not start
Attention-DCGAN unless the experiment spec explicitly enforces strict LOSO.

Recommended sequence:
1. Original-protocol smoke reproduction for MultiResLinkNet to verify the
   implementation.
2. Common-protocol CR_RVS 5-fold reproduction for MultiResLinkNet.
3. Decide LifWavNet only after MultiResLinkNet is audited.
