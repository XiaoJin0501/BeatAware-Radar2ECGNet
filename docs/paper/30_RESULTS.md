# Results (draft prose)

Headline numbers use the multi-seed frozen REM-Net (MMECG 0.885±0.002, CR_RVS
0.886±0.001). A single-seed 150-epoch confirmation run (peak06) gives MMECG
fold-macro 0.888, consistent with the multi-seed result (robust to loss-weight /
epoch choices). Region-fidelity from `docs/MORPHOLOGY_FIDUCIAL_BOOSTER.md`;
efficiency from the model profile. Obey `docs/PAPER_CLAIM_LEDGER.md`.

## 5. Results

### 5.1 MMECG beat-morphology reconstruction

Under 11-fold strict LOSO, REM-Net reconstructs beat morphology with a Shape-200
PCC of **0.885 ± 0.002** across three seeds (Table T1, Fig. F1). This is the
highest among all strict subject-independent methods and a large margin over the
strongest adapted baseline, radarODE-MTL evaluated in its native ECG_shape
morphology space (0.728 ± 0.133), as well as the representation and event-only
controls (WSST-TCN 0.273, REM-Event 0.279) and adapted RSSRNet (0.262) and AirECG
(0.180). The ICONIP BeatAware model uses target-subject calibration and is reported
separately (Block B, 0.671); it is not comparable in the strict-LOSO block. The
across-seed standard deviation of 0.002 shows the result does not depend on a lucky
initialization.

**Region-wise morphology.** To show that the gain is not merely a smooth global
correlation, we report fixed cycle-phase morphology metrics (Table T1b, Fig. F-region).
REM-Net attains QRS-edge PCC 0.962 and ST/T-region PCC 0.787, whereas radarODE-MTL
is competitive at the QRS-edge (0.961) but collapses in the ST/T (0.157) and
diastolic (0.034) regions. This localizes REM-Net's advantage: it preserves broader
beat structure (ST/T phase) rather than only detecting the R/QRS edge. The
low-amplitude diastolic phase remains the weakest region for all methods
(REM-Net 0.510), which we report as a limitation rather than hide behind
cherry-picked overlays; qualitative panels (Fig. F5) use a deterministic
high/median/hard selection rule.

**Ablation.** The component chain WSST-TCN (0.273) → +cardiac-event branch
(REM-Event, 0.279) → +morphology multi-task decoder (REM-Net, 0.885) shows the
morphology decoder is the decisive component. The event-conditioned alignment
module, evaluated with a matched recipe, changed Shape-200 PCC by within ±0.005
with fold-varying sign and was removed from the final model (Section 3.6).

**Efficiency.** REM-Net uses ≈1.79M parameters and ≈1.1 ms per 8-second window on a
single GPU, roughly 2.7× smaller than the ICONIP BeatAware model (≈4.9M), while
achieving substantially higher morphology fidelity (Table T5).

### 5.2 External validation on CR_RVS (CW radar)

The morphology-first principle transfers to a different radar modality. On the
public CR_RVS 24-GHz CW dataset, under verified 5-fold subject-independent
evaluation, REM-Net reaches Shape-200 PCC **0.886 ± 0.001** across three seeds
(Table T3, Fig. F2), far above the local CW-STFT+TCN baseline under the same
protocol (0.550 ± 0.075). Region-wise, REM-Net improves QRS-edge (0.941 vs 0.464)
and ST/T (0.793 vs 0.744) morphology over the local baseline; the diastolic phase
remains weak for both (0.380 vs 0.358), consistent with the lower SNR of
low-amplitude morphology under CW radar.

For external context, published Nature/CR_RVS results are quoted in a separate
reported-only block (VMD-CBNeT 0.856, CNN+LSTM 0.647). Under our subject-independent
5-fold morphology protocol, REM-Net's PCC (0.886) is numerically above the reported
VMD-CBNeT PCC (0.856); because the protocols and PCC spaces differ, we present this
as an as-reported contextual comparison, not a same-protocol head-to-head. We make
no claim of clinical interval (PR/QRS/QT) superiority; our full-window interval
errors are in fact worse and are discussed as a limitation.

### 5.3 Mechanistic analysis of the continuous gap

To explain why full-window continuous ECG remains weak, we probe continuous
assembly from the learned morphology (Fig. F4). Scored against the original ECG, a dense full-window decoder
sits at PCC 0.394 at full coverage. A previously reported oracle-anchor assembly
figure of 0.93–0.98 is withdrawn (audit C, 2026-09-09): it was scored against
per-cycle min-max normalized morphology targets rather than raw ECG, at ~44%
coverage, and reproduces the Shape-200 score to within 0.002. When anchors are predicted from radar, per-window assembly reaches 0.22–0.51
on the easiest fold but does not generalize: on held-out subjects (folds 2–3) it
collapses to ≈0.02. A diagnosis shows that beat detection recall can be recovered by
threshold calibration (the model fires at ≈88% of beats), yet the continuous metric
does not recover, because sub-50 ms localization precision does not transfer across
subjects (matched-beat error 40–55 ms on hard folds vs 16–20 ms on the easy fold).

Morphology is high and predicted-anchor precision fails under LOSO, but the
relative size of the timing and amplitude/baseline contributions is **not yet
measured**; a validation-only decomposition audit is registered. We present this as a bounded
mechanistic discovery and an open direction, not as a solved continuous model.
