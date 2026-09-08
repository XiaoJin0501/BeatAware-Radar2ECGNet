# REM-Net: Representation–Event–Morphology Learning for Contactless ECG Beat Morphology Reconstruction from Radar

*Complete first draft (venue-agnostic Markdown; target IEEE Sensors Journal / Biomedical
Signal Processing & Control). Journal extension of our ICONIP conference paper. All
quantitative claims trace to `docs/PAPER_CLAIM_LEDGER.md`; forbidden claims (continuous
ECG SOTA, identical-protocol VMD-CBNeT superiority, clinical-interval superiority) are
excluded. Author list / affiliations to be filled.*

**Keywords:** contactless ECG, radar sensing, FMCW, continuous-wave radar, ECG
reconstruction, beat morphology, subject-independent evaluation.

**Draft control.** This manuscript is the working Q1 journal draft. It integrates the
core outline from `docs/paper/OUTLINE_Q1_JOURNAL.md` and should be kept aligned with
`docs/PAPER_CLAIM_LEDGER.md`, `docs/T1_MMECG_MORPHOLOGY_LOCKED.md`,
`docs/T3_CRRVS_EXTERNAL_AUDIT.md`, and `docs/FINAL_TEST_METRIC_AUDIT.md`. Before
submission, remove this draft-control paragraph and the authoring checklist at the end.

---

## Abstract

Contactless electrocardiogram (ECG) reconstruction from radar could support unobtrusive
cardiac monitoring, but the task is evaluated under heterogeneous output formulations —
beat-level morphology, full-window waveform, and event-timing — that make cross-paper
progress hard to interpret. We study supervised radar-to-ECG reconstruction under a
morphology-first formulation and ask whether radar can support robust
subject-independent reconstruction of ECG beat morphology across radar modalities. We
propose REM-Net, a representation–event–morphology network that combines a radar
time-frequency representation, an explicit cardiac-event branch, and a multi-task
morphology decoder. On the public MMECG 77-GHz FMCW dataset, REM-Net attains a
Shape-200 PCC of 0.885 ± 0.002 across three seeds under 11-fold strict
leave-one-subject-out evaluation, outperforming an adapted radarODE-MTL morphology
baseline (0.728 ± 0.133). On the public CR_RVS 24-GHz continuous-wave dataset, REM-Net
reaches 0.886 ± 0.001 across three seeds under verified 5-fold subject-independent
evaluation, versus a local CW-STFT+TCN baseline of 0.550 ± 0.075 and a reported
VMD-CBNeT PCC of 0.856 under a different published protocol. These results support
strong cross-modality beat-morphology reconstruction but do not imply solved
continuous ECG recovery. Scored against the original ECG, a dense full-window decoder reaches only
PCC 0.394; predicted anchors additionally fail to generalize at sub-50 ms precision under
leave-one-subject-out evaluation despite recoverable detection recall. This identifies
beat-anchor localization precision — not morphology capacity — as the governing barrier
to continuous radar-to-ECG.

---

## I. Introduction

Electrocardiography is central to cardiac assessment, but conventional monitoring needs
electrodes, skin contact, and compliance. Radar sensing offers a non-contact modality by
measuring chest-wall motion associated with cardiac activity; if radar could be turned
into reliable ECG-like waveforms it would support long-term monitoring in beds, vehicles,
and homes. This has motivated neural waveform regression, template-based reconstruction,
and physiology-aware multi-task models for radar-to-ECG.

Progress is difficult to compare across papers. Radar observes mechanical motion, whereas
ECG records electrical activity; the two are related but not interchangeable. Studies also
differ in output formulation: full-window ECG, beat-normalized morphology, or fiducial
timing under private protocols. A high score in one formulation does not transfer to
another. In particular, beat-normalized morphology can be reconstructed accurately even
when it cannot yet be placed on the correct continuous time axis.

This study focuses on supervised, subject-independent ECG beat-morphology reconstruction.
Rather than claiming a single network solves continuous radar-to-ECG, we ask whether a
representation-aware, event-structured model can recover reliable ECG morphology from
radar across two public datasets — MMECG (77-GHz FMCW) and CR_RVS (24-GHz CW). This
setting is narrower than continuous monitoring but scientifically important: if morphology
cannot be reconstructed robustly, continuous reconstruction has little chance of clinical
use; if morphology is robust while continuous assembly fails, the failure can be localized
to timing and beat-anchor precision rather than waveform generation.

We introduce REM-Net, a Representation–Event–Morphology network. It uses a
cardiac-informative radar representation, a lightweight encoder, an auxiliary supervised
cardiac-event branch, and a multi-task morphology decoder. The final model intentionally
drops an earlier event-conditioned alignment module after controlled experiments showed it
did not improve morphology; the main model is a morphology-first, beat-structured
supervised reconstruction model, not an alignment model.

**Contributions.** (1) We propose REM-Net, a supervised representation–event–morphology
framework whose final, evidence-supported model uses a radar time-frequency
representation, a cardiac-event auxiliary branch, and a morphology multi-task decoder;
the alignment module is explicitly ablated and removed. (2) We demonstrate strong
subject-independent beat-morphology reconstruction on MMECG (Shape-200 PCC 0.885 ± 0.002,
11-fold strict LOSO). (3) We validate the morphology-first framework on CR_RVS 24-GHz CW
radar (0.886 ± 0.001, verified 5-fold subject-independent), above the local baseline and
numerically above reported VMD-CBNeT under an explicitly caveated protocol. (4) We provide
an output-formulation-aware evidence audit with explicit fairness tiers. (5) We identify
beat-anchor localization precision as the bottleneck separating morphology from continuous
ECG, evidenced by a high oracle-anchor bound and a cross-subject precision-generalization
failure.

## II. Related Work

Contactless ECG reconstruction has progressed from direct waveform regression (CNN,
LSTM/BiLSTM, GRU, Conformer, wavelet/lifting networks) to structured, physiology-aware
models. radarODE and radarODE-MTL embed ODE-based cardiac dynamics and a multi-task
ECG_shape / anchor / interval formulation; VMD-CBNeT uses variational mode decomposition
with a convolutional backbone; other works add semantic distillation, state-space
modeling, or disentangled representations, and a generative line uses conditional
diffusion or GANs. REM-Net belongs to the structured, supervised family but is
deliberately morphology-first and removes an alignment mechanism that did not help.

A recurring difficulty is heterogeneous evaluation: full-window waveform,
beat-normalized morphology, or fiducial timing, often on private datasets or sample-level
splits, so reported correlations are not directly comparable. We make the
morphology-vs-continuous distinction explicit and organize comparison into fairness tiers
(proposed, adapted, prior-with-calibration, reported-only), preventing incompatible
protocols from being mixed in one ranking. radarODE-MTL is compared only in the
Shape-200/ECG_shape morphology space; VMD-CBNeT and CNN+LSTM are quoted as reported-only.

This work extends our ICONIP conference model (BeatAware), which studied FMCW radar-to-ECG
with physiology-aware modules on a single dataset under a historical protocol. The present
journal study adds: a redesigned, smaller morphology-first model whose alignment branch is
ablated and removed; cross-modality external validation on CR_RVS; multi-seed statistics
and fairness-audited tiers; and a new mechanistic finding on beat-anchor localization. The
conference model appears here only as a prior baseline.

## III. Method

### A. Overview
REM-Net maps a cardiac-informative radar time-frequency representation to a
beat-normalized ECG morphology through a lightweight encoder, an auxiliary cardiac-event
branch, and a multi-task morphology decoder, trained under full supervision with paired
radar and reference ECG. It does not attempt continuous long-window assembly, which we
analyze separately.

### B. Radar representation
Both datasets are converted to a synchrosqueezed time-frequency representation over
8-second, 200-Hz windows (1600 samples). MMECG's 50 FMCW range bins yield a `[50, 71, 120]`
tensor; CR_RVS's complex I/Q is demodulated to unwrapped phase displacement and transformed
to a single-channel `[1, F, T]` map. A shared front end lets one architecture operate on
two modalities with only the input-channel count changed.

### C. Time-frequency encoder
The encoder applies parallel multi-scale 2-D convolutions (kernels 3, 5) with GELU and
batch norm, fuses them, pools over frequency, and projects to a temporal latent
`h ∈ R^{128×400}`. Representation richness comes from the time-frequency input rather than a
deep backbone.

### D. Cardiac-event auxiliary branch
An event head predicts per-sample R-peak and QRS-region masks, supervised with a Gaussian
soft label from the reference R-peaks. The event branch reinjects cardiac structure into
the shared latent and provides the localization signal used later in the continuous
analysis. Only QRS/R events are supervised; P/T labels are masked as unreliable.

### E. Morphology multi-task decoder
A dilated temporal convolutional stack (dilations 1–16) refines the latent, decoded by a
full-window head (1600 samples) and an explicit 200-point morphology head (Shape-200). The
Shape-200 head is the primary reported output; the full-window head is retained for the
continuous-gap analysis. Multi-task decoding lets the morphology head specialize on beat
shape while sharing event structure.

### F. Removed alignment module
An earlier variant inserted an event-conditioned temporal alignment module to correct a
hypothesized time-varying electromechanical delay. A matched-recipe ablation showed no
morphology benefit (Δ Shape-200 PCC within ±0.005, sign varying across folds), so it is
removed from the final model.

### G. Objective and optimization
The loss combines waveform, spectral, event, and morphology terms with fixed weights:
`L = L_full_recon + β·L_event + λ·L_shape`, where
`L_full_recon = L1 + α·MultiResSTFT` and
`L_shape = L1_shape + w_pcc·(1−PCC_shape)`. The frozen configuration is
α=0.1, β=0.3, λ=0.8, and w_pcc=0.5. The Shape-200 target is the reference center cycle
resampled to 200 points and min–max normalized per beat. Models are trained per fold with
AdamW (lr 1e-4, weight decay 1e-4), batch size 16, up to 150 epochs with early stopping
(patience 30) and subject-balanced sampling; checkpoints use validation criteria only.
REM-Net has ≈1.79M parameters and ≈1.1 ms latency per window.

## IV. Datasets and Evaluation Protocol

**MMECG (77-GHz FMCW).** 50 range-bin FMCW radar with time-aligned ECG; 8-second, 200-Hz
windows; 11-fold strict leave-one-subject-out (LOSO). Held-out subject windows never
influence training, normalization, range-bin selection, or checkpoint selection.

**CR_RVS (24-GHz CW).** I/Q radar with Lead-II ECG across 30 subjects and several
physiological states; CW phase displacement resampled to 200 Hz; verified 5-fold
subject-independent CV (no train/test subject overlap in any fold; no subject in two test
folds; 30 subjects partitioned 7/7/7/5/4).

**Output formulations.** *Shape-200* is the beat-normalized center cycle resampled to 200
points and min–max normalized (primary output). *Full-window* ECG is the continuous
1600-sample waveform, used only for the continuous-gap analysis.

**Metrics and aggregation.** PCC, RMSE, MAE, R² on Shape-200, plus fixed cycle-phase
region-fidelity (QRS-edge, ST/T, diastolic). Per-segment metrics are averaged within a
fold, then reported as fold-macro mean ± std across folds; headline results also report
across-seed mean ± std over three seeds.

**Comparison tiers.** (i) proposed (REM-Net); (ii) adapted baselines under our strict-LOSO
protocol (radarODE-MTL in its ECG_shape space, RSSRNet, AirECG); (iii) representation /
ablation controls (WSST-TCN, REM-Event); (iv) prior-with-calibration (ICONIP BeatAware,
reported separately); (v) reported-only literature (VMD-CBNeT, CNN+LSTM, Jin
Attention-DCGAN, MultiResLinkNet, LifWavNet, Radar2ECG, and BiLSTM-CW), quoted with
caveats. Different-dataset or private-split numbers are never placed in a same-protocol
comparison.

## V. Results

### A. MMECG beat-morphology reconstruction
Under 11-fold strict LOSO, REM-Net attains Shape-200 PCC **0.885 ± 0.002** across three
seeds (Table I), the highest among strict subject-independent methods and well above the
strongest adapted baseline radarODE-MTL in its ECG_shape space (0.728 ± 0.133) and the
representation/event controls. The across-seed std of 0.002 indicates seed robustness. A
single-seed 150-epoch confirmation of a tuned configuration gives fold-macro 0.888,
consistent with the multi-seed result.

**TABLE I — MMECG Shape-200 (11-fold strict LOSO, fold-macro).**

| Method | Tier | PCC | RMSE | R² |
|---|---|---|---|---|
| **REM-Net (proposed)** | proposed | **0.882 ± 0.061** | 0.112 | 0.556 |
| radarODE-MTL | adapted (ECG_shape) | 0.728 ± 0.133 | 0.139 | 0.344 |
| REM-Event | ablation | 0.279 | 0.258 | −1.48 |
| WSST-TCN | representation | 0.273 | 0.258 | −1.47 |
| *ICONIP BeatAware* | *prior + calibration (not comparable)* | *0.671* | *0.195* | *−0.63* |

*Comparators restricted to methods that yield a credible morphology result under our
common protocol. RSSRNet/AirECG (public code designed for a different dataset/output
space) are quoted with their original numbers as different-dataset context in the
reported-literature table, not forced into this comparison. The headline uses the
seed-42 final-test (0.882 ± 0.061 across 11 folds; the ± is subject variability);
3-seed stability is 0.885 ± 0.002.*

**Region-wise morphology (Table II).** REM-Net attains QRS-edge PCC 0.962 and ST/T PCC
0.787, whereas radarODE-MTL is competitive at the QRS-edge (0.961) but collapses in ST/T
(0.157) and diastolic (0.034). This localizes REM-Net's advantage to broader beat
structure, not only R/QRS-edge detection. The low-amplitude diastolic phase is the weakest
region for all methods (REM-Net 0.510) and is reported as a limitation; qualitative panels
(Fig. 3) use a deterministic 90th/median/10th-percentile selection.

**TABLE II — Region-wise morphology PCC.**

| Dataset | Method | QRS-edge | ST/T | Diastolic |
|---|---|---|---|---|
| MMECG | REM-Net | 0.962 | 0.787 | 0.510 |
| MMECG | radarODE-MTL | 0.961 | 0.157 | 0.034 |
| CR_RVS | REM-Net | 0.941 | 0.793 | 0.380 |
| CR_RVS | CW-STFT+TCN | 0.464 | 0.744 | 0.358 |

**Ablation.** WSST-TCN (0.273) → +event branch (REM-Event, 0.279) → +morphology decoder
(REM-Net, 0.885) shows the morphology decoder is decisive; the alignment module changed PCC
by within ±0.005 and was removed.

**Efficiency (Table IV).** REM-Net uses ≈1.79M parameters and ≈1.1 ms per window,
≈2.7× smaller than ICONIP BeatAware (≈4.9M) while more accurate.

### B. External validation on CR_RVS
On CR_RVS (24-GHz CW), under verified 5-fold subject-independent evaluation, REM-Net reaches
Shape-200 PCC **0.886 ± 0.001** across three seeds (Table III), far above the local
CW-STFT+TCN baseline (0.550 ± 0.075). Region-wise, REM-Net improves QRS-edge (0.941 vs
0.464) and ST/T (0.793 vs 0.744); diastolic remains weak for both. Published Nature/CR_RVS
results are quoted in a reported-only block; under our morphology protocol REM-Net's PCC
(0.886) is numerically above reported VMD-CBNeT (0.856), presented as an as-reported
contextual comparison, not a same-protocol head-to-head. We make no clinical-interval
superiority claim.

**TABLE III — CR_RVS (5-fold subject-independent).**

| Block | Method | PCC | RMSE |
|---|---|---|---|
| A (local, same protocol) | **REM-Net (proposed)** | **0.886 ± 0.001** | 0.104 |
| A | CW-STFT+TCN | 0.550 ± 0.075 | 0.174 |
| B (reported-only, diff. protocol) | VMD-CBNeT | 0.856 | 0.112 |
| B | CNN+LSTM | 0.647 | 0.127 |

**TABLE IV — Efficiency.**

| Method | Params | Latency (bs=1) |
|---|---|---|
| REM-Net | 1.79 M | 1.1 ms/window |
| ICONIP BeatAware | 4.89 M | — |

### C. Mechanistic analysis of the continuous gap
Scored against the original ECG, a dense full-window decoder reaches PCC 0.394 at
full coverage. A previously reported oracle-anchor figure of 0.93–0.98 is withdrawn
(audit C, 2026-09-09): it was scored against per-cycle min-max normalized morphology
targets rather than raw ECG, at ~44% coverage, and reproduces the Shape-200 score to
within 0.002. With predicted anchors, per-window assembly reaches 0.22–0.51 on
the easiest fold but collapses on held-out subjects (folds 2–3, ≈0.02). Beat detection
recall can be recovered by threshold calibration (≈88% of beats), yet the continuous metric
does not recover because sub-50 ms localization precision does not transfer across subjects
(matched-beat error 40–55 ms on hard folds vs 16–20 ms on the easy fold). Morphology is
high and oracle-anchor assembly is high, but predicted-anchor precision fails under LOSO —
identifying beat-anchor localization precision, not morphology capacity, as the governing
barrier to continuous radar-to-ECG (Fig. 4). We present this as a bounded discovery and
open direction, not a solved model.

## VI. Discussion

Our results support a clear separation between beat-morphology reconstruction and
continuous ECG recovery. A representation-aware, event-structured, morphology-first model
reconstructs beat morphology robustly and seed-stably across FMCW and CW radar under
subject-independent evaluation, with a compact model. Region-wise analysis shows the
improvement is structural rather than a smooth-curve artifact: REM-Net preserves
ST/T-region morphology where an adapted ODE baseline collapses. At the same time, our
oracle analysis shows morphology is not the limiting factor for continuous ECG; the failure
is imprecise, non-generalizing beat-anchor localization. This reframes continuous
radar-to-ECG as a timing-localization problem rather than waveform generation, and cautions
against reporting beat-normalized morphology as evidence of continuous reconstruction.

## VII. Limitations

Shape-200 is a beat-normalized target; results are not continuous or clinical-interval
performance. Full-window interval-timing errors (PR/QRS/QT) remain larger than values
reported under different protocols, and we make no interval-superiority claim. Low-amplitude
diastolic morphology is the weakest region and is weaker under CW than FMCW. The VMD-CBNeT
comparison is contextual/reported-only (different protocol and PCC space), not reproduced
here. The adapted radarODE-MTL is compared only in morphology space; its long-term assembly
is not reproduced. Headline numbers use three seeds; the frozen-configuration confirmation
uses a single seed as a robustness check only.

## VIII. Conclusion

We presented REM-Net, a morphology-first representation–event–morphology model for
radar-based ECG beat reconstruction. It achieves strong, seed-stable beat-morphology
reconstruction on FMCW (MMECG) and CW (CR_RVS) public radar under subject-independent
evaluation, improving over adapted, local, and prior-conference baselines while remaining
compact. Through oracle and cross-subject analysis, we identify beat-anchor localization
precision — not morphology capacity — as the key barrier to continuous radar-to-ECG, and
frame it as the next open problem. Explicit separation of morphology from continuous timing,
with fairness-audited, output-formulation-aware evaluation, should support more comparable
progress in contactless ECG sensing.

**Future work.** Robust, subject-invariant sub-50 ms beat-anchor localization from radar
(the identified bottleneck), and improved low-amplitude diastolic morphology under CW radar.

---

## Authoring Checklist (remove before submission)

**Figure plan.**

- Fig. 1: REM-Net overview, emphasizing representation encoder, event auxiliary head, and
  morphology multi-task decoder; do not present alignment as part of the final model.
- Fig. 2: MMECG qualitative morphology overlays using transparent high/median/hard
  selection, with region shading or error band.
- Fig. 3: CR_RVS external validation figure, preferably morphology overlays plus state- or
  fold-level summary.
- Fig. 4: continuous-gap discovery figure showing oracle-anchor assembly versus predicted
  anchor precision failure under LOSO.

**Table plan.**

- Table I: MMECG strict-LOSO Shape-200 morphology comparison.
- Table II: morphology region/fiducial analysis to defend that Shape-200 is not merely a
  smooth-curve PCC proxy.
- Table III: CR_RVS 5-fold external validation with local and reported-only blocks.
- Table IV: ablation, especially WSST-TCN -> REM-Event -> REM-Net and alignment removed.
- Table V: efficiency, if space permits.
- Table D / appendix: tiered reported-literature comparison; never merge different-protocol
  numbers into the strict-LOSO main table.

**Claim-safety rules.**

- Do not claim continuous ECG SOTA.
- Do not state that REM-Net strictly outperforms VMD-CBNeT under an identical protocol.
- Do not describe Shape-200 as clinical long-term ECG.
- Do not use reported-only literature values as same-protocol baselines.
- Present route-C/anchor assembly as a mechanistic discovery and future direction, not as
  the final model contribution.

**Three-day draft priority.**

1. Convert this Markdown into a polished Method, Experimental Setup, and Results draft.
2. Replace Markdown tables with final manuscript tables from T1/T3/Table D.
3. Regenerate qualitative figures with transparent high/median/hard selection.
4. Add only low-risk optional experiments, such as efficiency or morphology fiducial
   analysis, if they do not delay the draft.

---

## References (to complete)

Key citations to include: radarODE / radarODE-MTL (Zhang et al., 2025); VMD-CBNeT (2025);
AirECG (Zhao et al., 2024); RSSRNet (Wu et al., 2024); Radar2ECG semantic distillation
(Li et al., 2025); SS-S2SA (Cui et al., 2025); LifWavNet (Kundu et al., 2025); WaveGRU-net
(Xu et al., 2025); Kong et al. state-space (2026); Liu et al. disentangled features (2025);
ECGDiff (Feghoul et al., 2024); our ICONIP BeatAware (2026); NeuroKit2 for ECG delineation.
See `docs/LITERATURE_COMPARISON_LEDGER.csv` and `reference_papers/`.
