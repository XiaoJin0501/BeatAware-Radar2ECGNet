# Draft Paper Spine: Abstract, Introduction, and Contributions

Status: draft spine for positioning only. This file is grounded in
`docs/PAPER_CLAIM_LEDGER.md`, `docs/T1_MMECG_MORPHOLOGY_LOCKED.md`,
`docs/T3_CRRVS_EXTERNAL_AUDIT.md`, `docs/PAPER_CLAIM_LEDGER.md`, and
`docs/Q1_JOURNAL_MASTER_PLAN.md`. It deliberately separates beat-morphology evidence
from continuous ECG evidence. Do not copy this into a manuscript without a final
claim audit.

## Working Title

REM-Net: Representation-Event-Morphology Learning for Subject-Independent
Radar-Based ECG Beat Reconstruction

## Abstract

Contactless electrocardiogram reconstruction from radar has the potential to
support unobtrusive cardiac monitoring, but the task is often evaluated under
heterogeneous signal formulations that mix beat-level morphology, full-window
waveform reconstruction, and event-timing objectives. This makes it difficult to
determine whether a method improves clinically meaningful cardiac structure or
only benefits from a favorable output representation. In this work, we study
supervised radar-to-ECG reconstruction under a morphology-first formulation and
ask a narrower but critical question: can radar measurements support robust
subject-independent reconstruction of ECG beat morphology across different radar
modalities?

We propose REM-Net, a representation-event-morphology network that combines a
time-frequency radar representation, an explicit cardiac-event branch, and a
multi-task morphology decoder for beat-normalized ECG reconstruction. On the
public MMECG FMCW radar dataset, REM-Net achieves a Shape-200 PCC of
0.885 +/- 0.002 across three random seeds under 11-fold strict leave-one-subject-out
evaluation, outperforming an adapted radarODE-MTL morphology baseline
(0.728 +/- 0.133 across folds). On the public CR_RVS 24-GHz CW radar dataset,
REM-Net reaches 0.886 +/- 0.001 across three seeds under 5-fold
subject-independent evaluation, compared with a local CW-STFT+TCN baseline of
0.550 +/- 0.075 and a reported VMD-CBNeT PCC of 0.856 under a different published
protocol. These results support strong cross-modality beat-morphology
reconstruction, but they do not imply solved long-term continuous ECG recovery.

To clarify this boundary, we further analyze continuous ECG assembly from learned
beat morphology. Scored against the original ECG, a dense full-window decoder reaches only
PCC 0.394; predicted anchors additionally fail to generalize at sub-50 ms
precision under LOSO despite recoverable detection recall. We decompose the
remaining continuous radar-to-ECG gap into morphology, beat-timing and per-cycle
amplitude/baseline components rather than attributing it to beat-anchor localization
precision rather than morphology capacity. The resulting contribution is therefore
twofold: a robust supervised morphology reconstruction framework across FMCW and
CW radar, and an evidence-based characterization of why continuous radar-to-ECG
remains harder than beat-normalized reconstruction.

## Introduction Draft

Electrocardiography is a central tool for cardiac rhythm and morphology
assessment, but conventional ECG monitoring requires electrodes, skin contact,
and user compliance. Radar sensing offers a complementary non-contact modality by
measuring subtle chest-wall motion associated with cardiac and respiratory
activity. If radar signals could be transformed into reliable ECG-like waveforms,
they could support long-term monitoring in beds, vehicles, homes, and other
settings where adhesive electrodes are inconvenient. This promise has motivated
recent work on radar-to-ECG reconstruction, including neural waveform regression,
template-based reconstruction, and physiology-aware multi-task models.

Despite this progress, radar-to-ECG reconstruction remains difficult to compare
across papers. Radar observes mechanical motion, whereas ECG records cardiac
electrical activity; the two are related but not interchangeable. Existing
studies also differ in output formulation: some reconstruct a full temporal ECG
window, some evaluate beat-normalized morphology, and some report fiducial timing
or interval errors under private preprocessing and segmentation protocols. A high
score in one formulation does not automatically transfer to another. In
particular, beat-normalized morphology can be reconstructed accurately even when
the same morphology cannot yet be placed on the correct continuous time axis.
This distinction is central to interpreting radar-to-ECG results honestly.

The present study focuses on supervised, subject-independent ECG beat-morphology
reconstruction. Rather than claiming that a single network solves continuous
radar-to-ECG, we ask whether a representation-aware and event-structured model
can recover reliable ECG morphology from radar across two public radar datasets:
MMECG, a 77-GHz FMCW radar dataset, and CR_RVS, a 24-GHz continuous-wave radar
dataset. This setting is deliberately narrower than long-term continuous ECG
monitoring, but it is scientifically important: if morphology cannot be
reconstructed robustly, continuous ECG reconstruction has little chance of being
clinically useful. If morphology can be reconstructed robustly while continuous
assembly fails, the failure mode can be localized to timing and beat-anchor
precision rather than waveform generation.

We introduce REM-Net, a Representation-Event-Morphology network for radar-based
ECG beat reconstruction. The model uses a cardiac-informative radar
representation as input, extracts temporal features through a lightweight
encoder, predicts cardiac event structure as an auxiliary supervised task, and
decodes beat-normalized ECG morphology through a multi-task morphology head. The
final model intentionally drops the earlier event-conditioned alignment module
after controlled experiments showed that it did not improve morphology
reconstruction. This choice keeps the method aligned with the evidence: the
paper's main model is not an alignment model, but a morphology-first,
beat-structured supervised reconstruction model.

Our evaluation is organized around output-formulation-aware comparisons. On
MMECG, we use 11-fold strict leave-one-subject-out evaluation and report
Shape-200 morphology metrics as fold-macro statistics. REM-Net achieves
0.885 +/- 0.002 PCC across three seeds, with low seed variance and strong
improvement over adapted strict-LOSO baselines, including radarODE-MTL evaluated
in the same morphology space. On CR_RVS, we use verified 5-fold
subject-independent evaluation and show that the same morphology-first principle
transfers to 24-GHz CW radar, reaching 0.886 +/- 0.001 PCC across three seeds.
Published VMD-CBNeT numbers are reported in a separate caveated block because
they come from a different protocol; we do not claim an identical-protocol
comparison.

Finally, we use the learned morphology model to probe why full-window continuous
ECG reconstruction remains weak. A previously reported oracle-anchor assembly result is
withdrawn: it was scored against beat-normalized morphology targets rather than
the original ECG, at partial cycle coverage. When anchors are predicted from radar under LOSO, detection recall can be
recovered by threshold calibration, but sub-50 ms localization precision does not
generalize to harder unseen subjects, causing continuous assembly to collapse.
This negative result is not presented as a failed main model. Instead, it is a
mechanistic discovery that explains the morphology-continuous gap and identifies
robust beat-anchor localization as the next open bottleneck for radar-based
continuous ECG.

## Contributions

1. We propose REM-Net, a supervised representation-event-morphology framework for
   radar-based ECG beat reconstruction. The final evidence-supported model uses a
   radar time-frequency representation, a cardiac-event auxiliary branch, and a
   morphology multi-task decoder; the earlier alignment module is explicitly
   ablated and removed from the main model.

2. We demonstrate strong subject-independent beat-morphology reconstruction on
   the public MMECG FMCW radar dataset. Under 11-fold strict LOSO evaluation,
   REM-Net reaches Shape-200 PCC 0.885 +/- 0.002 across three seeds, improving
   over adapted morphology baselines including radarODE-MTL.

3. We validate the morphology-first framework on the public CR_RVS 24-GHz CW
   radar dataset. Under verified 5-fold subject-independent evaluation, REM-Net
   reaches Shape-200 PCC 0.886 +/- 0.001 across three seeds, substantially above
   the local CW-STFT+TCN baseline and numerically above reported VMD-CBNeT PCC
   under a different, explicitly caveated protocol.

4. We provide an output-formulation-aware evidence audit for radar-to-ECG
   reconstruction. Fair strict-subject-independent results, calibrated prior
   results, adapted baselines, and reported-only literature numbers are separated
   into explicit tiers, reducing the risk of unfair SOTA claims.

5. We separate the components that stand between beat morphology and continuous
   ECG: morphology fidelity, beat-anchor timing, and per-cycle amplitude/baseline,
   the last of which a beat-normalized target cannot supply by construction.
   Predicted-anchor assembly additionally fails under LOSO when sub-50 ms
   precision does not generalize. This frames continuous radar-to-ECG as
   an open problem rather than a solved waveform-decoding
   problem.

## Claim Boundaries

This paper may claim state-of-the-art or leading performance only for
subject-independent beat-morphology reconstruction under the stated Shape-200
protocols. It must not claim continuous ECG SOTA, identical-protocol superiority
over VMD-CBNeT, clinical interval superiority, or faithful reproduction of
radarODE-MTL long-term assembly. The continuous ECG analysis is a discovery and
limitation section, not the headline model claim.
