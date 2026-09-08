# Q1 Journal Paper Outline

Working title:

**REM-Net: Representation-Event-Morphology Learning for Contactless ECG Beat
Morphology Reconstruction from Radar**

## Abstract Spine

1. Contactless ECG reconstruction from radar is promising but difficult because
   radar observes mechanical chest motion rather than cardiac electrical activity.
2. Existing methods often mix different output formulations and protocols, making
   it unclear whether improvements come from representation, model design, or
   evaluation setup.
3. We propose REM-Net, a supervised representation-aware and event-structured
   morphology reconstruction framework.
4. REM-Net uses radar time-frequency representation, an event auxiliary head, and
   a morphology multi-task decoder to reconstruct normalized beat morphology.
5. On MMECG, REM-Net reaches Shape-200 PCC 0.882 +/- 0.061 under 11-fold strict
   LOSO; on CR_RVS, it reaches 0.881 +/- 0.013 under 5-fold subject-independent
   validation. Multi-seed stability is 0.885 +/- 0.002 and 0.886 +/- 0.001.
6. We separate the unsolved continuous-ECG gap into morphology, beat-timing and
   per-cycle amplitude/baseline components, without ranking them: predicted anchors
   do not generalize at sub-50 ms precision, and a beat-normalized target cannot
   supply per-cycle amplitude by construction.
7. The paper contributes a strong supervised morphology model, cross-modality
   validation, and a fairness-audited evaluation framework that separates
   morphology, continuous reconstruction, adapted baselines, and reported-only
   literature.

## 1. Introduction

Paragraph plan:

1. Why contactless ECG matters: long-term monitoring, comfort, home/ICU/sleep
   scenarios.
2. Why radar-to-ECG is not ordinary regression: radar measures mechanical motion;
   ECG measures electrical activation.
3. Why current comparisons are confusing: sample-level splits, private datasets,
   reported-only numbers, morphology vs continuous windows.
4. Our reframing: supervised beat morphology as the reliable first target, with
   continuous ECG analyzed honestly as a harder downstream assembly problem.
5. Contributions:
   - REM-Net, a supervised representation-event-morphology model.
   - Subject-independent validation on MMECG and CR_RVS.
   - Output-formulation-aware comparison against reproduced and reported baselines.
   - Mechanistic discovery that continuous failure is anchor localization precision,
     not morphology.

Forbidden in Introduction:

- Do not claim continuous ECG SOTA.
- Do not say our method strictly outperforms VMD-CBNeT under identical protocol.
- Do not present Shape-200 as clinical long-term ECG.

## 2. Related Work

Subsections:

1. Radar-based vital sign and ECG reconstruction.
2. Public datasets: MMECG and CR_RVS/Schellenberger.
3. Deep radar-to-ECG models:
   - radarODE/radarODE-MTL.
   - VMD-CBNeT and CR_RVS/Nature line.
   - RSSRNet/AirECG/RF2ESG/LifWavNet/MultiResLinkNet/Radar2ECG.
   - Cross-modal autoencoder and event-guided methods.
4. Evaluation mismatch:
   - private vs public datasets;
   - sample-level vs subject-independent;
   - beat-normalized morphology vs continuous ECG;
   - original-protocol vs common-protocol reproduction.

Use `docs/TABLE_D_REPORTED_LITERATURE_AUDIT.md` and
`docs/BASELINE_REPRODUCTION_POLICY.md`.

## 3. Method

### 3.1 Problem Formulation

- Input: radar representation from FMCW MMECG or CW CR_RVS.
- Output: normalized ECG beat morphology, Shape-200.
- Auxiliary outputs: cardiac event masks.
- Optional/diagnostic output: full-window ECG and anchor-guided assembly.

### 3.2 Radar Representation

- MMECG: WSST/STFT-like time-frequency representation.
- CR_RVS: CW phase/displacement-derived representation.
- Explain representation as a controlled input layer, not the sole novelty.

### 3.3 REM-Net Architecture

Core modules:

1. Representation encoder.
2. Event auxiliary head.
3. Morphology multi-task decoder.

Final supervised model uses alignment disabled. Alignment/anchor analysis moves to
mechanistic section.

### 3.4 Loss Function

Report the final loss:

```text
L = L_full_recon + beta_peak L_event + lambda_shape200 L_shape200
L_full_recon = L1 + 0.1 MR-STFT
L_shape200 = L1 + w_pcc(1 - PCC)
```

Keep it simple: time-domain, frequency-domain, event, morphology. Do not present
loss weighting as the main innovation.

### 3.5 Baseline Reproduction and Fairness Policy

Define:

- reproduced-common-protocol;
- reproduced-original-protocol;
- adapted;
- reported-only;
- different-dataset context.

This section protects the comparison tables.

## 4. Experimental Setup

### 4.1 Datasets

- MMECG public: strict 11-fold LOSO.
- CR_RVS/Schellenberger: subject-independent 5-fold.

### 4.2 Metrics

- Shape-200 PCC, RMSE, MAE, R2.
- Fold-macro aggregation.
- Multi-seed stability.
- Optional continuous diagnostics: full-window PCC, long-OLA, anchor precision.

### 4.3 Baselines

Main fair/adapted:

- radarODE-MTL adapted.
- WSST-TCN.
- REM-Event.
- RSSRNet adapted.
- AirECG adapted.
- CW-STFT+TCN.

Reported-only:

- VMD-CBNeT/CNN+LSTM.
- Jin Attention-DCGAN.
- MultiResLinkNet.
- LifWavNet.
- Radar2ECG.
- BiLSTM-CW.

Optional reproduced baseline:

- MultiResLinkNet on CR_RVS common protocol.

## 5. Results

### 5.1 MMECG Morphology

Main claim: REM-Net achieves the best strict-LOSO Shape-200 morphology result.

Numbers:

- REM-Net: 0.882 +/- 0.061.
- radarODE-MTL adapted: 0.728 +/- 0.133.
- G1 stability: 0.885 +/- 0.002.

### 5.2 CR_RVS External Validation

Main claim: REM-Net transfers to CW radar and strongly exceeds the local
CW-STFT+TCN baseline.

Numbers:

- REM-Net: 0.881 +/- 0.013.
- CW-STFT+TCN: 0.550 +/- 0.075.
- G1 stability: 0.886 +/- 0.001.
- Reported VMD-CBNeT: 0.856, caveated.

### 5.3 Ablation

Show:

- WSST-TCN -> REM-Event -> REM-Net.
- Alignment module does not improve morphology and is removed.
- Optional: event/morphology contributions if available.

### 5.4 Reported Literature Context

Use a tiered table, not a ranking table.

### 5.5 Continuous Gap Discovery

Show:

- morphology strong;
- oracle anchor assembly high;
- predicted anchor precision fails LOSO;
- therefore continuous ECG remains an open bottleneck.

## 6. Discussion

Key points:

1. Why morphology-first is meaningful but limited.
2. Why subject-independent evaluation changes the story.
3. Why many reported high numbers are not directly comparable.
4. What would be needed to solve continuous ECG: robust sub-50 ms anchor precision,
   better radar representation, possible personalization or calibration.
5. Why this still extends ICONIP: from raw-input physiology-aware model to
   representation-event-morphology framework plus external validation and
   fairness-audited comparison.

## 7. Conclusion

Conclude with:

- supervised REM-Net gives strong morphology reconstruction on two radar modalities;
- fair comparison and literature audit clarify the field;
- continuous reconstruction is not solved but its bottleneck is now characterized.

## Figure and Table Checklist

- Fig. 1: REM-Net overview.
- Fig. 2: MMECG high/median/hard qualitative morphology overlays.
- Fig. 3: CR_RVS qualitative or state-level result.
- Fig. 4: continuous gap discovery: oracle vs predicted anchor.
- Table 1: MMECG strict-LOSO morphology.
- Table 2: CR_RVS external validation.
- Table 3: ablation.
- Table 4: reported-literature tiered comparison.
- Table 5: efficiency, if time allows.

## Three-Day Draft Priority

1. Finish Method and Experimental Setup from locked protocol.
2. Produce final tables from T1/T3/Table D.
3. Regenerate qualitative figures with transparent high/median/hard selection.
4. Add optional MultiResLinkNet reproduction only if it will not delay the draft.
