# RQ1 Ablation and Multi-Metric Plan

This document locks the next supervised RQ1 evaluation step after advisor feedback:
the paper must not be driven by PCC alone. Claims about REM-Net must be supported
by controlled ablations and by metrics that reflect morphology, beat timing, and
monitoring reliability.

## 1. Motivation

The current main result is strong Shape-200 PCC on MMECG and CR_RVS. This is useful
but insufficient for a Q1 paper if it is reported as the only performance target.
Radar-to-ECG reconstruction is not only a curve-fitting problem; it must preserve:

1. beat morphology,
2. event timing,
3. interval consistency,
4. monitoring usability,
5. robustness and deployability.

The earlier conference evaluation was scientifically sensible because it reported
PCC/RMSE together with QMR, R-F1, and fiducial timing errors. RQ1 should keep this
spirit, while clearly separating beat-normalized morphology from full-window or
long-term continuous ECG.

## 2. Metric Ladder

### Level 1: Morphology fidelity

Used for Shape-200 and beat-normalized reconstruction.

| Metric | Why it matters | Report rule |
|---|---|---|
| PCC | Shape similarity, insensitive to scale | Fold-macro mean plus std |
| RMSE | Amplitude error in the declared normalization space | Same aggregation as PCC |
| MAE | Robust absolute error | Same aggregation as PCC |
| R2 | Explained variance | Same aggregation as PCC |
| Region PCC/RMSE | QRS, ST/T, and non-QRS morphology reliability | Report when segmentation is reliable |

PCC remains important, but a model cannot be claimed better if the gain is only a
global correlation gain and region/interval metrics degrade.

### Level 2: Event and timing reliability

Inherited from the earlier evaluation logic and computed on full-window outputs
or event-head outputs when available.

| Metric | Definition |
|---|---|
| R-F1@150/100/50 ms | R-peak detection F1 under progressively stricter tolerance windows |
| RR error | Absolute RR-interval error on matched consecutive beats |
| HR error | Window-level heart-rate error from matched or valid detected beats |
| Q-Q/R-R/S-S/T-T error | Fiducial timing error in ms when NeuroKit2 delineation coverage is sufficient |

R-F1@150 ms is the historical monitoring-friendly number. R-F1@100/50 ms reveals
whether the result remains precise enough for continuous assembly and clinical
timing claims.

### Level 3: Monitoring usability

| Metric | Definition |
|---|---|
| QMR | Qualified Monitoring Rate: reference has at least two R peaks, prediction has at least two R peaks, and at least one valid RR match exists |
| Coverage | Fraction of windows with valid matched-beat or delineation metrics |
| Failure rate | Fraction of windows with missing, duplicate, or unusable peaks |

QMR and coverage are not decoration. They prevent a method from reporting good
waveform scores on only easy or successfully delineated windows.

### Level 4: Stress and efficiency

| Metric | Definition |
|---|---|
| Noise degradation | Shape-200 PCC under controlled input-representation Gaussian noise |
| Parameters | Trainable parameters |
| Latency | Single-window inference time, hardware stated |
| FLOPs | Optional if fast and robust to compute |

The noise test is an input-corruption stress test only. It must not be described
as real body-motion robustness.

## 3. Required Ablation Before Causal Claims

No causal statement about representation, event supervision, morphology decoding,
or loss design is allowed until the corresponding controlled ablation is run and
audited.

### A. Main architecture ablation: MMECG gate

Run first on a representative 3-fold gate, then promote to all 11 folds only if
the pattern is stable. Recommended gate folds: one easy, one medium, one hard fold
selected from the locked fold-level distribution, not from test cherry-picking.

| Arm | Model | Purpose | Claim enabled only if it wins |
|---|---|---|---|
| A0 | WSST-TCN | Representation-only temporal decoder | WSST alone is insufficient |
| A1 | REM-Event | Encoder plus event branch | Event structure helps timing/usability |
| A2 | REM-Net, beta_peak=0 | Same architecture without event supervision | Event loss contributes beyond capacity |
| A3 | REM-Net, shape PCC off | Shape auxiliary loss without PCC component | PCC term adds morphology fidelity |
| A4 | REM-Net final | Event-guided morphology decoder | Final model balances morphology and timing |
| A5 | REM-Net + alignment | Optional, already mostly resolved | Alignment retained only if it improves multi-metric results |

Promotion rule: A4 must improve morphology metrics over A0/A1 and must not degrade
R-F1/QMR/RR relative to the simpler event-aware arms. If A4 only improves Shape-200
PCC while timing or monitoring reliability worsens, the paper may claim morphology
improvement but not clinical monitoring improvement.

### B. Loss ablation: minimal and interpretable

| Arm | Loss setting | Purpose |
|---|---|---|
| L0 | Full-window reconstruction only | Baseline supervised regression |
| L1 | L0 + event loss | Test beat-event supervision |
| L2 | L1 + Shape-200 L1 | Test beat-normalized morphology supervision |
| L3 | L2 + Shape-200 PCC | Test correlation term in morphology space |

Do not make the loss function the main novelty. It is an explanation of why the
final supervised model is stable.

### C. CR_RVS external ablation

CR_RVS is primarily external validation, not the place to tune the model. Run only
the minimum mirror needed for claim hygiene:

| Arm | Purpose |
|---|---|
| C0: CW-STFT+TCN | Local CW baseline |
| C1: REM-Net final | Cross-modality morphology validation |
| C2: REM-Event or beta_peak=0 | Optional if the manuscript claims event supervision transfers to CW |

Do not claim CR_RVS event localization strength unless R-F1/coverage support it.
Current evidence says morphology transfers but beat localization does not.

## 4. Go / No-Go Rules

### Claims that require ablation support

| Claim | Required evidence |
|---|---|
| Event structure improves reconstruction | A1/A4 better than A0 on at least one timing/usability metric and no morphology collapse |
| Morphology decoder is useful | A4 better than A1 on Shape-200 PCC/RMSE/MAE and region metrics |
| Shape loss matters | L2/L3 better than L0/L1 on morphology, not only on global PCC |
| PCC term is justified | L3 better than L2 without harming QMR/R-F1 |
| Alignment is useful | A5 improves multi-metric results beyond A4 across representative folds |

### Forbidden shortcuts

- Do not infer module value from the final model alone.
- Do not use test metrics to select ablation checkpoints.
- Do not present Shape-200 PCC as evidence of continuous ECG reliability.
- Do not merge reported-only literature rows into same-protocol ablation tables.
- Do not claim clinical interval superiority unless delineation coverage and
  fiducial error audits pass.

## 5. Recommended Execution Order

1. Eval-only metric expansion on frozen REM-Net outputs:
   PCC/RMSE/MAE/R2, R-F1@150/100/50, RR/HR, QMR, region morphology, and coverage.
2. MMECG 3-fold ablation gate:
   A0/A1/A2/A3/A4, with the same training epochs, validation-best checkpoint rule,
   preprocessing, seed, and output directory convention.
3. Promote only successful arms to 11-fold:
   keep failed/probe outputs as non-publishable and clean them after documentation.
4. CR_RVS mirror:
   C0/C1 final table, C2 only if an event-transfer claim remains in the manuscript.
5. Update the paper claim ledger:
   every claim must point to the ablation row and metric group that supports it.

## 6. Manuscript Consequence

The Q1 story should shift from:

> REM-Net gets high PCC.

to:

> REM-Net improves beat morphology under subject-independent FMCW and CW radar
> protocols, and ablation plus multi-metric evaluation show which parts improve
> morphology, which parts affect event timing, and where continuous ECG still fails.

This is a stronger and more defensible journal narrative because it separates
performance, mechanism, and limitation instead of compressing them into a single
correlation score.
