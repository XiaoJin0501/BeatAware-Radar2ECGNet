# CR_RVS recent methods and next experiment direction

Reviewed 2026-09-08 from the two local primary PDFs in reference_papers.
No new training or reproduction results are reported here.

## Primary sources

- Chen et al., A Non-Contact ECG Reconstruction Method Based on FMCW Radar and Adaptive VMD-Assisted Deep Learning. DOI: 10.1109/OJSP.2026.3726823. Local accepted author manuscript, Sections IV.B-D and Table II.
- Xiao et al., Millimeter-Wave Radar-Based ECG Reconstruction Using Respiratory Harmonic Suppression and CA-WTBNet. DOI: 10.3390/bioengineering13070731. Sections 5.1-5.5, Tables 1 and 3-8.

Both identify the Schellenberger public dataset used by this project as CR_RVS. This is the Scientific Data source dataset, rather than an unspecified "Nature dataset". The public-source evaluation must be distinguished from Chen et al.'s separate hospital FMCW acquisition.

## Reported protocol and results

| Item | Chen et al. | Xiao et al. |
|---|---|---|
| Subject partition | 20 train / 5 validation / 5 test | 25 development / 5 test; development segments split 85/15 |
| Input sequence | 2048 samples | 2048 samples at 200 Hz, non-overlapping; discard initial 4096 samples |
| Conditions | Dataset conditions described; exact condition subset needs confirmation | Resting only |
| Frontend | ACO-optimized VMD | MODWT, harmonic suppression, cardiac-frequency protection, five input channels |
| Decoder | CNN, attention, recurrent decoding | Residual convolutions, window Transformers, attention, BiLSTM, skip paths |
| Main waveform PCC | 0.956 | 0.9641 |
| Error metric | RMSE 0.16, table labels mV | NRMSE 0.0458; denominator is GT window amplitude range |
| Network parameters | Not established in this audit | 1,033,833 (Table 1) |

Neither paper can be dismissed as using train/test subject overlap: both explicitly state subject-independent testing. Exact subject IDs, implementation availability, preprocessing parameters and checkpoint policies still need a reproduction audit. Reported values must remain contextual until common-protocol reproduction.

CA-WTBNet detects waveform R peaks after resampling 200 Hz outputs to 2000 Hz. Its F1 tolerance is 50 ms; timing error is conditional on matched peaks. Reported F1 is 0.9956 and timing error 3.13 ms. These are not directly interchangeable with this project's auxiliary-event-head measurements. Interpolation does not add measured temporal information.

## What the ablations actually support

- Chen Table II: no VMD / fixed VMD / ACO-VMD PCC = 0.856 / 0.901 / 0.943. The ablation PCC 0.943 differs from the overall 0.956; preserve this distinction.
- Xiao Table 7: full / raw only / no suppression / no cardiac bandpass channel PCC = 0.9641 / 0.9530 / 0.9583 / 0.9101. The largest listed frontend removal effect concerns the bandpass channel, not suppression alone.
- Xiao Table 8: removing PCC loss gives PCC 0.9539; removing frequency loss gives PCC 0.9638 but NRMSE increases from 0.0458 to 0.0517. Evaluate error metrics as well as correlation.
- CA-WTBNet is compact in parameter count but contains multiple architectural mechanisms. Describe it as compact, not as a trivial network.

## Implications for REM-Net

The current three-fold A2/A4 ablation shows a metric tradeoff: A4 PCC 0.8350 versus A2 0.8279, but A2 has lower RMSE/MAE and higher R2. This is compatible with competing supervision objectives; it does not establish gradient interference or its cause.

The large A1-to-A2 morphology difference changes both decoder/output formulation and supervision. It does not isolate a single architectural mechanism. A normalized beat objective also does not by itself establish absolute-time waveform reconstruction ability.

Published continuous-waveform results require narrowing manuscript claims: localization is a bottleneck in the evaluated REM-Net/assembly pipeline, not an established field-wide information limit. Separate within-dataset training on two modalities from zero-shot cross-dataset transfer.

## Ordered next experiments (planned, not launched)

1. Audit best.pt selection, output normalization, identical evaluation targets, and validation policies across A2/A4. Examine validation amplitude bias and local-region errors. Do not select checkpoint policy from test rankings.
2. Build a CR_RVS Resting development benchmark with subject-disjoint train/validation and reserved test subjects. Existing inspected folds are development evidence; do not describe them as an untouched final test.
3. Establish a compact 1-D continuous-waveform decoder baseline on phase input. Hold decoder, split, temporal sampling, training budget and validation selection fixed while comparing phase input, phase plus cardiac bandpass, and phase plus bandpass plus harmonic suppression. Use established signal-processing libraries; avoid a custom VMD solver.
4. Select frontend choices using validation PCC, amplitude error and R-F1 jointly. Measure waveform-derived peaks and event-head peaks separately, with matching coverage. Include preprocessing in end-to-end latency.
5. Only if frontend improvements are insufficient, test one temporal-resolution change with an unchanged target and frontend. A broad backbone search is not the next experiment.
6. Reproduce the two published methods under an explicit shared protocol when their specifications permit; document deviations as adaptations. Promote no literature-reproduction claim before execution and audit.

Every non-smoke training run requires its own committed implementation and registered experiment specification. These recommendations do not replace the existing frozen reference run.
