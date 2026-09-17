# Q1 Native-RCG Resolution Gate Audit

## Scope and provenance

This document freezes the completed six-run development gate
`q1_native_rcg_resolution_gate`. The gate used MMECG `loso_v2` folds 1/5/9,
training and validation only; the test split was never opened. Both arms used
the same fold-paired seed, 80/20 maximum-epoch/patience budget, validation
checkpoint composite, train-fold robust scaler, and locked L3 objective:
Huber + 0.20 event BCE + 0.50 MR-STFT (128/256/512).

- C0 `B2a-wide`: WSST-only capacity control, 1,487,551 parameters.
- C1 `B2b`: WSST + native 200-Hz RCG direct fusion, 1,487,667 parameters.
- Parameter delta: 116 (0.008%); downstream decoder shapes and paired
  initialization were controlled.
- Implementation commit: `089d9e1fcf24ac44074fbb102d9d67c2a433e57a`.
- Runtime head recorded by the runs: `6149a98785e004697bc62d9f9fc05b6a0e156c4f`.
- Source artifacts:
  `experiments_mmecg/q1_native_rcg_resolution_gate/`.

These are pilot/development results and are not manuscript performance
evidence.

## Complete six-run matrix

Metrics are the values at the validation-selected checkpoint. PCC, NRMSE, R2,
variance ratio, relative >10-Hz energy, and region MAEs are computed per
window and averaged within each fold. Shape-200 PCC is auxiliary and was not
used for checkpoint selection.

| Arm | Fold | Best/stop epoch | PCC@0 | NRMSE | R2 | Var ratio | >10-Hz ratio | F1@50 | F1@100 | F1@150 | Event F1@100 | RR MAE (ms) | HR MAE (bpm) | QRS MAE | ST/T MAE | Shape-200 PCC (aux) | Composite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B2a-wide | 1 | 27/47 | 0.1711 | 2.3045 | -0.0661 | 0.0006 | 0.1844 | 0.0403 | 0.0776 | 0.1673 | 0.0000 | 75.76 | 9.30 | 1.8346 | 0.5177 | 0.2020 | 0.0801 |
| B2b | 1 | 32/52 | 0.1226 | 2.2988 | -0.0614 | 0.0164 | 0.0944 | 0.4186 | 0.5490 | 0.5743 | 0.0011 | 651.20 | 30.93 | 1.8515 | 0.4707 | 0.1323 | 0.1314 |
| B2a-wide | 5 | 2/22 | 0.1487 | 2.1227 | -0.0208 | 0.0081 | 0.1357 | 0.1503 | 0.3045 | 0.4451 | 0.3764 | 206.84 | 13.07 | 1.6883 | 0.8984 | 0.1787 | 0.1052 |
| B2b | 5 | 1/21 | 0.0942 | 2.1524 | -0.0503 | 0.0220 | 2.3835 | 0.1850 | 0.5668 | 0.6903 | 0.4301 | 135.39 | 9.00 | 1.7029 | 0.8956 | 0.1342 | 0.1227 |
| B2a-wide | 9 | 2/22 | 0.0465 | 2.2773 | -0.0864 | 0.0013 | 0.1274 | 0.1042 | 0.2219 | 0.3341 | 0.3052 | 275.17 | 16.48 | 1.8907 | 0.5311 | 0.0628 | 0.0519 |
| B2b | 9 | 2/22 | 0.0423 | 2.2767 | -0.0865 | 0.0097 | 0.5045 | 0.2857 | 0.5302 | 0.7163 | 0.4561 | 156.32 | 11.78 | 1.8930 | 0.5218 | 0.0550 | 0.0964 |

## Fold-macro summary

| Metric | B2a-wide | B2b | B2b - B2a-wide | Direction |
|---|---:|---:|---:|---|
| PCC@0 | 0.1221 | 0.0863 | -0.0358 | worse |
| NRMSE | 2.2348 | 2.2427 | +0.0078 | worse |
| R2 | -0.0577 | -0.0661 | -0.0083 | worse |
| Amplitude variance ratio | 0.0033 | 0.0160 | +0.0127 | closer to 1, still severely compressed |
| Relative >10-Hz energy ratio | 0.1492 | 0.9941 | +0.8449 | unstable; fold 5 overshoots to 2.3835 |
| Waveform F1@50 | 0.0983 | 0.2965 | +0.1982 | better |
| Waveform F1@100 | 0.2013 | 0.5487 | +0.3473 | better |
| Waveform F1@150 | 0.3155 | 0.6603 | +0.3448 | better |
| Event-head F1@100 | 0.2272 | 0.2958 | +0.0685 | better |
| RR MAE (ms) | 185.92 | 314.30 | +128.38 | worse |
| HR MAE (bpm) | 12.95 | 17.24 | +4.29 | worse |
| QRS-region MAE | 1.8045 | 1.8158 | +0.0113 | worse |
| ST/T-region MAE | 0.6491 | 0.6294 | -0.0197 | slightly better |
| Shape-200 PCC (auxiliary) | 0.1478 | 0.1072 | -0.0407 | worse |
| Validation composite | 0.0790 | 0.1168 | +0.0378 | higher because F1 dominates the gain |

## Preregistered decision

| Gate | Required | Observed | Decision |
|---|---|---|---|
| Waveform | PCC improves on 3/3 folds, macro delta >= +0.05; R2 improves on 3/3 | PCC improved on 0/3, macro delta -0.0358; R2 improved on 1/3 | **FAIL** |
| Mechanism | QRS MAE decreases on 3/3 and a smoothing ratio moves toward 1 on 3/3 | QRS MAE decreased on 0/3; spectral ratio is non-monotonic and overshoots on fold 5 | **FAIL** |
| Event safety | Macro F1@100 decline <=0.02 and no fold decline >0.05 | Macro F1@100 increased by 0.3473; all folds increased | **PASS** |
| Promote | All three gates pass | Two required gates fail | **FAIL — route closed** |

## Interpretation

Native RCG direct fusion clearly changes the output and improves tolerant
waveform/event peak detection, but it does not improve continuous waveform
fidelity. The same arm reduces PCC on every fold, leaves fold-macro R2 more
negative, worsens QRS-region MAE on every fold, and worsens RR/HR error in the
macro aggregate. The high-frequency ratio cannot be interpreted as recovered
QRS morphology: it ranges from 0.094 to 2.383 while PCC and QRS MAE regress.

The defensible conclusion is narrow: **under this capacity-matched architecture,
preprocessing, and repaired three-fold validation protocol, direct fusion of
native 200-Hz RCG does not add transferable continuous-ECG morphology
information beyond WSST.** This does not prove that radar contains no such
information, and it does not invalidate other denoising or array-processing
formulations. Per the preregistered stop rule, no B2b fusion variant or 11-fold
scaling is authorized.
