# MMECG Split Audit and LOSO Diagnostics

## Purpose

This note records the current audit of the MMECG preprocessing and split protocols. The goal is to distinguish three different issues:

1. Hard data leakage: identical records or windows appear in multiple splits.
2. Subject overlap: the same subject appears in train/val/test under a samplewise or record-level split.
3. Strict subject-independent generalization: test subjects are completely unseen during training.

This distinction is important for Q1-level paper framing because published radar-to-ECG results may use different split protocols.

## Samplewise Split Audit

The MMECG `samplewise` split is produced by:

```text
/home/qhh2237/Datasets/MMECG/03A_create_samplewise_splits.py
```

Despite the name, the split is performed at the record level:

```text
train: 63 records
val:   14 records
test:  14 records
```

### Record and Window Leakage

The processed H5 files show:

```text
train-val exact window overlap: 0
train-test exact window overlap: 0
val-test exact window overlap: 0

train-val record overlap: 0
train-test record overlap: 0
val-test record overlap: 0
```

Conclusion:

There is no hard leakage of identical records or identical `(record_id, start_idx, end_idx)` windows across splits.

### Subject Overlap

However, subjects appear in multiple splits:

```text
test-train subject overlap:
S1, S2, S13, S14, S16, S29, S30

test-val subject overlap:
S10, S13, S14, S16, S29, S30

train-val subject overlap:
S5, S9, S13, S14, S16, S17, S29, S30
```

Conclusion:

The samplewise split is useful for measuring within-subject or record-level reconstruction ability, but it should not be presented as subject-independent evidence.

Suggested wording:

> The samplewise protocol is record-disjoint and window-disjoint, but not subject-disjoint. Therefore, it evaluates within-subject generalization across records rather than strict subject-independent generalization.

## Samplewise Processed H5 Statistics

Processed windows:

```text
train: 1639 windows, 63 records
val:   1190 windows, 14 records
test:  1190 windows, 14 records
```

Physiological state distribution:

```text
train: NB 420, IB 408, SP 403, PE 408
val:   NB 510, IB 255, SP 255, PE 170
test:  NB 595, IB 255, SP 170, PE 170
```

The training set is deliberately balanced by physiological state through adaptive stride, while validation and test use a fixed 2-second stride.

## LOSO Split Audit

The LOSO split is produced by:

```text
/home/qhh2237/Datasets/MMECG/03B_create_loso_splits.py
```

For each fold:

```text
test: one held-out subject
val: one subject from the remaining subjects
train: the other nine subjects
```

Audit result:

```text
train-val subject overlap: 0
train-test subject overlap: 0
val-test subject overlap: 0
```

Conclusion:

The LOSO split is strict subject-independent evaluation.

## Current LOSO Failure Pattern

Existing class-balanced experiment:

```text
experiments_mmecg/mmecg_reg_loso
use_diffusion=False
balance_by=class
narrow_bandpass=False
```

### Fold 01

Test subject:

```text
S1, NB only, 170 segments
```

Global metrics:

```text
PCC mean:        0.2540
RMSE norm mean:  0.1804
R2 mean:        -0.1841
R error mean:   29.01 ms
RR error mean:  11.83 ms
QMR:            65.88%
PCC<0.60 rate:  86.47%
```

PCC quantiles:

```text
min:  -0.2018
10%:  -0.1324
25%:  -0.0485
50%:   0.2870
75%:   0.5266
90%:   0.6114
max:   0.8231
```

Interpretation:

Fold 01 has mixed behavior: many failed waveform segments, but a subset of segments reaches high PCC. The failure is not uniform random collapse.

### Fold 02

Test subject:

```text
S2, NB only, 170 segments
```

Global metrics:

```text
PCC mean:        0.1843
RMSE norm mean:  0.1819
R2 mean:        -0.7343
R error mean:   33.66 ms
RR error mean:  18.80 ms
QMR:            99.41%
PCC<0.60 rate:  100.00%
```

PCC quantiles:

```text
min:  -0.0472
10%:   0.0368
25%:   0.1016
50%:   0.1750
75%:   0.2544
90%:   0.3552
max:   0.4672
```

Interpretation:

Fold 02 preserves monitoring viability better than waveform morphology. QMR is high, but waveform PCC is consistently low.

## Key Interpretation for Paper Framing

The current evidence supports a protocol-gap story:

1. Samplewise is record-disjoint but subject-overlapping.
2. LOSO is subject-disjoint and substantially harder.
3. Early LOSO failures occur especially on S1/S2, which are small-record subjects with NB-only test data.
4. Waveform morphology and monitoring reliability can diverge: low PCC does not necessarily imply complete failure in R/RR monitoring metrics.

This supports a multi-level evaluation story rather than a PCC-only story.

## Current Subject-Balanced Control

Completed experiment:

```text
experiments_mmecg/mmecg_reg_loso_subject
use_diffusion=False
balance_by=subject
narrow_bandpass=False
fold_idx=1
```

Early validation progress:

```text
epoch 5:  val_pcc=0.0830
epoch 10: val_pcc=0.1473
epoch 15: val_pcc=0.2118
epoch 20: val_pcc=0.2427
epoch 25: val_pcc=0.2731
epoch 35: val_pcc=0.2778
```

Final training:

```text
early stopped at epoch 130
best checkpoint: epoch 70, val_pcc=0.3055
```

Test results on fold 01:

```text
PCC mean:        0.4055
RMSE norm mean:  0.1857
R2 mean:        -0.2502
R error mean:   24.03 ms
RR error mean:  13.96 ms
QMR:            95.29%
Avg F1@150ms:   0.8033
PCC<0.60 rate:  98.82%
```

Compared with class-balanced fold 01:

```text
PCC:       0.2540 -> 0.4055
QMR:       65.88% -> 95.29%
R error:   29.01 ms -> 24.03 ms
```

Interpretation:

Subject-balanced training substantially improves monitoring reliability and fold 01 waveform correlation on the held-out S1 subject, even though the absolute PCC remains below a morphology-perfect ECG reconstruction level.

## Record-Disjoint Subject-Overlapping Samplewise Control

Completed experiment:

```text
experiments_mmecg/mmecg_reg_samplewise_subject
use_diffusion=False
balance_by=subject
narrow_bandpass=False
protocol=samplewise
```

Final validation:

```text
best checkpoint: epoch 60, val_pcc=0.4421
```

Test results:

```text
PCC mean:        0.2667
RMSE norm mean:  0.1642
R2 mean:        -0.2432
R error mean:   25.91 ms
RR error mean:  16.86 ms
QMR:            86.05%
Avg F1@150ms:   0.7600
PCC<0.60 rate:  82.18%
```

Scene-wise PCC:

```text
NB:  0.3941
IB:  0.1123
PE:  0.3470
SP: -0.0281
```

Subject-wise PCC:

```text
S10: 0.7402
S14: 0.6210
S13: 0.5238
S30: 0.3470
S2:  0.3273
S1:  0.2151
S16: 0.0749
S29:-0.0153
```

Interpretation:

The samplewise test set is not uniformly easy despite subject overlap. Its global score is strongly affected by difficult subject-scene combinations, especially S29 under IB/SP and S16 under NB. This means the current architecture has partial learning ability but does not yet provide stable morphology reconstruction across all record-level and scene-level shifts.

Decision rule:

The next factors to test should be:

```text
narrow-band RCG preprocessing
ECG normalization and metric scale
electromechanical delay alignment range
samplewise vs LOSO protocol gap
subject/scene-specific failure analysis
```

## Lag-Aware Training Follow-Up

Lag-adjusted PCC analysis showed that many low-PCC failures are timing dominated rather than pure morphology collapse. A lag-aware waveform loss has been added as an optional training objective:

```text
use_lag_aware_loss
lag_max_ms
lambda_lag_pcc
lambda_lag_l1
```

Recommended first diagnostic run:

```bash
python scripts/train_mmecg.py \
  --exp_tag mmecg_reg_samplewise_lag100 \
  --protocol samplewise --epochs 80 \
  --use_diffusion false \
  --balance_by subject \
  --narrow_bandpass false \
  --use_lag_aware_loss true \
  --lag_max_ms 100 \
  --lambda_lag_pcc 0.2 \
  --lambda_lag_l1 0.05
```

Interpretation target:

```text
raw PCC should increase;
lag-adjusted PCC gap should shrink;
R/QMR should not degrade.
```

## Lag100 Result and Revised Loss Hypothesis

Completed experiment:

```text
experiments_mmecg/mmecg_reg_samplewise_lag100
protocol=samplewise
balance_by=subject
narrow_bandpass=False
use_lag_aware_loss=True
lag_max_ms=100
lambda_lag_pcc=0.2
lambda_lag_l1=0.05
```

Final validation:

```text
best checkpoint: epoch 80, val_pcc=0.4363
```

Test result:

```text
raw PCC mean:     0.2287
shifted PCC mean: 0.7233  (max_lag=300 ms diagnostic)
PCC gain mean:    0.4946
QMR:              73.70%
RR error mean:     9.81 ms
Avg F1@150ms:      0.8128
```

Comparison with the no-lag samplewise baseline:

```text
raw PCC:      0.2667 -> 0.2287  (worse)
shifted PCC:  0.5585 -> 0.7233  (much better)
PCC gain:     0.2918 -> 0.4946  (worse timing gap)
QMR:          86.05% -> 73.70%  (worse)
F1@150ms:     0.7600 -> 0.8128  (better fiducial tolerance)
```

Interpretation:

The naive best-shift lag-aware loss is not the final method. It proves that the model can learn ECG morphology much better than raw PCC suggests, but it also allows the model to place that morphology at the wrong zero-lag timing. In paper terms, this is an important negative control: shift-invariant optimization improves latent morphology while worsening deployable synchronized reconstruction.

The revised hypothesis is:

```text
good training objective = bounded shifted morphology matching
                        + explicit zero-lag PCC anchor
                        + soft penalty for large best-lag mass
```

This has been implemented with additional optional loss terms:

```text
lambda_zero_pcc
lambda_lag_penalty
lag_softmax_tau
```

Current follow-up run:

```bash
python scripts/train_mmecg.py \
  --exp_tag mmecg_reg_samplewise_lag_anchor \
  --protocol samplewise --epochs 80 \
  --use_diffusion false \
  --balance_by subject \
  --narrow_bandpass false \
  --use_lag_aware_loss true \
  --lag_max_ms 100 \
  --lambda_lag_pcc 0.1 \
  --lambda_lag_l1 0.02 \
  --lambda_zero_pcc 0.2 \
  --lambda_lag_penalty 0.05 \
  --lag_softmax_tau 0.05
```

Decision rule after `mmecg_reg_samplewise_lag_anchor` finishes:

```text
If raw PCC >= 0.30 and shifted PCC remains >= 0.65:
  keep the anchor objective and tune weights.

If shifted PCC remains high but raw PCC stays <= 0.25:
  the model learns morphology but still lacks alignment capacity;
  consider an explicit alignment head or sequence-level lag estimator.

If both raw and shifted PCC drop:
  the anchor is too strong or conflicts with the shifted morphology term;
  lower lambda_zero_pcc / lambda_lag_penalty and retry.
```

## Lag Anchor Result

Completed experiment:

```text
experiments_mmecg/mmecg_reg_samplewise_lag_anchor
protocol=samplewise
balance_by=subject
narrow_bandpass=False
lag_max_ms=100
lambda_lag_pcc=0.1
lambda_lag_l1=0.02
lambda_zero_pcc=0.2
lambda_lag_penalty=0.05
lag_softmax_tau=0.05
```

Final validation:

```text
best checkpoint: epoch 50, val_pcc=0.4382
```

Test result:

```text
raw PCC mean:     0.2221
shifted PCC mean: 0.6752  (max_lag=300 ms diagnostic)
PCC gain mean:    0.4531
QMR:              80.92%
RR error mean:    13.80 ms
Avg F1@150ms:      0.8245
```

Three-way comparison:

| Experiment | Raw PCC | Shifted PCC | PCC Gain | Mean Abs Lag | Timing Failure | Morphology Failure | QMR | F1@150ms | RR Error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mmecg_reg_samplewise_subject` | 0.267 | 0.559 | 0.292 | 48.7 ms | 56.6% | 21.7% | 86.1% | 0.760 | 16.9 ms |
| `mmecg_reg_samplewise_lag100` | 0.229 | 0.723 | 0.495 | 54.3 ms | 73.8% | 8.1% | 73.7% | 0.813 | 9.8 ms |
| `mmecg_reg_samplewise_lag_anchor` | 0.222 | 0.675 | 0.453 | 49.5 ms | 76.1% | 8.5% | 80.9% | 0.825 | 13.8 ms |

Interpretation:

The anchor loss did not close the raw-vs-shifted PCC gap. It partially recovered monitoring viability compared with naive lag100 (`QMR 73.7% -> 80.9%`) and preserved strong morphology (`shifted PCC 0.675`, morphology failure only `8.5%`), but raw synchronized waveform PCC stayed below the no-lag baseline.

This is now a stronger conclusion than the original lag100 negative control:

```text
The current model can reconstruct ECG-like morphology,
but the learned waveform is not reliably synchronized at zero lag.
Loss-only lag tolerance is insufficient.
```

Subject-scene evidence:

```text
S13 NB: raw 0.725, shifted 0.823, small lag -> good morphology and synchronization
S14 NB: raw 0.711, shifted 0.750, small lag -> good morphology and synchronization
S29 SP: raw -0.073, shifted 0.800, median lag -60 ms -> excellent morphology but wrong timing
S29 IB: raw -0.005, shifted 0.681, median lag 50 ms -> timing-dominated failure
S16 NB: raw 0.074, shifted 0.588, mean abs lag 95 ms -> severe timing instability
S13 IB: shifted only 0.439 -> residual morphology failure
```

Next methodological step:

```text
Stop treating timing as only a loss-function nuisance.
Add an explicit per-segment alignment mechanism.
```

Recommended next controls:

1. Estimate and report dataset-level radar-ECG lag distribution directly from input RCG vs ECG for each subject-scene. This checks whether the lag is already present before the model or is introduced by the model.
2. Add a light scalar lag head that predicts a bounded per-segment shift from radar features, then applies differentiable temporal warping/rolling before waveform loss.
3. Keep the current lag-adjusted diagnostic as a paper contribution, but do not claim `LagAwareWaveformLoss` alone is the final contribution unless the alignment head succeeds.
4. Before full LOSO lag-aware training, run the scalar-lag-head control on samplewise first. Full LOSO is too expensive until raw PCC improves in the fast protocol.

## Input RCG-vs-ECG Lag Diagnostic

Added diagnostic:

```bash
python scripts/analyze_mmecg_input_lag.py \
  --protocol samplewise \
  --split test \
  --max_lag_ms 300
```

Output directory:

```text
experiments_mmecg/input_lag_samplewise_test/
```

Global result:

```text
best_channel median lag:        -75.0 ms
best_channel mean |lag|:        102.5 ms
best_channel |lag| >= 50ms:      80.3%
mean_abs median lag:            -70.0 ms
mean_abs mean |lag|:            121.8 ms
mean_abs |lag| >= 50ms:          87.0%
```

Subject-scene examples:

```text
S1 NB:   best_channel median lag   40 ms, mean |lag|  44.6 ms
S16 NB:  best_channel median lag -190 ms, mean |lag| 172.8 ms
S29 SP:  best_channel median lag  -80 ms, mean |lag|  91.0 ms
S13 IB:  best_channel median lag  180 ms, mean |lag| 143.0 ms
S14 NB:  best_channel median lag  -80 ms, mean |lag|  85.1 ms
```

Interpretation:

Large timing offsets are already visible at the input RCG-vs-ECG level, before the reconstruction model. The failure is therefore not only caused by the decoder placing ECG morphology at the wrong time. The input signal itself carries subject/scene/segment-dependent electromechanical phase offsets.

This supports the scalar lag-head direction:

```text
static EMD FIR alignment is insufficient
because the required correction is segment-dependent.
```

## Output Scalar Lag-Head Control

Implemented optional model flag:

```text
use_output_lag_align
output_lag_max_ms
```

The model predicts one bounded lag value per segment from fused radar features and applies a differentiable temporal shift to the reconstructed ECG. This directly tests whether explicit per-segment alignment can close the raw-vs-shifted PCC gap.

Current running experiment:

```bash
python scripts/train_mmecg.py \
  --exp_tag mmecg_reg_samplewise_output_align \
  --protocol samplewise --epochs 80 \
  --use_diffusion false \
  --balance_by subject \
  --narrow_bandpass false \
  --use_output_lag_align true \
  --output_lag_max_ms 200 \
  --use_lag_aware_loss true \
  --lag_max_ms 100 \
  --lambda_lag_pcc 0.0 \
  --lambda_lag_l1 0.0 \
  --lambda_zero_pcc 0.2 \
  --lambda_lag_penalty 0.0
```

Decision rule:

```text
If raw PCC improves above the no-align baseline (0.267)
and shifted PCC remains high, keep this branch and tune lag head/loss.

If raw PCC stays low but shifted PCC remains high, the head is not learning
usable lag from radar features; next step is explicit auxiliary lag
supervision from input-lag pseudo-labels.

If raw and shifted PCC both drop, the output shift is destabilizing
and should be moved earlier into feature alignment instead of output alignment.
```

## Output Lag-Head Result

Two output scalar lag-head controls were tested:

```text
mmecg_reg_samplewise_output_align
  output_lag_max_ms=200
  no output-lag regularization

mmecg_reg_samplewise_output_align_reg
  output_lag_max_ms=100
  lambda_output_lag_l1=0.02
  zero-initialized lag-head final layer
```

Findings:

```text
No regularization:
  best checkpoint at epoch 5, val_pcc=0.4699
  predicted lag saturated near +197 ms for almost every segment
  conclusion: learned a global boundary-shift shortcut, not adaptive alignment

With regularization:
  best checkpoint at epoch 5, val_pcc=0.4120
  predicted lag collapsed near -2.7 ms for almost every segment
  conclusion: regularization prevented saturation but also prevented useful lag learning
```

Full test result for the regularized control:

```text
raw PCC mean:     0.2088
shifted PCC mean: 0.5318
PCC gain mean:    0.3230
QMR:              88.4%
RR error mean:    14.4 ms
F1@150ms:          0.7100
```

Comparison with no-align baseline:

```text
raw PCC:      0.2667 -> 0.2088  (worse)
shifted PCC:  0.5585 -> 0.5318  (worse)
QMR:          86.1%  -> 88.4%   (slightly better)
F1@150ms:     0.760 -> 0.710    (worse)
```

Conclusion:

An unsupervised output scalar lag head is not sufficient. Without constraints,
it finds a global boundary shift. With simple L1 regularization, it collapses to
near-zero shift and does not learn subject/scene/segment-specific alignment.

Next step:

```text
Use input-lag diagnostic estimates as pseudo-labels for auxiliary lag supervision,
or move the alignment mechanism earlier to feature-level alignment with explicit
lag supervision.
```
