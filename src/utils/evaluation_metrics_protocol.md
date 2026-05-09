# Evaluation Metrics Protocol for Radar-to-ECG Reconstruction

## 1. Purpose

This document defines a unified evaluation protocol for radar-to-ECG reconstruction and radar-based ECG fiducial analysis. The purpose is to support fair comparison with prior studies, including AirECG, radarODE-MTL, Zhang et al. conditional diffusion model, Kong et al. state-space model, and Cao et al. fiducial-point detection model. Since many of these studies do not release source code, the comparison should be based on clearly defined, reproducible metrics. When baseline values are extracted directly from papers, their original metric meanings must be carefully aligned with the proposed evaluation protocol.

The evaluation is organized into four levels. Level 1 evaluates waveform-level ECG reconstruction quality. Level 2 evaluates beat-level timing and peak localization accuracy. Level 3 evaluates fiducial-point detection performance. Level 4 evaluates clinical interval fidelity. The first three levels are mainly used for direct comparison with existing studies, while Level 4 is used as an extended clinical evaluation because most prior radar-to-ECG papers do not systematically report PR, QRS duration, QT, or QTc interval errors.

| Level | Category | Main Purpose | Comparable Studies |
|---|---|---|---|
| Level 1 | Waveform-level reconstruction | Evaluate global ECG waveform similarity and amplitude reconstruction quality | AirECG, radarODE-MTL, Kong et al., Zhang et al. |
| Level 2 | Beat-level timing / peak localization | Evaluate R-peak, RR/PPI, T-wave, and Q/R/S/T timing consistency | AirECG, radarODE-MTL, Kong et al., Zhang et al. |
| Level 3 | Fiducial-point detection | Evaluate detection of clinically relevant ECG fiducial points | Cao et al. |
| Level 4 | Clinical interval fidelity | Evaluate PR, QRS duration, QT, and QTc interval preservation | Proposed extended clinical evaluation |

The most important conceptual distinction is:

```text
Q/R/S/T peak timing error = peak localization error
PR / QRS / QT / QTc error = clinical interval error
```

These two metric families should not be mixed. Q/R/S/T peak timing error measures the temporal location difference of individual ECG peaks. It does not measure QRS duration, QT interval, or PR interval.

## 2. Common Notation

Let:

```text
x        = ground-truth ECG segment
x_hat    = reconstructed ECG segment
T        = segment length in samples
T_seg    = segment length in samples
fs       = sampling rate
t_p      = ground-truth location of peak or fiducial point p
t_hat_p  = predicted / reconstructed location of peak or fiducial point p
```

All timing errors should be converted from samples to milliseconds by:

```math
error_{ms} = error_{samples} \times \frac{1000}{f_s}
```

The recommended ECG peak and fiducial detection tool is:

```python
NeuroKit2: ecg_peaks + ecg_delineate
```

The recommended output files are:

```text
segment_metrics.csv
beat_metrics.csv
subject_summary.csv
global_summary.json
```

The evaluation should preserve segment-level and beat-level raw values rather than only reporting global averages. This is important because many prior studies report CDFs, subject-level distributions, medians, and IQRs.

## 3. Level 1: Waveform-Level Reconstruction Metrics

Waveform-level metrics evaluate the global similarity between the reconstructed ECG and the ground-truth ECG. These metrics are suitable for comparison with AirECG, radarODE-MTL, Kong et al., and Zhang et al.

### 3.1 Pearson Correlation Coefficient

Metric name:

```python
pcc_raw
```

Definition:

```math
PCC(x, \hat{x}) =
\frac{\sum_{t=1}^{T}(x_t-\bar{x})(\hat{x}_t-\bar{\hat{x}})}
{\sqrt{\sum_{t=1}^{T}(x_t-\bar{x})^2}
\sqrt{\sum_{t=1}^{T}(\hat{x}_t-\bar{\hat{x}})^2}}
```

Interpretation:

```text
Range: [-1, 1]
Higher is better
```

PCC measures waveform-shape similarity between reconstructed ECG and reference ECG. The main comparison table should report raw PCC. DTW-aligned PCC may be reported as an optional supplementary metric, but it should not replace raw PCC in the main comparison table because most radar-to-ECG papers report ordinary segment-level PCC.

Optional supplementary metric:

```python
pcc_dtw_aligned
```

### 3.2 Root Mean Square Error

Metric names:

```python
rmse_norm
rmse_mV
```

Definition:

```math
RMSE(x, \hat{x}) =
\sqrt{\frac{1}{T}\sum_{t=1}^{T}(x_t-\hat{x}_t)^2}
```

Interpretation:

```text
Lower is better
```

RMSE measures point-wise amplitude reconstruction error. Two versions should be saved if possible:

```text
rmse_norm = RMSE computed on normalized ECG
rmse_mV   = RMSE computed on ECG in physical mV scale
```

Use `rmse_norm` when comparing with studies that evaluate normalized ECG signals. Use `rmse_mV` when comparing with studies that report physical ECG amplitude error.

### 3.3 Mean Absolute Error

Metric names:

```python
mae_norm
mae_mV
```

Definition:

```math
MAE(x, \hat{x}) =
\frac{1}{T}\sum_{t=1}^{T}|x_t-\hat{x}_t|
```

Interpretation:

```text
Lower is better
```

MAE measures average absolute amplitude error between reconstructed ECG and ground-truth ECG. Two versions should be saved if possible:

```text
mae_norm = MAE computed on normalized ECG
mae_mV   = MAE computed on ECG in physical mV scale
```

### 3.4 Coefficient of Determination

Metric name:

```python
r2
```

Definition:

```math
R^2 =
1 -
\frac{\sum_{t=1}^{T}(x_t-\hat{x}_t)^2}
{\sum_{t=1}^{T}(x_t-\bar{x})^2}
```

Interpretation:

```text
Higher is better
```

R² measures how much variance of the ground-truth ECG is explained by the reconstructed ECG. This metric is especially useful when comparing with radarODE-MTL, which reports R² together with RMSE and PCC.

## 4. Level 2: Beat-Level Timing and Peak Localization Metrics

Beat-level metrics evaluate whether the reconstructed ECG preserves important cardiac timing events. Before computing these metrics, ECG peaks or fiducial points should be detected from both ground-truth ECG and reconstructed ECG.

Recommended detection pipeline:

```text
1. Detect R peaks from ground-truth ECG.
2. Detect R peaks from reconstructed ECG.
3. Delineate Q, S, T peaks or relevant fiducial points.
4. Match predicted events to ground-truth events beat by beat.
5. Compute timing errors on matched events.
6. Count unmatched events in detection-based metrics.
```

Recommended matching principle:

```text
Each GT event can be matched with at most one predicted event.
Each predicted event can be matched with at most one GT event.
Nearest-neighbor matching within a physiological tolerance is recommended.
```

### 4.1 R-Peak Localization Error

Metric name:

```python
r_peak_error_ms
```

Definition:

```math
e_R^{(i)} =
|\hat{t}_R^{(i)} - t_R^{(i)}|
\times \frac{1000}{f_s}
```

where:

```text
t_R      = R-peak location in ground-truth ECG
t_hat_R  = matched R-peak location in reconstructed ECG
fs       = sampling rate
```

Interpretation:

```text
Unit: ms
Lower is better
```

Recommended aggregation:

```text
mean
median
standard deviation
IQR
CDF values
```

### 4.2 Q/R/S/T Peak Timing Error

Metric names:

```python
q_peak_error_ms
r_peak_error_ms
s_peak_error_ms
t_peak_error_ms
qrst_peak_error_ms_mean
```

Definition:

```math
e_p^{(i)} =
|\hat{t}_p^{(i)} - t_p^{(i)}|
\times \frac{1000}{f_s},
\quad p \in \{Q,R,S,T\}
```

Interpretation:

```text
Unit: ms
Lower is better
```

This metric measures the absolute location error of Q, R, S, and T peaks. It is peak localization error, not clinical interval error.

Correct interpretation:

```text
Q peak error = location error of Q peak
R peak error = location error of R peak
S peak error = location error of S peak
T peak error = location error of T peak
```

Incorrect interpretation:

```text
Q/R/S/T peak error ≠ QRS duration error
Q/R/S/T peak error ≠ QT interval error
Q/R/S/T peak error ≠ PR interval error
```

This metric is useful for comparison with studies that report Q/R/S/T or QRST peak timing errors, such as Zhang et al. and radarODE-MTL.

### 4.3 Q/R/S/T Relative Peak Timing Error

Metric names:

```python
q_peak_error_rel_percent
r_peak_error_rel_percent
s_peak_error_rel_percent
t_peak_error_rel_percent
qrst_peak_error_rel_percent_mean
```

Definition:

```math
e_{p,rel}^{(i)} =
\frac{|\hat{t}_p^{(i)} - t_p^{(i)}|}{T_{seg}}
\times 100\%,
\quad p \in \{Q,R,S,T\}
```

where:

```text
T_seg = segment length in samples
```

Interpretation:

```text
Unit: %
Lower is better
```

This metric measures peak localization error normalized by segment length. It is useful when comparing with studies that report Q/R/S/T timing errors in percentage form rather than milliseconds.

### 4.4 RR Interval Error

Metric name:

```python
rr_interval_error_ms
```

Definition:

First compute RR intervals:

```math
RR_i = t_R^{(i+1)} - t_R^{(i)}
```

```math
\hat{RR}_i = \hat{t}_R^{(i+1)} - \hat{t}_R^{(i)}
```

Then compute:

```math
e_{RR}^{(i)} =
|\hat{RR}_i - RR_i|
\times \frac{1000}{f_s}
```

Interpretation:

```text
Unit: ms
Lower is better
```

RR interval error measures cardiac cycle timing error. Use this metric when comparing with AirECG, Kong et al., Zhang et al., and other studies that report RR interval error.

### 4.5 PPI Error

Metric name:

```python
ppi_error_ms
```

Definition:

In this evaluation protocol, PPI error is computed in the same way as RR interval error:

```python
ppi_error_ms = rr_interval_error_ms
```

Interpretation:

```text
Unit: ms
Lower is better
```

PPI refers to peak-to-peak interval or cardiac cycle length. It is mainly used to align with radarODE-MTL terminology. Use `ppi_error_ms` when comparing with radarODE-MTL. Use `rr_interval_error_ms` when comparing with AirECG, Kong et al., and Zhang et al.

### 4.6 T-Wave Timing Error

Metric name:

```python
t_wave_timing_error_ms
```

Definition:

```math
e_T^{(i)} =
|\hat{t}_T^{(i)} - t_T^{(i)}|
\times \frac{1000}{f_s}
```

Interpretation:

```text
Unit: ms
Lower is better
```

T-wave timing error measures whether the reconstructed ECG preserves T-wave timing. This metric is useful for comparison with AirECG, which reports T-wave timing error.

### 4.7 Qualified Monitoring Rate

Metric name:

```python
qualified_monitoring_rate
```

Definition:

```math
QMR =
\frac{N_{valid}}{N_{total}}
\times 100\%
```

where:

```text
N_total = total number of test segments
N_valid = number of segments where valid ECG features can be detected
```

Recommended validity rule:

```text
A segment is considered valid if:
1. GT ECG has at least 2 detected R peaks.
2. Reconstructed ECG has at least 2 detected R peaks.
3. At least one matched RR interval exists.
4. Optional: Q/R/S/T or Pon/Qon/Rpeak/Soff/Toff can be partially detected.
```

Interpretation:

```text
Unit: %
Higher is better
```

Qualified monitoring rate measures whether the reconstructed ECG is sufficiently recognizable by ECG feature detection tools. This metric is mainly used for comparison with AirECG.

### 4.8 Missed Detection Rate

Metric names:

```python
rpeak_mdr_event
segment_failure_rate_pcc60
```

Event-level MDR:

```math
MDR_{event} =
\frac{FN}{TP+FN}
```

Equivalent to:

```math
MDR_{event} = 1 - Recall
```

Segment-level failure rate:

```math
MDR_{segment} =
\frac{N_{failed}}{N_{total}}
\times 100\%
```

Recommended failure rule:

```text
failed segment = PCC < 0.60 or R-peak detection failed
```

Interpretation:

```text
Unit: %
Lower is better
```

MDR measures missed detection or reconstruction failure. This metric is useful for comparison with radarODE-MTL.

## 5. Level 3: Fiducial-Point Detection Metrics

This level is designed for comparison with studies that directly detect ECG fiducial points from radar-derived signals, especially Cao et al.

The evaluated fiducial points are:

```text
Pon   = onset of P-wave
Qon   = onset of QRS complex
Rpeak = peak of R-wave
Soff  = offset of QRS complex
Toff  = offset of T-wave
```

These fiducial points are clinically important because they define PR interval, QRS duration, QT interval, and RR interval.

### 5.1 Tolerance-Based Matching

Use tolerance windows around ground-truth fiducial points.

Recommended tolerance settings:

```python
tolerance_list_ms = [150, 100, 50]
```

For each fiducial type:

```text
TP = predicted point within tolerance window of a GT point
FP = predicted point outside all GT tolerance windows
FN = GT point without matched prediction
```

Use one-to-one matching:

```text
Each GT point can be matched with at most one predicted point.
Each predicted point can be matched with at most one GT point.
```

### 5.2 Precision

Metric names:

```python
pon_precision_150ms
qon_precision_150ms
rpeak_precision_150ms
soff_precision_150ms
toff_precision_150ms
average_precision_150ms
```

The same format should be used for 100 ms and 50 ms tolerance.

Definition:

```math
Precision =
\frac{TP}{TP+FP}
```

Interpretation:

```text
Higher is better
```

Precision measures how many predicted fiducial points are correct.

### 5.3 Recall

Metric names:

```python
pon_recall_150ms
qon_recall_150ms
rpeak_recall_150ms
soff_recall_150ms
toff_recall_150ms
average_recall_150ms
```

The same format should be used for 100 ms and 50 ms tolerance.

Definition:

```math
Recall =
\frac{TP}{TP+FN}
```

Interpretation:

```text
Higher is better
```

Recall measures how many ground-truth fiducial points are successfully detected.

### 5.4 F1-Score

Metric names:

```python
pon_f1_150ms
qon_f1_150ms
rpeak_f1_150ms
soff_f1_150ms
toff_f1_150ms
average_f1_150ms
```

The same format should be used for 100 ms and 50 ms tolerance.

Definition:

```math
F1 =
\frac{2 \cdot Precision \cdot Recall}
{Precision + Recall}
```

Interpretation:

```text
Higher is better
```

Recommended output:

```python
average_f1_150ms
average_f1_100ms
average_f1_50ms
```

## 6. Level 4: Clinical Interval Fidelity Metrics

Clinical interval metrics evaluate whether the reconstructed ECG preserves clinically meaningful ECG intervals. This level is not always directly comparable with prior radar-to-ECG papers, because many of them do not report PR, QRS, QT, or QTc interval errors. However, these metrics are important for demonstrating clinical relevance.

### 6.1 PR Interval Error

Metric name:

```python
pr_interval_error_ms
```

Clinical definition:

```math
PR = t_{Qon} - t_{Pon}
```

Error definition:

```math
e_{PR}^{(i)} =
|\hat{PR}^{(i)} - PR^{(i)}|
\times \frac{1000}{f_s}
```

Interpretation:

```text
Unit: ms
Lower is better
```

PR interval error measures atrioventricular conduction interval preservation.

### 6.2 QRS Duration Error

Metric name:

```python
qrs_duration_error_ms
```

Clinical definition:

```math
QRS = t_{Soff} - t_{Qon}
```

Error definition:

```math
e_{QRS}^{(i)} =
|\hat{QRS}^{(i)} - QRS^{(i)}|
\times \frac{1000}{f_s}
```

Interpretation:

```text
Unit: ms
Lower is better
```

QRS duration error measures ventricular depolarization duration preservation.

### 6.3 QT Interval Error

Metric name:

```python
qt_interval_error_ms
```

Clinical definition:

```math
QT = t_{Toff} - t_{Qon}
```

Error definition:

```math
e_{QT}^{(i)} =
|\hat{QT}^{(i)} - QT^{(i)}|
\times \frac{1000}{f_s}
```

Interpretation:

```text
Unit: ms
Lower is better
```

QT interval error measures ventricular depolarization and repolarization interval preservation.

### 6.4 QTc Interval Error

Metric name:

```python
qtc_interval_error_ms
```

Clinical definition:

Using Bazett correction:

```math
QTc = \frac{QT}{\sqrt{RR}}
```

where QT and RR should be measured in seconds.

Error definition:

```math
e_{QTc}^{(i)} =
|\hat{QTc}^{(i)} - QTc^{(i)}|
\times 1000
```

Interpretation:

```text
Unit: ms
Lower is better
```

QTc interval error measures heart-rate-corrected QT interval preservation.

## 7. Recommended CSV Outputs

### 7.1 segment_metrics.csv

Each row corresponds to one ECG segment.

```python
subject_id
scene
segment_id
fs
segment_length_sec

pcc_raw
rmse_norm
rmse_mV
mae_norm
mae_mV
r2

qualified_flag
num_r_gt
num_r_pred
num_matched_beats

q_peak_error_ms_mean
r_peak_error_ms_mean
s_peak_error_ms_mean
t_peak_error_ms_mean
qrst_peak_error_ms_mean

q_peak_error_rel_percent_mean
r_peak_error_rel_percent_mean
s_peak_error_rel_percent_mean
t_peak_error_rel_percent_mean
qrst_peak_error_rel_percent_mean

rr_interval_error_ms_mean
ppi_error_ms_mean
t_wave_timing_error_ms_mean

pon_f1_150ms
qon_f1_150ms
rpeak_f1_150ms
soff_f1_150ms
toff_f1_150ms
average_f1_150ms

pon_f1_100ms
qon_f1_100ms
rpeak_f1_100ms
soff_f1_100ms
toff_f1_100ms
average_f1_100ms

pon_f1_50ms
qon_f1_50ms
rpeak_f1_50ms
soff_f1_50ms
toff_f1_50ms
average_f1_50ms

pr_interval_error_ms_mean
qrs_duration_error_ms_mean
qt_interval_error_ms_mean
qtc_interval_error_ms_mean

segment_failed_pcc60
```

### 7.2 beat_metrics.csv

Each row corresponds to one matched beat.

```python
subject_id
scene
segment_id
beat_id
fs

q_peak_error_ms
r_peak_error_ms
s_peak_error_ms
t_peak_error_ms

q_peak_error_rel_percent
r_peak_error_rel_percent
s_peak_error_rel_percent
t_peak_error_rel_percent

rr_interval_error_ms
ppi_error_ms

pr_interval_error_ms
qrs_duration_error_ms
qt_interval_error_ms
qtc_interval_error_ms
```

### 7.3 subject_summary.csv

Each row corresponds to one subject.

```python
subject_id
scene

pcc_mean
pcc_median
pcc_iqr

rmse_mean
rmse_median
rmse_iqr

mae_mean
mae_median
mae_iqr

r2_mean
r2_median
r2_iqr

rr_error_mean_ms
rr_error_median_ms
rr_error_iqr_ms

r_peak_error_mean_ms
r_peak_error_median_ms
r_peak_error_iqr_ms

t_wave_error_mean_ms
t_wave_error_median_ms
t_wave_error_iqr_ms

average_f1_150ms
average_f1_100ms
average_f1_50ms

pr_error_mean_ms
pr_error_median_ms
pr_error_iqr_ms

qrs_error_mean_ms
qrs_error_median_ms
qrs_error_iqr_ms

qt_error_mean_ms
qt_error_median_ms
qt_error_iqr_ms

qtc_error_mean_ms
qtc_error_median_ms
qtc_error_iqr_ms

qualified_monitoring_rate
segment_failure_rate_pcc60
```

### 7.4 global_summary.json

Recommended fields:

```json
{
  "num_subjects": 0,
  "num_segments": 0,
  "num_valid_segments": 0,
  "num_matched_beats": 0,

  "pcc_mean": 0,
  "pcc_median": 0,
  "pcc_iqr": 0,

  "rmse_mean": 0,
  "rmse_median": 0,
  "rmse_iqr": 0,

  "mae_mean": 0,
  "mae_median": 0,
  "mae_iqr": 0,

  "r2_mean": 0,
  "r2_median": 0,
  "r2_iqr": 0,

  "rr_interval_error_ms_mean": 0,
  "rr_interval_error_ms_median": 0,
  "rr_interval_error_ms_iqr": 0,

  "r_peak_error_ms_mean": 0,
  "r_peak_error_ms_median": 0,
  "r_peak_error_ms_iqr": 0,

  "t_wave_timing_error_ms_mean": 0,
  "t_wave_timing_error_ms_median": 0,
  "t_wave_timing_error_ms_iqr": 0,

  "average_f1_150ms": 0,
  "average_f1_100ms": 0,
  "average_f1_50ms": 0,

  "pr_interval_error_ms_mean": 0,
  "pr_interval_error_ms_median": 0,

  "qrs_duration_error_ms_mean": 0,
  "qrs_duration_error_ms_median": 0,

  "qt_interval_error_ms_mean": 0,
  "qt_interval_error_ms_median": 0,

  "qtc_interval_error_ms_mean": 0,
  "qtc_interval_error_ms_median": 0,

  "qualified_monitoring_rate": 0,
  "segment_failure_rate_pcc60": 0
}
```

## 8. Recommended Main Tables for Paper

### Table 1. Waveform-Level ECG Reconstruction

| Method | PCC ↑ | RMSE ↓ | MAE ↓ | R² ↑ |
|---|---:|---:|---:|---:|
| AirECG |  |  |  |  |
| radarODE-MTL |  |  |  |  |
| Kong et al. |  |  |  |  |
| Zhang et al. |  |  |  |  |
| Proposed |  |  |  |  |

### Table 2. Beat-Level Timing Fidelity

| Method | R-Peak Error (ms) ↓ | RR/PPI Error (ms) ↓ | T-Wave Error (ms) ↓ | Q/R/S/T Peak Error (%) ↓ | QMR (%) ↑ | MDR (%) ↓ |
|---|---:|---:|---:|---:|---:|---:|
| AirECG |  |  |  |  |  |  |
| radarODE-MTL |  |  |  |  |  |  |
| Kong et al. |  |  |  |  |  |  |
| Zhang et al. |  |  |  |  |  |  |
| Proposed |  |  |  |  |  |  |

Important:

```text
Q/R/S/T Peak Error = peak localization error
RR/PPI Error = cardiac cycle interval error
```

### Table 3. Fiducial-Point Detection

| Method | Pon F1 @150ms ↑ | Qon F1 @150ms ↑ | Rpeak F1 @150ms ↑ | Soff F1 @150ms ↑ | Toff F1 @150ms ↑ | Avg F1 @150ms ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Cao et al. |  |  |  |  |  |  |
| Proposed |  |  |  |  |  |  |

Optional additional columns:

```text
Avg F1 @100ms
Avg F1 @50ms
```

### Table 4. Clinical Interval Fidelity

| Method | PR Error (ms) ↓ | QRS Duration Error (ms) ↓ | QT Error (ms) ↓ | QTc Error (ms) ↓ |
|---|---:|---:|---:|---:|
| AirECG | Not reported | Not reported | Not reported | Not reported |
| radarODE-MTL | Not reported | Not reported | Not reported | Not reported |
| Kong et al. | Partially reported / not directly comparable | Not fully reported | Not fully reported | Not fully reported |
| Zhang et al. | Not reported | Not reported | Not reported | Not reported |
| Proposed |  |  |  |  |

This table demonstrates whether the reconstructed ECG preserves clinically meaningful interval information. It may not be directly comparable with many existing radar-to-ECG studies, because most prior works mainly report waveform similarity, RR interval error, or peak localization error rather than full clinical interval fidelity.

## 9. Recommended Function Names in metrics.py

```python
compute_waveform_metrics()
compute_r_peak_metrics()
compute_qrst_peak_timing_errors()
compute_relative_peak_timing_errors()
compute_rr_interval_error()
compute_ppi_error()
compute_t_wave_timing_error()
compute_qualified_monitoring_rate()
compute_mdr()
compute_fiducial_detection_f1()
compute_clinical_interval_errors()
summarize_segment_metrics()
summarize_subject_metrics()
summarize_global_metrics()
```

## 10. Key Implementation Rules

Rule 1: Do not confuse peak localization error with clinical interval error.

Correct:

```text
Q/R/S/T peak timing error = location error of individual Q/R/S/T peaks
```

Incorrect:

```text
Q/R/S/T peak timing error = QRS/QT/PR interval error
```

Rule 2: Report both mean and distributional statistics.

For each metric, save:

```text
mean
median
standard deviation
IQR
raw values for CDF
```

Median and IQR are particularly useful for subject-wise and beat-wise evaluation.

Rule 3: Keep raw segment-level values.

Do not only save global averages. Always keep:

```text
segment_metrics.csv
beat_metrics.csv
subject_summary.csv
global_summary.json
```

This allows later generation of:

```text
CDF plots
box plots
subject-wise bar plots
case selection
best / median / worst waveform visualization
```

Rule 4: Use raw PCC as the main metric.

Use:

```python
pcc_raw
```

as the main PCC metric.

Optional supplementary metric:

```python
pcc_dtw_aligned
```

DTW-aligned PCC should not replace raw PCC in the main comparison table.

Rule 5: Use separate terminology for different papers.

| Paper | Preferred Comparable Metrics |
|---|---|
| AirECG | PCC, RMSE, qualified monitoring rate, RR interval error, T-wave timing error |
| radarODE-MTL | RMSE, PCC, R², PPI error, timing error, MDR |
| Zhang et al. conditional diffusion | PCC, RMSE, RR interval error, Q/R/S/T peak timing error |
| Kong et al. state-space model | PCC, MAE, RR interval MAE, HRV-related error |
| Cao et al. fiducial detection | Pon/Qon/Rpeak/Soff/Toff Precision, Recall, F1 under 150/100/50 ms tolerance |

## 11. Recommended Final Evaluation Strategy

The final evaluation should include four complementary perspectives:

```text
1. Waveform similarity
   PCC, RMSE, MAE, R²

2. Beat-level timing
   R-peak error, RR/PPI error, T-wave timing error, Q/R/S/T peak error

3. Fiducial-point detection
   Pon/Qon/Rpeak/Soff/Toff F1 under 150/100/50 ms tolerance

4. Clinical interval fidelity
   PR error, QRS duration error, QT error, QTc error
```

This structure allows direct comparison with prior radar-to-ECG studies while also highlighting clinically meaningful ECG reconstruction quality beyond simple waveform similarity.

## 12. Practical Recommendation for Current Implementation

When the implementation asks:

```text
1. Clinical interval error
2. Peak localization error
3. Both
```

Choose:

```text
3. Both
```

Reason:

```text
Peak localization error is required for direct comparison with most prior radar-to-ECG papers.
Clinical interval error is useful for strengthening the clinical relevance of the proposed method, even if many baselines do not report it.
```

Recommended priority:

```text
Priority 1: waveform-level metrics
Priority 2: peak localization and RR/PPI timing metrics
Priority 3: fiducial-point detection F1
Priority 4: clinical interval fidelity
```

## 13. Final Metric Naming Summary

Use the following metric names consistently in code, CSV files, and paper tables.

Waveform-level metrics:

```python
pcc_raw
rmse_norm
rmse_mV
mae_norm
mae_mV
r2
```

Peak-level timing metrics:

```python
q_peak_error_ms
r_peak_error_ms
s_peak_error_ms
t_peak_error_ms
qrst_peak_error_ms_mean

q_peak_error_rel_percent
r_peak_error_rel_percent
s_peak_error_rel_percent
t_peak_error_rel_percent
qrst_peak_error_rel_percent_mean
```

Interval-level timing metrics:

```python
rr_interval_error_ms
ppi_error_ms
t_wave_timing_error_ms
```

Monitoring validity and missed detection metrics:

```python
qualified_monitoring_rate
rpeak_mdr_event
segment_failure_rate_pcc60
```

Fiducial-point detection metrics:

```python
pon_precision_150ms
qon_precision_150ms
rpeak_precision_150ms
soff_precision_150ms
toff_precision_150ms
average_precision_150ms

pon_recall_150ms
qon_recall_150ms
rpeak_recall_150ms
soff_recall_150ms
toff_recall_150ms
average_recall_150ms

pon_f1_150ms
qon_f1_150ms
rpeak_f1_150ms
soff_f1_150ms
toff_f1_150ms
average_f1_150ms
```

Repeat the same naming format for:

```text
100ms tolerance
50ms tolerance
```

Clinical interval fidelity metrics:

```python
pr_interval_error_ms
qrs_duration_error_ms
qt_interval_error_ms
qtc_interval_error_ms
```

## 14. Final Writing Statement for Paper

A concise statement for the Methods or Evaluation section:

```text
To ensure fair comparison with prior radar-based ECG reconstruction studies, we evaluate the reconstructed ECG from four complementary perspectives: waveform-level reconstruction quality, beat-level timing fidelity, fiducial-point detection accuracy, and clinical interval fidelity. Waveform-level metrics include PCC, RMSE, MAE, and R². Beat-level timing metrics include R-peak localization error, RR/PPI interval error, T-wave timing error, and Q/R/S/T peak timing error. Fiducial-point detection is evaluated using tolerance-based precision, recall, and F1-score for Pon, Qon, Rpeak, Soff, and Toff. Finally, clinical interval fidelity is assessed using PR interval error, QRS duration error, QT interval error, and QTc interval error. Importantly, Q/R/S/T peak timing error is treated as a peak localization metric rather than a clinical interval metric.
```
