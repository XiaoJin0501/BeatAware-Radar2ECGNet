# Q1 Event-Label Learnability Gate 0 Audit

Date: 2026-10-04. Experiment: `q1_event_label_learnability_v1`. This was a
CPU-only, read-only evaluation over all 16,682 unique ECG windows in the audited
MMECG 8-second subject pool. It did not read WSST/radar tensors, model
predictions, checkpoints, validation metrics, or test results.

## Locked question and method

Before any model result was generated, the gate asked whether fixed automatic
Q/R/S/T labels were computationally complete and repeatable enough to supervise
a structured event head. NeuroKit2 0.2.12 `prominence` independently cleaned,
detected R peaks, and delineated every 8-second window at 200 Hz. For consecutive
overlapping windows from the same record, local coordinates were converted to
record-absolute samples and compared only on common support after excluding
300 ms at both edges. Matching was order preserving within 150 ms.

Repeated overlapping observations test algorithmic consistency only. They are
not independent clinical observations, and this gate does not establish Q/S
peak validity against cardiologist annotations.

## Results

| Fiducial | Availability median | Minimum subject | Repeatability coverage | Repeatability MedAE | Repeatability P95 | Decision |
|---|---:|---:|---:|---:|---:|:---:|
| Q | 1.000 | 1.000 | 0.999 | 0.0 ms | 5.0 ms | PASS |
| R | 1.000 | 1.000 | 0.998 | 0.0 ms | 0.0 ms | PASS |
| S | 1.000 | 1.000 | 0.994 | 0.0 ms | 0.0 ms | PASS |
| T | 0.969 | 0.898 | 0.981 | 0.0 ms | 0.0 ms | PASS |

The subject-median and maximum Q<R<S<T violation rates were both 0.000. All
pre-registered completeness, overlap-repeatability, and ordering clauses passed.

## Decision and boundary

**Gate 0 passed.** A separately specified folds 1/5/9 Event-Head experiment may
therefore be designed. The result does not authorize a clinical-validity claim,
does not promote any existing timing result, and does not authorize full 11-fold
training. The next experiment must first determine whether held-out-subject WSST
contains enough information to predict these targets directly. If that event-only
gate fails against the frozen Chen-TCN waveform-derived readout, development of
the event-conditioned waveform decoder stops.

Machine-readable artifacts, subject-level denominators, source hashes, and ECG
metadata fingerprints are in `docs/audits/q1_event_label_learnability_v1/`.
