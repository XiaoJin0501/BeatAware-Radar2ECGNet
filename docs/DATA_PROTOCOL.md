# Data Protocol

This document centralizes dataset and preprocessing rules. Code, experiment
specs, and manuscripts should refer here instead of redefining protocol details.

## Datasets

| Dataset | Radar | Protocol | Role |
|---|---|---|---|
| MMECG public | 77-GHz FMCW, 50 range bins | 11-fold strict LOSO | Main benchmark |
| CR_RVS / Nature | 24-GHz CW I/Q | 5-fold subject-independent CV | External validation |

## MMECG Working Definition

- Window length: 8 seconds.
- ECG output length: 1600 samples.
- Nominal sampling rate: 200 Hz.
- Radar input: 50 range bins unless transformed to WSST/STFT representation.
- Physiological states: use stored dataset state labels and report state-wise
  aggregation only after subject/fold grouping is clear.

## CR_RVS Working Definition

- Use raw `.mat` files as the source of truth.
- Derive CW phase/displacement from I/Q before WSST/STFT feature generation.
- Use subject-independent 5-fold CV.
- State mappings must be recorded in each experiment spec and manifest.

## Allowed Ground-Truth Use

Allowed:

- training supervision;
- validation loss/metric computation inside the training split protocol;
- oracle upper-bound analysis if explicitly labeled.

Forbidden in final inference:

- ECG-correlation range-bin selection on test subjects;
- GT R-peaks for predicted long-term assembly;
- test-subject-derived normalization statistics;
- test labels for checkpoint/model selection.

## Shape-200

Shape-200 is a beat-normalized morphology protocol. It can support morphology
claims but cannot by itself support final long-term continuous ECG claims.

## Long-Term Assembly

Long-term assembly protocol is not yet locked. The next formal task is to
define:

- anchor source: oracle or predicted;
- cycle placement;
- overlap handling;
- trial/window stitching;
- metric aggregation;
- failure handling.

Until this is done, all long-term claims remain untested.
