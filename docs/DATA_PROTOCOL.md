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

## Current output protocols

The active protocol has three explicitly separated tiers:

1. Shape-200 is a known-cycle, beat-normalized morphology upper bound.
2. Strict blind continuous ECG is the native 8-second, 1600-sample full-window
   output under repaired LOSO, without ECG-derived alignment at inference.
3. Temporal-tolerance analysis permits GT-selected lag only as a diagnostic
   upper bound and must never be reported as achievable performance.

The former anchor-guided Route C assembly and learned delay-warp designs are
closed. They are not pending interfaces or next experiments. Historical
artifacts are retained in `archive/deprecated_docs/` and the claim ledger.
