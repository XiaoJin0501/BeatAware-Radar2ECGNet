# Model Interfaces

This document records stable model I/O expectations.

## Continuous ECG Models

Expected forward contract:

```text
input -> (ecg_pred, auxiliary_outputs)
ecg_pred: [B, 1, 1600]
```

Models using this contract include BeatAware, simple temporal baselines, and
REM-AlignNet variants.

## WSST/STFT Models

Expected radar representation input:

```text
[B, C, F, T]
```

Current WSST models use 50 range channels as `C`, compact frequency bins as
`F`, and representation time frames as `T`.

## Shape-200 Heads

Shape-200 outputs represent beat-normalized morphology:

```text
ecg_shape: [B, 1, 200]
```

Shape-200 is not a replacement for continuous ECG output. If a model uses both
continuous and Shape-200 heads, the manifest must record which output was used
for each metric.

## Long-Term Assembly

The long-term assembly interface is pending. It must define:

- input cycle predictions;
- anchor predictions;
- optional confidence estimates;
- assembled ECG output;
- failure indicators.
