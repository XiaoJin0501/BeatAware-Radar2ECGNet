# Calibrated continuous ECG probe

The Q1 development route now tests explicitly disclosed held-out-subject
calibration for the primary 8-second continuous ECG output. Shape-200 remains
auxiliary morphology evidence and is absent from this probe.

The first measurement freezes the existing B1 continuous checkpoints. One
complete validation-subject record estimates a subject-level lag and affine
mapping; all other records are query-only and their ECG is never consumed by
calibration. This is a non-publishable development probe used to decide whether
a learned calibration-conditioned REM-Net is justified.

This changes neither the manuscript nor existing experiment artifacts.

## Measured outcome

The one-record subject calibration was run first and failed its registered
gate: fold 1 improved, folds 5 and 9 worsened, and mean delta PCC was +0.090.
That result is retained in `q1_calibrated_continuous_probe`.

A separate session-prefix hypothesis was then registered before measurement.
It uses the first 20 seconds of every record for calibration and begins query
windows at 20 seconds, so no calibration/query window shares any source sample.
It passed: all 3 validation subjects improved, 26/31 records improved, and the
mean fold delta PCC was +0.326. Calibrated continuous PCC was 0.497/0.589/0.580
on folds 1/5/9. These are development-probe numbers, not manuscript evidence.

Scientific consequence: a single subject constant is insufficient, while a
short session-specific timing calibration is promising enough to justify a
learned calibration-conditioned REM-Net. Shape-200 remains absent from the
selection and gate.

The one-record subject calibration was run first and failed its registered
gate: only fold 1 improved, folds 5/9 worsened, and mean delta PCC was +0.090.
That result is retained. A separate session-prefix probe was then registered to
test record-varying calibration without rewriting the failed hypothesis.
