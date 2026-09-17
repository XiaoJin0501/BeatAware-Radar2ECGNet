# Add validation-only temporal-tolerance audit

The frozen repaired-LOSO B1 checkpoints can now be evaluated under bounded
GT-assisted shifts from 0 to +/-100 ms. The evaluator reports the original raw
full-window PCC@0 separately from a zero-phase 0.5--40 Hz, per-window-standardized
NCC curve. Every lag uses the same 1560-sample support, preventing a changing
denominator from favoring larger tolerances.

The audit is hard-limited to validation data. Max-lag NCC is explicitly an
oracle diagnostic because the reference ECG selects the lag; it cannot be
reported as deployable reconstruction performance or evidence that the lag is
physiological EMAT.
