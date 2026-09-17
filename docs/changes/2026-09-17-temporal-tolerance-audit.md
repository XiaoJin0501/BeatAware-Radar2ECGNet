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

## Result

Across repaired `loso_v2` validation folds 1/5/9, raw full-window PCC@0 is
0.2231 and filtered fixed-support NCC@0 is 0.2229. The fold-macro max-lag NCC
rises to 0.3661 at +/-20 ms, 0.5633 at +/-50 ms, 0.6243 at +/-70 ms, and 0.6353
at +/-100 ms. Fold-specific median selected lags are +65, +10, and +25 ms.
Only 2.1% of pooled windows select a search boundary.

The curve establishes a substantial GT-assisted timing-sensitive ceiling and
largely saturates by 70--90 ms. It does not establish that all improvement is
physiological EMAT: lag is selected separately for each validation window and
cannot be obtained at deployment without a radar-only estimator.
