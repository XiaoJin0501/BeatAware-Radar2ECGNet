# REM-Net-only two-dataset CDF

Author requested adding CR_RVS to the existing MMECG CDF, with no baselines;
PR only if present in the main table. The current table has no PR column.
New entry: `scripts/figures/plot_remnet_two_dataset_cdf.py`.

Output preserves the requested `fig_mmecg_clinical_error_cdf` basename, now
two rows × four panels,7.16in wide. Top MMECG11subjects, bottom CR_RVS30subjects;
columns Q/R/S/T position errors, RR, Q–S peak spacing and Q–T peak spacing.
QRS/QT clinical durations are NOT available from a Q/R/S/T peak head. Therefore
the last two panels explicitly identify peak intervals, rather than mislabeling
them as clinical durations or mixing old continuous-model MMECG durations with
new dual-head CR_RVS spacings. This correction was announced before evaluation.
Original figure outputs were backed up before replacement. Main table and raw
benchmark files are unchanged.

Use only original seed42 dual-head REM-Net formal predictions for both datasets.
Reference NeuroKit2 prominence uses the frozen clinical-endpoint function;
structured prediction decoding reuses the archived exact function. Match beats
by R, order-preserving one-to-one DP, maximum cardinality then minimum total
error within150ms; Q/S/T use the corresponding matched beat. RR requires adjacent
matches on both sides. Finite positive Q–S/Q–T spacings are evaluated in matched
beats. No GRACE/GT shift, model inference/training or threshold tuning.

Repeated observations from overlapping windows are collapsed by median error
at exact record/global reference R; RR keys contain both reference endpoints.
Median de-duplication can introduce2.5ms values from native5ms errors; these are
not sub-sample event localization. CDF=equal subject weight, conditional on
matched evaluable observations. P50/P90 use the first exact empirical support
node attaining the requested probability, with1e-12 numerical tie tolerance;
no percentile interpolation. Missing subjects, counts and window-event coverage
are retained explicitly. CDF values are NOT detection probabilities or coverage.

Fiducial curves use monotonic PCHIP through empirical nodes only as a visual
connection. Interval curves connect observed support nodes directly with small
markers; zero-error probability has a true vertical jump at zero, and finite
150ms axes do not renormalize or force tails to1. PDF/SVG are vector; PNG600dpi.

All sources/hashes, deduplicated per-subject errors, exact nodes, quantiles and
coverage are in `docs/paper/sections/figures/source_data/remnet_two_dataset_cdf_20261026/`.
The CDF is subject-balanced and de-duplicated, unlike some main-table seed/fold
averages; it supplements rather than reproduces those summary values. The
current main table's mixed-source/naming issues remain in the separate baseline
audit and were not silently repaired here.
