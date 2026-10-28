# Frozen phase_pd inference: clinical interval roundtrip gate and agreement

Author request2026-10-29. No training, tuning or checkpoint selection. Use A4
selected seed-base42 weights on MMECG11 and CR_RVS5 test folds. First roundtrip
raw-record references through per-RR200point min-max shapes and inverse resampling.
Stop clinical durations if any dataset/metric abs bias>5ms or repeated-measures
LoA half-width>20ms; switch main figure to HR/RR/Q–S/Q–T peak intervals.

Decode a distinct shape for every adjacent valid ordered predicted R pair using
unchanged phase_pd weights and full-window context; zero-baseline fallback for
uncovered regions. Score central1024samples of available8s test windows. Frozen
NeuroKit prominence defaults and cleaning; supply own R anchors on synthesized
waveforms. No raw reference or amplitude used in deployable synthesis.

Unique beat observations selected by centre proximity independent of predictions;
HR uses individual available central windows. Paired differences, unequal-count
ANOVA variance components (Bland/Altman2007 §3), subject bootstrap10000seed42,
naive LoA, pooled ICC(A,1), proportional bias and coverage exported. Original
manuscript and outputs preserved; new source_data/phase_pd_bland_altman_20261029.

Entry point: docs/paper/sections/figures/plot_phase_pd_bland_altman.py.
Tests: tests/test_phase_pd_bland_altman.py. Formal results must retain all failures
and qualify conditional endpoint coverage; no clinical claim if gate fails.

Completed: all16 folds,7735 MMECG and5605 CR_RVS windows.171 raw-reference
alignment audits have best lag0 and correlation>0.999999999999998. All4
roundtrip checks pass: QRS/QT biases0.56/0.49ms (MMECG),1.32/0.60ms (CR_RVS);
LoA half-widths3.85/10.42 and6.42/7.16ms. Clinical route therefore evaluated.
Native soft-refined fractional R defines phase grids and inverse sampling.
QRS/QT MAE7.54/18.98ms (MMECG),11.61/22.44ms (CR_RVS), with conditional
coverage87.3/83.2% and85.4/80.0%. QRS ICC is negative in both datasets and QT
ICC low: report these limitations, do not claim clinical interchangeability.
Central fallback8.38%/14.85%. All paired observations and clipping counts saved.
Five tests pass. PDF515.52x244.8pt (7.16x3.4in), embedded STIX TrueType,
rasterized points; PNG4296x2040 at600dpi. Main and supplementary plots inspected.
Existing manuscript untouched; TableIII clinical addendum separately exported.
Integer-grid implementation preflight is archived locally and excluded from
all final statistics. Source and output hashes in new manifest. No training.
