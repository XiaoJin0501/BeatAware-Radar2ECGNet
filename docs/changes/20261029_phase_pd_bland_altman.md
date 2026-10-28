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
