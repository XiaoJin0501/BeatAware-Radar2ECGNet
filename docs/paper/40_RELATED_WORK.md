# Related Work (draft prose)

Grounded in `docs/LITERATURE_COMPARISON_LEDGER.csv`,
`docs/TABLE_D_REPORTED_LITERATURE_AUDIT.md`, and `reference_papers/`. Reported-only
numbers are used as context, never as same-protocol head-to-head.

## 2. Related Work

### 2.1 Radar-to-ECG reconstruction

Contactless ECG reconstruction from radar has progressed from direct waveform
regression to structured, physiology-aware models. Early and baseline approaches
regress an ECG-like waveform from radar features with generic sequence backbones
(CNN, LSTM/BiLSTM, GRU, Conformer) or wavelet/lifting networks. A second line
introduces cardiac structure or signal decomposition: radarODE and radarODE-MTL
embed ODE-based cardiac dynamics and a multi-task ECG_shape / anchor / interval
formulation; VMD-CBNeT uses variational mode decomposition with a convolutional
backbone; several works add semantic distillation, state-space modeling, or
disentangled representations. A third line targets high-fidelity generation with
conditional diffusion or GAN models. Our model belongs to the structured,
supervised family but is deliberately morphology-first: it does not claim to solve
continuous long-window ECG, and it removes an alignment mechanism that did not help
morphology.

### 2.2 Evaluation formulations and comparability

A recurring difficulty is that radar-to-ECG methods are evaluated under
heterogeneous output formulations — full-window waveform, beat-normalized
morphology, or fiducial/interval timing — often on private datasets or sample-level
splits. Reported correlation scores therefore are not directly comparable across
papers. Radar observes mechanical wall motion while ECG records electrical activity,
so beat morphology and continuous timing are distinct sub-problems. We make this
distinction explicit and organize comparison into fairness tiers (proposed, adapted,
prior-with-calibration, reported-only), which prevents mixing incompatible protocols
in one ranking. Under this discipline, radarODE-MTL is compared only in the
Shape-200 / ECG_shape morphology space (its long-term assembly is not reproduced),
and VMD-CBNeT / CNN+LSTM are quoted as reported-only external references.

### 2.3 Relation to our prior conference work

This work extends our ICONIP conference model (BeatAware), which studied FMCW
radar-to-ECG with physiology-aware modules on a single dataset under a historical
protocol. The present journal study differs substantially: (i) a redesigned,
smaller morphology-first model (REM-Net) whose alignment branch is ablated and
removed based on controlled evidence; (ii) cross-modality external validation on a
second public dataset (24-GHz CW CR_RVS) under verified subject-independent
splits; (iii) multi-seed statistics and fairness-audited comparison tiers; and
(iv) a new mechanistic finding that identifies beat-anchor localization precision as
the barrier to continuous radar-to-ECG. The conference model appears here only as a
prior baseline, not as the contribution.
