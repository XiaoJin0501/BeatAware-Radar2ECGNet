# Q1 Experiment Matrix

Paper-first experiment design for the electromechanical-alignment thesis. Each
table/figure exists to prove a specific claim (see `docs/PAPER_CLAIM_LEDGER.md`).
Do not add experiments that do not map to a claim.

## Central thesis & claims (recap)

Radar↔ECG has a time-varying electromechanical delay `δ(t)` (≈PEP). Modeling and
self-emergently learning it closes the morphology↔continuous gap.

- **C1** Explicit event structure + learned time-varying electromechanical
  alignment (`δ` field + lag-aware) framework for continuous ECG.
- **C2** Dual-track evaluation quantifies "morphology easy, timing hard" and the
  alignment closes the gap.
- **C3** Learned `δ` is interpretable and varies with physiological state.
- **C4** Holds across FMCW (MMECG) and CW (CR_RVS) under subject-independent
  protocols.

## Protocols & aggregation (locked)

- MMECG public: 11-fold strict LOSO. CR_RVS/Nature: 5-fold subject-independent CV.
- **Continuous metric = full-window per-window PCC (primary) + within-record
  long-OLA PCC (secondary).**
- Aggregation: **fold-macro, report mean ± std**. Key conclusions: **≥3 seeds**.
- Checkpoint selection: **validation-best only** (never test-set).

## Main tables

### T1 — MMECG morphology (Shape-200)
Supports C1 (morphology capability).

| Method | Tier | PCC | RMSE | MAE | R² |
|---|---|---|---|---|---|
| REM-AlignNet-MTL | proposed | | | | |
| radarODE-MTL | adapted | | | | |
| ICONIP BeatAware | prior-conf | | | | |
| WSST-TCN | repr-baseline | | | | |
| (appendix) RSSRNet / AirECG | adapted | | | | |

### T2 — MMECG continuous ★
Supports C1 + C2 (the gap-closing core).

| Method | full-window PCC/RMSE/MAE/R² | long-OLA PCC/coverage | R-F1 / RR(ms) | QRS/QT(ms)† |
|---|---|---|---|---|
| REM-AlignNet-MTL (proposed) | | | | |
| ICONIP BeatAware | | | | |
| WSST-TCN | | | | |

radarODE-MTL appears only as a footnote: its long-term assembly is not faithfully
reproducible locally, so it is not fairly comparable in continuous space.

### T3 — CR_RVS external validation
Supports C4; aligned to VMD-CBNeT reported metrics.

| Method | Tier | PCC | RMSE | RR | QT | PR | QSR |
|---|---|---|---|---|---|---|---|
| CW-STFT + REM-AlignNet-MTL | proposed | | | | | | |
| CW-STFT + TCN | local | | | | | | |
| VMD-CBNeT / CNN+LSTM | reported (caveat) | | | | | | |

### T4 — Ablation (mechanism) ★
Supports C1 + C2 causally; proves the alignment.

| Variant | Shape-200 PCC (should stay flat) | full-window PCC (should rise) | RR ms (should drop) |
|---|---|---|---|
| B0 (no align, no lag / WSST-TCN) | | | |
| B1 (+δ_sys) | | | |
| B2 (+lag) | | | |
| B3 (+δ_sys + r) | | | |
| B4 (full) | | | |

### T5 — Efficiency
| Method | Params (M) | FLOPs (G) | Model latency (ms) | Preproc time | GPU mem |
|---|---|---|---|---|---|
| REM-AlignNet-MTL | | | | | |
| radarODE-MTL | | | | | |
| ICONIP BeatAware | | | | | |

Always separate model-only time from representation-preprocessing time.

## C3 figures (evidence, not tables)

- **Fig-A** gap-closing curve: continuous PCC across B0→B4, morphology flat.
- **Fig-B** δ_sys vs pseudo-GT lag (correlation).
- **Fig-C** δ_sys vs HR (PEP shortens with rate).
- **Fig-D** δ_sys distribution across CR_RVS states (resting/Valsalva/apnea).
- **Fig-E** qualitative continuous ECG before/after alignment, overlaid with GT.

## Fair-comparison tiers (table footnotes)

- **Main comparison**: same public dataset + subject-independent (proposed,
  BeatAware, our adapted radarODE, CW-STFT+TCN).
- **Reported-only (caveat)**: VMD-CBNeT / CNN+LSTM (CR_RVS same dataset, different
  protocol).
- **Adapted reproduction**: RSSRNet / AirECG (originally private datasets).
- **Appendix pool**: generic CNN/LSTM/BiLSTM/Conformer — not in main tables.

## Claim → evidence map

| Claim | Evidence |
|---|---|
| C1 | T1 + T2 + T4 (B1/B4) |
| C2 | T2 + Fig-A + T4 |
| C3 | Fig-B / Fig-C / Fig-D |
| C4 | T3 (+ CR_RVS ablation) |

† Clinical intervals (QT/QTc) enter tables only if delineation quality is
confirmed reliable (see `docs/METRIC_CONTRACT.md`).

## Minimal closed loop (before scaling)

MMECG fold_01 only, variants **B0 / B1 / B3 / B4**. Decision on three signals:

1. Shape-200 PCC roughly preserved (not dropping).
2. full-window PCC rises.
3. RR / R-F1 improve.

**Go/no-go**: if B4 continuous does not clearly improve on fold_01, stop-loss or
adjust the design. If it clearly improves, extend to 11-fold LOSO + CR_RVS.
Only after this does the project move from idea to the Q1 experiment framework.
