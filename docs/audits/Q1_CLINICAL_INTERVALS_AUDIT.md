# Q1 Clinical Intervals Audit (NeuroKit2)

Date: 2026-09-25. Evaluator: `scripts/evaluate_clinical_intervals.py` (read-only over frozen
test predictions). Outputs: `Q1_CLINICAL_INTERVALS_fold.csv` (251 rows),
`Q1_CLINICAL_INTERVALS_scenario.csv` (887 rows). Table I: `scripts/make_table1_benchmark.py`
-> `docs/paper/sections/Table_I_Benchmark.tex` (generated, never hand-edited).

## Delineation method and why

NeuroKit2 0.2.12, `ecg_clean` then `ecg_delineate(method="peak")`, identically on reference and
reconstruction. At 200 Hz the `dwt` method places `ECG_R_Offsets` on the T wave and
`ECG_R_Onsets` near the P wave (reference "QRS" median 375 ms); `cwt` returns no QRS boundaries.
Q and S peaks agree across all three methods, so:

- QRS duration = S peak - Q peak (Q-S interval)
- QT = T offset - Q peak; QTc = QT / sqrt(RR) (Bazett)

Reference plausibility on all folds: QRS 90-115 ms, QT 335-410 ms, QTc 374-427 ms; reference
delineation succeeds in 98-100 % of windows.

## Consistency

R-MAE, cycle-length error and PCC reproduce `Q1_CORE_METRICS_fold.csv` exactly (251/251 rows,
max |diff| = 0).

## Coverage (windows where the reconstruction can be delineated)

MMECG: every baseline >= 99 %; REM-Net 92.6 %, with **zero** coverage in two seed-folds
(seed 42 fold 7, seed 2024 fold 2 -- waveform R-peak detection fails). REM-Net's QRS/QT errors
therefore exclude its weakest folds and are favourable to it; the Table I caption says so.
CR_RVS: 97-98 % for all three systems.

## Findings (MMECG, three seeds; CR_RVS, seed 42)

| | QRS err. (ms) | QT err. (ms) | best |
|---|---:|---:|---|
| MMECG REM-Net | 40.8 | 47.6 | Cross-Modal Conformer: 31.4 / 30.8 |
| CR_RVS REM-Net | 42.9 | 72.0 | QRS: REM-Net (42.9 vs 43.1); QT: WaveGRU-Net 42.1 |

REM-Net is not the most accurate system on clinical intervals on either dataset. Per state
(Fig. 3, per-subject medians) REM-Net has the lowest QT error in all four MMECG states; the
fold-mean in Table I is higher because it is pulled by outlying windows. Both are reported.

## Timing spread of the trained systems (Fig. 4(d))

Standard deviation of matched anchor offsets, MMECG, three seeds: REM-Net anchor head 50.1 ms,
Chen-TCN 50.5 ms, REM-Net waveform peaks 51.7 ms, radarODE-MTL 53.8 ms. The robust MAD is
quantised to whole samples and ties every system at 51.9 ms, so the s.d. is used. The
architectures do not differ materially in timing spread; all trained systems sit ~0.35 R^2
above the jittered-template curve at that spread, the amplitude-shrinkage hedge of Eq. (4).
