# T1 — MMECG Beat-Morphology (Shape-200) — LOCKED evidence

Foundation table for the Q1 paper (claim M1). This file locks the numbers,
aggregation, tier labels, and per-method fairness verdicts. Source:
`experiments_mmecg/q1_paper_style_report/summary_all_methods.csv`
(built by `scripts/summarize_shape200_results.py`).

## Aggregation contract (locked)

- Protocol: MMECG public, **11-fold strict LOSO** (except where noted).
- Output space: **Shape-200** (beat-normalized center-cycle morphology, 200 pts).
- Metric path: per test segment → per-fold mean (within-fold segment mean, in each
  `fold_*/results_shape200/global_summary_shape200.json`) → **fold-macro mean ± std
  across the 11 folds**. `std` = across-fold standard deviation.
- Checkpoint selection: validation-best only.
- Shape-200 is morphology evidence only (never long-term continuous ECG).

## Locked table — Block A: strict subject-independent LOSO (11-fold, fold-macro)

This is the fair-comparison block. All rows share the strict-LOSO protocol.

| Method | Tier | PCC (mean±std) | PCC median | RMSE | MAE | R² | n_seg |
|---|---|---|---|---|---|---|---|
| REM-AlignNet-MTL | **proposed** | **0.890 ± 0.054** | 0.903 | 0.110 | 0.088 | 0.556 | 7735 |
| radarODE-MTL | adapted (ECG_shape) | 0.728 ± 0.133 | 0.770 | 0.139 | 0.106 | 0.344 | 7735 |
| REM-Event | ablation (event only) | 0.279 ± 0.151 | 0.348 | 0.258 | 0.183 | −1.48 | 7735 |
| WSST-TCN | representation baseline | 0.273 ± 0.151 | 0.349 | 0.258 | 0.183 | −1.47 | 7735 |
| RSSRNet | adapted | 0.262 ± 0.107 | 0.256 | 0.290 | 0.226 | −2.15 | 7735 |
| AirECG | adapted | 0.180 ± 0.062 | 0.188 | 0.245 | 0.173 | −1.14 | 7735 |

## Block B: LOSO + target-subject calibration (NOT subject-independent)

Reported separately with an explicit caveat; **not comparable** to Block A (uses
target-subject calibration data, different preprocessing, 7185 vs 7735 segments).
Decision (2026-08-10): keep as a caveated row, never merged into Block A.

| Method | protocol | PCC (mean±std) | RMSE | MAE | R² | n_seg |
|---|---|---|---|---|---|---|
| ICONIP BeatAware | LOSO + calib (40/10), narrow_bandpass=False | 0.671 ± 0.159 | 0.195 | 0.145 | −0.626 | 7185 |

Generic backbones (CNN/LSTM/BiLSTM/Conformer/GRU) have no Shape-200 evaluation
(0 folds); they stay in the continuous/appendix pool, not T1.

## Tier definitions (locked)

- **proposed**: our model, strict LOSO.
- **adapted**: baseline re-implemented and run under OUR strict-LOSO protocol; not
  the official implementation. radarODE-MTL is compared only in Shape-200/ECG_shape
  space (its native morphology output); we do NOT claim to reproduce its long-term
  assembly.
- **representation baseline / ablation**: WSST-TCN, REM-Event (our controls).
- **prior-conf**: our ICONIP BeatAware — see fairness flag below.

## Per-method fairness verdicts

| Method | protocol (config) | Verdict |
|---|---|---|
| REM-AlignNet-MTL | `loso`, 8-s WSST (_wsst.h5), narrow_bandpass=True, minmax | OK (proposed, strict) |
| radarODE-MTL | `strict_loso`, **calibration segments = 0** (verified all folds), 4-s WSST (native radarODE) | OK — genuinely subject-independent; fair in Shape-200/ECG_shape space; input window is 4 s by radarODE design (state it); no long-assembly claim |
| RSSRNet | `strict_loso` | OK (adapted); **preprocessing not recorded in config** — state as limitation or confirm from training script |
| AirECG | `strict_loso` | OK (adapted); **preprocessing not recorded in config** — state as limitation or confirm from training script |
| WSST-TCN, REM-Event | `loso`, 8-s WSST | OK (ablation, our pipeline) |
| ICONIP BeatAware | `loso_calib`, calib_n_train=40/val=10, narrow_bandpass=False | **NOT COMPARABLE** — target-subject calibration (not subject-independent); different preprocessing; 7185 vs 7735 segments (calib holdout) → Block B |

## Hygiene flags & required actions (before final table)

1. **BeatAware — RESOLVED (2026-08-10):** kept as a separate caveated Block B row
   ("LOSO+calibration, not subject-independent"), never merged into Block A. A
   strict-LOSO BeatAware re-run is optional future work (a new experiment), not
   required for T1 to be honest.
2. **Traceability gap:** registry `commit_id` is `unknown` for the core T1 rows
   (proposed, radarODE, baselines). Backfill commit IDs or explicitly mark them as
   pre-governance snapshots; a paper table needs each number traceable.
3. **Preprocessing parity — partially resolved (2026-08-10):**
   - Proposed / WSST-TCN / REM-Event: 8-s WSST, narrow_bandpass=True, minmax.
   - radarODE-MTL: 4-s WSST (native radarODE), strict LOSO, **calibration=0 verified**.
     Fair in Shape-200 output space; note the 4-s vs 8-s input-window difference.
   - RSSRNet / AirECG: preprocessing NOT recorded in their configs; either confirm
     from their training scripts or state as a documented limitation. (Low scorers;
     does not affect the proposed's lead.)
4. **Aggregation note:** across-fold std uses n=11; report as mean±std. If the paper
   prefers subject-macro or CIs, recompute from `fold_level_all_methods.csv`.

## Locked vs pending

- Locked: numbers, fold-macro aggregation definition, tier labels, fairness verdicts.
- Pending (no new modeling; hygiene only): resolve BeatAware protocol, backfill
  commit IDs, confirm preprocessing parity. Then T1 is paper-ready.
