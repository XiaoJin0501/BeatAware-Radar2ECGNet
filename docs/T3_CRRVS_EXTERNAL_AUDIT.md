# T3 — CR_RVS External Validation — DESIGN / AUDIT (pre-lock)

Audit-first per the T3 plan (T3 mixes local results with reported-only literature
and is fairness-sensitive). This file locks the protocol facts, metric
correspondence, fairness verdicts, claim language, and the two-block structure.
No new experiments; evidence-chain consolidation only.

## Verified facts

- **CR_RVS is 5-fold subject-independent — VERIFIED.** No train∩test subject
  overlap in any fold; no cross-fold test overlap; 30 subjects partitioned across
  the 5 test folds (7/7/7/5/4). Source: subject-overlap audit on
  `/home/qhh2237/Datasets/CR_RVS/processed/q1_shape_kfold`.
- Aggregation: **fold-macro, mean ± std over 5 folds** (matches METRIC_CONTRACT).
- ECG-derived Q/S/T fiducials exist for CR_RVS test (valid rate 1.0) and enable
  timing diagnostics, but are NOT radar inputs and must not drive test-time
  selection (`q1_crrvs_fiducial_audit`).

## Our CR_RVS numbers (local, 5-fold subject-independent)

Morphology (Shape-200, beat-normalized):
| Method | PCC (mean±std) | RMSE | MAE | R² |
|---|---|---|---|---|
| CW-STFT + REM-AlignNet-MTL (proposed) | 0.887 ± 0.023 | 0.104 | 0.088 | 0.26 |
| CW-STFT + TCN (local baseline) | 0.550 ± 0.075 | 0.174 | 0.124 | −0.99 |

Full-window (8-s continuous, diagnostic; NeuroKit delineation on reconstructed ECG):
| Method | PCC | RR ms | PR ms | QRS ms | QT ms | F1@150 |
|---|---|---|---|---|---|---|
| CW-STFT + REM + lag-refine (proposed, pilot) | 0.500 | 14.65 | 55.5 | 96.4 | 75.5 | 0.878 |
| CW-STFT + REM (proposed, shape pilot) | 0.347 | 27.2 | 71.8 | 121.0 | 107.2 | 0.580 |
| CW-STFT + TCN (local baseline) | 0.438 | 17.69 | 63.2 | 123.1 | 94.4 | 0.749 |

## Reported-only literature (VMD-CBNeT paper; NOT locally reproduced)

| Method | PCC | RMSE | RR ms | PR ms | QRS ms | QT ms |
|---|---|---|---|---|---|---|
| VMD-CBNeT (reported) | 0.856 | 0.112 | 5.625 | 4.911 | 9.643 | 5.625* |
| CNN+LSTM (reported) | 0.647 | 0.127 | 48.534 | 7.414 | 29.052 | 48.534* |

`*` **Transcription red flag:** RR == QT for BOTH methods (5.625 / 48.534). Implausible;
verify against the original VMD-CBNeT paper before publishing the QT (and possibly RR)
comparison. Do not use these QT values until re-sourced.

## Metric correspondence & fairness verdicts

- **VMD-CBNeT / CNN+LSTM = reported-only, DIFFERENT PROTOCOL** (their leave-one-out,
  their preprocessing/segmentation; not reproduced here). Block B only.
- **PCC space mismatch:** our headline 0.887 is **Shape-200 beat-normalized
  morphology** (an easier target); VMD-CBNeT's 0.856 is their reconstruction PCC
  (space/protocol unstated). These are **NOT the same quantity** — do not place in
  one comparable cell. Our closest analog to a reconstruction PCC is the full-window
  0.500 (still different protocol).
- **Clinical intervals:** our full-window RR 14.65 ms is in a reasonable range, but
  our PR/QRS/QT errors (55/96/75 ms) are **much worse** than VMD-CBNeT's reported
  (4.9/9.6/5.6 ms). Do NOT claim interval superiority; if reported at all, show ours
  honestly alongside the protocol caveat (and the RR==QT transcription flag).

## Allowed claim language (locked)

- ALLOWED: "Under our subject-independent 5-fold CR_RVS **morphology** protocol, the
  proposed model reaches PCC 0.887 / RMSE 0.104, **exceeding the reported** VMD-CBNeT
  PCC 0.856 / RMSE 0.112 (reported-only, different protocol; not a same-protocol
  comparison)."
- FORBIDDEN: "strictly outperforms VMD-CBNeT under identical protocol"; any claim of
  interval (PR/QRS/QT) superiority; using the RR==QT reported values without
  re-sourcing.

## Locked T3 structure (two blocks)

- **Block A — Local CR_RVS 5-fold subject-independent (ours):** proposed (morphology
  + full-window rows, clearly labeled by space) and CW-STFT+TCN local baseline.
- **Block B — Published reported-only (Nature/CR_RVS):** VMD-CBNeT, CNN+LSTM, each
  tagged reported-only + "different protocol" caveat.
- Never merge Block A and Block B into one comparable ranking.

## Required actions before T3 is paper-ready

1. Re-source VMD-CBNeT RR/QT (and confirm QSR vs QRS definition) from the paper;
   fix the RR==QT transcription or drop those columns.
2. State the PCC evaluation space for VMD-CBNeT if the paper specifies it; otherwise
   caveat as unknown.
3. Decide the headline: recommend leading with morphology PCC (Block A) + the
   protocol-caveated reported reference (Block B); present full-window as a
   secondary diagnostic, not the headline.
4. Backfill CR_RVS registry commit ids (several are `unknown`/pre-governance).
