# Final-Test Metric Audit

Status: updated after the locked incumbent REM-Net seed-42 final-test re-run.

## Source Runs

| Dataset | Experiment ID | Commit | Protocol | Folds |
|---|---|---|---|---|
| MMECG | `q1_final_remnet_incumbent_mmecg_seed42` | `ebb726a` | strict LOSO | 11 |
| CR_RVS | `q1_final_remnet_incumbent_crrvs_seed42` | `ebb726a` | subject-independent CV | 5 |

Outputs:

- `experiments_mmecg/q1_final_remnet_incumbent/final_summary.csv`
- `experiments_mmecg/q1_final_remnet_incumbent/final_fold_metrics.csv`

## Metric Contract

- Output space: Shape-200 beat-normalized morphology.
- Metric: per-segment Shape-200 PCC from `evaluate_remalign_shape_head.py`.
- Aggregation: per-fold mean PCC, then fold-macro mean +/- across-fold std.
- Checkpoint: `best.pt`, selected by validation full-window PCC in the training
  script. No test fold is used for checkpoint selection.
- Interpretation: morphology evidence only; not full-window continuous ECG.

## Locked Final-Test Numbers

| Dataset | PCC mean +/- std | n folds |
|---|---:|---:|
| MMECG | 0.882 +/- 0.061 | 11 |
| CR_RVS | 0.881 +/- 0.013 | 5 |

## Stability Check

G1 multi-seed remains a stability supplement, not a replacement for the seed-42
final-test row:

| Dataset | 3-seed PCC mean +/- std |
|---|---:|
| MMECG | 0.885 +/- 0.002 |
| CR_RVS | 0.886 +/- 0.001 |

## Claim Rules

- Allowed: report seed-42 final-test rows as the main table values and mention
  multi-seed stability in text or appendix.
- Allowed: compare MMECG final-test Shape-200 PCC against adapted radarODE-MTL
  Shape-200/ECG_shape.
- Allowed with caveat: CR_RVS morphology PCC exceeds reported VMD-CBNeT PCC under
  our morphology protocol, but this is not a same-protocol reproduction.
- Forbidden: claim full-window continuous ECG SOTA from Shape-200 results.
