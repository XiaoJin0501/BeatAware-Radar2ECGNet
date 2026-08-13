# Significance & State-Robustness Audit (traceable sources)

Locks the two claims that condition on statistics or per-state breakdowns to
reproducible sources, so the manuscript wording is backed by data, not assertion.

## 1. Paired significance: REM-Net vs radarODE-MTL (MMECG Shape-200)

- REM-Net per-fold PCC (locked final-test): `experiments_mmecg/q1_final_remnet_incumbent/final_fold_metrics.csv`
  (11 folds, mean 0.882).
- radarODE-MTL per-fold PCC: `experiments_mmecg/q1_teacher_report_remalign_mtl_shapefirst/fold_level_rem_vs_radarode.csv`
  column `radarode_pcc` (11 folds, mean 0.728).
- Test: paired over the 11 folds (`scipy.stats.ttest_rel`, `wilcoxon`).

Reproduced values (this audit):

| Quantity | Value | Manuscript |
|---|---|---|
| REM > radarODE folds | 11 / 11 | "on every fold" |
| mean paired diff | +0.1539 | +0.154 |
| paired t-test p | 9.803×10⁻⁵ | 9.8×10⁻⁵ |
| Wilcoxon signed-rank p | 9.766×10⁻⁴ | 9.8×10⁻⁴ |

Exact match. The abstract's weaker "paired $p<10^{-3}$" is also satisfied.

## 2. Physiological-state robustness (tab:states)

State-wise Shape-200 PCC is the segment-pooled mean over the `state` column of the
frozen per-fold segment CSVs
(`experiments_mmecg/q1_final_remnet_incumbent_mmecg_seed42/fold_*/results_shape_head/segment_metrics_shape_head.csv`;
state map 0=NB,1=IB,2=SP,3=PE).

Reproduced (MMECG): NB 0.912, IB 0.903, SP 0.850, PE 0.834 — exact match to tab:states.
CR_RVS states come from the corresponding CR_RVS incumbent segment CSVs (NB 0.887,
IB 0.877, SP 0.869). The manuscript wording "robust across physiological states" is
backed by this table; degradation is described as graceful, not as invariance.

## Rule

Do not alter these manuscript numbers without re-running the recompute above against
the same frozen run directories.
