# Dual-head full-LOSO aggregation correction

The seed-42 11-fold training and one-time test evaluations completed before
aggregation. Fold 7 produced no matched structured Q/R/S/T event, so its event
MedAE is undefined (`None`). The initial aggregator attempted to average that
value with floats and stopped without writing `summary.json`.

The correction converts undefined event metrics to explicit missing values,
aggregates only finite folds, and records `n_valid_folds`, `n_missing_folds`,
and the per-fold missing entry. It does not alter checkpoints, predictions,
per-fold metrics, thresholds, or any test inference. Undefined MedAE is never
converted to zero; Fold 7 remains an explicit detection failure through its
coverage/MDR values.
