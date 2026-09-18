# Freeze model exploration and pivot to matched baselines

## Decision

REM-Net hyperparameter tuning and new-mechanism exploration are stopped. The
completed seed-42 11-fold result is frozen as the proposed-model endpoint.

The 60-second paired-ECG calibration failed general promotion and is retained
only as Discussion evidence. Its evaluator and regression test were moved to
the legacy archive; its immutable audit and machine-readable result remain.

## Active work

The sole active research path is fair baseline alignment under the repaired
`loso_v2` continuous protocol. Baselines must use matched subjects, 8-second
support, validation-only checkpoint selection, one-time test evaluation, and
the common metric/aggregation contract. Old-split and reported-paper values
remain separate contextual evidence.

## Files moved

- `scripts/eval_60s_calibration_loso.py` ->
  `archive/legacy_scripts/audits/eval_60s_calibration_loso.py`
- `tests/test_eval_60s_calibration.py` ->
  `archive/legacy_tests/test_eval_60s_calibration.py`
