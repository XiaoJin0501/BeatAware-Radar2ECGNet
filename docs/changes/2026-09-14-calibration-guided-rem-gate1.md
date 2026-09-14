# Calibration-guided REM Gate 1

This independently registered successor placed the deployable per-record
20-second lag+affine calibration in front of REM residual learning.  Epoch 0
exactly reproduced the conventional control and was eligible for checkpoint
selection.  Validation and test policy were unchanged; Shape-200 was absent.

Fold-1 full arm selected epoch 23 and early-stopped at epoch 43.  Conventional
to learned metrics were: PCC 0.4970 -> 0.5286, RMSE 0.8700 -> 0.8514, MAE
0.5074 -> 0.4853, R2 0.2356 -> 0.2639, F1@50 0.9513 -> 0.9591, and F1@100
0.9682 -> 0.9805.  The improvement is directionally consistent across every
reported metric, but delta PCC is +0.0316, below the preregistered +0.05 gate.
Therefore Gate 1 fails and folds 5/9 plus the factorial arms remain unlaunched.

The result bounds the current global-token design rather than calibration as a
whole.  A defensible successor would preserve calibration ECG as an explicit
morphology memory or dictionary and use event-conditioned retrieval/synthesis,
instead of compressing the paired prefix to a single 64-dimensional token.
