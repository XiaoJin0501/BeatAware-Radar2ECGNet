# Calibration-conditioned REM Gate 1

The validation-only fold-1 full-arm pilot completed at the registered 80/20
budget.  The frozen B1 backbone and the adapter were evaluated only on
sample-disjoint post-20-second query windows; MMECG test data and Shape-200 were
not opened or computed.

Best validation checkpoint was epoch 77: PCC 0.0226, RMSE 1.2010, R2 -0.4488,
F1@100ms 0.9666, composite 0.1540.  The registered conventional lag+affine
control on the identical query set is PCC 0.4970, RMSE 0.8700, R2 0.2356.
Therefore the stage-1 requirement (PCC >= conventional + 0.05 without worsening
RMSE or R2) fails on every required dimension.  Folds 5/9 and the factorial
arms are not authorised by this experiment.

The train-only Stage-0 overfit check passed (loss -69.2%, PCC 0.6825 -> 0.9294),
so the failure is not a dead residual path.  It bounds this architecture's
ability to transfer a learned calibration transform across held-out subjects.
Any successor must be registered separately rather than modifying this result
post hoc.  A scientifically motivated successor is to make the already-validated
conventional session lag+affine correction the identity-initialized control and
ask whether Event x Morphology residual learning adds value beyond it.
