# Close zero-shot successor gates

Two bounded development-validation gates were completed on repaired `loso_v2`
folds 5 and 9. Neither authorises a successor architecture.

## Signed-RCG phase fusion

The parameter-matched P0/P1 comparison failed. P1 minus repaired B1 macro PCC
was -0.0535, and R2 did not improve on both folds. Direct zero-shot phase
fusion is closed under the registered rule.

## Population residual dictionary

PCA32 represented only 25.7% and 30.1% of training residual energy. Even an
oracle projection using validation ECG reached PCC 0.6245 and 0.3734, below
the registered 0.75 ceiling on both folds. The deployable radar-ridge readout
changed macro PCC by -0.0015 versus B1. A deep residual dictionary is therefore
not authorised by this audit.

Both gates are validation-only, use no held-out-subject calibration, leave the
test split unopened, and do not compute Shape-200. These are route-selection
results, not headline paper results.
