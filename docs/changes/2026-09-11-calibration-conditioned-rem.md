# Calibration-conditioned continuous REM-Net

The passed session-prefix probe authorises a learnable continuous adapter. The
new module retains the REM interpretation: a paired calibration prefix encodes
morphology (`M`), the radar model's event field supplies structure (`E`), and
their interaction modulates a full-window residual on the radar representation
(`R`). It outputs 1600 samples and is identity-initialized.

All factorial controls are parameter-matched. Shape-200 is absent from training,
checkpoint selection, and promotion gates. Formal training remains blocked until
the episodic loader/trainer, smoke overfit, registry, and preflight are complete.
