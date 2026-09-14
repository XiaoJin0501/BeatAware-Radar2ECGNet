# Twenty-percent supervised calibration probe

Fold-1 was evaluated under a subject-calibrated few-shot protocol, not strict
zero-shot LOSO.  For each held-out record, the chronological first 20% supplied
paired radar/ECG calibration and exactly 100 adapter-only optimization steps;
the final 80% was source-sample-disjoint and never selected steps or checkpoints.
The MMECG test split and Shape-200 were not opened.

Continuous query PCC was 0.5063 for conventional lag+affine, 0.5376 for the
source-trained calibration-guided REM, and 0.5518 after 20% supervised
adaptation.  Adaptation changed MAE 0.4956 -> 0.4105 and F1@100 0.9666 ->
0.9834 relative to conventional, but delta PCC was only +0.0455 (gate +0.05),
delta over source REM was +0.0142 (gate +0.03), and R2 0.2518 was slightly below
the conventional 0.2530.  The preregistered gate therefore fails.

This result does not authorize tuning the number of steps on the query set.
It suggests that ordinary parameter adaptation underuses the paired ECG;
the next model hypothesis should retain calibration beats explicitly as a
morphology memory and retrieve/synthesize them from radar-derived event context.
