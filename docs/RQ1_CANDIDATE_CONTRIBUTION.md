# Candidate journal extension: morphology-to-waveform consistency

Status: implemented experimental objective, not a demonstrated contribution.
Date: 2026-09-08. Scope: supervised RQ1 on both MMECG and CR_RVS.

## Motivation and lineage

The inspected working conference manuscript, ICONIP2026/ICONIP_2026.tex,
already contains cardiac-prior conditioning, QRS supervision, a temporal
backbone, phase alignment, waveform L1 and multi-resolution STFT loss.
Its final submitted version still needs to be identified before a definitive
conference-to-journal novelty assessment. An event loss or spectral loss alone
should not be presented as a new journal contribution.

REMAlignMTLNet currently produces a continuous waveform and a 200-point
morphology prediction from shared features, with separate supervised losses.
It does not constrain the continuous head's center cycle to agree with the
morphology head. The candidate tests whether an explicit training connection
improves actual waveform fidelity without changing inference or adding weights.

## Objective and limited mathematical justification

For one valid training cycle, let C crop its ECG-derived interval and linearly
resample it to 200 points. Let a = min(C(y)), b = max(C(y))-a, where y is the
reference waveform. Define P(z) = (C(z)-a)/b, with b > 1e-4. The candidate is

    L_cons = mean(abs(P(predicted_waveform) - stopgrad(predicted_shape))).
    L_total = L_existing + lambda_cons * L_cons.

Training uses reference coordinates and amplitude statistics. Inference remains
radar-only and does not call this loss. Predictions are not individually
min-max normalized inside the loss, so amplitude and offset errors remain
penalized. Invalid or flat reference cycles are excluded with reported coverage;
no synthetic fallback interval is used.

The triangle inequality gives

    ||P(predicted_waveform)-P(y)||_1
      <= ||P(predicted_waveform)-predicted_shape||_1
         + ||predicted_shape-P(y)||_1.

This explains the objective only: both terms must be small to bound the
normalized center-cycle error. It is not a generalization theorem, a bound for
all beats, a timing guarantee, or a proof that optimizing it improves PCC.
Detaching the shape target blocks direct gradients into its decoder from this
term, but the encoder remains shared. The teacher is neither frozen nor assumed
correct. An inaccurate shape head can transfer errors.

## Required controls on BOTH datasets

- B0: existing REM-Net, consistency weight zero.
- B1: identical network and training, lambda_cons = 0.05, target = detached head.
- B2: identical network and weight, target = reference normalized cycle. This
  distinguishes cross-output coupling from simply increasing local supervision.

Keep event weight fixed at 0.3 for this experiment. Resolve the independent
A2/A4 event-supervision question separately, rather than changing both factors.
Preserve each dataset's established inputs and subject splits. Use seed 42 for
the initial development comparison. Fix validation checkpoint policy before
launch; never select on test metrics. Prefer validation full-waveform loss for
this objective, with all arms re-evaluated under that same policy.

Primary evidence is continuous-window amplitude error and PCC, accompanied by
waveform-derived R-F1 at 50/100/150 ms, timing error and coverage. Shape-200 and
region metrics check whether shape quality deteriorates. Event-head timing is
reported separately. Report paired subject differences, failure rates, compute
cost and convergence on MMECG and CR_RVS. Inspected test folds remain development
evidence, not a fresh untouched test.

## Decision rules

- If B1 improves both datasets and outperforms B2, investigate the coupling as
  a substantive methodological extension and confirm across seeds/folds.
- If B1 only beats B0 but not B2, the evidence supports extra local supervision,
  not a special benefit from the morphology head.
- If B1 only improves normalized morphology or harms timing/amplitude, do not
  promote it as a continuous reconstruction contribution.
- If it fails, retain the frozen reference and record the negative result.

Novelty is provisional. Related multi-task approaches already exist, e.g.
radarODE-MTL (https://arxiv.org/abs/2410.08656), and time-frequency supervision
exists in LifWavNet (https://arxiv.org/abs/2510.27692). A preliminary literature
search is not evidence of first use of consistency or distillation. Domain
formulation and controlled empirical benefit must carry any eventual claim.

## Implementation

- src/losses/morphology_consistency.py: parameter-free training loss and coverage.
- configs/mmecg_config.py: lambda_morphology_consistency defaults to zero.
- scripts/train_mmecg.py: optional loss plus head/reference control and logging.
- tests/test_morphology_consistency.py: numerical, gradient and model smoke checks.

Only engineering checks have been run. Dataset training, checkpoint-policy audit,
canonical experiment registration and performance comparison remain pending.
No current manuscript result or frozen checkpoint is replaced by this candidate.
