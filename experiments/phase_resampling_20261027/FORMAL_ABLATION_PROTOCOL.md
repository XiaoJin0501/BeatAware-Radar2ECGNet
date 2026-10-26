# Full-paper ablation — author option1

2026-10-27. Author selected retain existing architecture dimensions/checkpoints,
not capacity-matched retraining. Approved scope is A0–A4 + S1/S2 on MMECG11 folds
and CR_RVS5 folds, seed-base42 (actual42+fold, matching original runners).
This does not authorize former Stage2 Chen/generalization/multi-seed routes.

## Seven variants

| ID | Input to shape decoder | Phase | Event probabilities | Window context |
|---|---|---|---|---|
| A0 | H_window,128ch | no | no | yes |
| A1 | [H_window;P_window],132ch | no | yes | yes |
| A2 | [H_phase;H_window],256ch | yes | no | yes |
| A3 | [H_phase;P_phase],132ch | yes | yes | no |
| A4 | [H_phase;P_phase;H_window],260ch | yes | yes | yes |
| S1 | A0 window shape path, no event loss | no | no | yes |
| S2 | Event-only loss; no meaningful shape output | NA | NA | NA |

All H_window/P_window interpolation uses incumbent align_corners=False;
H_phase/P_phase uses the validated exclusive-right-endpoint grid and
align_corners=True full-grid sampling. All probability/anchor inputs to shape
are detached. No bidirectional shape-to-event gradient coupling is claimed.

S1/S2 retain inactive heads only for architectural compatibility; Table3 marks
these outputs unassessable (NA), never scores random unused heads as model
outputs. Report both allocated and active parameters. S1 has no anchor-based
shape generation or GT substitution. S1/S2 versus A0 assesses removing task
supervision; A4 versus A0 assesses the entire forward-conditioned architecture.

## Audited reuse and training budget

112 variant-fold combinations =7 variants×16 folds. Reuse25:

* A0:16 original validation-selected checkpoints.
* A3/A4:6 Stage1 selected checkpoints (MM1/2,CR1).
* S1:3 original MM shape-only checkpoints, folds1/5/9. State keys/shapes,
  stem/TCN/heads, seed, inputs, shape target/loss, optimization and validation
  selection are compatible; retain old weights untouched.

New training87: A1/A2/S2 48, remaining S1 13, remaining A3/A4 26. Machine-readable
source paths/hashes and strict state-dict load validation: `reuse_audit.json`.
No old test result or new test result chooses a reused checkpoint.

Recipe unchanged: AdamW lr1e-4 wd1e-4, batch16, max100 epochs, patience15, no
scheduler/augmentation changes; MM drop_lastFalse, CR True. Same100-epoch anchor
curriculum for phase paths; it is not shortened based on observed early stops.
Multi-task loss/selector exactly as Stage1. S1: shape loss, validation PCC/-RMSE/R2.
S2:0.3 anchor BCE, validation independent macroF1@100ms then -anchorBCE; no
meaningless shape output enters S2 selection.

## Locked evaluation and inference statistics

ALL weights are trained/frozen before ANY final test in this protocol. Test
class is separate; Stage1's no-test guard is not modified. Test generation uses
only predicted anchors/probabilities, including failures. Reference bounds may
only label diagnostic statuses or form scoring targets, never repair inference.
No oracle or continuous-model readout enters the requested main tables.

Per subject: Fisher-z mean of window Shape-PCC; arithmetic mean window RMSE;
formal frozen R-based one-to-one ±150ms matching for Q/R/S/T errors. Preserve
all subjects/windows and all eligible event denominators; unavailable conditional
event errors remain NA and their coverage/counts are explicitly exported.
Legacy arithmetic-window PCC is retained in per_subject.csv for provenance;
its difference from Fisher-z reporting is not an architecture improvement.

PCC in percent, ΔPCC in percentage points, variant−A4. Bootstrap10000 paired
subject resamples, seed42 local RNG, percentile95%CI of MEAN paired differences
(not Hodges–Lehmann). N=11 MM and30 CR subjects; neither beats/windows nor CV
folds are independent bootstrap units. Single seed, not multi-seed evidence.

Wilcoxon two-sided, wilcox zeros, auto method; all-zero differences yieldp1.
Table1: A0/A1/A2/A3 vsA4 constitute four Holm tests PER dataset. Table2: four
preplanned subject-level PCC contrasts form a separate Holm family PER dataset:

* Phase:0.5[(A2−A0)+(A4−A1)].
* Events:0.5[(A1−A0)+(A4−A2)].
* Interaction:(A4−A1)−(A2−A0).
* Window context:A4−A3.

Holm p<.05 gets*, including inferiority (direction shown by Δ). Table2 R-MAE
effects are descriptive, with available complete-subject counts; negative means
improvement. Table3 descriptive; no extra posthoc testing. No deletions based on
unfavorable results. Different capacities mean these are architecture contrasts,
NOT strictly capacity-controlled causal estimates. Report parameters in S1.

## Outputs and stopping condition

Independent root: `formal_ablation_20261027/` under this directory. Weight/source
selection records, per-epoch logs/curves, per-window test CSV/NPZ, per-subject
CSV, three paper tables (LaTeX+CSV), TableS1 macro and fold-level coverage,
fallback/unmatched/parameter summaries. Report each subject and failure, not
only the star. Existing manuscript/figures/results are never overwritten.

Queue trains/freezes, tests once, generates tables/report, then STOPS. No
additional architecture or baseline route is automatically started.
