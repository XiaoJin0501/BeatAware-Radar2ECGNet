# Stage1 author approval and locked gate

2026-10-27 author confirmed six validation-only trainings: MMECG physical folds
1/2 and CR_RVS fold1, phase_p/phase_pd, seed-base42 (actual42+fold, original recipe).
Author separately confirmed equal arithmetic weight for all three development
folds when applying the supplied gate; individual fold values are also exported.
No Stage2/test access is authorized.

Runner and spec: `experiments/phase_resampling_20261027/train_stage1.py` and
`stage1_spec.json`. Original loaders, targets, optimizer parameters and original
harmonic validation selector are reused. True MAE is obtained from the frozen
formal R-matching accumulator (the incumbent selector's scorer returns MedAE,
not MAE); original rank() remains unchanged. Subject Fisher-z PCC is additional
reporting only. Reference labels in eval affect status diagnostics only.

12 tests passed, including gradient blocking, original interp compatibility,
equal-case aggregation, phase_p priority and oracle PlanA stop. A 16-train/
16-validation-window GPU end-to-end CR_RVS phase_pd engineering run passed.
Engineering checkpoints are isolated and never used as results/initialization.
No old artifacts or results changed. Queue uses a clean dedicated worktree,
serial GPU runs and atomic resume states, then STOP after validation-only gate.
