# Author option1 — complete paper ablations

Author confirmed retain existing variant dimensions/reuse, full MMECG11+CR5,
seed-base42. Added window_p(A1) and phase_d(A2); original phase_p/phase_pd and
interp code paths unchanged. S1/S2 supervise only the corresponding task.

Protocol/spec freeze before training:112 variant-folds,25 audited reusable
checkpoints,87 new trainings. All training frozen before final test. Independent
test class/barrier, immutable selected weights, separate outputs. Exact original
optimizer/data/loss/validation selection for each applicable task retained.

Subject-level bootstrap10000/95%CI, two-sided Wilcoxon and prespecified Holm
families; Δ variant−A4 in percentage points. Factorial contrasts match author
formulae. Capacities differ; no capacity-controlled causal or bidirectional
gradient-coupling claim. Single-seed evidence. Existing manuscript unchanged.

Tests cover new path shapes/stop-gradient, single-task masks/selectors, test
barrier, bootstrap RNG isolation/formulae/Holm, and synthetic export of all
three tables/S1 with11/30 paired subjects. GPU16-window A1 engineering run
passes; engineering weights never reused. Dedicated clean worktree/committed
registry precede formal queue launch.
