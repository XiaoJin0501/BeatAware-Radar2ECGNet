# Retained checkpoint dependencies

The active final phase_pd/ablation runs are under `experiments/phase_resampling_20261027/`.
This directory retains the original checkpoint families still referenced by final selections,
adapted baseline inference and paper figure helpers:

- `q1_dual_endpoint_baselines_mmecg/`
- `q1_dual_endpoint_baselines_crrvs/`
- `q1_dual_endpoint_remnet_crrvs/`
- `q1_dual_head_balanced_full_loso_seed42/`
- `q1_dual_head_component_ablation_seed42/`

Every selected checkpoint was SHA256 checked before and after cleanup. Completed-run
`last.pt`/`last_state.pt` optimizer states are recoverably archived, while best checkpoints,
predictions and source metadata remain. Historical continuous-regression, calibration,
closed architecture routes and old multi-seed families were relocated externally.

`EXPERIMENT_REGISTRY.csv` records relocated experiment directories as `Archived` with
an external path. Its pre-existing edits were retained. See `../docs/CLEANUP_20261104.md`.
