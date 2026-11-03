# Current script entry points

The final paper model is `phase_pd`; its training/ablation code lives in
`experiments/phase_resampling_20261027/`, not in a new top-level trainer.

| Entry point | Purpose |
|---|---|
| `evaluate_hrv_agreement.py` | Frozen final checkpoints;60s HR/HRV agreement, exact correction and retention accounting |
| `profile_model_complexity.py` | Six trained models, native inputs, FLOP and timing audit |
| `figures/fill_verified_baseline_table.py` | Verified main comparison table |
| `figures/build_merged_ablation_table.py` | Unified architecture/task ablation table |
| `figures/evaluate_baseline_clinical_intervals.py` | Frozen baseline per-beat clinical inference |
| `figures/finalize_baseline_clinical_table.py` | Complete-result guard and clinical-table update |
| `figures/run_baseline_clinical_queue.py` | Existing inference supervisor; not a training queue |
| `figures/plot_*` | Retained paper figures; helpers/versioned modules remain when imported by current plots |
| `export_paper_source_data.py` | Original dual-head paper source bundle; do not relabel as phase_pd |
| `stats_subject_level.py` | Shared subject-level statistics |
| `validate_delineator_public.py`, `validate_fiducial_annotations.py` | Existing reference/annotation validation utilities |

`run_dual_head_interaction.py`, `train_continuous_ecg_pilot.py` and
`train_anchor_head.py` are retained **dependencies**, not the current research route.
They are imported by the frozen/common loaders and evaluators. Removing an old-looking
filename without following imports can break the final model.

Older experiment/training entry points were relocated outside the project. See
`../docs/CLEANUP_20261104.md` for exact paths and recovery commands.
No model/loss/metric/checkpoint-selection change is implied by this cleanup.
