# Current experiments and dependencies

- `phase_resampling_20261027/`: final REM-Net phase_pd architecture and full formal ablation.
  Keep `formal_ablation_20261027/`, Stage0 evidence and Stage1 selected weights: some formal
  cells reuse Stage1 checkpoints. Completed optimizer/resume states were archived externally;
  selected best weights, predictions and logs remain.
- `development/structured_decoder_v0_20261017_ab/`: still a runtime dependency despite its
  historical name. Contains raw-reference caches, common inference/preprocessing and frozen
  WSST source. Do not delete it merely because other development routes are closed.
- `specs/`: only retained current/provenance/runtime-required specifications.

Other closed development routes and specifications are recoverably stored outside this tree.
See `../docs/CLEANUP_20261104.md`. No training was started by cleanup.
