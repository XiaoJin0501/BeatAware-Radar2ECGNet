# Resilient formal-training state

Date: 2026-09-18

## Scope

This is an engineering-reliability and registered-budget update for
`q1_remnet_continuous_strict_loso_seed42`. It does not change the model, loss,
data, checkpoint score, or evaluation metrics. The formal budget is now 100
maximum epochs with early-stopping patience 15.

## Guarantees

- `last_state.pt` is atomically replaced after every epoch and contains model,
  optimizer, explicit scheduler state (`None` for the frozen constant-LR
  protocol), patience counter, best validation score/epoch, history, initial
  losses, and Python/NumPy/Torch CPU/Torch CUDA RNG states. The train-loader
  generator state is also retained.
- Checkpoints, JSON, CSV, and compressed NPZ artifacts use same-directory
  `.tmp` files followed by `os.replace()`.
- The per-fold state machine permits only
  `TRAINING -> CHECKPOINT_FROZEN -> TEST_EVALUATING -> COMPLETE`; COMPLETE is
  terminal. Test construction is guarded by `TEST_EVALUATING` and occurs only
  after validation freezes the checkpoint.
- The 11-fold orchestrator resumes incomplete folds and skips only a fold with
  both terminal COMPLETE state and atomic test metrics.

## Verification

Eight focused regression tests passed. In addition, a real fold-1 CPU smoke
run trained epoch 1, exited immediately after the atomic epoch checkpoint with
code 75, and restarted from epoch 2. Its final state was CHECKPOINT_FROZEN,
history was exactly `[1, 2]`, no `.tmp` file remained, and neither test metrics
nor test predictions existed. The temporary smoke tree was removed afterward.

No formal 11-fold training was launched.
