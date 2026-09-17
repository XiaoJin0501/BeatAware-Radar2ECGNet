# Native-RCG closure and continuous final-candidate preparation

Date: 2026-09-17

## Scientific decision

The capacity-matched B2a-wide/B2b gate completed 6/6 development-validation
runs. B2b improved tolerant beat-event F1 but reduced PCC on every fold and
failed the preregistered waveform and mechanism requirements. Direct native
RCG fusion is closed without post-hoc variants. The complete evidence is
frozen in `docs/audits/Q1_NATIVE_RCG_GATE_AUDIT.md`.

The official continuous candidate is frozen as WSST-driven REM-Net B1 with the
locked L3 MR-STFT objective. A registered 11-fold `loso_v2` seed-42 runner was
added. It selects checkpoints on validation and opens test only afterward,
saving test predictions and a fold-macro summary.

## Verification and blocker

- Native-RCG regression tests: 5 passed.
- New Python entry points compile and expose the expected CLI.
- Formal preflight correctly refuses a dirty tracked worktree.
- Data preflight found that full-window WSST companions exist only for
  train/validation on folds 1/5/9; test companions are absent.
- Approximately 37 GiB is free. Repeating full WSST tensors in every fold is
  estimated at 50–60 GiB and is therefore unsafe.

No 11-fold training was launched. The next implementation action is a
numerically verified deduplicated full-window WSST cache (or added storage).
The known 4-second WSST cache is not an allowed fallback.
