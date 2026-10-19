# Continuous-regression screening v2 (author final approval2026-10-20)

Author authorization2026-10-20: commit final v2 rules, archive v1 runs, then
execute all36 MMECG runs fresh. No old checkpoints/metrics reused for screening.
Folds1/2/3, exact seeds42/43/44, arms1/2/3/4. These exact seeds supersede the older
seed-base-plus-fold convention. Full training splits, full validation records;
no test access.11-fold confirmation remains BLOCKED even if screening passes.

Architecture/loss: unchanged tested continuous four-arm model. Arm3 gates use
detached event probabilities; arm4 alone accepts GT event Gaussian oracle masks.
No Shape head. L1+0.5(1-PCC)+adjacent-difference L1+0.3event BCE on1600points.
AdamW1e-4, batch16, max100/patience15, drop_last=True. No scheduler/AMP/clipping.
Select subject-macro harmonic(max(0,M1)*matched coverage,waveform R-F1).
Epochs1–10 consume no patience. No0.8eligibility floor or extra F1 gate.
Arm2/3 differ516parameters(0.03217%).

`metrics.py` is shared by training, final validation and A supplements. Source and
dependencies are hashed by `freeze.py` BEFORE supplements/training; any later
change aborts. `metric_contract.json` resolves RR definition, boundary coverage,
lag support/sign/tie breaks. M1 uses waveform-detected R (same as waveform R-F1),
not event-head R. It is matched/R-aligned morphology, not unaligned continuous
fidelity. M3 is reference-selected timing-tolerance analysis, not deployable
alignment. Strict native beat PCC and RMSE remain visible.

M1 matched coverage includes all GT R in retained support. Full-window boundary
failures remain in that denominator; scored coverage and counts are separate.
M2 unchanged A complete1024sample score. M3 compares fixed1004sample common support
for lags-10..10; positive lag means prediction delayed. No interpolation shift
search beyond those integer samples. No metric tuning after screening begins.

Screen rule: at least8of9 arm3-minus2 record-optimal PCC deltas positive;
seed-averaged then subject-averaged record-optimal PCC AND M1×coverage gains
>=.01; each original unshifted endpoint MAE degradation<=5ms; mean coverage
degradation<=.02. One GT-selected lag per record±150ms on fixed common support.
M2/M3/strict beat PCC are report-only. No Wilcoxon/p-values here;
seeds are not independent subjects. Arm1/oracle4 are descriptive only.

Run from clean committed isolated source worktree; old dirty user worktree is
read-only resource root. `REM_REFERENCE_ROOT` identifies that root;
`REM_SCREEN_OUT` optionally relocates NEW outputs only. Default results live in
the original project's new event_conditioned_screen_20261019/results subtree.
No old A artifacts, checkpoints, manuscript tables or figures are overwritten.
Default output is results_v2/. v1 runs are recoverably archived with hashes.

```sh
python -B -m pytest experiments/development/event_conditioned_screen_20261019/test_metrics.py experiments/development/event_conditioned_screen_20261019/test_design.py experiments/development/event_conditioned_screen_20261019/test_v2.py -q
python -B experiments/development/event_conditioned_screen_20261019/check_v2.py
# Commit code + registry, then verify git status is clean before the next commands.
python -B experiments/development/event_conditioned_screen_20261019/prepare_v2.py
python -B -u scripts/run_continuous_screen_v2.py
```

Each epoch atomically saves full last state and best validation checkpoint.
Resume restores optimizer/history/patience and all four RNG states plus loader
generator. Validation isolates RNG and uses eval/no_grad. Final best predictions
are saved as native window outputs and event probabilities per record. Completed
runs are not retrained on restart within v2. v1 is never resumed. Reports list
all36 rows; missing rows abort. Three disjoint fold workers own12runs each.

Local output artifacts: METRIC_FREEZE.json; AUTHOR_APPROVED.json; INPUT_FINGERPRINTS.json;
data/fold_XX validation features; runs/fold_XX/seed_YY/arm_Z result/checkpoints;
screening_results.csv, screening_decision.json, summary.json and REPORT.md.
**This is screening, NOT statistical evidence. Stop after report.**
