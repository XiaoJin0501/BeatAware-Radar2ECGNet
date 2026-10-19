# Approved continuous screening v2 — fixed before comparisons and training

Author final approval2026-10-20 supersedes the unexecuted draft. Exactly MMECG
folds1/2/3 × seeds42/43/44 × four arms =36 fresh runs. No test access, no
confirmation stage, no between-arm results inspected to choose these rules.

## Checkpoint

Per validation subject, use the existing M1 (R-aligned PCC, Fisher-z within
subject), matched coverage c, and waveform R-F1 f from the unchanged detector
and ±150ms matcher. a=max(0,M1)*c; score=2af/(a+f), zero if a+f=0 or M1 missing.
Take equal-subject arithmetic mean. No0.8coverage eligibility floor and no new
F1 pass/fail guard: these were in the draft but NOT in the author's final rules.
Missing raw M1 is still missing in logs and cannot pass screening. Exact
checkpoint ties keep the earliest epoch. Track best from epoch1, but do not
accumulate patience during epochs1–10; epoch11 is the first eligible failure.

## New record-level metric, frozen now

One optimal lag per entire retained native record. Scan61 integer lags−30..30
samples (±150ms at200Hz). Compare y[318:L−318] to prediction[318+k:L−318+k]
for every candidate k; fixed common support prevents varying sample counts.
Positive k means prediction late. Ties: smallest |k| then negative k. No
amplitude fit or polarity correction. Fisher-z average across records within
subject, then equal-subject arithmetic mean. This is GT-assisted screening,
not a deployable waveform correction or proof of strict timing superiority.

## Screening — ALL required

Primary contrast is arm3 minus arm2. Arm1 and oracle arm4 are descriptive.

1. Record-optimal PCC: ≥8/9fold×seed paired deltas strictly positive, and
   subject-then-seed macro mean gain≥0.01.
2. M1×matched coverage: same aggregation, gain≥0.01. No separate M1-only gate.
3. Every original, unshifted event-head Q/R/S/T MAE macro degradation≤5ms.
4. Matched coverage macro degradation≤0.02.

M2, M3, strict beat PCC remain reported but never enter checkpoint/gate.
All36 rows required; missing required endpoints cannot pass. No p-values:
this is screening, not statistical evidence. Confirmation11fold stays blocked.

## Unchanged

Model, initialisation, full8s WSST support, train/val split, native reference,
M1–M3, strict beat and event metrics, loss, λ0.3, AdamW1e−4, batch16,
drop_last=True, patience15, maximum100epochs, no scheduler/AMP/clipping.
Validation preserves RNG and mode. Added CPU metrics consume no training RNG.
Isolated fold processes have disjoint runs; no shared GPU RNG state.

## Traceability

Source worktree: BeatAware-Radar2ECGNet-continuous-screen-v2, branch
exp/mmecg-event-conditioned-screen-v2. New output results_v2/. Old runs moved
recoverably to archive/experiments/event_conditioned_screen_v1_20261020/runs,
with per-file hashes verified. Old diagnostic reports, A artifacts and raw
feature caches remain untouched; v1 pause remains. No old weights resumed.
Commit implementation and registry before production training; freeze source,
dependencies, package versions, split inventories and input-cache hashes.
