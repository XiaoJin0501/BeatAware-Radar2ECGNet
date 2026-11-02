# Frozen final-model HR/HRV agreement — 2026-11-03

Author-authorized inference only, no training/checkpoint selection. Specification fixed before running in `docs/paper/hrv_agreement_20261103/spec.json`.

- Same phase_pd A4 selected checkpoints as main table; verify SHA against saved clinical manifest and main-table prediction hashes.
- Recompute only REM stem/TCN/event logits (FP32 eval/no_grad), use original ordered decoder and ±10sample/tau1 soft refinement. All five adapted baselines output R events; reuse their full-precision native refined-anchor caches with source checkpoint hashes.
- Record-level reference R from raw_reference event_r and frozen source hashes.
- Nonoverlap60s at recording time zero. Stitch overlapping native full8s windows by reference-free nearest-center ownership. No GT matching/offset correction. No shape or continuous-waveform generation is needed. Native edge anchors are retained; this differs explicitly from central5.12s clinical waveform scoring. Unsupported full segments are counted as unavailable, not quietly scored from shorter time series. Incomplete record tails listed separately.
- One-pass correction, RR inclusive0.3–2.0s then >20% relative to centered5 original-index local median (including current; physiologic RR only in median). >=80% retained independently for both signals; >=2 intervals for sampleSD/RMSSD. No interpolation. RMSSD uses adjacent retained intervals, explicitly fixed.
- All candidate accepted/discarded segment rows exported. Pairwise intersection per method, same denominators definitions, support and artifact failures separately. HR=60/meanRR; SDNN sampleSD; RMSSD in ms.
- Reuse unchanged repeated_loa from phase_pd clinical evaluator; pooled individual-segment MAE/bias/Pearson, repeated-measures LoA using between+within subject variance. No new p-value tests or training conclusions.
- Short60s RR statistics are not independently confirmed sinus/NN HRV. Method-dependent accepted sets and failures limit interpretation; removed-RR fraction is not detection coverage.

Tests: `python -B -m pytest -q tests/test_hrv_agreement.py`:7 passed (known metrics/sampleSD, exactly20% and80% boundaries, one-pass outlier rejection, ownership/dedup boundaries,60s boundaries, missing predictions retained as failed rows, retained-gap RMSSD convention).

Outputs will be written under the fixed directory; `docs/paper/RESULTS_MANIFEST.md` will include all numeric summary fields, the producing script, commit and hashes. Existing results/figures remain unchanged.

Coverage-completion engineering correction: the initial sparse-only pass could support just71/822 complete CR segments; those provisional outputs are archived under `archive_sparse_test_windows/` and do not constitute final results. Fill missing4s-grid native8s contexts for the same80 formal-test CR recordings using identical raw phase/STFT preprocessing and all six frozen checkpoints. No artifact, eligibility, statistic or selection threshold changes. Existing MM native windows cover all182 complete segments. Original predictions remain in the union; every method gets identical owned input windows. No new subjects or CRRVS_0024 are added. This correction follows the user's request for continuous60s recording segments, rather than accepting fragmented outputs as60s HRV.

Final export checks: all6024 method-segment rows (1004 recording segments ×6) unique and fully supported;36 dataset/method/metric summaries; independent recomputation confirms MAE,bias,Pearson and ANOVA LoA. Ref RR/correction/metrics identical across methods; removals equal physiologic+local outliers. Main checkpoints96 SHA verified. Original16 REM anchor sets reproduce frozen clinical central-window HR to max0.00026401bpm with no finite-status mismatch. Native CR completion adds7147 contexts across60 records (remaining20 need no added windows), exact RCG and bitwise float16 STFT audits. Original sparse outputs retained as superseded engineering audit.

Added reproducible subject_retention and zero-retained-subject report, plus native-grid full60s-support test;8 tests pass. REM CR subject22 has0/30 retained segments (30 prediction correction failures,1 reference failure); keep all30 in discard denominator and report29/30 subjects contributing to agreement, rather than imply all30 remain. No threshold was altered.
