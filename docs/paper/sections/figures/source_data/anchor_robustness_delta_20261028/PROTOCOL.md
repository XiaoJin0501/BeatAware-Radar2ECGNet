# Paired-delta revision, fixed before extended-noise evaluation

Reuse original sigma0..80 per-window observations EXACTLY, without re-inference.
All old source data remain unchanged. Archive old figure files before replacing
the three author-requested fig_anchor_robustness assets.

Y =100*(A4 jitter subject PCC − same subject A0 PCC), percentage points. Repeat
average within subject FIRST as before. Bootstrap10000 paired subjects, local
seed42, same existing bootstrap_mean. Star uses ALL-window per-subject A4−A0 and
paired CI; its x remains mean anchor error on matched central-RR windows.
Dashed Context-only baseline becomes zero, not absolute PCC. All16 frozen A4
weights, features/targets/probabilities/interpolation/constraints/fallback stay fixed.

Extend nominal sigma128ms on both datasets,5 repeats0..4, same noise protocol.
If either dataset lacks a statistical crossover, extend both datasets in fixed
sequence256,512,1024ms and stop when both have a first crossing. If still none
at1024, report censored range/fallback limitations rather than claiming an
unobserved threshold. Do not tune noise seeds, losses or model. No training.

Crossover = lower paired95%CI first reaching0, interpolated in REALISED error;
retain old sigma0..80 points exactly. Report mean crossing separately.
Use shared y limits if ratio of the panels' plotted CI ranges <=2; otherwise
independent. Same3.5x1.75in style/font/color; no extra data lines.

Collect both signed endpoint errors e0,e1 of the USED predicted grid bounds.
RR span error=|e1−e0| ms. Common component c=(e0+e1)/2; differential component
d=(e1−e0)/2. Report matched-window primary values on same subset as star x,
plus all-valid and all-window diagnostics. Per-window quantities averaged within
subject then equal subjects; subject bootstrap for span error. Include same-sign
rate and squared common-energy share to assess, not assume, common-mode dominance.
Common-mode errors within a pair do NOT establish a constant record/device offset.
No assertion that 'mostly common offset' applies outside the measured subset.

Original absolute-PCC CSV/REPORT remain historical artifacts. New delta CSV,
per-subject data, REPORT.md and checkpoints/source hashes are in this directory;
the user-visible new source CSV is anchor_robustness_delta.csv in figures/.
