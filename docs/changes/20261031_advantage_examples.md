# Shared-window advantage examples (author requested)

Original v4 preserved. No training,weight changes or benchmark-table updates.
Author approved newly selecting actual test cases where REM-Net beats BOTH
Chen-TCN and radarODE-MTL, with all methods/input/GT sharing each window.
No advantage-selection annotation inside the figure; disclosure is inREPORT,
source CSV,manifest and optional caption for manuscript integration.

Entry point:scripts/figures/plot_examples_advantage.py. Original frozen dual-head
seed42 lineage,not phase_pd. Representative-window PCC recomputed with frozen
target routine and verified against previous candidate scores; cached reference
R differing from frozen prominence handled by SAME frozen delineator. Rank
eligible REM>=.85 with positive gap to both by descending min gap,then REM score,
fold,index. Apnea retains lower-quartile reference HF display screen. Inference
independently decodes EVERY reference beat for ALL3 methods. Accept first ranked
case with actual mean per-beat PCC above BOTH baselines; actual per-beat REM PCC
is not guaranteed>=.85. All attempted cases retained; one insufficient-context
candidate skipped. This is explicitly post-hoc advantage-case selection,NOT
median/representative examples or population superiority evidence.

New outputs figures/fig_examples_advantage.{pdf,svg,png},same7.16x6.3in/600dpi
v4layout. All7 scenarios newly selected consistently. Source complete35waves in
fig_examples_advantage_source.csv; portable single-CSV renderer
plot_fig_examples_advantage.py. Source/selection/provenance REPORT,NPZ,per-window,
per-beat,attempt and candidate logs in source_data/examples_advantage_20261031.

Validation:all7 actual paired wins confirmed,35CSV traces exact,reference-RR
coverage100% on all1024 displayed points,CSV-only redraw pixel-identical,oldv3/v4
sixfigure hashes unchanged. Same preprocessing GT used for scoring/display,
raw-reference alignment verified,no added filter/smoothing. Q/R/S/T labels only
reference first columns; no prediction markers or advantage banner inimage.
