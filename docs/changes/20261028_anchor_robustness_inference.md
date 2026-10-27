# Author-requested anchor robustness — inference only

Final A4 phase_pd seed-base42, all16 test folds; original reference shapes,
PCC and Fisher-z subject aggregation unchanged. Inject only shape-grid anchors;
predicted event probabilities and context features unchanged. Sigma0/5/10/20/
40/80ms,5 repeats seeds0–4;10 pair-rejection attempts then original GT-free
fallback; no filtering windows, no training or tuning.

Protocol fixed in figures/source_data/anchor_robustness_20261028/PROTOCOL.md.
All reference RR bounds checked valid. Clarify crossover=first paired lowerCI
reaching0,0 if oracle already not demonstrably above context; never-crossed
reports both nominal sigma>80 and realised-error bounds. Predicted star x uses
matched-window error while y retains all-window PCC; report denominators.

New plot/inference script and CSV/PNG/vectorPDF/SVG alongside existing figures;
REPORT and diagnostic arrays in dedicated source-data directory. Existing
model code, checkpoints, figures and old results unchanged. Unit tests cover
zero-noise identity, independent draws, max-retry fallback and crossover rules.
