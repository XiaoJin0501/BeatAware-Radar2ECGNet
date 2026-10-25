# Per-beat Fig4 and predicted-anchor FigS1

Author confirmed per-beat centered inference for every plotted method. Added
`plot_per_beat_morphology.py` and an optional morphology mode to the existing
grid renderer. Reused original best checkpoints and original target/PCC code;
no training, checkpoint selection, original table edits or template replication.

Seven median-selected scene examples contain51 distinct reference beats;
each of three methods has an independently inferred shape for each beat.
FigS1 uses43 separately inferred predicted-interval contexts. None of these
selected panels required flat fallback support (all seven masks have0 samples),
but masks and gray shading are implemented and logged. No gray segment is
fabricated to suggest a failure. Seven unit/artifact tests pass. Sixteen original
REM-Net fold score reconstructions match within1.3933e-6. PDF/SVG are vector,
7.16in wide. Source arrays/checkpoint hashes/selection/captions are in
`docs/paper/sections/figures/source_data/fig4_per_beat_20261026/`.

Full adaptation/source audit:
`docs/audits/Q1_TABLE_BASELINE_FORMAT_AUDIT_20261026.md`.
It records the current table's mixed continuous/dual-head sources and incorrect
interval labels; these were identified, not silently repaired. All old figures
were backed up before replacement. No unrelated worktree edits were staged.
