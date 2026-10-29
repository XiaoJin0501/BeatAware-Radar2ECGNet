# Scenario-distribution figure: IEEE double-column layout only

Author requested slight layout adjustment to fig_mmecg_subject_wise_results.
Retained2x3 layout,1.8:1:1 column widths, all box/point values, RNGseed42 and
small-group symmetric offsets, event colors, axes ranges, n annotations, panels
and full frames. Updated physical canvas to7.16x4.05in, slightly increased
column/row separation, reduced height, standardized STIX8pt labels/7pt ticks.
Removed tight-bbox export, which previously changed the actual physical width.
PDF now515.52x291.6pt with embedded STIX TrueType; PNG4296x2430 at600dpi.
Also exports SVG. Existing complete scene-event cache reused; no event rescoring,
model inference or training. Plot-source CSV hash unchanged byte-for-byte.
PNG visually checked for label/legend clipping and per-scene sample counts.

Renderer: scripts/figures/plot_scenario_results.py (existing local script now
tracked). IEEEtran figure* inclusion snippet saved alongside figure source CSV.
