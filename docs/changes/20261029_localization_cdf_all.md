# Six-method localization CDF from unchanged existing nodes

Author request:2x4 Q/R/S/T panels, MMECG and CR_RVS; six methods, fixed palette
and line styles, REM-Net drawn last/highest zorder; STIX,8pt labels/7pt ticks,
7.16in wide; one top legend. Existing two-method figure retained byte-for-byte.

No inference, training, error calculation or matching. Uses exact_cdf_nodes.csv
from baseline_event_cdf_20261027, the SAME original dual-head seed42 benchmark
underlying the referenced two-method figure (not newer phase_pd).48 distributions,
equal-subject source weighting and overlap-median errors unchanged. Exact step
plot, no interpolation, smoothing, tail exclusion or renormalization at150ms.

Source definition requires caption correction:150ms is the one-to-one R-match
tolerance; Q/S/T inherit beat matches and can exceed150ms. Caption explicitly
states this rather than asserting all individual endpoints were within150ms.
Common event-head architecture means independent weights and backbone-specific
input-channel counts, not shared weights. R-MDR main-table reference is included
as requested; report flags its missing literal heading in the current local table.

Canonical styles: scripts/figures/paper_method_styles.py. Entry point:
scripts/figures/plot_localization_cdf_all.py. Output new fig_localization_cdf_all
PDF/SVG/600dpi PNG and source_data/localization_cdf_all_20261029 with preserved
source nodes, displayed nodes, tail fractions, caption, report and hash manifest.
Visually checked single-row legend, all48 curves and panel labels. PDF
515.52x277.2pt, embedded STIX TrueType. Data-integrity tests cover retained tail
denominator, exact boundary CDF and rejection of invalid empirical nodes.

Follow-up: --main exports fig_localization_cdf_main with2 curves per panel:
REM-Net vs Chen-TCN for MMECG, vs Conformer for CR_RVS. Fixed dataset-wide
selection uses lowest mean of the four EXISTING subject-macro Q/R/S/T MAEs
among baselines (40.8858ms Chen-MMECG;23.1893ms Conformer-CR_RVS); same
baseline in all four panels, no per-panel selection. No errors are recomputed.
Selection ranking is exported from the precomputed metric table. Both main and
supplement captions explicitly state the display selection rule. Existing
fig_localization_cdf_all.pdf/svg/png retained byte-for-byte as the supplement;
original two-method plot also unchanged. Main16 distribution node arrays
verified equal to corresponding six-method arrays. Three data-integrity tests
pass; main PNG visually inspected, one shared three-entry legend with dataset
qualifiers. Input model/data lineage remains the original dual-head seed42.

Main-figure beautification:7.16x3.2in,0–120ms ticks0/40/80/120, y ticks0/.5/1.
Removed top legend; first panel of each row has its own lower-right2-entry
legend, vertically separated from6.3pt median text. Row titles name the baseline;
x labels bottom row only; CDF label first column only. All8 median annotations
reuse existing p50_ms (equal-subject CDF median, full error support), never a
median of120ms-cropped observations. Existing error data remain unchanged; only
display nodes clip at120ms with original tail denominator. Medians exported in
median_annotations.csv. Main caption states selection, weighting and actual
R-anchored150ms matching; no falsely endpoint-truncated claim. Supplement graph
files unchanged. Three existing data-integrity tests pass; main PNG visually
checked and PDF dimensions515.52x230.4pt confirmed.

Latest requested display revision: main7.16x3.65in to accommodate x labels
Error(ms) on all8 panels and complete four-sided frames. Main curves now use
monotone PCHIP through original empirical knots (including original nodes
beyond120ms to avoid an artificial flat edge); caption/report explicitly say
inter-knot values are interpolated, not additional measured data or an exact
ECDF. Source errors, nodes, medians, aggregation and tail denominators unchanged.
The six-method supplementary plot remains an unchanged exact-step version.
