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
