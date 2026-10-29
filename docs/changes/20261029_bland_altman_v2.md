# Bland–Altman v2 presentation and HR matching-error attribution

Author requested presentation-only2x3 RR/QRS/QT at7.16x3.6in. Original paired
values, all original figures and six selected statistical rows remain unchanged
(verified against original output SHA256 manifest and exact CSV row equality).
Main points are subject means (11/30), sidebars ALL paired beat differences,
5ms bins. LoA remain original beat-level Bland/Altman2007 variance components;
regression remains original beat-level slope, drawn only if stored CI excludes0.
No triangles or discarded statistical observations. CR_RVS QT range expands to
show subject22's mean−95.90ms; symmetric limits include both LoA and all subject
means plus15% margin. All display-overflow counts and fractions exported.

HR attribution reads existing A4 probability outputs and original reference-R
audit, recovers native soft-refined R and verifies all HR pairs within0.001bpm.
No checkpoint loading, model inference or training.150ms one-to-one central-R
matching: any-missed/any-extra overlap; exclusive missed-only/extra-only/both/
neither also reported. These associations are not causal attribution and can
include temporal mislocalization or window boundary crossings.

MMECG:1367 finite-HR windows with error>5bpm; missed-only1075, extra-only1,
both291, neither0;91 unavailable HR windows separately retained. CR_RVS:1052
bad windows; missed-only875, extra-only0, both177, neither0;232 unavailable.
Any-missed99.93/100%, any-extra21.36/16.83% (MMECG/CR_RVS).

Outputs: fig_bland_altman_v2.pdf/svg/png, bland_altman_stats_v2.csv, new
source_data/phase_pd_bland_altman_v2_20261029/REPORT.md, subject means, display
overflow and per-window/summary HR CSVs and source-hash manifest. Three tests
passed; visual inspection complete; all original output hashes preserved.

Follow-up author revision: lower-left annotations preferred with rendered marker
and LoA clearance checks; occupied panels use upper-left with symmetric expanded
headroom. Both original beat-level cluster-bootstrap slope/CI and NEW equal-weight
subject-mean OLS/bootstrap10000seed42 are exported; the plotted line now uses only
the subject-mean fit and its CI. Legacy original columns are unchanged.
MMECG QRS has beat slope+0.8334 but subject-mean slope−1.8375 (CI−4.0011,
−0.2457); these are different levels of variation, not interchangeable fits.
Worst absolute CR_RVS QT subject mean:22, officialGDN0022, mean−95.90ms,
MAE101.95ms, coverage167/828=20.17%. Scenes: Resting80windows/47matched of339
eligible, Valsalva80/69of342, Apnea37/51of147. Matched error median−100ms,
IQR[−115,−85]ms, range[−175,+130]ms;97.01% negative. Full eligible pairs,
5ms error histogram and scene summaries separately exported. No outliers removed
or offset subtracted; this is a conditional automated-delineation result.
Seven tests pass, including equal-subject fitting despite unequal beat counts,
preservation of original columns, and lower-left annotation preference.

2026-10-30 author display-only revision: remove every orange regression line.
Added --render-only, reading saved pairs and stats without refitting or running
HR attribution. Preserved previous x limits via data-limit updates, retaining
all other elements. SHA256 checks confirm unchanged stats CSVs, paired values,
subject means and display-overflow CSV. SVG has no orange strokes. Report
explains that regression_line_drawn is now a historical eligibility column;
all numerical slope/CI values remain unchanged. Exported7.16x3.6in PDF/SVG/PNG
and IEEEtran figure* inclusion snippet. Render-only manifest saved separately.
