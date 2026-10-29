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
