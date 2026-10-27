# Anchor robustness: final figure and report formatting

Uses the saved paired-subject observations; no inference or training. Fixed x ticks
at 0, 20, 40, 60, 80, 100 ms without relocating data, enabled Unicode minus, and
limited the one-line reference-anchor annotation to panel (a). Full requested
legend labels use two shared rows to fit the 3.5-inch canvas.

The report lists mean and lower-CI zero crossings separately, predicted-anchor
paired delta CIs, and median absolute first-R and RR-span errors. Primary medians
use equal-subject empirical CDF weighting; pooled-window medians and matched,
valid, and all-window subsets are explicitly distinguished. Existing robustness
observations and crossover definitions are unchanged.

Validation: 7 tests passed using tests/test_anchor_robustness_delta.py and
tests/test_anchor_robustness.py. Visually checked PNG; PDF is vector 3.5 by
1.75 inches with embedded TrueType fonts. No unrelated manuscript files changed.
