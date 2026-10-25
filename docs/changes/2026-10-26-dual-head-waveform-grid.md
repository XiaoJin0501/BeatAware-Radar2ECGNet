# Fig4: two-dataset dual-head waveform grid

Author requested original dual-head REM-Net, not the later continuous-decoder
screening model. No training, inference, checkpoint selection or aggregate table
change is performed. Input is the immutable2026-10-16 source-data snapshot.
Compare its seed42 REM-Net, adapted dual-head Chen-TCN and radarODE-MTL outputs.
Do not label Chen here as the not-yet-trained native categorical AR model.

Entry: `scripts/figures/plot_dual_head_waveform_grid.py`.
Tests: `tests/test_dual_head_waveform_grid.py`.

Outputs replace only the requested `fig_mmecg_waveform_examples` PDF/PNG, adding
SVG. Existing files are backed up under
`docs/paper/sections/figures/source_data/fig4_dual_head_20261026/previous_*`.
The figure is exactly7.16in wide,6.65in high; PDF/SVG are vector graphics and
PNG is600dpi. RQ1.tex and all other figures/tables remain untouched.

## Scientific display contract

Each method's one representative200sample shape is min-max normalized and
uniformly inverse-resampled to its OWN predicted R-R intervals. Predicted
structured anchors use the exact archived decoder, loaded as a pure function
from its AST without importing neural training code. A first/last interval is
extended using its nearest predicted RR; fewer than twoR anchors yields a flat
explicit failure. No GT anchors enter prediction assembly. No Savitzky–Golay,
gain/offset fit, lag optimization, polarity correction, Q/S/T warping or selected
high-quality subregions. The central1024points (5.12s) are retained.

Reference ECG is normalized independently within its OWN GT R-R intervals.
Markers on REM-Net are event-head predictions, not delineated extrema of the
assembled waveform. They therefore need not coincide with Q/S/T waveform peaks.
Panel PCC is computed on the actual displayed5.12s normalized signals, not the
representative Shape-200 PCC. GT self-PCC=1. The figure neither proves native
continuous regression nor changes original population reconstruction metrics.

## Selection, fixed independently of baselines

For every subject/scenario, Fisher-z-average all REM-Net displayed-window PCCs;
choose the subject closest to the median across subjects. Within that subject,
choose its window nearest its median window PCC. Flat failures are retained.
Break ties by subject ID, then record/start/index. No baseline score influences
selection. Both baselines use exactly the selected record and window row.
MMECG scenarios NB/IB/SP/PE map to normal/irregular breathing, sleep and
post-exercise. CR_RVS NB/IB/SP map to Resting/Valsalva/Apnea.

The zoom reuses the Resting example; choose predicted R nearest window center
with full support, then crop -.25RR to+.75RR: oneRR duration, not boundaries
exactly at twoR peaks. The same absolute crop is applied to both traces, without
independent alignment. The zoom is the only exception to the5.12s x-scale and
is explicitly labeled. Its PCC is a local-crop PCC, not the window score.

Radar is existing preprocessed test RCG, exact record/start metadata verified.
Filter .8–3Hz over the entire8s, choose the channel with maximum cardiac-band
to total AC energy, then retain center5.12s. No ECG-dependent channel choice.
Pink/green grid y spacing is decorative in dimensionless units, not mV.

Selection IDs, candidate and subject scores, source SHA256s, channel/band and
plotted arrays are saved in `source_data/fig4_dual_head_20261026/`.

## Suggested caption

Representative radar inputs and ECG visualizations from the original dual-head
framework on (a) MMECG and (b) CR_RVS. The same5.12s central input window is used
for all methods. Subjects and windows are selected by a predeclared median rule,
not by best reconstruction quality. Model traces are representative-beat shapes
assembled using each method's predicted R anchors, not native continuous network
outputs. ECG amplitude is independently normalized beat-wise; reported PCCs
compare the displayed signals without temporal alignment. Gray dashed lines
indicate reference R peaks, and colored markers indicate REM-Net's predicted
Q/R/S/T events. The rightmost panel in (b) shows one predicted-RR-duration crop
of the Resting example on a different, labeled time scale. Grid amplitudes are
dimensionless and do not imply calibrated millivolts. These panels illustrate
typical model behavior and are not selected superiority examples.
