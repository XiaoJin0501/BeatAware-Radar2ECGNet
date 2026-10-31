# Shared-subject conditional localization CDF illustration

Author asked for another CDF showing best REM versus worst Chen/Conformer.
Do not splice different subjects/observations into an apparently same-population
comparison. Implement a shared-subject descriptive alternative: MMECG vsChen,
CR_RVS vsConformer; select the common held-out subject maximizing mean of four
baseline-minus-REM MAE gaps. Same subject for both methods and allfour panels.
No per-panel selection. Disclose post-hoc selection inREPORT/caption,not an
in-plot banner. Original full-population main/supplement figures unchanged.

Read ONLY existing deduplicated_errors.json for original dual-head seed42,not
phase_pd. No inference,training,new detection/matching,offset removal or scalar
error change. Conditional matches remain method-specific; raw eventIDs absent,
so identical paired-event sets are NOT claimed. Include counts and coverage.

MMECG S16 best shared-subject mean gap2.289ms;CR_RVS S13 gap19.804ms. These are
NOT population estimates or claims of first-order CDF dominance. Median/curve
crossings retained. Existing deduplication,full tails and exact inverse ECDF
median convention preserved; only new descriptive subset ECDFs constructed.
Same7.16x3.65in layout/style as main,display-only PCHIP through measured knots.

Outputs:fig_localization_cdf_advantage.pdf/svg/png,exact13871 scalar errors CSV,
CDF-nodeCSV;source_data/localization_cdf_advantage_20261031 includesREPORT,
allsubject selection ranks,16 distributions' metrics/counts/coverage,caption,
manifest/source hashes. Entry:scripts/figures/plot_localization_cdf_advantage.py.
Validation:16 arrays exactly equal source;all13871 errors unchanged;case subject
consistent across4panels andbothmethods;old6 main/supplement hashes unchanged;
render visually inspected. No manuscript metrics or original figures replaced.
