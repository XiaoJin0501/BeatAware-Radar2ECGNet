# Replace illustrative numbers with verified full-test results

Author asked to ignore filled screenshot values and substitute real experimental
results. Update local paper Table_baseline_compare.tex without changing its
layout/citations. Backup exact original table under results_verified_20261101.
No training,inference,new delineation/checkpoint selection or selected advantage
case reuse. Generator:scripts/figures/fill_verified_baseline_table.py.

Model:phase_pd A4 current full seed-base42;five original adapted dual-head seed42
baselines. Same formal cached Shape-200 targets;all16test folds. PCC within each
subject uses formal Fisher-z average,then mean over11/30subjects,matching A4
ablation. This intentionally harmonizes old baseline window-arithmetic/fold-macro
aggregates;do not mix old84.57%Chen with new86.54%Fisher result. Preserve alternate
window-arithmetic PCC in per-subjectCSV for audit. All6same target/count/map.

Events/RR:raw subject_seed_metrics42 for baselines,same frozen accumulator on
cached A4 probabilities/reference points forREM. A4 Q/R/S/T,RMSE andFisherPCC
reproduce formal saved subject rows to1e-7. RR uses adjacent matched event-head
R peaks,not legacy waveform/continuous RR. Conditional MAEs include lowcoverage;
Conformer MMsubject9 has0matches,so5conditional means use10/11 subjects and are
explicitly marked dagger. Subject9 stays in morphology/coverage,never0MAE.

Clinical QRS/QT:only existing phase_pd assembled onset/offset paired_values.csv
with PASS roundtrip. Reaggregate matched absolute errors within subject before
macro mean/populationSD. No pooledBA-MAE substitution. Baselines lack this same
verified clinical evaluation;20cells intentionally unavailable,not filled with
QS/QTpeak or CONTINUOUS_ROOT results. Do not highlight a sole available clinical
method as a comparative winner. Means/SDs1decimal,PCCpercent/RMSE2decimals,noSD.

Correct old Q-Q/R-R/S-S/T-T column names to Q/R/S/T(localization). Rankings based
on unroundedmeans,gray/boldbest,underlinesecond;realdata doNOT putREMfirst on all
metrics. Minimal header/note changes necessary to describe actual numbers.

Outputs:results_verified_20261101/ contains verifiedTex,108-cellsummaryCSV,
246-subject-methodCSV,REPORT,manifest and originalTex backup. Sourcepaths/hashes,
counts,coverage,aggregation and modelversion explicit. Manuscript file local
untracked beforetask;do not stage unrelatedpaper files.

Validation:108cells,246subject rows;20intended unavailableclinical cells;
5disclosed10/11conditionalcells;12proper10columnrows;matching outputhashes and
formalA4reproduction. No LaTeX engine available;structural audit only,notcompiled.
