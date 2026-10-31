# Unified two-dataset component/task ablation table

Author requested first-image layout:one full-model reference row,architecture
ablation block and training-task block,shared3components+two5metric dataset
blocks. No duplicated full/A0 rows. Fill six missingCov and S1 delta/statistics
from existing full-test per_subject.csv;no training,inference or model changes.

Coverage EXACT original definition:within each subject totalQ/R/S/T matched /
totaleligible,then subjectmean,not equal mean of endpoint percentages. NewCov:
A1MM84.08/CR85.46,A2MM81.95/CR84.80,A3MM67.97/CR76.63. OriginalA0/A4/S2Cov
exactly reproduced. N/A output cells remain N/A(S1event,S2shape).

S1 deltaMM−1.779490pp,CI[−2.546719,−0.943600];CR−1.043497pp,
CI[−2.116574,+0.166923]. Pairedsubject bootstrap10000,seed42. S1Wilcoxon raw
MM.0048828125,CR.0037444327;Holm acrossthese2 additionalposthoccomparisons
=.0074888654 each. Mark*dagger anddisclose NEWposthocfamily. PreserveALL8
originalarchitecture pHolm/stars,four comparisons per dataset. DoNOT pretend
existingcorrectedpvalues belong to one newcombinedfamily. CRbootstrapmeanCI
crosses0 despite signedranktest;different estimands,doNOT modifyeither result.

Main rankings jointly acrossall applicable7rows from unroundedmeans. CR A2PCC
slightlyhigher thanA4 despitebothrounding90.45;A4second. Fullrowshade identifies
reference,not universalbest. Note unequalheadinputwidth/capacity explicitly.

GenerateTable_ablation.tex+Table_ablation_S1.tex;originalpaper tablebackup in
ablation_merged_20261101 alongside14rowCSV,newS1statistics,REPORT,spec,manifest.
Oldexperiments Table_modules/multitask/S1 unchanged. 14columns,IEEEdoublecolumn
resizebox textwidth. Validation:all6Cov filled,noTBD,14armdataset rows,7proper
14columnlines,originalplannedtests unchanged,source hashesunchanged,N/Afields
notfabricated. NoLaTeXengine;structural checkonly. PriorbaselineclinicalGPUjob
kept running independently as requested.
