# Authorized frozen-baseline clinical-interval completion

User asked to fill missing QRS/QT cells for five adapted baselines. Frozen
seed42 checkpoints,allMM11+CR5test folds. No training,new model,head change,
checkpoint selection,offset correction or selected example use.

Entry:evaluate_baseline_clinical_intervals.py;supervisor
run_baseline_clinical_queue.py;completion callback
finalize_baseline_clinical_table.py. Runtime spec,per-record resumes/progress,
checkpoint hashes and complete sourcepairs in baseline_clinical_20261101.
Reuse phase_pd frozen clinical delineate/make_rows/dedup and raw references;
assert identical reference interval sets/values against REM-Net before scoring.
All valid predictedRR pairs in full1600 synthesized BEFORE central1024 scoring.
RR limits60–400,Rmatching30samples,both interval-endpoint matching30samples,
fixed-zero missing-region fallback. All failures stay in eligible denominator.

Baseline head natively emits one beat per8s context:independently rerun its OWN
shape readout for each predictedRR-centred8s radar context. Never copy a single
representative beat. Shape-only forward asserted identical to nativefullforward.
Native RCG from frozen raw preprocessing verifiedagainst testH5;radarODE needs
TF,including CR65x93->71x240bilinear. CPU50rangebin WSSTvectorization bitwise
float16-verified against original per-bin implementation on2records and native
cached TF at each evaluated record. Frozen CNN/GRU/Conformer/DCN weights.

IMPORTANT context distinction:baselines use separately centred8s radar contexts
per beat;REM phase_pd reuses its parent8s features. Endpoint scoring/matching
are common,but radar support and total synthesis inference cost differ;do not
claim identicalfixed-input-budget generation or reuse per-pass latency as total
clinical assembly latency. This native-interface difference is explicit in
REPORT/mainnote/coverage supplement.

Detected early implementation bug before any completed population/table result:
initial predictor synthesized only RRpairs intersecting central ROI,which can
bias boundary delineation. Corrected to all validfull-windowpairs,regression
2tests pass. Provisionalpairs/metas moved to archive_initial_central_support,
not included in final collect. Unchanged per-beat native model outputs safely
reused by context/checkpoint hash;missing full-window contexts added. No metric
threshold or model was tuned after results. Formal main table still unavailable
until COMPLETEmanifest;no provisional clinical cells published.

All5baseline subject clinical MAEs/coverage/assessability will be exported and
merged by the verified-table generator. Per-subject matched MAE thenequal-subject
mean/populationSD;no pooledBA-MAE substitution,no0error imputation. Main Pcc,
RMSE,Q/R/S/T,RR values must remain exactly unchanged(asserted bycallback).
Clinical no-match subjects disclosed,not silently treated as fullcohort.
GenerateTable_baseline_clinical_coverage.tex for n/coverage/fallback. Callback
protects any concurrentmanuscript edits by writing an independent updatedTex
instead. Inference error stopswithout filling partialdata. Current job is still
RUNNING atthiscommit;no new completedclinical baseline results claimed yet.
