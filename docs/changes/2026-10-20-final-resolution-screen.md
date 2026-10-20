# Final bounded architecture screen — author-approved, pre-result rules

This is the author's explicitly authorized final architecture experiment. It
does not revive any other closed design. New code/output are isolated under
`experiments/development/final_resolution_20261020`; historical v2 checkpoints,
metrics, figures and results are read-only. Spec: same directory `spec.json`.

## Architecture and loss

Baseline A is exactly v2 arm1, reused at the same three folds and seeds.
B changes only waveform loss; C adds native-resolution fusion with ordinary
loss; D combines both. MMECG screening has 27 fresh runs. The native path has
five stride-one kernel7 dilated convolutions at 1/2/4/8/16 with BN/GELU,
widths 50→32→32→32→32→128. CR_RVS uses one input channel with the same design.
All native features retain 1600 samples. Fusion is
`H_tf + sigmoid(Conv1x1(H_tf))*Conv1x1(H_td)`. The event head's architecture,
absolute-time labels and existing H_tf input stay unchanged; waveform fusion
is not inserted into the event readout. Waveform gradients can still update
the shared TF stem/TCN. Parameter budget uses stem+TCN only (conservative).

Input uses audited baseline `rawio.record_data` and `radar_window`: MMECG
clean archive RCG, per-channel per-window standardization; CR_RVS unwrapped,
detrended, downsampled200Hz CW phase, .6–12Hz bandpass, robust window scaling.
This is baseline RCG, NOT unfiltered sensor ADC data. Exact record/start/stop
keys match WSST windows. No delay compensation, ECG-selected channel or lag fit.

Ordinary objective is reused exactly from v2. Tolerant waveform loss uses all
41 integer lags -20..20 at200Hz, temperature0.1, per-example soft-min and batch
mean. Every lag uses the same1560sample support: prediction[20:1580],
reference[20+tau:1580+tau]. No wrapping/padding; no favorable variable overlap.
The formula is UNNORMALIZED as requested: no +T log41. It can be negative and
need not equal ordinary loss numerically. Event BCE stays full1600, weight0.3.
The 50ms shift test checks the tolerant loss remains approximately unchanged.
The finite41-lag soft-min is not mathematically exactly shift invariant: shifting
changes the tail candidates. The initial1e-4 unit-test tolerance failed with
an absolute difference0.0006001 on the deterministic synthetic signal, before
any training. The approximately-equal test uses absolute tolerance0.001;
the requested loss, lag set and screening thresholds were not changed.

## Frozen append-only metrics

`regional_metrics.py` extends the unified frozen v2 interface, not an alternate
implementation of M1/M2/M3. QRS uses native offsets -12..12 inclusive (±60ms),
T uses 61 samples on offsets30..90 (150..450ms), both around the separate
matched waveform R and reference R from the EXACT scored M1 pair. For short
RR, T offsets scale by min(1,.75*RR_samples/90), keeping its right endpoint
inside the M1 beat. Full aligned200sample beats determine std(pred)/std(ref).
PCC aggregation is within-subject Fisher-z; amplitude ratios arithmetic mean.
Unmatched M1 beats are not assigned artificial morphology scores: coverage is
always reported and multiplied into the screening endpoint. Flat prediction
PCC is0; flat-reference amplitude ratio is unavailable and counted.

New regional scores are also computed on the saved v2 arm1 predictions, in the
new output directory only. Baseline checkpoints are not reranked or retrained.
Full native-record,1024center-crop stitching, detection and existing metrics
remain identical to v2. Checkpoint rule, seeds, optimizer, batch/drop_last,
patience15 and first10epochs without patience are unchanged.

## Selection locked before new training

Compare each candidate with A at the same fold/seed. Average seeds within each
validation subject, then average the three subjects. A candidate passes iff
M1×coverage gain≥.01, QRS-PCC gain≥.02 and at least7/9 paired product gains
are strictly positive. Missing required scores cannot pass. No other metric
chooses the winner. Priority is **D, then C, then B, otherwise A**, confirmed
by the author before implementation/training. M2, record-optimal±150ms PCC,
T morphology, amplitude ratios, Q/R/S/T MAE and waveform/head R-F1 are reported.
This remains screening, NOT statistical evidence. No further design is queued.

Chen's nine runs are deferred until this27-run screen completes and its pending
GPU validation timing/cadence prerequisites are satisfied. Formal expansion
scope requires the author's separate clarification; no test access is inferred.

## Pre-training verification

MMECG total parameters: baseline1,438,789; C/D1,533,573. Added94,784 /
original stem+TCN1,355,008 =6.9951%. CR_RVS: baseline1,382,341;
C/D1,466,149; added83,808 /1,298,560 =6.4539%. Both are below15%.
The first training window in each development fold was compared with the
existing baseline split's `rcg` cache: max absolute difference0 in all3folds.
Twelve focused unit tests pass, including common initialization parity, loss
enumeration, exact ordinary objective, fusion gradients, MM/CW output shapes,
regional identity/amplitude tests, strict input support and D>C>B>A selection.
The real-input GPU check completed B/C/D, two one-batch epochs each, validating
one full record each epoch. It saved no weights and verified validation
preserves CPU/CUDA RNG and restores train mode. No screening results were
opened to set or modify the design or selection gates.
