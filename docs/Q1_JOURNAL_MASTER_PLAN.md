# Q1 Journal Master Plan

This document is the single authoritative roadmap for the IEEE Q1 journal
paper. When an older roadmap, experiment note, or agent recap conflicts with
this file, this file governs. Scientific definitions remain subject to
`RESEARCH_CONTRACT.md`, `DATA_PROTOCOL.md`, and `METRIC_CONTRACT.md`.

## Paper objective and model identity

The sole objective is a rigorous Q1 paper on radar-to-ECG reconstruction. The
model is **REM-Net: Representation–Event–Morphology Network**, with explicit
event–morphology factorization:

- **Representation:** a shared radar representation of cardiac mechanical
  activity;
- **Event:** cardiac-event timing and cycle structure;
- **Morphology:** beat-local P–QRS–T morphology recovery under waveform and
  spectral constraints.

`REM-AlignNet` is a deprecated working name. Do not create a new model name or
change the core idea merely to accommodate an unfavorable experiment.

## Factorized three-tier benchmark

### Tier 1 — Beat-normalized morphology (Shape-200)

Purpose: measure the morphology recoverable when the cardiac cycle is known and
mapped to a normalized 200-sample grid. This is a morphology upper-bound task,
not continuous deployment.

Required metrics: beat-wise PCC, RMSE, and R², reported under an explicit
normalization and fold/subject aggregation contract.

### Tier 2 — Strict blind continuous stream

Purpose: measure real blind deployment under repaired strict LOSO, without ECG
priors at inference. The primary output is the full 8-second, 1600-sample ECG.

Required metrics: continuous PCC@0, R-F1 with stated tolerance, R-peak MAE, and
IBI MAE. Checkpoint selection is validation-only; test ECG cannot influence
normalization, alignment, or model selection.

### Tier 3 — Temporal-tolerance diagnostics

Purpose: quantify sensitivity of continuous reconstruction to residual timing.
Max-Lag NCC uses ground truth to choose the lag and is therefore a diagnostic
upper bound, never deployable performance.

Required readouts: Max-Lag NCC at ±20, ±50, ±70, and ±100 ms, accompanied by
the selected-lag distribution and boundary-hit rate.

## Frozen Step-1 evidence

On repaired `loso_v2` folds 1/5/9 with frozen B1 checkpoints:

| Readout | Fold-macro value |
|---|---:|
| Continuous PCC@0 | 0.223 |
| Max-Lag NCC @ ±20 ms | 0.366 |
| Max-Lag NCC @ ±50 ms | 0.563 |
| Max-Lag NCC @ ±70 ms | 0.624 |
| Max-Lag NCC @ ±100 ms | 0.635 |

The curve bends and begins to saturate around 50–70 ms. This time scale is
compatible with electromechanical-delay magnitudes discussed as EMAT, but the
experiment does **not** identify EMAT: lag is selected using ground-truth ECG
per window and may also contain model error, preprocessing delay, and
cross-subject generalization error. The defensible result is a large
timing-sensitive diagnostic ceiling.

Artifacts live in
`experiments_mmecg/q1_temporal_tolerance_curve_audit/` and are registered in
`EXPERIMENT_REGISTRY.csv`.

## Step 2 — loss gate completed

The `q1_loss_gate` completed all nine formal runs on 2026-09-17. It compared on
folds 1/5/9, with all other variables fixed:

- L1: Huber + event BCE + PCC loss;
- L2: Huber + event BCE + first-derivative L1;
- L3: Huber + event BCE + MR-STFT using windows/FFTs 128, 256, 512.

The fold-macro L0/L3 comparison is PCC 0.2231→0.2935, R² -0.1356→+0.0121,
NRMSE 2.2719→2.1319, QRS MAE 1.8622→1.8092, and ST/T MAE 0.5720→0.5294.
However, waveform F1@100 falls 0.9027→0.8688, and the registered smoothing
ratios do not consistently move toward 1. L3 is therefore the waveform winner
but not a clean pass of the original joint multi-metric promotion rule. L4 is
not authorized, and Step 2 is not being expanded directly to 11 folds.

The complete nine-run matrix and exact interpretation are frozen in
`docs/audits/Q1_LOSS_GATING_AUDIT.md`.

**Locked development decision:** MR-STFT is the waveform core loss for the
next controlled model experiment. This locks the best observed waveform
objective; it does not erase the measured F1 trade-off or convert the pilot
into manuscript evidence.

## Step 2.5 — shared-gradient audit completed

A read-only audit measured gradients of MR-STFT and event BCE on the parameters
actually shared by B1 (`wsst_stem + decoder.tcn`) across every validation batch.
Global cosine means were -0.137/+0.276/+0.088 on folds 1/5/9. Only fold 1 was
significantly negative; folds 5 and 9 were significantly positive, and no
predeclared module reached mean cosine ≤ -0.30 on any fold. The preregistered
conflict rule failed. PCGrad and branch-gradient isolation are therefore not
authorized from this hypothesis. Full evidence:
`docs/audits/Q1_SHARED_GRADIENT_CONFLICT_AUDIT.md`.

## Step 3 — native-RCG resolution gate completed and closed

The capacity-matched B2a-wide/B2b gate completed all six validation runs on
2026-09-17. B2b reduced PCC on 3/3 folds (fold-macro 0.1221 to 0.0863), made R2
more negative (-0.0577 to -0.0661), and worsened QRS-region MAE on 3/3 folds.
It substantially increased waveform F1@100 (0.2013 to 0.5487), but RR/HR error
worsened in the macro aggregate and the high-frequency ratio was unstable,
including a 2.383 overshoot on fold 5. Thus the waveform and mechanism gates
failed even though event safety passed.

Per the preregistered rule, the B2b native-RCG direct-fusion route is
**closed**. Do not tune another fusion variant or scale B2b to 11 folds. This
failure bounds the tested direct-fusion formulation; it does not establish
that native radar contains no useful information. The immutable six-run table
and decision are in `docs/audits/Q1_NATIVE_RCG_GATE_AUDIT.md`.

## Official continuous model and completed benchmark

The official continuous candidate is now frozen as **WSST-driven REM-Net
(continuous B1) with the L3 MR-STFT waveform core**. “Official” means the
architecture, loss, normalization, checkpoint rule, and evaluation contract
are frozen. It does not by itself claim publishable superiority.

`q1_remnet_continuous_strict_loso_seed42` completed all 11 repaired `loso_v2`
folds with validation-only checkpoint selection followed by one-time held-out
test evaluation. Its fold-macro continuous results are PCC@0 0.305 ± 0.173,
R² 0.018 ± 0.181, NRMSE 2.151 ± 0.360, and waveform F1@100 ms 0.750 ± 0.265.
These values freeze the proposed-model seed-42 endpoint; they do not establish
superiority or authorize further model tuning.

### Metric audit of the completed benchmark (2026-09-18)

`docs/audits/Q1_REMNET_CONTINUOUS_11FOLD_METRIC_AUDIT.md` audited definitions,
units, aggregation, and NaN handling for this run; the manifest is
`experiments/manifests/q1_remnet_continuous_strict_loso_seed42.json`. Protocol
checks passed: 11/11 folds `COMPLETE`, all validation and all test file hashes
distinct, test opened once per fold after checkpoint freezing, zero NaN.

Locked wording for this result:

- R² is **centred near zero across folds**, not reliably positive: mean 0.018,
  population std 0.181, negative on folds 1, 2, 3, 10 and 11.
- The **amplitude variance ratio 0.312 ± 0.150** is the mechanism evidence that
  the prediction retains roughly a third of ground-truth variance, i.e.
  substantial residual regression toward the mean. Amplitude collapse is
  mitigated, not resolved.
- RMSE/MAE/region MAE are in per-record **z-score units**, not millivolts.
- The validation composite is partially degenerate — its NRMSE term is
  identically zero because NRMSE > 1 throughout — and the methods section must
  say so rather than paraphrase the formula as fully active.

Two reporting rules are binding for every continuous table (now in
`docs/METRIC_CONTRACT.md`): RR/HR aggregate only over folds with
`rr_coverage >= 0.80`, with fold 7 (coverage 0.0039) reported as a failure case
and the gated values RR 44.5 ± 30.1 ms / HR 3.39 ± 1.83 bpm over 10 folds; and
`above_10hz_energy_ratio` is reported as median 0.122, IQR [0.083, 0.195],
because fold 5 reaches 1.702.

These rules apply unchanged to the matched-protocol baselines below, so that
baseline and REM-Net numbers remain directly comparable.

**Data preparation complete (2026-09-18):** the deduplicated subject-centric
8-second WSST pool contains all 11 subjects and 16,682 unique windows in 23.95
GiB. All 33,257 train/validation/test fold references resolve without a missing
key; reused formal-cache values and the loader smoke batch are bit-identical.
The data blocker was removed and the completed run used this pool. Four-second
WSST remains prohibited. Full preparation evidence is in
`docs/audits/Q1_WSST8S_SUBJECT_POOL_AUDIT.md`.

The formal budget is frozen at **100 maximum epochs / patience 15**. The runner
uses epoch-level atomic recovery and the irreversible state sequence
`TRAINING -> CHECKPOINT_FROZEN -> TEST_EVALUATING -> COMPLETE`. Validation
alone selects `best_composite.pt`; the test split cannot be constructed before
the checkpoint is frozen, and a COMPLETE fold is terminal.

## Matched-protocol baselines — completed 2026-09-19

`q1_matched_baselines_continuous_strict_loso_seed42` ran five published
families on the same 11 `loso_v2` folds, split hashes, scaler, evaluator,
budget and checkpoint rule as REM-Net (`docs/MATCHED_BASELINE_PROTOCOL.md`).
radarODE-MTL runs as a genuine four-task model under LibMTL EGA with the
authors' settings; its earlier backbone-only folds are archived and not
reportable. Continuous fold-macro PCC@0 (population std, 11 folds):

| Model | PCC@0 | R² | Paired Wilcoxon vs REM-Net (PCC) |
|---|---:|---:|---|
| Chen-TCN (adapted) | 0.327 ± 0.170 | 0.019 | Δ −0.023, p = 0.520, REM-Net wins 5/11 |
| REM-Net | 0.305 ± 0.173 | 0.018 | — |
| Cross-modal Conformer (adapted) | 0.291 ± 0.154 | −0.014 | Δ +0.013, p = 0.638, 6/11 |
| radarODE-MTL (adapted, EGA) | 0.290 ± 0.140 | −0.008 | Δ +0.015, p = 0.577, 6/11 |
| WaveGRU-Net (adapted) | 0.201 ± 0.134 | −0.060 | Δ +0.104, p = 0.001, 11/11 |
| Attention-DCGAN (adapted) | 0.129 ± 0.105 | −0.346 | Δ +0.176, p = 0.001, 11/11 |

Locked reading: on continuous Tier 2, REM-Net is **statistically
indistinguishable** from the three strongest families and significantly better
than two. Every method sits at R² ≈ 0. REM-Net has the lowest amplitude
variance ratio (0.312) and the lowest >10 Hz energy ratio (median 0.122), i.e.
the most smoothing, and it is the only model whose rate metrics need the
coverage gate (fold 7). "REM-Net outperforms state of the art on continuous
ECG" is not supported and must not be written.

## Current priority — Q1 manuscript evidence plan (2026-09-19)

Target venues: IEEE TIM (primary), IEEE JBHI, IEEE IoT-J.

**Paper positioning, locked 2026-09-19: a measurement and benchmark paper.**
Title: *Contactless Radar-to-ECG Reconstruction: A Rigorous Leak-Free Benchmark
and Factorized Deep Architecture*. The contributions, in order of weight:

1. **Protocol correction.** A split defect (two validation subjects across eleven
   folds, cross-fold selection bias) is identified and repaired (`loso_v2`), and a
   leak-free, state-machine-enforced evaluation contract is released.
2. **Metric correction.** Beat-normalised PCC -- the Shape-200 / ECG_shape family --
   is saturated by a radar-free template built from training labels alone
   (0.879–0.887 on FMCW MMECG and CW CR_RVS). Template-referenced reporting is
   proposed and applied.
3. **Matched benchmark.** Six models from five families on identical folds, scaler,
   evaluator, budget and checkpoint rule: strict blind continuous reconstruction is
   bounded near R² ≈ 0 for all of them.
4. **Mechanism.** Where the continuous gap comes from (timing versus morphology),
   established by template diagnostics rather than inferred.

REM-Net is presented as a representative factorized architecture evaluated inside
the benchmark, not as a state-of-the-art claim. No sentence may say REM-Net
outperforms the matched baselines unless a table in this plan supports it.

The alternative title *REM-Net: Physically Factorized Time-Frequency Decoupling
for Non-Contact ECG Reconstruction Under Electromechanical Latency Drift* is
rejected as unsupported: the learned delay collapsed to a constant, the
delay-warp route was falsified, and Tier 3 does not identify EMAT.

Workstreams, in priority order. A single RTX 4080 serialises all training.

**W1 — Tier 1 morphology under `loso_v2` (highest priority).** The MMECG
Shape-200 headline (0.882 vs radarODE 0.728) is BLOCKED: it came from the old
`loso` split with two validation subjects, 4-second WSST (now prohibited) and
`narrow_bandpass`. It cannot be cited until reproduced.

- **W1a — Tier-1 readout of the frozen continuous models.** Every one of the
  66 matched test runs saved raw-unit predictions, ground truth and GT R peaks.
  Beat morphology (centre cycle cut by GT R peaks, resampled to 200, min-max
  normalised on both sides: the Tier-1 definition) is computed for all six
  models with one registered evaluator and compared by paired Wilcoxon. No
  training, no model choice, identical for every arm.
  Spec: `experiments/specs/q1_tier1_readout_from_continuous_seed42.json`.
- **W1b — Tier-1 trained rerun.** Models trained for Shape-200 on `loso_v2`
  with full 8-second WSST support. radarODE-MTL keeps its native ECG_shape head
  (its home task); the other baselines need a declared Shape-200 head. Spec to
  be written after W1a, because W1a tells us whether a trained rerun is needed
  to settle the morphology claim or merely to confirm it.

**W1a result (2026-09-19) — Tier 1 is saturated by a radar-free template.** The
training fold's mean beat, predicted for every test window, scores Shape-200 PCC
0.879 ± 0.070 on the old split, 0.881 ± 0.071 on `loso_v2` and 0.887 ± 0.013 on
CR_RVS -- the level of both Tier-1 headlines. REM-AlignNet-MTL is indistinguishable
from it per fold (p = 0.413); radarODE-MTL is below it on 11/11 folds. The frozen
continuous models read 0.14-0.35 under the same definition, far below the template.
Tier-1 claims M1/M2/S1/S3 are suspended and **W1b is paused** pending an author
decision on whether and how Tier 1 can carry a positive, template-referenced claim.
Full record: `docs/audits/Q1_TIER1_TEMPLATE_FLOOR_AUDIT.md`.

**Mechanism result (2026-09-19) — the continuous gap is timing.** A training-fold
template beat placed at ground-truth R peaks reaches PCC 0.856 and R² 0.730 on
`loso_v2`, against REM-Net's 0.305 / 0.018 (11/11 folds, p = 0.001). Placed at
radar-derived peaks (REM-Net event head or waveform, Chen-TCN waveform) it drops to
PCC 0.19–0.20 and R² below zero, significantly under REM-Net's own output. Beat
timing, not beat morphology, bounds continuous reconstruction; amplitude shrinkage
is the loss-optimal hedge against timing error. Full record:
`docs/audits/Q1_TEMPLATE_TIMING_AUDIT.md`.

**W2 result (2026-09-21) — three seeds complete, 198 sealed folds.** Across seeds
42/123/2024: Chen-TCN PCC 0.329 ± 0.018, REM-Net 0.324 ± 0.015 (Δ −0.005, p = 0.831,
5/11 folds), radarODE-MTL 0.290 ± 0.001, Conformer 0.282 ± 0.010, WaveGRU-Net
0.203 ± 0.007, Attention-DCGAN 0.113 ± 0.012. REM-Net holds the only clearly positive
R² (0.041 ± 0.016) and the best NRMSE, is behind radarODE-MTL (0.819) and Chen-TCN
(0.808) on waveform F1@100 with 0.732, and is the only model whose rate metrics need
the coverage gate (31/33 entries). Single-seed rankings among the top three were not
reliable: radarODE-MTL varies by ±0.001 across seeds, Chen-TCN by ±0.018. Full record:
`docs/audits/Q1_MULTISEED_BENCHMARK_AUDIT.md`.

**W2 — Three-seed statistical audit.** Seeds 123 and 2024 for REM-Net and all
five baselines on the continuous benchmark, same everything else. Requires an
additive seed-base argument; seed 42 results stay frozen. About 38 GPU hours,
dominated by radarODE-MTL (EGA, ~14 h per seed).

**W3 — CR_RVS protocol alignment (scope revised 2026-09-19).** The CR_RVS
Shape-200 build min-max normalises every window, so a continuous target was rebuilt
(`processed/q1_continuous_kfold`, same windows, per-record z-scored ECG). Matched
arms: REM-Net, Chen-TCN (native time-domain family) and WaveGRU-Net (MODWT wavelet
family), 5 folds, queued after W2 on branch `exp/w3-crrvs-alignment`. radarODE-MTL,
Attention-DCGAN and the Conformer are MMECG-only. The other CR_RVS methods -- Wu 2024
RSSRnet, Xiao 2026 CA-WTBNet, Li 2025 Radar2ECG, Chowdhury 2024 MultiResLinkNet --
enter as reported literature with their protocols stated; the audit is
`docs/audits/Q1_CRRVS_LITERATURE_PROTOCOL_AUDIT.md`. Its central fact: RSSRnet
reports PCC 0.959 under a data-level split of 80 %-overlapping windows and per-subject
medians of 0.39-0.90 under leave-one-subject-out in the same paper. After W3, the
CR_RVS results are also reported for the Resting scenario alone, the scenario every
cited paper uses.

**W3 result (2026-09-19).** CR_RVS seed 42: Chen-TCN PCC 0.506 / R² 0.254,
WaveGRU-Net 0.503 / 0.252, REM-Net 0.442 / 0.182 (REM-Net 1/5 folds). Resting-only
subject-level PCC 0.53-0.61, the regime of RSSRnet's own leave-one-subject-out
result. Template floor 0.887 with 0/5 folds above it; template at GT timing reaches
R² 0.764. Timing error is within-subject jitter plus missed beats; an oracle
per-subject latency calibration recovers only 6-17 % of the gap on both radars. Full
record: `docs/audits/Q1_CRRVS_DIAGNOSTICS_AUDIT.md`.

**W4 — Manuscript restructure.** Rewrite `docs/paper/RQ1.tex` around the three
tiers; rebuild `docs/PAPER_CLAIM_LEDGER.md` from the matched evidence; retire
every "resolves collapse" or "outperforms SOTA" sentence that the tables above
do not support. Runs in parallel with W1–W3.

The completed 60-second paired-ECG calibration is retained only as Discussion
evidence. It improved fold-macro PCC from 0.305 to 0.327 but improved only 5/11
folds and degraded R², NRMSE, and F1 on average. Its evaluator is archived at
`archive/legacy_scripts/audits/eval_60s_calibration_loso.py`; it is not an
active method, tuning surface, or alternative headline protocol.

## Non-negotiable reporting rules

- Never present Shape-200 as continuous ECG reconstruction.
- Never present Max-Lag NCC as achievable radar-only performance.
- Never mix reported literature numbers with matched reproductions without
  labeling the protocol difference.
- Never use test subjects to select a model, hyperparameter, affine map, lag,
  or checkpoint.
- Do not fabricate expected numbers or retrofit a story to a favorable result.
