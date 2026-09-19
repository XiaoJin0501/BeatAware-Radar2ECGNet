# Q1 CR_RVS Diagnostics Audit: Scenario Readout, Template Floor, Timing

Date: 2026-09-19. Experiment: `q1_crrvs_diagnostics_seed42` (spec with pre-declared
readings: `experiments/specs/q1_crrvs_diagnostics_seed42.json`). Inputs: the frozen
seed-42 CR_RVS test predictions of REM-Net, Chen-TCN and WaveGRU-Net
(`q1_remnet_continuous_crrvs_seed42`, `q1_matched_baselines_continuous_crrvs_seed42`),
row-aligned with `processed/q1_continuous_kfold` (verified by exact ground-truth
equality). Nothing was trained or selected.

Evaluators: `scripts/evaluate_scenario_readout.py` (A1),
`scripts/evaluate_tier1_readout.py --dataset crrvs` (A2),
`scripts/evaluate_template_timing.py --dataset {crrvs,mmecg}` (A3-A5; the MMECG
rerun reproduces T0-T3 of `docs/audits/Q1_TEMPLATE_TIMING_AUDIT.md` exactly).

CR_RVS has five folds, so a two-sided Wilcoxon cannot reach p < 0.05; fold wins
and effect sizes are reported. Population std across folds throughout. Units: stored
per-record z-scored ECG.

## Headline CR_RVS continuous results (all scenarios)

| Model | PCC | R² | NRMSE | F1@100 | Var. ratio |
|---|---:|---:|---:|---:|---:|
| Chen-TCN | 0.506 ± 0.080 | +0.254 ± 0.095 | 1.854 | 0.824 | 0.380 |
| WaveGRU-Net | 0.503 ± 0.103 | +0.252 ± 0.114 | 1.834 | 0.782 | 0.418 |
| REM-Net | 0.442 ± 0.081 | +0.182 ± 0.093 | 1.951 | 0.723 | 0.342 |

REM-Net wins 1/5 folds on PCC against either baseline. Unlike MMECG, where every
family sits at R² ≈ 0, all three reach a positive R² of 0.18-0.25 on CW radar.

## A1 — Scenario readout and the like-for-like literature comparison

| Scenario | REM-Net PCC / R² | Chen-TCN PCC / R² | WaveGRU-Net PCC / R² |
|---|---|---|---|
| Resting | 0.477 / +0.215 | **0.552 / +0.296** | 0.543 / +0.289 |
| Valsalva | 0.443 / +0.184 | 0.496 / +0.249 | 0.499 / +0.255 |
| Apnea | 0.366 / +0.111 | 0.414 / +0.163 | 0.416 / +0.157 |

Difficulty orders Resting > Valsalva > Apnea for every model.

**Resting, per-subject median PCC over 30 subjects** -- the quantity Wu et al. 2024
report under leave-one-subject-out (RSSRnet: 0.39-0.90, more than half above 0.7):

| Model | Range | Median | Subjects above 0.7 |
|---|---|---:|---:|
| Chen-TCN | 0.30-0.80 | 0.60 | 7/30 |
| WaveGRU-Net | 0.15-0.84 | 0.61 | 9/30 |
| REM-Net | 0.04-0.72 | 0.53 | 1/30 |

Under subject-independent evaluation on resting data, subject-level PCC sits around
0.5-0.6 -- the regime of RSSRnet's own leave-one-subject-out result and of
Chowdhury 2024's ~0.66 -- not the 0.96-0.99 reported under data-level splits. Models,
preprocessing and subject sets differ from Wu 2024, so this is an order-of-magnitude
agreement, not a matched comparison.

## A2 — Tier-1 template floor on CR_RVS

| | Shape-200 PCC | RMSE | MAE | Folds above template |
|---|---:|---:|---:|---:|
| WaveGRU-Net | 0.529 ± 0.106 | 0.187 | 0.134 | 0/5 |
| Chen-TCN | 0.528 ± 0.079 | 0.189 | 0.135 | 0/5 |
| REM-Net | 0.429 ± 0.082 | 0.281 | 0.221 | 0/5 |
| **Radar-free template** | **0.887 ± 0.014** | **0.103** | **0.087** | — |

Same cycle definition and test beats as the models. As on MMECG, every model reads far
below a training-label template when the cycle is cut from its continuous output.
REM-Net is worse than both baselines on RMSE and MAE in 5/5 folds.

## A3 — Template timing on CW versus FMCW

| Variant | CR_RVS PCC / R² | MMECG PCC / R² |
|---|---|---|
| Best model output | 0.506 / +0.254 (Chen-TCN) | 0.327 / +0.019 (Chen-TCN) |
| REM-Net output | 0.442 / +0.182 | 0.305 / +0.018 |
| **T0 template @ GT peaks** | **0.877 / +0.764** | **0.856 / +0.730** |
| T1 template @ REM-Net event head | 0.298 / −1.582 | 0.203 / −0.711 |
| T2 template @ REM-Net waveform peaks | 0.263 / −0.535 | 0.186 / −0.477 |
| T3 template @ Chen-TCN waveform peaks | 0.376 / −0.227 | 0.197 / −0.485 |
| T4 template @ WaveGRU-Net waveform peaks | 0.375 / −0.217 | 0.120 / −0.670 |

## A4 — What the timing error is made of

Radar-derived peaks matched to GT R peaks within ±150 ms; signed offset = radar −
GT. Jitter is the robust within-subject spread (1.4826 × MAD, median over subjects);
the sampling step is 5 ms.

| Peak source | Dataset | Recall | Radar / GT peaks | Global bias | Median \|err\| | Between-subject bias SD | Within-subject jitter |
|---|---|---:|---:|---:|---:|---:|---:|
| REM-Net event head | CR_RVS | 0.724 | 0.739 | 0 ms | 15 ms | 21.6 ms | 20.4 ms |
| REM-Net waveform | CR_RVS | 0.743 | 0.905 | 0 ms | 20 ms | 14.9 ms | 22.2 ms |
| Chen-TCN waveform | CR_RVS | 0.850 | 0.990 | 0 ms | 15 ms | 19.8 ms | 7.4 ms |
| WaveGRU-Net waveform | CR_RVS | 0.856 | 1.064 | 0 ms | 15 ms | 13.9 ms | 7.4 ms |
| REM-Net event head | MMECG | 0.802 | 0.838 | 0 ms | 35 ms | 30.4 ms | 29.7 ms |
| REM-Net waveform | MMECG | 0.797 | 0.883 | 0 ms | 35 ms | 32.0 ms | 37.1 ms |
| Chen-TCN waveform | MMECG | 0.851 | 0.942 | 0 ms | 35 ms | 32.3 ms | 37.1 ms |
| WaveGRU-Net waveform | MMECG | 0.656 | 0.986 | 0 ms | 45 ms | 23.9 ms | 59.3 ms |

Per-subject median offsets span roughly −80 to +55 ms across both datasets.

## A5 — Would a perfect per-subject latency calibration help?

Each test subject's own median offset (ground-truth-derived: an **oracle**) is removed
from its radar peaks before the template is placed.

| Peak source | Dataset | R² raw | R² calibrated | R² @ GT timing | Gap recovered |
|---|---|---:|---:|---:|---:|
| REM-Net event head | CR_RVS | −1.582 | −1.445 | +0.764 | 5.8 % |
| Chen-TCN waveform | CR_RVS | −0.227 | −0.054 | +0.764 | 17.4 % |
| REM-Net event head | MMECG | −0.711 | −0.536 | +0.730 | 12.2 % |
| Chen-TCN waveform | MMECG | −0.485 | −0.281 | +0.730 | 16.8 % |

## CW versus FMCW: what is shared and what differs

**Shared.**

1. **Beat timing bounds continuous reconstruction on both radars.** A fixed training
   template at exact beat times explains 73-76 % of ECG variance on both datasets (T0),
   while every model is far below it. Beat morphology beyond a training template is
   not the limiting factor on either modality.
2. **No global electromechanical delay survives training.** The median offset of every
   radar-derived peak source is 0 ms on both datasets: supervised training on the ECG
   absorbs any population-level mechanical-to-electrical latency.
3. **Per-subject latency exists but is a minor term.** Per-subject offsets vary (SD
   14-32 ms, range about −80 to +55 ms), which is compatible with subject-specific
   electromechanical delay among other causes. Removing each subject's offset with an
   oracle recovers only **6-17 %** of the R² gap on both radars (pre-declared reading 2):
   the dominant error is **within-subject, beat-to-beat** -- jitter and missed or spurious
   beats -- which no per-subject constant removes. The EMAT hypothesis in its
   calibratable, per-subject form is therefore not the bottleneck, and EMAT itself is not
   identified by these analyses.

**Different.**

1. **Timing precision.** On CW the native-rate baselines place matched beats with
   ~7 ms within-subject jitter and a 15 ms median error; on FMCW jitter is 30-60 ms and
   the median error 35-45 ms. This is consistent with the gap in continuous quality:
   R² 0.18-0.25 on CR_RVS against ≈ 0 on MMECG.
2. **What remains once jitter is small.** On CW, Chen-TCN timing with 7 ms jitter and
   oracle calibration still gives a template R² of −0.05 against 0.76 at GT timing. The
   residual is beat **detection** -- 15 % of GT beats have no radar peak within ±150 ms --
   because one missed or extra beat shifts every following cycle of the template. On
   FMCW, jitter and detection errors act together.
3. **REM-Net's timing is the weak link on CW.** REM-Net's peaks have ~3x the jitter
   (20-22 ms) and lower recall (0.72-0.74) of the native-rate baselines on CR_RVS,
   matching its last place there. REM-Net reads a log-STFT with an 80 ms hop (93 frames
   per 8 s), the baselines read the 200 Hz phase. This is consistent with input temporal
   resolution limiting beat timing on CW; it is a hypothesis, not a controlled result.
   On MMECG, where REM-Net's WSST has ~33 ms frames, its jitter matches Chen-TCN's.

## Consequences for the claims

- **C2** (template saturation) holds on CW radar: 0/5 folds above the template.
- **C4** (timing-dominated gap) holds on CW radar, and is sharpened on both radars: the
  bottleneck is within-subject beat placement and beat detection, not a calibratable
  per-subject latency.
- **C3** must be scoped to MMECG: the universal R² ≈ 0 does not transfer; on CW every
  model reaches a positive R².
- REM-Net is the weakest of the three matched models on CR_RVS (1/5 folds on PCC);
  no REM-Net advantage may be claimed on CW radar.
