# Q1 Three-Seed Matched Benchmark (MMECG, Table 1)

Date: 2026-09-21. Experiment: `q1_multiseed_benchmark_aggregation`.
Aggregator: `scripts/aggregate_multiseed_benchmark.py`.
Outputs: `experiments_mmecg/q1_multiseed_benchmark_summary/`.
Runs: `q1_remnet_continuous_strict_loso_seed{42,123,2024}` and
`q1_matched_baselines_continuous_strict_loso_seed{42,123,2024}`.

**198 folds sealed**: six models x three seeds (42 / 123 / 2024) x eleven `loso_v2`
folds. Every fold reached stage `COMPLETE` with test metrics present; the aggregator
refuses to run otherwise.

Conventions (`docs/METRIC_CONTRACT.md`): per seed, fold-macro over the eleven test
subjects; the headline is **mean ± std across the three seeds** of those values.
RMSE / MAE / region MAE are in per-record **z-score units**. Waveform F1 uses peaks
detected on the reconstruction; `event_f1_*` is not applicable to the baselines and
is not tabulated.

**Parameter counts** are the `params` field of each model's frozen
`fold_01/config.json` in the sealed seed-42 runs, i.e. the trainable parameters of
the network that produced these numbers. They are exact: REM-Net 1,438,019
(WSST stem 205,568 + decoder 1,232,451), Chen-TCN 607,965, WaveGRU-Net 705,553,
Cross-Modal Conformer 422,737, Attention-DCGAN 4,345,170, radarODE-MTL 41,618,306
(four-task EGA configuration; the earlier backbone-only variant was 26.77 M and is
not the model benchmarked here). These are **continuous-track** models and must not
be exchanged with the Shape-200 counts in `docs/EFFICIENCY_AUDIT.md`, which profile
a different network (REM-Net Shape-200, 1.788 M).

## Table 1 — continuous ECG, MMECG `loso_v2`, three seeds

| Model | Params | PCC@0 ↑ | R² ↑ | NRMSE ↓ | Waveform F1@100 ↑ | QRS MAE ↓ | ST/T MAE ↓ | Var. ratio | >10 Hz (median) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Chen-TCN (adapted) | 0.61 M | **0.329 ± 0.018** | 0.003 ± 0.038 | 2.153 ± 0.041 | 0.808 ± 0.033 | 1.733 | 0.601 | 0.433 | 0.362 |
| **REM-Net (ours)** | 1.44 M | 0.324 ± 0.015 | **0.041 ± 0.016** | **2.125 ± 0.018** | 0.732 ± 0.035 | 1.690 | 0.607 | 0.331 | 0.122 |
| radarODE-MTL (adapted, EGA) | 41.6 M | 0.290 ± 0.001 | −0.016 ± 0.022 | 2.183 ± 0.020 | **0.819 ± 0.010** | 1.692 | 0.652 | 0.368 | 0.274 |
| Cross-Modal Conformer (adapted) | 0.42 M | 0.282 ± 0.010 | −0.053 ± 0.030 | 2.216 ± 0.029 | 0.767 ± 0.012 | 1.739 | 0.620 | 0.447 | 0.547 |
| WaveGRU-Net (adapted) | 0.71 M | 0.203 ± 0.007 | −0.057 ± 0.015 | 2.239 ± 0.017 | 0.557 ± 0.004 | **1.683** | 0.698 | 0.283 | 0.557 |
| Attention-DCGAN (adapted) | 4.35 M | 0.113 ± 0.012 | −0.474 ± 0.128 | 2.625 ± 0.093 | 0.675 ± 0.003 | 1.840 | 0.796 | 0.638 | 0.806 |

Rate metrics, gated at `rr_coverage >= 0.80` (entries = model x seed x fold):

| Model | RR error | HR error | Entries used |
|---|---:|---:|---:|
| radarODE-MTL | **45.5 ms** | 3.90 bpm | 33/33 |
| REM-Net | 49.1 ms | **3.53 bpm** | 31/33 |
| Cross-Modal Conformer | 54.5 ms | 4.75 bpm | 33/33 |
| Chen-TCN | 77.9 ms | 5.47 bpm | 33/33 |
| WaveGRU-Net | 106.4 ms | 8.81 bpm | 33/33 |
| Attention-DCGAN | 161.3 ms | 16.76 bpm | 33/33 |

REM-Net is the only model that loses entries to the gate: seed 42 fold 7
(`rr_coverage` 0.0039) and seed 2024 fold 2 (0.0118), both waveform-level R-peak
detection collapses. Its rate advantage is computed on 31 of 33 entries and must be
reported with that fact.

## Paired statistics

Metrics are averaged over the three seeds per fold; a paired two-sided Wilcoxon runs
on the eleven seed-averaged fold values. **Five baselines x five metrics = 25
uncorrected tests**; with Holm correction at the family level nothing below p ≈ 0.002
survives, so only the WaveGRU-Net and Attention-DCGAN results are robust to
multiplicity. Everything else is descriptive.

| REM-Net versus | ΔPCC (p) | ΔR² (p) | ΔNRMSE (p) | ΔF1@100 (p) |
|---|---|---|---|---|
| Chen-TCN | −0.005 (0.831), 5/11 | +0.038 (0.365), 7/11 | −0.028 (0.465), 7/11 | −0.077 (0.206), 4/11 |
| radarODE-MTL | +0.034 (0.083), 8/11 | +0.057 (0.032), 9/11 | −0.058 (0.042), 9/11 | −0.087 (0.067), 3/11 |
| Cross-Modal Conformer | +0.042 (0.083), 9/11 | +0.093 (0.054), 10/11 | −0.091 (0.054), 10/11 | −0.035 (0.966), 7/11 |
| WaveGRU-Net | +0.121 (0.003), 10/11 | +0.097 (0.042), 7/11 | −0.114 (0.032), 8/11 | +0.175 (0.014), 10/11 |
| Attention-DCGAN | +0.211 (0.002), 10/11 | +0.515 (0.001), 11/11 | −0.500 (0.001), 11/11 | +0.057 (0.147), 8/11 |

Δ is REM-Net minus baseline; the fold count is folds where REM-Net is better.

### Cycle Length Estimation — paired tests (added 2026-09-24)

The paired family above covers PCC, R2, NRMSE and Anchor F1@100 only. Cycle Length was
not tested. Computed here from the same frozen per-fold `test_metrics.json`, seeds
averaged within each fold over entries passing `rr_coverage >= 0.80`, then a paired
two-sided Wilcoxon over the eleven folds.

| REM-Net versus | REM-Net | baseline | change | p | folds won |
|---|---:|---:|---:|---:|---:|
| radarODE-MTL | 48.8 ms | **45.5 ms** | **-7.3 % (worse)** | 0.966 | 4/11 |
| Cross-Modal Conformer | 48.8 ms | 54.5 ms | +10.4 % | 0.577 | 6/11 |
| Chen-TCN | 48.8 ms | 77.9 ms | +37.4 % | 0.206 | 7/11 |
| WaveGRU-Net | 48.8 ms | 106.4 ms | **+54.1 %** | **0.001** | 11/11 |
| Attention-DCGAN | 48.8 ms | 161.3 ms | **+69.7 %** | **0.001** | 11/11 |

REM-Net loses 2 of 33 entries to the coverage gate; every baseline loses none.

**Binding reading.** REM-Net does **not** reduce Cycle Length error relative to the
strongest comparator: against radarODE-MTL it is 7.3 % worse and statistically
indistinguishable (p = 0.966, 4/11 folds). Only the WaveGRU-Net and Attention-DCGAN
contrasts are significant, and they survive Holm correction. No manuscript sentence may
claim a Cycle Length reduction "relative to the strongest baseline". The supportable
statements are (i) parity with a 41.6 M multitask comparator at 1.44 M parameters, and
(ii) the scenario-level result in `Q1_SCENARIO_BREAKDOWN_AUDIT.md`, which is measured
against Chen-TCN and WaveGRU-Net only and must name those two comparators.

## What three seeds change

1. **REM-Net and Chen-TCN are tied on correlation.** PCC 0.324 ± 0.015 against
   0.329 ± 0.018, Δ −0.005 with p = 0.831 and 5/11 folds. No ordering between them is
   supportable.
2. **REM-Net holds the only clearly positive R² (0.041 ± 0.016) and the best NRMSE.**
   Chen-TCN's R² is 0.003 ± 0.038 and swings from +0.040 (seed 123) to −0.049
   (seed 2024); REM-Net's stays 0.018-0.055. Against radarODE-MTL, the Conformer,
   WaveGRU-Net and Attention-DCGAN the R²/NRMSE differences reach p = 0.032-0.054,
   which does not survive multiplicity correction except for the last two.
3. **Event timing still favours the baselines.** radarODE-MTL 0.819 and Chen-TCN
   0.808 on waveform F1@100 against REM-Net's 0.732, and REM-Net is the only model
   whose rate metrics need the coverage gate.
4. **REM-Net remains the most smoothed model**: amplitude variance ratio 0.331 and a
   >10 Hz energy ratio median of 0.122, both furthest from 1 apart from WaveGRU-Net.
5. **Seed stability differs by an order of magnitude.** radarODE-MTL's PCC varies by
   ±0.001 across seeds, Chen-TCN's by ±0.018. Single-seed rankings between the top
   three were not reliable, which is why the metric contract requires three.

## Permitted wording

- "Under a leak-free matched protocol, REM-Net is statistically indistinguishable
  from the strongest baseline (Chen-TCN) on continuous PCC, achieves the highest R²
  and lowest NRMSE, and is outperformed on waveform beat timing by radarODE-MTL and
  Chen-TCN."
- Not permitted: "REM-Net outperforms state of the art"; any ranking of the top three
  by PCC; any rate-metric claim without the 31/33 entry note; any of the p = 0.03-0.06
  results described as significant without the multiplicity caveat.
