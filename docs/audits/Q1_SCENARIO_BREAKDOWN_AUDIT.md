# Q1 Scenario-Specific Breakdown

Date: 2026-09-21. Experiment: `q1_scenario_breakdown`.
Evaluator: `scripts/evaluate_scenario_breakdown.py`.
Outputs: `experiments_mmecg/q1_scenario_breakdown_{mmecg,crrvs}/`.
Frozen test predictions only; nothing trained or selected.

## Confound, stated first

Scenarios are not balanced over subjects. In MMECG, sleep appears in three test
subjects and post-exercise in six, while folds 1 and 2 contain normal breathing
only; one subject contributes 935 of the sleep windows. A window-pooled scenario
average would therefore measure subject identity as much as physiological state.

Every statistic below is computed **per (subject, scenario)** first -- median window
PCC and RR error within that subject's windows -- then averaged over seeds and
aggregated over subjects. The subject count accompanies every cell and no scenario
row may be read without it.

## MMECG (three seeds, median over subjects [IQR])

| Scenario | Subjects | REM-Net PCC | Chen-TCN PCC | WaveGRU-Net PCC | REM-Net RR | Chen-TCN RR | WaveGRU-Net RR |
|---|---:|---|---|---|---:|---:|---:|
| Normal breathing | 11 | **0.358** [0.111, 0.476] | 0.332 [0.160, 0.475] | 0.244 [0.108, 0.334] | **39.5 ms** | 50.4 ms | 77.0 ms |
| Irregular breathing | 7 | **0.437** [0.202, 0.471] | 0.422 [0.324, 0.470] | 0.212 [0.187, 0.233] | **56.4 ms** | 75.8 ms | 99.7 ms |
| Post-exercise | 6 | 0.261 [0.109, 0.447] | **0.307** [0.101, 0.366] | 0.143 [0.077, 0.186] | **49.8 ms** | 102.2 ms | 75.4 ms |
| Sleep | 3 | 0.265 [0.122, 0.420] | **0.316** [0.163, 0.436] | 0.138 [0.026, 0.190] | **10.9 ms** | 45.8 ms | 60.4 ms |

## CR_RVS (seed 42, median over subjects [IQR])

| Scenario | Subjects | REM-Net PCC | Chen-TCN PCC | WaveGRU-Net PCC | REM-Net RR | Chen-TCN RR | WaveGRU-Net RR |
|---|---:|---|---|---|---:|---:|---:|
| Resting | 30 | 0.525 [0.313, 0.621] | 0.599 [0.425, 0.683] | **0.609** [0.433, 0.722] | **42.7 ms** | 55.4 ms | 70.1 ms |
| Valsalva | 27 | 0.483 [0.346, 0.583] | 0.500 [0.355, 0.681] | **0.577** [0.373, 0.741] | **56.6 ms** | 105.9 ms | 119.7 ms |
| Apnea | 23 | 0.341 [0.247, 0.513] | 0.401 [0.316, 0.469] | **0.424** [0.290, 0.543] | 74.7 ms | 115.9 ms | **79.5 ms** |

## Readings

1. **Subject spread dwarfs scenario spread.** Within every scenario the per-subject
   PCC range spans roughly 0.5-0.8 in absolute terms (e.g. MMECG normal breathing
   −0.004 to 0.643 for REM-Net), while scenario medians move by 0.1-0.2. Reporting a
   single scenario mean without the per-subject dispersion would overstate how much
   physiological state explains.
2. **Rate error is where REM-Net leads.** Its RR error is the lowest in six of the
   seven scenarios across both radars, including post-exercise and sleep on MMECG,
   even where its PCC is not the highest. This is the event head doing what it was
   designed for and is the one consistent REM-Net advantage in the benchmark.
3. **Order by difficulty.** MMECG: irregular and normal breathing are easier than
   post-exercise and sleep for every model. CR_RVS: Resting > Valsalva > Apnea.
4. **CR_RVS ordering is stable across scenarios.** WaveGRU-Net and Chen-TCN lead
   REM-Net on PCC in all three CW scenarios, matching the aggregate result.
5. **Sleep and post-exercise carry three and six subjects.** No claim may rest on
   those rows alone.
