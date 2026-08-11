# F0 Supervised Tuning Summary

Purpose: freeze a supervised RQ1 REM-Net configuration for the 3-day Q1 draft
cycle. This is not SSL and does not introduce a new research question.

## Candidate Set

All candidates use the final morphology-first REM-Net backbone:

- `model_arch=rem_align_mtl`
- `use_alignment=false`
- WSST input (`_wsst.h5`)
- `target_norm=minmax`
- `use_shape200_loss=true`
- no lag-aware loss, no output lag alignment, no diffusion

Only supervised loss/model-capacity knobs were varied:

| Candidate | beta_peak | lambda_shape200 | shape200_pcc_weight | hidden |
| --- | ---: | ---: | ---: | ---: |
| incumbent | 0.3 | 0.8 | 0.5 | 128 |
| shape12 | 0.3 | 1.2 | 0.5 | 128 |
| pcc08 | 0.3 | 0.8 | 0.8 | 128 |
| peak01 | 0.1 | 0.8 | 0.5 | 128 |
| peak06 | 0.6 | 0.8 | 0.5 | 128 |
| hidden192 | 0.3 | 0.8 | 0.5 | 192 |

## Dev-Fold Results

F0 uses a small, predeclared dev-fold subset for fast supervised freeze tuning:
MMECG folds 1/5/9 and CR_RVS folds 1/3/5. Results are Shape-200 PCC from the
model's morphology head.

| Dataset | Best Candidate | Best PCC | Incumbent PCC | Read |
| --- | --- | ---: | ---: | --- |
| MMECG | peak06 | 0.847 +/- 0.074 | 0.831 +/- 0.080 | event weight helps morphology |
| CR_RVS | hidden192 | 0.884 +/- 0.009 | 0.881 +/- 0.007 | gains are tiny / overlapping |
| CR_RVS | peak06 | 0.881 +/- 0.004 | 0.881 +/- 0.007 | no meaningful regression |

Full aggregate source:
`experiments_mmecg/q1_freeze_f0_supervised_tuning/f0_candidate_summary.csv`.

## Freeze Decision

For the 3-day Q1 draft cycle, promote `peak06` to the final-candidate run:

```text
beta_peak = 0.6
lambda_shape200 = 0.8
shape200_pcc_weight = 0.5
baseline_hidden = 128
use_alignment = false
```

Rationale:

1. It gives the clearest MMECG gain in the dev folds.
2. It does not materially degrade CR_RVS relative to the incumbent.
3. Increasing hidden size gives only a small CR_RVS gain and a clear MMECG drop,
   so it is not a good cross-dataset final setting.
4. This keeps the paper story simple: representation + event supervision +
   morphology decoder, not a large-capacity model.

## Caveat

F0 is tuning evidence only. The paper result must come from the final-candidate
full-fold run plus metric audit. Do not cite F0 numbers as final table values.
