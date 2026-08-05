# Experiments Registry

This directory contains frozen baselines, active Q1 experiments, and generated
reports. Temporary smoke/probe/invalid artifacts have been removed; see
`../docs/DELETED_EXPERIMENTS_LOG.md`.

The source of truth is:

```text
EXPERIMENT_REGISTRY.csv
```

Each experiment must have exactly one category:

| Category | Meaning | Publishable |
|---|---|---|
| `smoke` | Code-running check | false |
| `probe` | Quick trend check, single fold or partial folds | false |
| `pilot` | Mostly complete but protocol/model claim not locked | false by default |
| `final` | Protocol-locked formal experiment | true |
| `invalid` | Wrong protocol, unfair comparison, broken run, or untraceable result | false |

## Active Pilot Results

These are useful for supervisor updates, but they are not final paper tables
until promoted to `category=final`.

| Directory | Dataset | Protocol | Role | Publishable |
|---|---|---|---|---|
| `q1_remalign_mtl_shapefirst_loso` | MMECG public | 11-fold strict LOSO | Proposed REM-AlignNet-MTL Shape-200 result | false |
| `q1_remalign_wsst_tcn_loso` | MMECG public | 11-fold strict LOSO | WSST + TCN baseline | false |
| `q1_remalign_event_lowpeak_loso` | MMECG public | 11-fold strict LOSO | Event ablation | false |
| `q1_crrvs_remalign_mtl_5fold` | CR_RVS / Nature | 5-fold subject-independent CV | External REM-AlignNet-MTL validation | false |
| `q1_crrvs_wsst_tcn_5fold` | CR_RVS / Nature | 5-fold subject-independent CV | External CW-STFT + TCN baseline | false |

## Frozen Baselines

| Directory | Role |
|---|---|
| `mmecg_radarode_mtl_ega_strict_loso` | Adapted radarODE-MTL ECG_shape reproduction; pilot, not original long-term result |
| `mmecg_beataware_fewshot40v10` | ICONIP BeatAware baseline; historical pilot |
| `mmecg_airecg_strict_loso_v4` | Adapted AirECG baseline; pilot |
| `mmecg_rssrnet_strict_loso_v2` | Adapted RSSRNet baseline; pilot |
| `mmecg_cnn_strict_loso_baseline` | Generic CNN baseline |
| `mmecg_lstm_strict_loso_baseline` | Generic LSTM baseline |
| `mmecg_bilstm_strict_loso_baseline` | Generic BiLSTM baseline |
| `mmecg_conformer_strict_loso_baseline` | Generic Conformer baseline |

## Report Artifacts

| Directory | Contents |
|---|---|
| `q1_supervisor_shape200_snapshot` | English supervisor-facing Shape-200 snapshot |
| `q1_paper_style_report` | MMECG Shape-200 paper-style figures/tables |
| `q1_crrvs_external_report_5fold` | CR_RVS 5-fold external validation figures/tables |
| `q1_teacher_report_remalign_mtl_shapefirst` | REM vs radarODE focused report |

## Deleted Runs

Temporary smoke, probe, single-fold, obsolete few-shot, or wrong-protocol runs
are not kept at the top level. Do not cite deleted runs in manuscripts. The
cleanup record is maintained in `../docs/DELETED_EXPERIMENTS_LOG.md`.
