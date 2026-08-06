# Q1 Project Structure and Naming Policy

This repository now separates the ICONIP baseline, frozen reproduced baselines,
Q1 candidate experiments, external CR_RVS validation, and generated reports.
Probe/smoke/invalid artifacts have been removed; deletions are tracked in
`docs/DELETED_EXPERIMENTS_LOG.md`. The goal is to keep the project
reproducible before adding new radarODE-style long-term ECG experiments.

## Current Paper Direction

The Q1 extension should not be framed as simply adding modules to the ICONIP
network. The current defensible direction is:

> Radar-to-ECG reconstruction should be evaluated as a representation-aware,
> event-structured, and morphology-guided reconstruction problem.

The current strongest evidence is in the beat-normalized Shape-200 morphology
space. The final Q1 claim must lock exactly which protocols are used in the
main table. If the paper claims superiority over radarODE-MTL's final long ECG
result, it requires a radarODE-style long-term ECG assembly protocol.

## Dataset Protocols

| Dataset | Protocol | Current Use | Important Caveat |
|---|---|---|---|
| MMECG public | 11-fold strict LOSO | Main reproduced Shape-200 benchmark | The local MMECG is the public MMECG dataset, not the unrelated MMECG name used in some external papers. |
| CR_RVS / Nature | 5-fold subject-independent CV | External validation on 24-GHz CW radar | This is not 30-fold LOSO. It is strict subject-independent 5-fold CV. |

## Experiment Categories

Every experiment must belong to exactly one category. The category must be
recorded in `experiments_mmecg/EXPERIMENT_REGISTRY.csv`.

| Category | Meaning | Publishable |
|---|---|---|
| `smoke` | Code-running check, usually 1-2 epochs or one tiny fold | false |
| `probe` | Quick trend check, usually single fold or partial folds | false |
| `pilot` | Mostly complete, but protocol/model claim is not locked | false by default |
| `final` | Protocol-locked formal experiment with complete reporting | true |
| `invalid` | Wrong protocol, unfair comparison, broken run, or untraceable result | false |

Rules:

- A result is not paper-ready unless `category=final` and `publishable=true`.
- A completed 11-fold or 5-fold run can still be `pilot` if the protocol has
  not been accepted as the final paper protocol.
- Reports generated from pilot experiments are also `pilot`, even if the
  figures look polished.
- `invalid` results must not be used in slides except as an internal caution.

## Active Experiment Tags

These directories should remain at the top level of `experiments_mmecg/`.

| Tag | Role | Category | Publishable |
|---|---|---|---|
| `q1_remalign_mtl_shapefirst_loso` | Proposed REM-AlignNet-MTL on MMECG | pilot | false |
| `q1_remalign_wsst_tcn_loso` | WSST + TCN representation baseline | pilot | false |
| `q1_remalign_event_lowpeak_loso` | Event-structured ablation | pilot | false |
| `mmecg_radarode_mtl_ega_strict_loso` | Adapted radarODE-MTL ECG_shape reproduction | pilot | false |
| `mmecg_beataware_fewshot40v10` | ICONIP BeatAware baseline | pilot | false |
| `mmecg_airecg_strict_loso_v4` | Adapted AirECG baseline | pilot | false |
| `mmecg_rssrnet_strict_loso_v2` | Adapted RSSRNet baseline | pilot | false |
| `q1_crrvs_remalign_mtl_5fold` | Proposed REM-AlignNet-MTL on CR_RVS | pilot | false |
| `q1_crrvs_wsst_tcn_5fold` | CR_RVS CW-STFT + TCN baseline | pilot | false |
| `q1_paper_style_report` | MMECG paper-style report package | pilot | false |
| `q1_crrvs_external_report_5fold` | CR_RVS 5-fold report package | pilot | false |
| `q1_supervisor_shape200_snapshot` | Supervisor-facing English snapshot | pilot | false |

These are useful for supervisor discussion, but they should not be treated as
final manuscript results until the long-term protocol is locked.

## Retained Generic Baselines

These are retained only as controlled MMECG strict-LOSO baselines. They should
not be extended unless a specific comparison requires them.

| Prefix / Tag | Role |
|---|---|
| `mmecg_cnn/lstm/bilstm/conformer_strict_loso_baseline` | Generic strict-LOSO baselines |

## Deleted Exploratory Runs

Smoke, probe, single-fold, obsolete few-shot, and incorrect-protocol
exploratory runs have been deleted from `experiments_mmecg/`. Do not cite these
in the paper. The deletion list is preserved in
`docs/DELETED_EXPERIMENTS_LOG.md` for traceability.

## Naming Rules for New Experiments

Use this pattern:

```text
q1_<dataset>_<model>_<protocol>[_<variant>]
```

Recommended dataset names:

```text
mmecg
crrvs
```

Recommended protocol names:

```text
loso              # MMECG strict leave-one-subject-out
5fold             # CR_RVS subject-independent 5-fold
radarode_long     # radarODE-style long-term ECG assembly protocol
shape200          # single-cycle morphology protocol only
```

Examples:

```text
q1_mmecg_remalign_mtl_radarode_long
q1_mmecg_radarode_mtl_radarode_long
q1_crrvs_remalign_mtl_5fold_long
```

Avoid final experiment names such as:

```text
fold1
probe
smoke
loso30
fewshot40v10
```

unless the run is intentionally temporary and will be deleted or explicitly
registered as `smoke`/`probe`.

Every new experiment must also add a row to:

```text
experiments_mmecg/EXPERIMENT_REGISTRY.csv
```

The row must explicitly set:

```text
category
publishable
reason
```

## Required Result Files for a Formal Experiment

A formal experiment must contain:

```text
fold_XX/config.json
fold_XX/logs/train.log
fold_XX/checkpoints/best*.pt
fold_XX/results*/global_summary*.json
fold_XX/results*/segment_metrics*.csv
```

An experiment is not considered paper-ready until a report script consumes the
fold results and writes a top-level CSV/Markdown summary.

## Current Limitation

The current high-performing result is Shape-200 morphology reconstruction:

```text
REM-AlignNet-MTL MMECG Shape-200 PCC = 0.8898 +/- 0.0538
REM-AlignNet-MTL CR_RVS Shape-200 PCC = 0.8865 +/- 0.0228
```

This should be presented as beat-morphology reconstruction, not as final
long-term continuous ECG reconstruction. The next formal task is to implement
a radarODE-style long-term ECG assembly evaluator for both radarODE-MTL and
REM-AlignNet-MTL.
