# BeatAware-Radar2ECGNet Q1 Workspace

This repository supports one Q1 journal objective: propose a new radar-to-ECG
reconstruction model and demonstrate strong performance on both MMECG and
CR_RVS, outperforming or comparing favorably with most relevant published and
reproduced baselines under fair protocols.

## Source of Truth

Read these files first:

```text
AGENTS.md
CLAUDE.md
docs/agent/SHARED_CONTEXT.md
docs/PROJECT_STATE.md
docs/RESEARCH_CONTRACT.md
docs/DATA_PROTOCOL.md
docs/METRIC_CONTRACT.md
docs/EXPERIMENT_POLICY.md
docs/PAPER_CLAIM_LEDGER.md
docs/PROJECT_STRUCTURE_Q1.md
docs/NEXT_EXPERIMENTS_Q1.md
docs/GITHUB_WORKFLOW.md
experiments_mmecg/EXPERIMENT_REGISTRY.csv
experiments/registry.csv
experiments_mmecg/README.md
experiments/README.md
scripts/README.md
src/README.md
docs/DELETED_EXPERIMENTS_LOG.md
reference_papers/README.md
```

## Current Claim Status

The current strongest local result is Shape-200 beat morphology reconstruction,
not final long-term continuous ECG reconstruction.

```text
MMECG REM-AlignNet-MTL Shape-200 PCC ~= 0.8898
CR_RVS REM-AlignNet-MTL Shape-200 PCC ~= 0.8865
```

All current results are treated as `pilot` unless explicitly promoted in
`experiments_mmecg/EXPERIMENT_REGISTRY.csv`.

## Main Model

`REM-AlignNet-MTL`:

```text
Rhythm-Event-Morphology Alignment Network with Multi-Task Learning
```

The intended Q1 framing is a new representation-aware, event/morphology-guided
radar-to-ECG model validated on two datasets.

## Current Protocols

| Dataset | Protocol | Use |
|---|---|---|
| MMECG public | 11-fold strict LOSO | Main benchmark |
| CR_RVS / Nature | 5-fold subject-independent CV | External validation |

## Next Formal Milestone

The next paper-critical experiment is to lock the final evaluation protocol.
If the manuscript directly compares with radarODE-MTL's final long-ECG result,
this requires a fair radarODE-style long-term ECG assembly evaluation:

```text
radarODE-MTL original-style long ECG protocol
REM-AlignNet-MTL under the same long ECG protocol
MMECG strict LOSO
CR_RVS 5-fold subject-independent validation
```

Only after this should results be promoted from `pilot` to `final`.

## Hygiene Rules

- Do not create unregistered experiment directories.
- Do not cite `smoke`, `probe`, or `pilot` as final manuscript results.
- Add each experiment to `experiments_mmecg/EXPERIMENT_REGISTRY.csv`.
- Keep temporary outputs out of paper-facing directories.
- Use `scripts/README.md` before running scripts.
- Use `agent_skills/` as the single source for shared Claude/Codex skills.
- Run `python3 scripts/agent/sync_skills.py --check` before PRs that touch
  agent skills.
