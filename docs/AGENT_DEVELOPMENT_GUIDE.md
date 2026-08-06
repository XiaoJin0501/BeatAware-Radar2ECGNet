# Agent Development Guide

This guide defines how Claude Code, Codex, and human collaborators should work
in this repository. The standard is a careful deep-learning researcher and
senior research engineer: hypothesis-driven, reproducible, skeptical of easy
numbers, and disciplined about evidence.

## Core Principle

Every action must serve the single Q1 objective:

```text
Develop and validate a new radar-to-ECG reconstruction model that performs
strongly on MMECG and CR_RVS against fair published/reproduced baselines.
```

Do not create side quests, broad platform work, or unrelated model families
unless the user explicitly changes the project scope.

## Before Any Work

Read:

1. `docs/agent/SHARED_CONTEXT.md`
2. `docs/PROJECT_STATE.md`
3. `docs/RESEARCH_CONTRACT.md`
4. `docs/DATA_PROTOCOL.md`
5. `docs/METRIC_CONTRACT.md`
6. `docs/EXPERIMENT_POLICY.md`
7. this guide

Then identify:

- active branch;
- active objective;
- change type: implementation, protocol, experiment, report, or paper;
- expected scientific impact;
- files likely to be affected.

## Research Behavior

- Start from the research question, not from a module idea.
- Prefer controlled comparisons over clever but untraceable changes.
- Change one important variable at a time for experiments intended to support
  a paper claim.
- Treat surprising good results as suspicious until leakage, checkpointing,
  aggregation, and data split are audited.
- Treat surprising bad results as information, not failure.
- Never rewrite the scientific story to fit a lucky number.
- Keep ablations compact and claim-driven.

## Engineering Behavior

- Preserve the current `main` baseline unless working on a branch.
- Follow `docs/GITHUB_WORKFLOW.md` for branch names and PR flow.
- Make small, coherent commits with clear messages.
- Do not mix cleanup, model changes, metric changes, and report edits in one
  commit unless the change is explicitly a governance/update commit.
- Prefer existing loaders, model factories, loss utilities, and report scripts
  over inventing new parallel paths.
- Do not duplicate logic across scripts when a shared utility is appropriate.
- Do not add dependencies unless they are necessary and documented.
- Keep generated experiment outputs, checkpoints, logs, and local datasets out
  of Git.

## Experiment Development

Before training:

- create or reference an experiment spec;
- confirm dataset split and subject independence;
- confirm the baseline;
- confirm the single changed variable;
- confirm checkpoint selection uses validation only;
- confirm output path uniqueness;
- run a smoke test for new code paths.

During training:

- record command, branch, commit, config, and output path;
- monitor failure modes, not only headline metrics;
- do not manually cherry-pick test results.

After training:

- create or update a manifest;
- aggregate metrics according to `docs/METRIC_CONTRACT.md`;
- update `docs/PAPER_CLAIM_LEDGER.md` only when evidence supports it;
- keep failed experiments traceable if they influenced the next decision.

## Code Review Standard

Review scientific code for:

- data leakage;
- oracle information in final inference;
- test-subject normalization leakage;
- subject overlap across splits;
- checkpoint selection from test metrics;
- metric aggregation drift;
- unfair baseline preprocessing;
- mismatched output spaces, especially Shape-200 vs continuous ECG;
- claims that exceed the evidence.

Style matters, but scientific validity matters more.

## Paper and Reporting Standard

- Do not write manuscript claims from memory.
- Every number must trace to a registry row, manifest, script, and commit.
- Every baseline must be labeled as official, reproduced, or adapted.
- Shape-200 results must be labeled as beat morphology evidence.
- Long-term continuous ECG claims require a locked continuous/assembly
  protocol.
- If a figure is generated from pilot results, label it as pilot/progress.

## Agent Handoff Standard

At the end of meaningful work, report:

- branch and commit;
- files changed;
- tests/commands run;
- scientific impact;
- remaining risks;
- next recommended action.

For protocol, metric, or model-behavior changes, add a change note under:

```text
docs/changes/
```

## What Not To Do

- Do not launch broad architecture sweeps without a hypothesis.
- Do not keep adding modules to hide an evaluation mismatch.
- Do not promote pilot results to final.
- Do not introduce a new model name casually.
- Do not compare methods across incompatible protocols as if they are fair.
- Do not let agent-governance work become the research project.
