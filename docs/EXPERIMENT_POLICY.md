# Experiment Policy

Every experiment must be registered and classified before it is launched.

Experiments must serve the single Q1 paper objective: validate a new
radar-to-ECG reconstruction model on MMECG and CR_RVS against strong published
or reproduced baselines. Avoid open-ended side experiments that do not directly
support this objective.

## Experiment Tiers

| Tier | Meaning | Publishable |
|---|---|---|
| smoke | Code-running check | false |
| probe | Quick trend check | false |
| pilot | Mostly complete, but protocol/model claim not locked | false by default |
| final | Protocol-locked formal experiment | true |
| invalid | Wrong, unfair, broken, or untraceable | false |

## Required Before Training

- Current code is committed before any non-smoke training starts.
- `git status --short` is clean or contains only intentionally ignored local
  files.
- Experiment spec exists in `experiments/specs/`.
- A registry row exists in the single canonical registry:
  `experiments_mmecg/EXPERIMENT_REGISTRY.csv`.
- The registry row records `commit_id` and `branch`. New experiments must not
  use `unknown`.
- Branch follows `docs/GITHUB_WORKFLOW.md`.
- Baseline and single changed variable are identified.
- Dataset split and normalization rules are stated.
- Checkpoint selection metric is stated.
- Output directory is unique.
- Smoke validation is done for new code paths.

## Required After Training

- Manifest exists in `experiments/manifests/`.
- Code commit and branch are recorded and match the pre-training code state.
- Dataset fingerprint or source version is recorded.
- Metrics and aggregation method are recorded.
- Prediction artifact paths are recorded if generated.
- Failure cases and known caveats are recorded.

## Promotion Rule

Only the `result-promotion` workflow may change:

```text
pilot -> final
publishable=false -> publishable=true
```

Promotion requires complete folds, locked protocol, clean commit, manifest,
prediction artifacts, metric audit, and fair baselines.

## Commit Before Train

No formal or pilot training run may start from an uncommitted code state. This
prevents training results from drifting away from the code that produced them.

Allowed exception:

- `smoke` runs used only to check syntax/runtime shape can run before commit,
  but they must not be reported as experiment results.
