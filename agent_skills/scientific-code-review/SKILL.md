---
name: scientific-code-review
description: Review code for scientific validity risks rather than ordinary style issues.
---

# Scientific Code Review

Use for code changes that affect data loading, training, evaluation, metrics,
or baselines.

Prioritize:

- data leakage;
- oracle information in final inference;
- normalization leakage;
- subject overlap;
- checkpoint cherry-picking;
- unfair baseline protocol;
- GT events used during inference;
- metric aggregation drift.

Findings should cite files and lines. If no issue is found, state remaining
test gaps and residual risk.
