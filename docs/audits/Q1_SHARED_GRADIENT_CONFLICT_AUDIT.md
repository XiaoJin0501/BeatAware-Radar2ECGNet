# Shared-Gradient Conflict Audit

**Experiment:** `q1_shared_gradient_conflict_audit`
**Date:** 2026-09-17
**Data:** repaired `loso_v2` validation folds 1/5/9; test unopened
**Operation:** read-only autograd on the completed L3 validation-best
checkpoints; no optimizer and no parameter update

## Actual computation graph

The completed L3 runs used `CENet("B1")`, not a B2 dual-stream model. The
parameters actually shared by waveform and event losses are `wsst_stem` and
`decoder.tcn.blocks.0..4`. The task-specific `decoder.head` and
`decoder.event` parameters were excluded. B2 fusion and upsampling modules do
not exist in this graph.

For every validation batch, the audit independently computed gradients of the
locked `0.50 × MR-STFT` waveform term and `0.20 × event BCE` with respect to
the shared parameters. It then measured global and module-group cosine
similarity. All validation batches were included: 11/32/123 batches for folds
1/5/9 (166 batches, 2,635 windows total).

## Preregistered decision

Conflict required both:

1. mean global cosine below zero on 3/3 folds, with the batch-bootstrap 95% CI
   upper bound below zero on at least 2/3 folds; and
2. at least one predeclared shared module group with mean cosine ≤ -0.30 on at
   least 2/3 folds.

Thresholds and groups were registered before the full audit.

## Results

| Fold | Batches | Global mean cosine | Median | Negative batches | Bootstrap 95% CI |
|---:|---:|---:|---:|---:|---:|
| 1 | 11 | -0.1371 | -0.1502 | 100.0% | [-0.1761, -0.0965] |
| 5 | 32 | +0.2755 | +0.2619 | 15.6% | [+0.1777, +0.3727] |
| 9 | 123 | +0.0876 | +0.0401 | 40.7% | [+0.0402, +0.1358] |

Fold 1 contains a reproducible negative gradient relationship, especially in
the WSST stem (-0.1687) and first TCN block (-0.1652). It does not generalize:
folds 5 and 9 have significantly positive global means. No predeclared module
has mean cosine ≤ -0.30 in even one fold, so the localized requirement also
fails.

## Decision

**Shared-gradient conflict is not supported as the cross-fold explanation of
the L3 F1 trade-off.** Both preregistered requirements fail. A PCGrad training
arm is therefore not authorized from this hypothesis, and the thresholds will
not be revised after observing the result.

This does not prove that optimization interactions never occur. It establishes
the narrower result that the completed L3 checkpoints do not show a consistent,
strong MR-STFT-versus-event-BCE directional conflict on B1's shared parameters.
The fold-specific negative result in fold 1 is heterogeneity, not evidence for
a general mechanism.

The next controlled architecture question is the repaired-protocol,
capacity-controlled B2a/B2b comparison. That experiment must treat native RCG
information as an unproven input hypothesis and must not prestate an expected
gain.

## Artifacts

- Specification: `experiments/specs/q1_shared_gradient_conflict_audit.json`
- Implementation: `scripts/audit_shared_gradient_conflict.py`
- Machine-readable summary: `docs/audits/Q1_SHARED_GRADIENT_CONFLICT_AUDIT.json`
- Per-batch matrix: `docs/audits/Q1_SHARED_GRADIENT_CONFLICT_AUDIT.csv`
