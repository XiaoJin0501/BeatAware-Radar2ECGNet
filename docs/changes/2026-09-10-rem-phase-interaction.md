# REM-Net phase-interaction candidate

## Scientific change

The candidate keeps the paper name and R-E-M thesis, but replaces the legacy
parallel morphology head with an explicit interaction. Predicted R events
define a differentiable R-to-R phase coordinate; representation features and
event fields are sampled on that coordinate and decoded jointly into a
beat-normalized morphology.

The model does not consume ECG-derived cycle bounds or ground-truth events at
inference. Shape-200 remains an explicitly beat-normalized morphology target,
not continuous ECG.

## Controlled evaluation

`experiments/specs/q1_remnet_phase_interaction_gate.json` defines a 2x2
event-pathway by phase-canonicalization design. All four factorial arms have
the same parameters. The interaction contrast, rather than a simple
leave-one-out comparison, is the pre-registered synergy test.

## Validation before training

- model output and factory contract covered by `tests/test_rem_phase_mtl.py`;
- phase anchors and differentiable resampling covered by a gradient test;
- existing morphology consistency tests remain green;
- only a smoke/data-path check is allowed before a clean code commit.
