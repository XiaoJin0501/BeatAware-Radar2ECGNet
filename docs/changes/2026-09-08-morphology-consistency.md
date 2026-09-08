# Optional morphology consistency objective

Added a default-off supervised training term linking the continuous output's
reference-defined center cycle to the detached morphology head. A reference
target control supports attribution against additional local waveform supervision.
No inference architecture, evaluation metric, data split or frozen result changes.

Reference boundaries and amplitude statistics are training-only. Invalid/flat
cycles are skipped and coverage is logged. The loss preserves amplitude errors
by using a shared reference affine normalization for both predictions.

Candidate rationale and rejection criteria: docs/RQ1_CANDIDATE_CONTRIBUTION.md.
Training studies are not yet launched or promoted. New training runs must retain
the existing manifest, registration and commit-before-train requirements.
