# Q1 Paper Comparison Strategy

## Core Principle

This project is paper-first. The goal is not to force every baseline into a
`1 x 1600` full-window output, because published radar-to-ECG methods use
different output formulations. The Q1 manuscript should compare methods under
the closest fair protocol supported by each paper and dataset.

## Manuscript Positioning

The proposed method should be positioned as a radar-to-ECG framework that can
support:

- beat morphology reconstruction;
- event/fiducial timing recovery;
- optional continuous segment reconstruction.

Full-window ECG is valuable, but it is not the only valid reconstruction
target in this literature.

## Dataset-Specific Comparison Policy

### MMECG Public Dataset

Primary comparison should use subject-independent protocols whenever possible.

Recommended tables:

1. Beat morphology / ECG-shape comparison.
   - Compare against radarODE-MTL `ECG_shape` and other cycle-level methods.
   - Use Shape-200 or the closest published morphology metric.
2. Continuous segment comparison.
   - Compare against methods that output continuous segments, such as adapted
     BeatAware, RSSRNet, AirECG, CNN/LSTM/Conformer baselines, and our model.
   - Clearly state this table is not the only comparison axis.
3. Event/fiducial comparison.
   - Report R/RR/QRS/QT timing where labels and delineation are reliable.

### CR_RVS / Nature Dataset

Primary comparison should use the same dataset identity and closest published
metrics. VMD-CBNeT-style results can be cited as published references if the
paper uses this dataset, but the manuscript must clearly state any protocol
differences.

Recommended tables:

1. CR_RVS beat morphology and waveform metrics.
2. CR_RVS fiducial timing metrics aligned to published tables.
3. Cross-state or subject-independent validation as an additional strength if
   feasible.

## Handling Sample-Level Published Results

Sample-level random split results are not directly comparable to strict LOSO or
subject-independent cross-validation. They should not be mixed into the main
fair-comparison table as equal baselines.

Use this policy:

- If a published method uses subject-independent split on the same dataset:
  include in the main fair-comparison table.
- If a published method uses sample-level/random split on the same dataset:
  include in a separate "reported literature results" table or supplementary
  discussion with a protocol caveat.
- If a published method uses a different private dataset:
  cite qualitatively or include only in a literature context table.
- If a published method releases code and can be adapted to our protocol:
  include as a reproduced/adapted baseline, clearly labeled as adapted.

Recommended wording:

> We prioritize subject-independent comparisons. Results reported under
> sample-level random splits are listed separately because subject/session
> overlap can overestimate cross-subject radar-to-ECG reconstruction
> performance.

## Baseline Classes

| Class | Examples | Main Use |
|---|---|---|
| Published same-dataset subject-independent | radarODE-MTL if evaluated fairly; other public MMECG/CR_RVS papers if protocol matches | Main fair comparison |
| Published same-dataset sample-level | Methods reporting random split only | Separate reported-results table |
| Reproduced/adapted baselines | RSSRNet, AirECG, CNN/LSTM/Conformer if run in our protocol | Main reproduced table |
| Proposed family | REM-AlignNet-MTL and later variants | Main result and ablation |

## Current Evidence Interpretation

- Shape-200 evidence is useful and should not be treated as junk.
- Shape-200 is not evidence for long-term continuous ECG by itself.
- Full-window `1 x 1600` output is an additional capability and should be
  reported where comparable.
- radarODE-MTL comparison is currently strongest and fairest at the ECG_shape /
  Shape-200 level because the released source does not expose a complete long
  ECG assembly evaluator.

## Immediate Next Experiments

1. Build a literature comparison ledger:
   - paper;
   - dataset;
   - split protocol;
   - output target;
   - metrics;
   - whether directly comparable, adapted, or citation-only.
2. Lock two main experiment tracks:
   - MMECG Shape-200 / morphology track;
   - CR_RVS Shape-200 / morphology and fiducial track.
3. Use full-window `1 x 1600` experiments as a secondary continuous-monitoring
   table, not as the single success criterion.
4. Only promote a new architecture to 11-fold/5-fold final runs if it improves
   the paper-relevant main track.
