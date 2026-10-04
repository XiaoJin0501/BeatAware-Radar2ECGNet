# Q1 Structured Q/R/S/T Event Gate Audit — Protocol-Invalid Run

**Withdrawn 2026-10-04:** the implementation used a 30-sample (150 ms) match
tolerance for the field named `f1_100ms`. Because that field selected the best
validation checkpoint, the run did not implement the registered F1@100-ms
selection rule. All numerical results below are retained for diagnosis only;
the FAIL decision is withdrawn pending a corrected from-scratch rerun.

Date: 2026-10-04. Experiment: `q1_structured_event_gate_v1`. Implementation
commit: `c3779457d11e848b94fd5f9417609c17f1a87167`; clean launch commit:
`d44ea8adb6b594db4985f1ae68cd175c293e8d01`. The run used repaired `loso_v2`
folds 1/5/9, seed 42, validation-only checkpoint selection, and one-time test
access after checkpoint freezing. All three state machines finished at COMPLETE.

## Question and controlled change

The event-only E1 arm retained the incumbent WSST stem and five-block TCN and
replaced the two-scale R-anchor read-out with four direct Q/R/S/T heatmaps. It
had no waveform decoder, interval loss, ordering loss, range attention, native
RCG, calibration, or GT information at inference. The comparator was the frozen
seed-42 Chen-TCN waveform-derived `prominence` read-out extracted before E1
training.

## Held-out test results

| Fold | Readout | Q MedAE | R MedAE | S MedAE | T MedAE | Macro MedAE | Macro F1@100 | Order violations |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Chen-TCN | 40 | 45 | 45 | 30 | 40.00 | -- | -- |
| 1 | E1 | 50 | 50 | 25 | 35 | 40.00 | 0.982 | 0.042 |
| 5 | Chen-TCN | 45 | 15 | 15 | 20 | 23.75 | -- | -- |
| 5 | E1 | 20 | 15 | 15 | 15 | 16.25 | 0.930 | 0.123 |
| 9 | Chen-TCN | 25 | 25 | 25 | 35 | 27.50 | -- | -- |
| 9 | E1 | 30 | 30 | 30 | 25 | 28.75 | 0.953 | 0.056 |

Errors are milliseconds. Across folds, median E1 versus Chen-TCN MedAE was:
Q 30 versus 40, R 30 versus 25, S 25 versus 25, and T 25 versus 30. Thus E1
strictly won two of four event metrics. Mean fold-macro MedAE decreased only
6.85%, below the locked 10% requirement, and improved on only one of three
folds. Coverage safety passed. Ordering safety failed because folds 5 and 9
exceeded the locked 5% violation ceiling.

## Pre-registered decision

| Clause | Result |
|---|:---:|
| At least 3/4 event MedAEs improve and macro reduction >=10% | FAIL |
| Fold-macro MedAE improves on at least 2/3 folds | FAIL |
| Promoted-event coverage no more than 0.03 below comparator | PASS |
| Complete-beat ordering violation <=5% on every fold | FAIL |
| **Overall** | **FAIL** |

The high F1@100 values coexist with 25--50 ms median localisation errors because
F1 uses a tolerant detection window and was the pre-registered validation
selection endpoint. They must not be interpreted as precise fiducial timing.

This invalid run neither closes nor promotes the direct four-heatmap route. No
event-conditioned decoder, range-aware encoder, loss retuning, new threshold,
11-fold expansion, or manuscript claim is authorised while the corrected rerun
is pending. Its artifacts are retained under `archive/invalid_experiments/`.
