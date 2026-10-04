# Q1 Structured Q/R/S/T Event Gate Audit

Date: 2026-10-04. Experiment: `q1_structured_event_gate_v1`. Corrected
implementation commit: `d234e33d433107e2589290f941ca01ec84d5c9f8`; clean
launch commit: `861b4f4f886108f9dc4f9c93bd8d9519bb547f6f`. The corrected
run used repaired `loso_v2` folds 1/5/9, seed 42, validation-only checkpoint
selection by macro F1 at exactly 100 ms, and one-time test access only after
checkpoint freezing. All three fold state machines reached `COMPLETE`.

## Question and controlled change

E1 retained the incumbent WSST stem and five-block TCN, replacing the
waveform-derived read-out with four direct Q/R/S/T heatmaps. It used no
waveform decoder, interval loss, ordering loss, range attention, native RCG,
calibration, or ground-truth information at inference. E0 was the frozen
seed-42 Chen-TCN waveform-derived `prominence` read-out. Test ECG was used only
for the final endpoint calculation.

Localization MedAE used order-preserving matching within 150 ms. The safety
F1 and validation checkpoint criterion used a separate 100-ms tolerance. This
distinction is covered by a regression test.

## Corrected held-out test results

| Fold | Readout | Q MedAE | R MedAE | S MedAE | T MedAE | Macro MedAE | E1 macro F1@100 | E1 order violations |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Chen-TCN | 40 | 45 | 45 | 30 | 40.00 | -- | -- |
| 1 | E1 | 35 | 40 | 15 | 30 | 30.00 | 0.978 | 0.0525 |
| 5 | Chen-TCN | 45 | 15 | 15 | 20 | 23.75 | -- | -- |
| 5 | E1 | 20 | 15 | 20 | 15 | 17.50 | 0.920 | 0.1154 |
| 9 | Chen-TCN | 25 | 25 | 25 | 35 | 27.50 | -- | -- |
| 9 | E1 | 30 | 30 | 30 | 25 | 28.75 | 0.941 | 0.0768 |

Errors are milliseconds. Across folds, median E1 versus Chen-TCN MedAE was
Q 30 versus 40, R 30 versus 25, S 20 versus 25, and T 25 versus 30. E1
therefore strictly won Q, S, and T, while losing R by 5 ms. The mean of the
three fold-macro MedAEs fell from 30.42 to 25.42 ms, a 16.44% reduction.
Fold-level macro MedAE improved on folds 1 and 5 and worsened by 1.25 ms on
fold 9. The registered coverage clause passed.

## Pre-registered decision

| Clause | Result | Evidence |
|---|:---:|---|
| At least 3/4 event MedAEs improve and macro reduction >=10% | PASS | 3/4; 16.44% |
| Fold-macro MedAE improves on at least 2/3 folds | PASS | folds 1 and 5 |
| Promoted-event coverage no more than 0.03 below comparator | PASS | registered median-coverage check |
| Complete-beat ordering violation <=5% on every fold | FAIL | 5.25%, 11.54%, 7.68% |
| **Overall** | **FAIL** | one of four clauses failed |

This is a bounded, informative failure rather than evidence that direct event
supervision is ineffective. The corrected E1 arm improves three endpoint
errors and passes accuracy, consistency, and coverage, but independent
heatmaps do not enforce the physiological topology Q<R<S<T. High F1@100 also
must not be presented as sub-20-ms localization precision: it is a tolerant
detection metric, whereas the primary MedAEs are 15--40 ms.

Per the registered rule, this pilot does not authorize an 11-fold expansion or
a manuscript performance claim. It does, however, localize the remaining
failure to the read-out topology rather than to label availability or an
absence of event signal. Any continuation requires a separately registered,
single-variable topology-constrained event read-out compared with E1; it must
not silently relax the 5% ordering threshold or reuse test results for design.

## Withdrawn first execution

The first execution is retained under
`archive/invalid_experiments/q1_structured_event_gate_v1_f1_tolerance_mismatch_2026-10-04/`.
It computed the field named `f1_100ms` using 30 samples (150 ms), so its
validation checkpoint selection violated the registered protocol. Its apparent
FAIL and all associated numbers are diagnostic only and are not combined with
the corrected run.
