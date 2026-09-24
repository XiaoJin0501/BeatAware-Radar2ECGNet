# Paper Storyline and Experimental Summary — IEEE TIM

**Title (locked):** Contactless Radar-to-ECG Reconstruction: A Rigorous Leak-Free
Benchmark and Factorized Deep Architecture

**Governing documents:** `docs/paper/MANUSCRIPT_ARCHITECTURE_TIM.md`,
`docs/PAPER_CLAIM_LEDGER.md` (Benchmark paper claims, 2026-09-19),
`docs/METRIC_CONTRACT.md`. Every number below traces to a registered experiment.

---

## The thesis in one paragraph

Contactless radar-to-ECG reconstruction reports ECG Shape Recovery correlations of 0.8-0.98,
which would imply diagnostic-grade morphology from a non-contact sensor. Under one
leak-free measurement procedure applied to six models from five families on two radar
modalities, none of that survives: the ECG Shape Recovery metric is saturated by a predictor
that never sees the radar, and strict blind Long-term ECG Reconstruction is bounded near
$R^2 \approx 0$. The residual error is not morphological but **temporal** -- a fixed
average beat placed at ground-truth cardiac anchors (ECG Anchor Estimation) reaches $R^2 = 0.730$, while the
same beat placed at radar-derived anchors falls below zero. Amplitude shrinkage, the
field's most-reported artefact, is therefore not a defect to be engineered away: it is
the loss-optimal hedge against irreducible beat-to-beat timing error. The paper's
contribution is the corrected procedure, the true state of the art under it, and a
factorized architecture whose explicit anchor read-out makes the timing term
**measurable** rather than merely hypothesised.

> **Narrative note.** An earlier framing attributed this timing error to
> electromechanical-delay (EMAT) jitter and claimed factorization *defeats* mean
> collapse. Both are contradicted by the measurements (Ledger C7; variance ratio
> 0.331) and must not appear in the manuscript. The defensible claim is weaker and
> more useful: the error is within-subject beat-to-beat jitter of unidentified
> physiological origin, and factorization does not remove shrinkage -- it *localises*
> the cause.

---

## 1. The measurement bottleneck and the core idea

### 1.1 What the field measures

Three faults in the prevailing procedure inflate reported performance:

| Fault | Mechanism | Evidence |
|---|---|---|
| **F1 Split leakage** | The widely used MMECG LOSO construction reuses **two validation subjects across all eleven folds**, so checkpoint selection sees the same held-out data every fold. | `docs/MMECG_SPLIT_PROVENANCE_AUDIT.md` (C1) |
| **F2 Saturated metric** | ECG Shape Recovery PCC is computed on beats that are cut, resampled and amplitude-normalised. A **radar-free template** -- the training fold's mean beat -- scores **0.879-0.887** on both datasets. | `docs/audits/Q1_TIER1_TEMPLATE_FLOOR_AUDIT.md` (C2) |
| **F3 Timing / morphology conflation** | ECG Shape Recovery scores are reported as morphology evidence although the beat-cutting step has already removed the timing error. | `docs/audits/Q1_TEMPLATE_TIMING_AUDIT.md` (C4) |

F2 is the decisive one. Every previously reported Tier-1 figure of this project
(0.882, 0.881) sits **at** the floor; those claims are suspended as radar evidence.

### 1.2 The architectural response

REM-Net factorises the estimation problem into one shared representation and two
decoupled read-outs:

- **Representation** — a single stem over 50 range bins of a synchrosqueezed
  time-frequency transform (`50 x 71 x 240`, 8 s), frequency-pooled to a 400-step
  latent sequence at 50 Hz;
- **Shared temporal backbone** — five dilated residual TCN blocks (k = 7, dilation
  1-16), receptive field ≈ 7.5 s, i.e. the entire window;
- **Anchor Head (E)** — two Gaussian anchor channels, `2 x 1600` logits;
- **Waveform Head (M)** — a linear `1 x 1600` output in the train-fold robust-scaled
  space.

Total **1.44 M parameters** (stem 0.21 M + decoder 1.23 M).

The factorization is not claimed to improve waveform fidelity. Its purpose is
instrumental: an explicit anchor read-out turns the timing term into a directly
measured quantity, which is what makes Section 5 possible at all.

> **Insight.** The paper's object of study is the *measurement procedure*; the
> architecture earns its place by making the dominant error term observable, not by
> winning a correlation column.

---

## 2. Experimental setup and the strict protocol

| | MMECG | CR_RVS |
|---|---|---|
| Radar | 77 GHz FMCW | 24 GHz CW |
| Protocol | strict LOSO, `loso_v2`, **11 distinct validation subjects** | subject-independent 5-fold |
| Measurand | continuous `1 x 1600` ECG @ 200 Hz, per-record z-scored | same |
| Systems | 6 models / 5 families x **3 seeds** x 11 folds = **198 sealed folds** | 3 models x 5 folds |

Protocol rules, all binding and all registered:

- **Train-fold-only scaling.** Robust scaler (median / IQR) fitted on the training
  fold; per-window min-max as a primary target is prohibited.
- **Validation-only selection.** A one-shot test-access state machine
  (`TRAINING → CHECKPOINT_FROZEN → TEST_EVALUATING → COMPLETE`) makes it structurally
  impossible for a test metric to select a checkpoint.
- **Identical budget** across all six systems; every reproduction adaptation is
  registered in an adaptation table (D1-D6).
- **Coverage gating.** Rate metrics aggregate only over folds with
  `rr_coverage >= 0.80`; excluded folds are reported individually as failures.
- **Heavy-tailed ratios** are reported as median with IQR, never mean alone.
- **Disclosed degeneracy.** The validation composite contains clipped terms that were
  inactive in the observed range; the methods section says so rather than editing a
  frozen composite after the fact.

This draws an explicit line against calibration-prior evaluation, in which a
subject-specific template or scaling is fitted with test-subject information.

> **Insight.** The procedure is the instrument. Stating its tolerances is what lets
> the later null results count as measurements rather than as failures to tune.

---

## 3. The matched Long-term ECG Reconstruction benchmark

### 3.1 Table 1 — MMECG `loso_v2`, three seeds

| Model | Params | PCC@0 ↑ | R² ↑ | NRMSE ↓ | Anchor F1@100 ↑ | Var. ratio | >10 Hz (med.) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Chen-TCN | 0.61 M | **0.329 ± 0.018** | 0.003 ± 0.038 | 2.153 ± 0.041 | 0.808 ± 0.033 | 0.433 | 0.362 |
| **REM-Net** | **1.44 M** | 0.324 ± 0.015 | **0.041 ± 0.016** | **2.125 ± 0.018** | 0.732 ± 0.035 | 0.331 | 0.122 |
| radarODE-MTL | 41.6 M | 0.290 ± 0.001 | −0.016 ± 0.022 | 2.183 ± 0.020 | **0.819 ± 0.010** | 0.368 | 0.274 |
| Conformer | 0.42 M | 0.282 ± 0.010 | −0.053 ± 0.030 | 2.216 ± 0.029 | 0.767 ± 0.012 | 0.447 | 0.547 |
| WaveGRU-Net | 0.71 M | 0.203 ± 0.007 | −0.057 ± 0.015 | 2.239 ± 0.017 | 0.557 ± 0.004 | 0.283 | 0.557 |
| Attention-DCGAN | 4.35 M | 0.113 ± 0.012 | −0.474 ± 0.128 | 2.625 ± 0.093 | 0.675 ± 0.003 | 0.638 | 0.806 |

**REM-Net does not win this table, and the manuscript must not say it does.** It is
statistically indistinguishable from Chen-TCN on PCC (**p = 0.831**), holds the highest
$R^2$ and lowest NRMSE, and is **behind** radarODE-MTL and Chen-TCN on waveform-derived Anchor F1@100.
With Holm correction over 25 tests, only the WaveGRU-Net and Attention-DCGAN contrasts
survive; everything else is descriptive.

### 3.2 Two readings that overturn standard intuitions

**Capacity does not help.** radarODE-MTL carries **29x** REM-Net's parameters
(41.6 M vs 1.44 M) and lands at $R^2 = -0.016$.

**High-frequency energy does not track quality.** REM-Net retains the *least* relative
>10 Hz energy (**0.122**) while Attention-DCGAN retains the *most* (**0.806**) with the
*worst* correlation (0.113). The textbook expectation -- that spectral objectives buy
sharp QRS edges and that sharpness implies fidelity -- fails on measured data.

### 3.3 Where factorization does pay

The one consistent advantage is interval-level. REM-Net's **Cycle Length Estimation**
error is lowest in **six of the seven** physiological scenarios across both radars,
including **10.9 ms during sleep** against **45.8** and **60.4 ms** for the two
comparators. In the pooled three-seed aggregate it is second (**49.1 ms** vs
**45.5 ms**), and it is the only system whose interval metrics require the anchor
coverage gate, losing **2 of 33** entries to anchor-detection collapse.

Clinically: interval accuracy at this level supports rate and rhythm monitoring, while
the waveform remains unsuitable for morphological diagnosis.

> **Insight.** Six architectures from five families converge on the same ceiling. A
> ceiling that is invariant to capacity, objective and inductive bias is a property of
> the measurement problem, not of any model.

---

## 4. Loss gate and end-to-end optimisation

### 4.1 What was actually ablated

The registered L1/L2/L3 loss gate selected the multi-resolution STFT objective at
weight **0.50** (windows 128/256/512, hops 32/64/128), with a fixed Huber term and a
fixed anchor BCE term at **0.20**. `EVENT_W = 0.20` is held identical across all arms
as a control variable and **was never itself ablated**; no sentence may therefore
credit the anchor head with a share of the waveform result.

### 4.2 Verified engineering fact

The model is **100 % end-to-end jointly optimised**. Verified directly:

```
total = 1,438,019   requires_grad=True → 1,438,019   in optimizer → 1,438,019
frozen tensors: none
wsst_stem 307.3 | decoder.tcn 884.6 | decoder.head 1712.3 | decoder.event 1731.5
```

`torch.optim.AdamW(model.parameters(), ...)` takes the full parameter set; there is no
`requires_grad = False` anywhere in the training path, no parameter groups, no staged
schedule. The two `load_state_dict` calls are resumption and post-selection reload, not
pretrained initialisation.

The consequence is structural: the anchor BCE gradient **flows back into the shared
TCN** and modulates the features the waveform head consumes. The architecture is
decoupled at the read-out; the optimisation is strongly coupled at the backbone.

> **Insight.** "Factorized" describes the read-out topology, not an independence
> assumption about training. Stating this prevents a reviewer from reading Fig. 2 as a
> two-stage or frozen-backbone design, and it bounds what the ablation can claim.

---

## 5. Physiological states: the scenario breakdown

Scenarios are **not balanced over subjects** — in MMECG, sleep appears in three test
subjects and post-exercise in six, while two folds contain normal breathing only, and a
single subject contributes 935 of the sleep windows. Every cell below is therefore
computed **per (subject, scenario)** first, then averaged over seeds and aggregated over
subjects, with the subject count attached. No row may be read without it.

The recorded MMECG states are **Normal breathing, Irregular breathing, Post-exercise and
Sleep** (codes NB/IB/PE/SP); CR_RVS records **Resting, Valsalva and Apnea**. There is no
breath-hold condition in MMECG.

### 5.1 MMECG (77 GHz FMCW, three seeds, median over subjects)

| Scenario | Subj. | PCC — REM-Net / Chen-TCN / WaveGRU | Var. ratio — REM-Net / Chen-TCN / WaveGRU | Cycle Length — REM-Net / Chen-TCN / WaveGRU |
|---|---:|---|---|---|
| Normal breathing | 11 | **0.358** / 0.332 / 0.244 | 0.338 / 0.403 / 0.329 | **39.5** / 50.4 / 77.0 ms |
| Irregular breathing | 7 | **0.437** / 0.422 / 0.212 | 0.360 / 0.367 / 0.236 | **56.4** / 75.8 / 99.7 ms |
| Post-exercise | 6 | 0.261 / **0.307** / 0.143 | 0.332 / 0.382 / 0.276 | **49.8** / 102.2 / 75.4 ms |
| Sleep | 3 | 0.265 / **0.316** / 0.138 | 0.381 / 0.353 / 0.249 | **10.9** / 45.8 / 60.4 ms |

### 5.2 CR_RVS (24 GHz CW, seed 42, median over subjects)

| Scenario | Subj. | PCC — REM-Net / Chen-TCN / WaveGRU | Var. ratio — REM-Net / Chen-TCN / WaveGRU | Cycle Length — REM-Net / Chen-TCN / WaveGRU |
|---|---:|---|---|---|
| Resting | 30 | 0.525 / 0.599 / **0.609** | 0.383 / 0.437 / 0.452 | **42.7** / 55.4 / 70.1 ms |
| Valsalva | 27 | 0.483 / 0.500 / **0.577** | 0.364 / 0.358 / 0.395 | **56.6** / 105.9 / 119.7 ms |
| Apnea | 23 | 0.341 / 0.401 / **0.424** | 0.309 / 0.275 / 0.325 | 74.7 / 115.9 / **79.5** ms |

### 5.3 Four readings, two of which overturn the expected account

**(a) Subject spread dwarfs scenario spread on MMECG.** Within a single scenario the
per-subject PCC range spans 0.5–0.8 in absolute terms (normal breathing: −0.004 to
0.643 for REM-Net), while scenario medians move by only 0.1–0.2. Physiological state
explains far less than subject identity on the FMCW dataset.

**(b) Amplitude shrinkage does *not* deepen under physiological stress on MMECG.**
REM-Net's variance ratio spans **0.332–0.381** across all four states and is **highest
in sleep (0.381)** — the state where its correlation is among the lowest (0.265). An
account in which respiratory or postural stress progressively deepens amplitude
collapse on this radar is contradicted by the measurement.

**(c) CR_RVS *does* show a monotone manoeuvre effect, and it is model-agnostic.**
Resting → Valsalva → Apnea degrades variance ratio, PCC and Cycle Length error together,
in the same order, for **all three** systems (REM-Net's variance ratio 0.383 → 0.364 →
0.309; PCC 0.525 → 0.483 → 0.341; Cycle Length 42.7 → 56.6 → 74.7 ms). This is a
property of CW measurement under breath manoeuvres, shared by every architecture — not
evidence about any one of them, and not attributable by these data to a named
physiological mechanism such as electromechanical delay.

**(d) Where REM-Net is robust is the interval, not the waveform.** Its Cycle Length
error is lowest in **six of the seven** scenarios across both radars, including the two
hardest MMECG states (post-exercise **49.8 ms** vs 102.2 ms, sleep **10.9 ms** vs
45.8 ms) — precisely where its correlation is *not* the highest. On CW it leads on
Cycle Length in two of three scenarios while trailing on PCC in all three.

> **Insight.** Factorization does not buy scenario-invariant morphology — amplitude
> shrinkage is essentially state-invariant on FMCW and declines with every model on CW.
> What it buys is that the interval read-out survives the states in which the waveform
> read-out does not, which is exactly the separation a factorized design predicts and
> the reason the two read-outs must be reported apart.

---

## 6. Mechanistic analysis of the Long-term ECG Reconstruction gap

### 6.1 The oracle-anchor decomposition

A fixed template beat is placed at two anchor sets and scored by the same code:

| Probe | MMECG (FMCW) | CR_RVS (CW) |
|---|---:|---:|
| **T0** template at **ground-truth** anchors | PCC 0.856 / **R² 0.730** | **R² 0.764** |
| template at **radar-derived** anchors | PCC ≈ 0.20 / **R² < 0** | — |
| best trained model | **R² ≈ 0.041** | R² 0.18-0.25 |

A radar-free average beat, given correct timing, outperforms every trained system by a
wide margin. The same beat, given radar timing, collapses.

### 6.2 The residual is within-subject jitter, and it is not a latency

| Component | Measurement |
|---|---|
| Global offset | **median 0 ms** on both radars — no systematic latency survives training |
| Per-subject offset | SD **14-32 ms**; removing it with an **oracle** recovers only **6-17 %** of the $R^2$ gap |
| Beat-to-beat jitter | **7-22 ms** (CW), **30-60 ms** (FMCW) |
| Anchor recall | **0.66-0.86** — missed and spurious beats |

**EMAT is not identified.** The learned-delay route (`δ_sys` / `δ(t)`) was tested and
collapsed to a constant; it is closed and appears only as labelled negative evidence.
Attributing the residual to electromechanical delay would exceed the measurement.

### 6.3 Why amplitude shrinkage is rational

Under an $L_2$-type objective with anchor uncertainty of tens of milliseconds, the
conditional mean of the target over the timing posterior is a **low-amplitude,
low-bandwidth** waveform. Shrinkage (variance ratio **0.331**), $R^2 \approx 0$ and the
oracle-timing bound are one phenomenon seen from three directions. This also explains
why **enlarging a network changes capacity but not anchor precision**, and therefore
does not move the task.

> **Insight.** The bottleneck is sub-50 ms anchor localisation that does not generalise
> across subjects. Factorization does not defeat mean collapse — it *explains* it, by
> exposing the timing term that makes collapse the optimal response.

---

## 7. Robustness and efficiency

**Robustness** (eval-only on frozen checkpoints, Shape-200 track): graceful degradation
under input corruption — MMECG **0.834 at 0 dB** (−5.4 %), **0.781 at −6 dB**; the CW
modality degrades **2-5x faster**, consistent with its single-channel geometry and
absence of range-bin redundancy. *Scope: an input-corruption test, not a motion-artefact
or breathing-mode study.*

**Efficiency**, stated per track because the two are different networks:

| Track | Model | Params | Latency |
|---|---|---:|---:|
| Continuous (Table 1) | REM-Net B1 | **1.44 M** | — |
| Continuous (Table 1) | radarODE-MTL (4-task EGA) | 41.6 M | — |
| Shape-200 | REM-Net Shape-200 | 1.788 M | **0.98 ms/window** (RTX 4080 SUPER) |

The two tables may never be mixed in one sentence.

> **Insight.** A 1.44 M network reaching the same ceiling as a 41.6 M multitask ODE
> model is the efficiency argument *and* the scientific argument: the ceiling is set by
> anchor precision, so parameters buy nothing against it.

---

## 8. External validity and the literature

- **CR_RVS (24 GHz CW).** The template floor (**0.887**) and the oracle-timing bound
  (**R² 0.764**) reproduce. The $R^2 \approx 0$ ceiling **does not** (0.18-0.25). REM-Net
  is the **weakest** of the three matched models on CW (**0.442** vs 0.506 / 0.503,
  winning 1 of 5 folds). Reporting this inconsistency is a requirement, not an option.
- **Reported literature.** One CR_RVS paper reports $\rho = 0.959$ under a five-fold
  window-level split of 80 %-overlapping segments and per-subject medians of
  **0.39-0.90** under LOSO **in the same paper**. Others report 0.96-0.99 from
  resting-only data with overlapping-window data-level splits, or from a single fixed
  five-subject test set. The field's own numbers document the inflation; no leakage is
  alleged where the authors disclosed their protocol.

> **Insight.** The contribution generalises across radar modality and is corroborated by
> the literature's internal contradictions, which is stronger evidence than any single
> benchmark win would have been.

---

## 9. Contribution statement

1. **C1** A leak-free measurement procedure for radar-to-ECG, repairing a split that
   shares two validation subjects across eleven folds and adding a one-shot test-access
   gate.
2. **C2** A radar-free **template floor** for ECG Shape Recovery metrics, showing the
   field's headline metric is saturated at **0.879-0.887** and prior results sit on it.
3. **C3** A matched benchmark of six models from five families over **198 sealed folds**
   and three seeds, establishing the true state of the art at $R^2 \approx 0$.
4. **C4/C7** A mechanistic decomposition localising the residual in **within-subject
   beat-to-beat anchor jitter**, with amplitude shrinkage shown to be the loss-optimal
   response, and with EMAT explicitly **not** identified.
5. **C5/C6** External validation on a second radar modality and a protocol-annotated
   audit of the reported literature.

**Forbidden sentences** (Ledger, binding): REM-Net outperforms the matched baselines;
REM-Net recovers beat morphology from radar; REM-Net resolves amplitude collapse; any
Tier-1 number without its template floor; any CR_RVS literature comparison presented as
a matched result.
