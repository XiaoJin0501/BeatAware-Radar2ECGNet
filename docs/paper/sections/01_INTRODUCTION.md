# I. Introduction

*Draft for IEEE TIM. Terminology follows `docs/paper/TERMINOLOGY.md`; every
quantitative sentence maps to a claim in `docs/PAPER_CLAIM_LEDGER.md`.*

---

Continuous electrocardiography outside the clinic is limited less by sensing physics
than by adherence. Electrodes irritate skin, wearables are removed at night, and the
populations that would benefit most from uninterrupted rhythm monitoring — people under
evaluation for paroxysmal arrhythmia, infants, patients with burns or dermatological
contraindications — are the least able to tolerate contact. Millimetre-wave radar
measures the chest-wall displacement produced by each cardiac contraction without
contact, at low cost and through clothing. A growing literature now asks the harder
question of whether the ECG *waveform*, and not only heart rate, can be estimated from
that displacement, and reports correlations of 0.8–0.98 that would place contactless
morphology within reach of diagnostic use.

This paper argues that the difficulty has been mislocated. The radar and the ECG are
not two views of one waveform: they are coupled through a *latent, subject-specific and
beat-varying timing relation*. Mechanical chest-wall motion follows electrical
depolarisation after a delay that is neither constant across subjects nor stable from
beat to beat. Any model trained to regress the ECG directly from radar must therefore
solve two problems at once — *what* the beat looks like and *when* it occurs — and the
second one is, under a leak-free protocol, by far the harder. We call this the
**morphology–timing entanglement**, and we show that it, rather than any deficit in
generative capacity, sets the ceiling that the field is currently reporting around.

## A. Why the entanglement is invisible in current evaluations

The entanglement is hidden by the way the field measures. The prevailing headline
quantity — here called **ECG Shape Recovery** — is computed on beats that are cut
between *reference* R peaks, resampled to a fixed length, and amplitude-normalised on
both sides. Each of those three steps removes exactly the information the radar was
supposed to supply, and the first one hands the model the timing answer for free.

The consequence is measurable. A predictor that never sees the radar at all — the
training fold's mean beat — scores **0.879–0.887** under this metric on two independent
datasets. Published values, and our own earlier ones, sit at that level. A metric whose
radar-free floor coincides with its reported ceiling cannot adjudicate a morphology
claim.

Two further protocol faults compound it. The leave-one-subject-out construction most
widely used on the public 77 GHz dataset reuses the *same two validation subjects across
all eleven folds*, so checkpoint selection sees identical held-out data every fold. And
beat-level scores obtained at reference timing are routinely discussed as evidence about
waveform fidelity under deployment, where no reference timing exists.

## B. REM-Net: factorising morphology from timing

We propose **REM-Net**, a network that makes the entanglement explicit by construction.
A single synchrosqueezed time–frequency representation of the range profile feeds a
shared temporal backbone, which then splits into two read-outs: an **anchor head**
performing **ECG Anchor Estimation**, and a **waveform head** performing **Long-term ECG
Reconstruction**. The two branches model cardiac events and morphology separately, and
are optimised **jointly and end-to-end** — no stage is frozen, and the anchor
supervision gradient reaches the shared backbone, so event supervision modulates the
features the waveform branch consumes. The whole model uses **1.44 M parameters**.

The factorization is not proposed as a way to generate better waveforms, and we do not
claim it does. Its value is instrumental and clinical: an explicit anchor read-out turns
the dominant error term into a directly measured quantity, and it yields an interval
estimate that survives conditions in which the waveform estimate does not.

## C. What a leak-free benchmark measures

We rebuild the split with eleven distinct validation subjects, fit all scaling on the
training fold only, select checkpoints on validation exclusively through a one-shot
test-access state machine, and benchmark **six systems from five architectural families**
over **198 sealed folds** (six models × three seeds × eleven folds) on 77 GHz FMCW, with
external validation on 24 GHz CW.

On strict blind Long-term ECG Reconstruction, all six are bounded near $R^2 \approx 0$;
the best value observed is **0.041 ± 0.016**. The ceiling is indifferent to the factors
that usually move deep-learning results. A 41.6 M-parameter multitask neural-ODE system
reaches $R^2 = -0.016$, while our 1.44 M-parameter model reaches the best value, and the
two strongest systems are statistically indistinguishable in correlation ($p = 0.83$).
Relative high-frequency energy — often invoked as evidence of preserved QRS sharpness —
does not track quality either: the system retaining the *most* energy above 10 Hz (0.806
of the reference share) has the *worst* correlation (0.113).

## D. Factor-wise decomposition: the error is timing, and it is jitter

To localise the ceiling we place a single fixed template beat — a radar-free average
shape carrying no subject-specific morphology — at two different anchor sets, scored by
the same code. At **reference** anchors it reaches $R^2 = 0.730$ on FMCW and $0.764$ on
CW. At **radar-derived** anchors the identical waveform falls below zero. A shape that
knows nothing about the radar outperforms every trained system by a wide margin when,
and only when, it is placed correctly in time.

Decomposing the anchor error further shows that it is **not a drift and not a
calibratable latency**. No global offset survives training (median **0 ms** on both
radars); per-subject offsets exist (SD **14–32 ms**) but removing them with an oracle
recovers only **6–17 %** of the $R^2$ gap; the residual is within-subject beat-to-beat
**jitter** (**7–22 ms** on CW, **30–60 ms** on FMCW) together with missed and spurious
beats (recall **0.66–0.86**). We therefore name the residual *anchor timing jitter* and
deliberately do **not** attribute it to electromechanical delay: a learned-delay model
was tested, collapsed to a constant, and is reported as negative evidence.

This reframes the field's most-reported artefact. Amplitude shrinkage — our system
retains **0.331** of the reference variance — is not a defect awaiting a better
generator. Under any squared-error-like objective with tens of milliseconds of anchor
uncertainty, the conditional mean of the target over the timing posterior *is* a
low-amplitude, low-bandwidth waveform. Shrinkage, $R^2 \approx 0$ and the oracle-timing
bound are one phenomenon seen from three directions, and together they explain why
enlarging a network — which changes capacity but not anchor precision — has not moved
this task.

## E. Contributions

**1) Problem formulation and method.** We identify the **morphology–timing
entanglement** as the key difficulty in radar-based ECG reconstruction, and propose
REM-Net, a physically factorized network that models morphology and cardiac events in
separate branches and optimises them jointly end-to-end with only **1.44 M parameters**.

**2) Cross-modality validation.** Under strict leave-one-subject-out on two radar
modalities (77 GHz FMCW and 24 GHz CW), REM-Net attains **Cycle Length Estimation
accuracy statistically indistinguishable from a 41.6 M-parameter multitask neural-ODE
comparator** (**48.8 vs 45.5 ms**, $p = 0.97$) using **29× fewer parameters**, and the
lowest error in **six of seven** physiological scenarios against the two matched-family
baselines evaluated per scenario. Its Long-term ECG Reconstruction fidelity is
statistically indistinguishable from the strongest comparator ($p = 0.83$) while holding
the best $R^2$ and NRMSE. The diagnostic findings — the template floor and the
oracle-timing bound — reproduce on both modalities; the $R^2 \approx 0$ ceiling does not,
and we report that inconsistency.

**3) Evaluation protocol and error analysis.** We establish a strict subject-independent
protocol for both modalities, together with a **radar-free template floor** showing that
the field's headline metric is saturated at **0.879–0.887**, and through factor-wise
error decomposition show that **beat-to-beat anchor timing jitter — not morphological
distortion and not a correctable timing drift — is the dominant error source** in
Long-term ECG Reconstruction.

Section II reviews radar cardiac sensing, learning-based reconstruction, and evaluation
practice. Section III specifies the measurement procedure. Section IV describes the
systems under test and Section V the datasets. Section VI reports the benchmark, the
protocol faults and the mechanistic analysis. Sections VII–IX discuss what the field's
numbers measure, what a credible continuous claim would require, and the limitations of
this study.
