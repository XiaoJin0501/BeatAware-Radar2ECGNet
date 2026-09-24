# VI. Protocol Pitfalls and the Limits of Long-term ECG Reconstruction

*Draft v2, 2026-09-21. Terminology follows `docs/paper/TERMINOLOGY.md`; every number
traces to a registered experiment and an audit document. Target: IEEE TIM.*

Reported accuracy in radar-to-ECG reconstruction spans a wide range: single-beat
correlations of 0.96–0.99 appear alongside subject-independent figures near 0.6. We
show that most of that spread is produced by the measurement procedure rather than by
the models, and then quantify what remains once the procedure is corrected. Each
pitfall is demonstrated on data: a partition defect measured on our own earlier split,
a metric ceiling measured with a radar-free predictor, and a separation of anchor
timing from morphology measured against an oracle-timing reference.

## A. Pitfall 1 — Selection leakage through the validation partition

Subject-independent testing is routinely claimed; the *validation* partition is rarely
described. In the MMECG split this project used before its audit, the rule choosing the
validation subject returned only two distinct subjects across eleven folds, and both
later served as test subjects in other folds. No test subject entered training, yet
checkpoints in eleven folds were selected on nearly the same two individuals, so the
folds were not eleven independent generalisation events and the eleven-fold aggregate
promised more than it measured.

The repaired partition (Section III-C) assigns eleven distinct validation subjects in
rotation. All aggregates computed on the defective split were withdrawn; per-fold
values remain individually valid. We report the defect in full because it is invisible
in any published table: two papers with identical "leave-one-subject-out" statements
can differ by this much, and only a per-fold listing exposes it.

## B. Pitfall 2 — The template saturation trap

**ECG Shape Recovery** — a cycle cut at reference anchors, resampled and min–max
normalised on both sides — is the most widely reported figure in this literature,
whether as a dedicated shape output or as the shape task of a multitask radar model.
We evaluated it against the radar-free template floor of Section III-G: the training
fold's mean beat, predicted for every test window, with no radar input and no test
data.

| Setting | Radar-free template | Model figure | Paired comparison |
|---|---:|---:|---|
| MMECG, defective split | 0.879 $\pm$ 0.070 | 0.882 $\pm$ 0.061 (ours, as previously reported) | per-fold values unavailable |
| MMECG, defective split | 0.879 $\pm$ 0.070 | 0.890 $\pm$ 0.054 (ours, earlier variant) | $\Delta=+0.011$, 7/11 folds, $p=0.413$ |
| MMECG, defective split | 0.879 $\pm$ 0.070 | 0.728 $\pm$ 0.133 (radarODE-MTL, adapted) | $\Delta=-0.151$, 0/11 folds, $p=0.001$ |
| MMECG, strict LOSO (`loso_v2`) | 0.881 $\pm$ 0.071 | — | — |
| CR_RVS, five folds | 0.887 $\pm$ 0.013 | 0.881 $\pm$ 0.013 (ours) | model below the floor |

A constant beat, carrying no radar information whatsoever, reaches
$\rho\approx0.88$ on both a 77 GHz FMCW and a 24 GHz CW dataset — the level of every
single-beat figure we had produced, and above the adapted multitask baseline. The
explanation is topological rather than technical: a single-lead beat is stereotyped, so
after per-cycle amplitude normalisation and resampling, most of the variance a
correlation can see is the shared P-QRS-T shape, which is available from training
labels alone.

The consequence for the literature is specific and should be stated without
overreaching. We do not claim that any particular published system learned nothing; we
have not reproduced those systems under their own protocols. What the floor
establishes is that **this metric cannot distinguish radar-derived morphology from a
prior on ECG topology**, so a single-beat score in the 0.88–0.96 range is not by itself
evidence of the former — and the CW studies reporting 0.96–0.99 do so on resting-only
data under window-level splits, where both pitfalls act together.

Applying the same definition to the long-term estimates of the six benchmarked systems
gives 0.14–0.34 on MMECG and 0.43–0.53 on CR_RVS, against floors of 0.881 and 0.887: no
system exceeds the floor in any fold, ours included. Cutting the cycle at reference
anchors removes amplitude and baseline error but not timing error — a QRS displaced by
tens of milliseconds still lands off the cycle.

We therefore report every single-beat result **template-referenced**: the floor beside
the number, a per-fold paired test against it, and the radar-derived margin stated
explicitly.

## C. Pitfall 3 — Anchor timing and morphology conflated in one scalar

If beat shape is not the limiting factor, what is? Placing the same training template
at different sources of anchor timing, and scoring it exactly as a model, separates the
two. With reference anchors the template reaches $\rho=0.856$, $R^2=0.730$ on MMECG and
$\rho=0.877$, $R^2=0.764$ on CR_RVS, while the best benchmarked system reaches
$R^2=0.041\pm0.016$ on MMECG and 0.18–0.25 on CR_RVS. With radar-derived anchors the
same template falls to $R^2<0$ on both radars. Anchor placement, not beat shape, bounds
Long-term ECG Reconstruction.

### The beat-to-beat timing jitter bottleneck

Matching radar-derived anchors to reference anchors within
$\pm\SI{150}{\milli\second}$ decomposes the error into three parts:

1. **No global latency survives training.** The median signed offset is
   \SI{0}{\milli\second} for every anchor source on both radars. Supervised training
   against the ECG absorbs any population-level mechanical-to-electrical delay, so a
   constant electromechanical offset is not present in the residual error.
2. **Per-subject offsets exist, and correcting them buys little.** Their spread across
   subjects is 14–\SI{32}{\milli\second} (range $-80$ to $+\SI{55}{\milli\second}$),
   consistent with a subject-specific electromechanical activation time among other
   causes. Removing each subject's own median offset with an **oracle** recovers only
   **6–17 %** of the $R^2$ gap to the oracle-timing bound, on both radars. Static
   per-subject latency calibration is therefore not a route to closing this gap.
3. **The dominant terms are within-subject and beat-to-beat.** Anchor timing jitter is
   7–\SI{22}{\milli\second} on CW and 30–\SI{60}{\milli\second} on FMCW radar, and
   14–34 % of reference beats have no radar anchor within the matching window. A missed
   or spurious anchor displaces every subsequent cycle, which is why the CW systems —
   whose jitter is only a few samples — still fall far below the oracle-timing bound.

Two caveats keep this reading defensible. The residual is measured as timing error, not
attributed: beat-to-beat variation of the electromechanical activation time is one
plausible contributor, alongside anchor-detector error, signal-to-noise fluctuation and
respiration-modulated chest-wall motion, and the present experiments do not separate
them. And part of the residual — the missed anchors — is an estimation failure rather
than a physical limit, which leaves room for method development that a pure
"physical ceiling" reading would foreclose.

### Why every system shrinks amplitude

The benchmarked systems retain only 0.28–0.64 of the reference variance, ours the least
(0.331). This is not a defect of one objective: under uncertain anchor placement a
sharp waveform is penalised twice, once where it is placed and once where it is
missing, so shrinking amplitude is the loss-optimal response for any squared- or
spectral-error criterion. Amplitude shrinkage, $R^2\approx0$ and the oracle-timing
bound are one phenomenon seen from three directions — and they explain why enlarging a
network, which changes capacity but not anchor precision, has not moved this task.

### What the decoupled architecture buys

The one consistent advantage of the proposed decoupled architecture in this benchmark
is interval-level: its **Cycle Length Estimation** error is the lowest in six of the
seven physiological scenarios across both radars, including \SI{10.9}{\milli\second}
during sleep against 45.8 and \SI{60.4}{\milli\second} for the two comparators, at
1.44 M parameters against the 41.6 M of the multitask ODE baseline. In the pooled
three-seed aggregate it is second (\SI{49.1}{\milli\second} against
\SI{45.5}{\milli\second}), and it is the only system whose interval metrics require the
anchor-coverage gate, losing two of thirty-three entries to anchor-detection collapse.
Clinically, interval accuracy at this level supports rate and rhythm monitoring while
the waveform itself remains unsuitable for morphological diagnosis — a distinction this
benchmark makes measurable rather than rhetorical.

## D. Reading the reported literature through these pitfalls

On the CW dataset used here, the three pitfalls account for the published spread. One
system reports $\rho=0.959$ under a five-fold window-level split of 80 %-overlapping
segments and per-subject medians of **0.39–0.90** under leave-one-subject-out in the
same paper. A second reports a cosine similarity of 0.987 after splitting
75 %-overlapping windows 8:2 with no stated subject separation. A third is genuinely
subject-independent but uses one fixed five-subject resting test set and places its
0.964 beside the first system's window-level figure. A fourth, the only one describing
subject-separated folds, reports temporal correlations of 0.60–0.66 — the regime our
matched systems occupy (subject-level medians 0.53–0.61 on resting data).

These are reported with their protocols — scenario, split unit, window overlap,
reference normalisation, metric — in a block kept separate from matched results. No
leakage is alleged where a paper does not describe one.

## E. What the corrected procedure leaves standing

Under strict LOSO, template-referenced single-beat metrics and timing-resolved
diagnostics, the position is narrow and stated plainly. Long-term ECG Reconstruction on
FMCW data is bounded near $R^2\approx0$ for all five published families and for the
architecture proposed here; on CW data it reaches $R^2=0.18$–$0.25$. No system in this
study recovers beat morphology beyond a training-label template. The gap to an
oracle-timed template is dominated by within-subject, beat-to-beat anchor error, which
no per-subject calibration removes. Progress is therefore an anchor-timing problem, and
a benchmark that cannot separate anchor timing from morphology will keep rewarding
systems that emit a smoothed average beat.
