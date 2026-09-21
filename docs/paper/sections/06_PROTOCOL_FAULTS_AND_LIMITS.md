# VI. Protocol Faults and the Reconstruction Ceiling

*Draft, 2026-09-21. Every number traces to a registered experiment and an audit
document; see `docs/PAPER_CLAIM_LEDGER.md`. Target: IEEE TIM.*

Reported accuracy in radar-to-ECG reconstruction spans a wide range: beat-level
correlations of 0.96–0.99 appear alongside subject-independent figures near 0.6. This
section shows that the spread is produced mainly by three properties of the
measurement procedure rather than by the models, and then quantifies what remains once
those properties are controlled. Each fault is demonstrated on data rather than
argued: a partition defect measured on our own earlier split, a metric floor measured
with a radar-free predictor, and a timing/morphology separation measured with an
oracle-timing reference.

## A. Fault 1 — Selection leakage through the validation partition

Subject-independent testing is routinely reported, but the *validation* partition is
rarely described. In the MMECG split this project used before the audit, the rule that
picked the validation subject returned only two distinct subjects across eleven folds,
and both later served as test subjects in other folds. The consequence is not test
contamination in the ordinary sense — no test subject appeared in training — but a
shared selection signal: checkpoints in eleven folds were chosen on nearly the same
two individuals, so the folds were not eleven independent generalisation events, and
the eleven-fold aggregate overstated what the mean $\pm$ standard deviation implies.

The repaired partition assigns eleven distinct validation subjects in rotation
(Section III-B). All aggregates computed on the defective partition were withdrawn and
recomputed; per-fold values remain individually valid. We record the defect in full
because it is invisible in any published table: two papers with identical
"leave-one-subject-out" statements can differ by this much, and only a per-fold hash
listing exposes it.

## B. Fault 2 — A beat-normalised metric saturated by a radar-free template

Beat-normalised correlation — a cycle cut at reference R peaks, resampled to a fixed
length and min–max normalised on both sides — is the most widely reported figure in
this literature, either as Shape-200 or as the ECG_shape task of multitask radar
models. We evaluated the **template floor** of Section III-F against it: the training
fold's mean beat, predicted for every test window, with no radar input and no test
data.

| Setting | Radar-free template | Previously reported model | Paired comparison |
|---|---:|---:|---|
| MMECG, defective split | 0.879 $\pm$ 0.070 | 0.882 $\pm$ 0.061 | per-fold values unavailable |
| MMECG, defective split | 0.879 $\pm$ 0.070 | 0.890 $\pm$ 0.054 (this architecture, earlier variant) | $\Delta=+0.011$, 7/11 folds, $p=0.413$ |
| MMECG, defective split | 0.879 $\pm$ 0.070 | 0.728 $\pm$ 0.133 (radarODE-MTL, adapted) | $\Delta=-0.151$, 0/11 folds, $p=0.001$ |
| MMECG, `loso_v2` | 0.881 $\pm$ 0.071 | — | — |
| CR_RVS, five folds | 0.887 $\pm$ 0.013 | 0.881 $\pm$ 0.013 | model below the floor |

The template reaches the level of every beat-normalised figure we or the adapted
baselines had produced. A single-lead ECG beat is stereotyped; after per-cycle
amplitude normalisation and resampling, most of the variance a correlation can see is
the shared P-QRS-T shape, which is available from training labels alone. The metric
therefore measures template likeness, and a model that emits a canonical beat scores
well whether or not it read anything from the radar.

Applying the same definition to the continuous outputs of the six benchmarked systems
gives 0.14–0.34 on MMECG and 0.43–0.53 on CR_RVS, against floors of 0.881 and 0.887 —
no system exceeds the floor in any fold. Cutting the cycle with reference R peaks
removes amplitude and baseline error but not timing error: a QRS displaced by tens of
milliseconds still lands off the cycle.

We therefore report beat-level results **template-referenced**: the floor beside every
such number, a per-fold paired test against it, and the radar-derived margin stated
explicitly. Under this rule, none of the beat-normalised results in this study — ours
included — supports a claim of morphology recovered from radar.

## C. Fault 3 — Timing and morphology conflated in a single scalar

If beat shape is not the limiting factor, what is? We place the same training template
at different sources of beat timing and score the result exactly as a model
(Table VI-3). With reference timing, a template carrying no subject-specific
morphology reaches $\rho=0.856$, $R^2=0.730$ on MMECG and $\rho=0.877$, $R^2=0.764$ on
CR_RVS. Every benchmarked system lies far below that bound: $R^2$ is $0.041\pm0.016$
at best on MMECG and 0.18–0.25 on CR_RVS. With radar-derived timing, the same template
falls to $R^2<0$ on both radars.

Beat placement, not beat shape, therefore bounds continuous reconstruction — and the
bound is modality-dependent, which the aggregate scores alone would not reveal.

### Decomposing the timing error

Matching radar-derived peaks to reference R peaks within $\pm\SI{150}{\milli\second}$
separates three components (Table VI-4):

1. **No global latency survives training.** The median signed offset is
   \SI{0}{\milli\second} for every peak source on both radars. Supervised training
   against the ECG absorbs any population-level electromechanical delay; it is not
   visible in the residual error.
2. **Per-subject offsets exist but are a minor term.** Their standard deviation across
   subjects is 14–\SI{32}{\milli\second} (range about $-80$ to $+\SI{55}{\milli\second}$),
   which is compatible with subject-specific electromechanical delay among other
   causes. Removing each subject's own median offset with an oracle recovers only
   **6–17 %** of the $R^2$ gap to the timing bound, on both radars. A calibratable
   constant latency is therefore not the bottleneck, and these analyses do not
   identify an electromechanical activation time.
3. **The dominant terms are within-subject.** Beat-to-beat jitter is
   7–\SI{22}{\milli\second} on CW radar and 30–\SI{60}{\milli\second} on FMCW, and
   14–34 % of reference beats have no radar peak within the matching window. A missed
   or spurious beat displaces every subsequent cycle, which is why the CW systems,
   whose jitter is small, still fall far below the timing bound.

### Why every system shrinks amplitude

The benchmarked systems retain only 0.28–0.64 of the reference variance, the proposed
architecture least of all (0.331). This is not a defect of one loss function: under
uncertain beat placement, a sharp waveform is penalised twice — once where it is
placed and once where it is missing — so shrinking amplitude is the loss-optimal
response for any squared- or spectral-error objective. Amplitude shrinkage, $R^2\approx
0$ and the timing bound are one phenomenon observed from three directions.

## D. Consequences for reported results in this field

The three faults are sufficient to explain the spread in the literature on the CW
dataset used here (Table VI-5). One published system reports $\rho=0.959$ under a
five-fold split of 80 %-overlapping windows that is not stated to be subject-wise, and
per-subject medians of **0.39–0.90** under leave-one-subject-out **in the same paper**;
that single contrast brackets the effect of the partition. A second reports a cosine
similarity of 0.987 after splitting 75 %-overlapping windows 8:2 without a stated
subject separation. A third is genuinely subject-independent but uses one fixed
five-subject resting test set and compares itself with the first system's data-level
figure. A fourth, the only one describing subject-separated folds, reports a temporal
correlation of 0.60–0.66 — the regime our matched systems occupy (subject-level medians
0.53–0.61 on resting data).

We do not restate these numbers as a ranking. They are reported with their protocol —
scenario, split unit, window overlap, reference normalisation, metric — in a block
kept separate from matched results, and no leakage is alleged where a paper does not
describe one.

## E. What the corrected procedure leaves

Under the repaired partition, template-referenced beat metrics and timing-resolved
diagnostics, the honest position is narrow and stated plainly. Continuous
subject-independent radar-to-ECG reconstruction on FMCW data is bounded near
$R^2\approx0$ for all five published families and for the architecture proposed here;
on CW data it reaches $R^2=0.18$–$0.25$. No system in this study recovers beat
morphology beyond a training-label template. The gap to a reference-timed template is
dominated by within-subject beat placement error, which no per-subject calibration
removes. Progress in this task is therefore a beat-timing problem, and a benchmark
that cannot separate timing from morphology will keep rewarding systems that emit a
smoothed average beat.
