# II. Related Work

*Draft for IEEE TIM. Citation keys refer to `docs/paper/references.bib`. Protocol
statements about individual papers are quoted from those papers and are audited in
`docs/audits/Q1_CRRVS_LITERATURE_PROTOCOL_AUDIT.md` and
`docs/TABLE_D_REPORTED_LITERATURE_AUDIT.md`; no leakage is alleged where authors
disclosed their protocol.*

---

## A. Radar-based contactless cardiac sensing

Radar cardiac sensing rests on a mechanical observable. A millimetre-wave transceiver
illuminates the thorax and recovers, from the phase of the returned signal, a
sub-millimetre displacement waveform of the chest wall — the *ballistic* consequence of
ventricular contraction and blood ejection. Surveys of the field
\cite{zhang2023overview, liComprehensiveSurveyContactless2025,
kouhalvandi2025advances, hassanpourContactlessVitalSign2025a} trace two decades of work
in which this displacement is used for respiration rate, heart rate and, more recently,
beat-to-beat interval estimation \cite{wuNonintrusiveHumanVital2024,
singhMultiResidentNonContactVital2021, noceraMachineLearningRADARbased2024}.

Two properties of the observable matter for this paper. First, the cardiac component is
one to two orders of magnitude smaller than the respiratory component, so every system
must separate a weak, broadband mechanical signature from a strong, slow one. Second,
and more consequentially, the mechanical event is *not* the electrical event. Ventricular
depolarisation precedes measurable wall motion by an interval that varies with
contractility, preload and posture, and that varies between consecutive beats within a
single recording. Cardiac sensing that stops at rate or interval estimation is largely
insulated from this fact. Waveform reconstruction is not, and Section VI-D quantifies
the consequence.

## B. Learning-based radar-to-ECG reconstruction

Attempts to estimate the ECG waveform itself fall into five architectural families,
each represented in our benchmark.

**Temporal-convolutional regression.** The reference system for the 77 GHz dataset used
here \cite{chenmmwave} maps the range-profile tensor through per-bin temporal
convolutions and a transformer fusion stage to a continuous ECG estimate, trained with
a time-domain objective. Related encoder–decoder and sequence-to-sequence formulations
\cite{kimEndtoendElectrocardiogramSignal2024, S2SA, cross-modalautoencoder} share this
shape.

**Structured physiological priors.** radarODE \cite{radarode} and its multitask
extension \cite{radarodemt} embed an ordinary-differential-equation model of the cardiac
cycle, decoding a morphology template that is then placed in time, and train several
heads — morphology, interval, anchor — under a gradient-balancing scheme. This family is
the closest prior art to our own factorization and our strongest comparator.

**Wavelet and multiresolution decompositions.** WaveGRU-Net \cite{xu2025wavegrunet},
LIFWavNet \cite{lifwavnet}, VMD-CBNet \cite{xu2025vmdcbnet} and MultiResLinkNet
\cite{chowdhury2024multireslinknet} decompose the radar signal into sub-bands before
sequence modelling, on the argument that QRS content occupies a distinct scale.

**Adversarial and generative models.** ECG-like GAN formulations
\cite{ecglikegan, ReconstructingECG-like}, diffusion decoders \cite{ecgdiff} and
attention-DCGAN variants optimise a distributional objective, on the argument that a
regression loss cannot produce sharp QRS complexes.

**Attention and cross-modal fusion.** Conformer and transformer encoders
\cite{conformer, Multi-ScaleBottleneck, liu2026mmjepaecg}, AirECG \cite{airecg} and
CardiacWave \cite{xuCardiacWaveMmWavebasedScheme2021} apply self-attention to the
time–frequency representation, some with disentanglement objectives \cite{disentangled}.

Several of these works report the intention to separate timing from shape — radarODE's
template-plus-placement decoding is explicitly such a design, and \cite{disentangled}
names disentanglement as its goal. What has not been done, and what this paper supplies,
is a *measurement* of how much of the residual error each factor accounts for under a
protocol in which neither factor has been given away by the evaluation.

## C. Evaluation practice, and why reported numbers span an order of magnitude

Reported correlations on radar-to-ECG reconstruction range from below 0.3 to 0.99. That
spread is largely explained by three protocol choices, which are frequently left
implicit.

**Split unit.** Where the cross-validation unit is the *window* rather than the
*subject*, and windows overlap, adjacent training and test segments share samples. The
strongest available evidence for the size of this effect comes from within a single
paper: RSSRnet \cite{rssrnet} reports $\rho = 0.959$ under a five-fold split of
80 %-overlapping windows and per-subject medians of **0.39–0.90** under
leave-one-subject-out *in the same work*. Other high figures on that dataset come from
resting-only data with window-level splits \cite{li2025cuheart, li2025non} or from a
single fixed five-subject test set \cite{11398845}.

**Target normalisation and beat normalisation.** Where the target is normalised per
window, or where evaluation is performed on beats cut at reference R peaks, resampled
and amplitude-normalised on both sides, the metric no longer measures what a deployed
system would produce. Section VI-B shows this construction is saturated: a radar-free
training-fold template reaches 0.879–0.887 on both datasets studied here.

**Selection discipline.** Cross-fold reuse of validation subjects, and checkpoint
selection informed by test metrics, both inflate subject-independent claims. We document
one widely used split in which two validation subjects are shared across all eleven
folds.

A smaller literature has begun to push back. Fiducial-level analyses \cite{caofiducial,
de2004automatic} report timing errors in milliseconds rather than correlation alone, and
\cite{caoNoncontactDetectionElectrocardiogram2024a} questions what non-contact ECG
figures represent. Our contribution to this strand is to make the reference levels
explicit — a radar-free template floor for beat-normalised morphology, and an
oracle-timing bound for continuous reconstruction — so that a reported number can be
read against what is achievable without the radar, and against what would be achievable
with perfect timing.

## D. Positioning

Relative to the closest prior art: radarODE-MTL \cite{radarodemt} also separates
morphology from timing and also trains multiple heads jointly, and in our benchmark it
is the strongest comparator on interval accuracy. It differs from REM-Net in three ways.
It imposes a parametric ODE prior on the cardiac cycle, where we impose only a
two-branch read-out over a shared representation; it uses 41.6 M parameters against our
1.44 M; and its factorization is presented as a means to better reconstruction, where
ours is presented — and evaluated — as an instrument for *measuring* the timing term.
Relative to the benchmark literature as a whole, this paper is the first to report six
systems from five families under one leak-free protocol on two radar modalities, with
radar-free reference levels attached to every morphology number.
