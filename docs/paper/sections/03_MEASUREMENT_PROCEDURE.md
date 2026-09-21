# III. Rigorous Measurement Procedure and a Leak-Free Protocol

*Draft v2, 2026-09-21. Terminology follows `docs/paper/TERMINOLOGY.md`; every number
traces to a registered experiment. Target: IEEE TIM.*

Radar-to-ECG reconstruction is a measurement problem: a system estimates a
physiological waveform from an indirect mechanical observation, and the estimate is
only as trustworthy as the procedure used to quantify it. This section defines that
procedure. It states the measurand, shows why the window-level partitions common in
this literature cannot support a subject-independent claim, specifies the strict
leave-one-subject-out (LOSO) protocol and the information barriers that make it
leak-free, fixes the selection rule and the one-time access to held-out data, and
defines the metrics with their aggregation units and their radar-free reference
levels. The same procedure is applied to all six systems under test and to both radar
modalities.

## A. Measurand

The measurand of **Long-term ECG Reconstruction** is the continuous single-lead ECG
over the same span as the radar observation: $\mathbf{y}\in\mathbb{R}^{1600}$ at
$f_s=\SI{200}{\hertz}$, i.e. an \SI{8}{\second} window estimated sample by sample,
without beat segmentation, without reference-derived alignment at inference, and
without per-window renormalisation of the reference. Amplitude errors are expressed in
standard deviations of each subject's own ECG, and every table says so.

Two auxiliary read-outs are defined but are never presented as the measurand.
**ECG Shape Recovery** measures single-beat morphology with the cycle cut at reference
anchors, resampled to 200 samples and min–max normalised; it presupposes knowledge of
the cycle and is therefore a diagnostic. **ECG Anchor Estimation** measures where
cardiac anchors are placed in time, with two strictly separate read-outs: anchors
detected on the reconstructed waveform, available for every system, and estimates from
a dedicated anchor head, which only the proposed architecture has.

## B. Why window-level partitions cannot support a subject-independent claim

The dominant partitioning practice in this field splits *windows*, not subjects: a
recording is cut into overlapping segments and a fixed fraction is held out at random.
With the overlaps in common use — 75 % and 80 % in two of the CW studies we survey —
adjacent windows share most of their samples, so a held-out window is a near-duplicate
of a training window from the same subject, the same posture and the same minutes of
recording. What such a protocol measures is interpolation within a recording, not
generalisation to a new person.

The effect is not hypothetical and does not require an external counter-example: one
published system reports a correlation of 0.959 under a five-fold window-level split
and, **in the same paper**, per-subject medians of 0.39–0.90 when the same model is
re-evaluated leave-one-subject-out. Any number produced under a window-level split is
therefore incomparable with a subject-independent one, and the two must never share a
ranking column.

## C. Strict LOSO and the repaired validation partition

Both datasets are partitioned by subject and by nothing else. MMECG (77 GHz FMCW, 11
subjects) uses eleven-fold strict LOSO; CR_RVS (24 GHz CW, 30 subjects) uses five
subject-independent folds holding out four to seven subjects each. Train, validation
and test subjects are disjoint in every fold, and the partition is audited rather than
asserted: each fold's three files are hashed (SHA-256) and the hashes are written into
every run's configuration, so subject disjointness and split identity remain verifiable
after the fact.

Subject disjointness alone is not sufficient, because the *validation* partition also
carries information. In the MMECG split this project used before its audit, the
validation subject was chosen by a rule that selected the subject whose record count
lay closest to the median; across eleven folds that rule returned only two distinct
validation subjects, each of which later served as a test subject in another fold.
No test subject ever entered training, yet checkpoints across eleven folds were
selected on nearly the same two individuals, so the folds were not eleven independent
generalisation events. The repaired partition assigns eleven distinct validation
subjects in a rotation in which every subject validates exactly once and tests exactly
once. Every aggregate computed on the defective partition was withdrawn.

## D. Information barriers: normalisation, representation, selection

**Reference scaling.** The reference ECG is z-scored once per recording and then mapped
into a training-fold robust scale — median and inter-quartile range computed over the
*entire training fold only* — with the estimate mapped back through the same fixed
transform before scoring. Per-window or per-cycle min–max normalisation of the
reference is prohibited. It is convenient and common, and it is a leak of the measurand
into the metric: it discards each window's true amplitude and baseline, so a
correlation computed against it is structurally blind to amplitude error and scores a
shape-matched but amplitude-wrong estimate as excellent. The CW release used here was
rebuilt for exactly this reason, with the reference z-scored per record while windows,
radar preprocessing and time–frequency maps stayed byte-identical to the earlier build.

**Radar representations.** Every input is computed from radar alone — a synchrosqueezed
transform over 50 range bins, a log-STFT of the CW phase, or the native-rate range-bin
signal — and each spans the full \SI{8}{\second} of the measurand. An earlier
configuration whose time–frequency map covered only the central \SI{4}{\second}, leaving
half of every output window without input support, was found and discarded before any
result reported here.

**Selection.** Checkpoints are chosen on validation data only, by a composite fixed in
writing before training,

$$S = 0.40\,\rho + 0.25\left(1-\min(\mathrm{NRMSE},1)\right) + 0.20\max(R^2,0) + 0.15\,F_1^{100\,\mathrm{ms}},$$

and reported as it behaved rather than as it reads: since NRMSE exceeds unity
throughout the observed range and $R^2$ is often negative, the second term is
identically zero and the third inactive on many epochs, leaving selection driven by
$\rho$ and waveform-derived anchor $F_1$. No test statistic — scaler, lag, threshold,
range-bin choice, checkpoint or model — is ever computed on held-out subjects.

## E. One-time access to held-out subjects

Training uses one budget for every system: at most 100 epochs, early stopping with
patience 15, AdamW at $10^{-4}$ with weight decay $10^{-4}$, batch size 16, and a
per-fold seed $s_\text{base}+k$ so that systems are paired within a fold. The benchmark
is repeated for $s_\text{base}\in\{42,123,2024\}$.

Access to held-out subjects is enforced mechanically by an irreversible state machine,
`TRAINING` $\to$ `CHECKPOINT_FROZEN` $\to$ `TEST_EVALUATING` $\to$ `COMPLETE`. The test
split is not constructed in memory until the checkpoint is frozen; a `COMPLETE` fold is
terminal and is never recomputed; per-epoch state is written atomically so that an
interrupted fold resumes instead of restarting with a fresh selection. The results
reported here comprise 198 sealed folds on MMECG (six systems $\times$ three seeds
$\times$ eleven folds) and 15 on CR_RVS.

## F. Metrics, aggregation units, dispersion

All systems are scored by one implementation, imported rather than duplicated, so no
part of the comparison can originate in evaluation code. Waveform metrics ($\rho$,
RMSE, MAE, NRMSE, $R^2$) are computed per window and averaged within the held-out
subject; anchor $F_1$ at 50/100/\SI{150}{\milli\second} pools counts across windows;
QRS ($R\pm\SI{50}{\milli\second}$) and ST-T ($R+80$ to $R+\SI{400}{\milli\second}$)
errors pool across beats. Each table states which unit applies.

Three reporting rules follow from the metric audit. **Cycle Length Estimation** and
heart-rate error are aggregated only over entries whose anchor coverage is at least
0.80, with excluded entries listed individually as failure cases rather than averaged
in. The relative above-\SI{10}{\hertz} energy ratio is a ratio of ratios and is
reported as a median with its range. Waveform-derived anchors and anchor-head estimates
keep separate columns and are never interchanged.

Dispersion is reported twice: the population standard deviation across folds within a
seed, and the standard deviation across the three seeds of the per-seed fold-macro
value, which is the headline uncertainty. Paired comparisons use the Wilcoxon
signed-rank test on seed-averaged fold values, with the number of comparisons stated
and no multiplicity correction implied.

## G. Radar-free reference levels

A measurement procedure needs references against which an estimate is judged
informative; two are defined here and computed with the same evaluation code as the
systems.

The **radar-free template floor** predicts the training fold's mean beat for every test
window. It uses training labels only — no radar, no test data — and therefore marks the
score reachable with zero radar-derived information about the individual. Every ECG
Shape Recovery result is reported beside it, with a per-fold paired test and the
radar-derived margin, for instance
$(\rho_\text{model}-\rho_\text{template})/(1-\rho_\text{template})$.

The **oracle-timing bound** places that same template at the reference anchors. It
isolates how much of the measurand is explained by anchor placement alone, with beat
shape fixed to a population average. It consumes reference timing, is therefore never
an achievable score, and exists to separate anchor-timing error from morphology error
in Section VI.

## H. Reproducibility

Every run is registered before it starts with a specification naming the question, the
folds, the budget, the metric set and the aggregation contract; the registry records
the commit and branch that produced it. Split hashes, per-fold configurations,
validation histories, frozen checkpoints, per-window test estimates and fold-macro
summaries are retained for all 213 sealed folds, and every analysis in Section VI is
recomputed from those frozen estimates rather than from any transcribed number.
