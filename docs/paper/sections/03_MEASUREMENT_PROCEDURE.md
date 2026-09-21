# III. Measurement Procedure

*Draft, 2026-09-21. Every number traces to `docs/PAPER_CLAIM_LEDGER.md` and a
registered experiment. Target: IEEE TIM.*

Radar-to-ECG reconstruction is treated here as a measurement problem: a system
estimates a physiological waveform from an indirect mechanical observation, and the
value of the estimate is only as trustworthy as the procedure used to quantify it.
This section specifies that procedure — the measurand, the partition of subjects, the
scaling of the reference signal, the selection rule, the one-time access to the
held-out data, the metric definitions with their aggregation units, and the
radar-free reference levels against which every reported number is read. The
procedure is applied identically to all six systems under test and to both radar
modalities.

## A. Measurand

The measurand is the continuous single-lead ECG over the same time span as the radar
observation: $\mathbf{y}\in\mathbb{R}^{1600}$ sampled at $f_s=\SI{200}{\hertz}$, i.e.
an \SI{8}{\second} window reconstructed sample-by-sample without beat segmentation,
without alignment to the reference at inference, and without any per-window
renormalisation of the reference. The stored reference is z-scored once per recording,
so all amplitude errors below are expressed in standard deviations of the subject's own
ECG rather than in millivolts; this is stated wherever such a unit appears.

Two output formulations are explicitly *not* the measurand. Beat-normalised
morphology, in which a cycle is cut with reference R peaks, resampled to a fixed
length and min–max normalised, is an auxiliary diagnostic whose reference level is
established in Section VI-B. Lag-compensated waveforms, in which a shift is selected
using the reference ECG, are upper-bound diagnostics and never deployable estimates.

## B. Subject partition and the repaired split

Both datasets are partitioned by subject. MMECG (77 GHz FMCW, 11 subjects) uses
eleven-fold leave-one-subject-out; CR_RVS (24 GHz CW, 30 subjects) uses five
subject-independent folds with four to seven held-out subjects each.

The partition itself is part of the instrument and is audited. In the MMECG split
originally used in this project, the validation subject was chosen by a rule that
selected the subject whose record count was closest to the median; across eleven folds
that rule returned only **two distinct validation subjects**, each of which later
appeared as a test subject in other folds. Checkpoints in different folds were
therefore selected on nearly the same data, and the eleven-fold aggregate no longer
estimated eleven independent generalisation events. The repaired partition,
`loso_v2`, assigns eleven distinct validation subjects in a rotation such that every
subject serves as validation exactly once and as test exactly once. Each fold's
train, validation and test files are hashed (SHA-256) and the hashes are recorded in
every run's configuration, so subject disjointness and split identity are verifiable
after the fact rather than asserted. For CR_RVS the five validation subjects are
likewise distinct; that each holds only three records (~195 windows) is a stated
limitation of that dataset, not a choice of this procedure.

## C. Reference scaling and information barriers

The reference ECG is mapped into a training-fold robust scale — median and
inter-quartile range computed over the *entire training fold only* — and the
prediction is mapped back through the same fixed transform before scoring. No
statistic of the validation or test subjects enters normalisation, range-bin
selection, thresholds, lags, affine corrections, checkpoints or model choice.
Per-window or per-cycle min–max normalisation of the reference is prohibited: it
discards cycle amplitude and baseline, and makes correlation structurally blind to
amplitude error. The CR_RVS release used in this work was rebuilt for this reason,
with the reference z-scored per record while the windows, radar preprocessing and
time–frequency maps were left byte-identical to the earlier build.

Radar inputs carry their own barrier. Every representation is computed from radar
alone: a synchrosqueezed transform over 50 range bins for MMECG, a log-STFT of the CW
phase for CR_RVS, or the native-rate range-bin signal for the time-domain systems.
Each representation spans the full \SI{8}{\second} of the measurand; an earlier
configuration in which the time–frequency map covered only the central
\SI{4}{\second}, leaving half of each output window without input support, was
identified and discarded before any result in this paper was produced.

## D. Selection rule and one-time access to held-out data

Training uses a fixed budget for every system: at most 100 epochs, early stopping with
patience 15, AdamW at learning rate $10^{-4}$ with weight decay $10^{-4}$, and batch
size 16. The per-fold seed is $s_{\text{base}}+k$ for fold $k$, so systems are paired
within a fold, and the whole benchmark is repeated for
$s_{\text{base}}\in\{42,123,2024\}$.

Checkpoints are selected on validation data only, by a composite score fixed in
writing before training:

$$S = 0.40\,\rho + 0.25\left(1-\min(\text{NRMSE},1)\right) + 0.20\max(R^2,0) + 0.15\,F_1^{100\,\text{ms}} .$$

Two terms must be described accurately rather than paraphrased: because NRMSE exceeds
unity throughout the observed operating range and $R^2$ is frequently negative, the
second term is identically zero and the third is inactive on many epochs, so selection
is in practice driven by $\rho$ and $F_1^{100\,\text{ms}}$. The rule was frozen before
the runs and is reported as it behaved, not as it reads.

Access to held-out subjects is enforced by an irreversible state machine —
`TRAINING` → `CHECKPOINT_FROZEN` → `TEST_EVALUATING` → `COMPLETE`. The test split is
not constructed in memory until the checkpoint is frozen, a `COMPLETE` fold is
terminal and is never recomputed, and per-epoch state is written atomically so that an
interrupted fold resumes rather than restarting with a fresh selection. The benchmark
reported here comprises 198 such sealed folds on MMECG (six systems $\times$ three
seeds $\times$ eleven folds) and 15 on CR_RVS.

## E. Metrics, aggregation units and dispersion

All systems are scored by one implementation, imported rather than duplicated, so that
no difference in the comparison can originate in evaluation code. Waveform metrics
($\rho$, RMSE, MAE, NRMSE, $R^2$) are computed per window and averaged within the
held-out subject; $F_1$ at 50/100/\SI{150}{\milli\second} pools true and false counts
across windows; QRS ($R\pm\SI{50}{\milli\second}$) and ST-T
($R+80$ to $R+\SI{400}{\milli\second}$) errors pool across beats. Each table states
which unit applies. NRMSE normalises by the ground-truth inter-quartile range of the
same window.

Three reporting rules follow from the metric audit. First, rate metrics (RR and HR
error) are aggregated only over entries whose beat coverage is at least 0.80, and
excluded entries are listed individually as failure cases rather than averaged in.
Second, the relative above-\SI{10}{\hertz} energy ratio is a ratio of ratios and is
reported as a median with its range, never as a bare mean. Third, two distinct timing
read-outs exist — peaks detected on the reconstructed waveform and peaks from an event
head, where a system has one — and they are never merged or interchanged; systems
without an event head leave that column empty rather than scoring zero.

Dispersion is reported at two levels: the population standard deviation across folds
within a seed, and the standard deviation across the three seeds of the per-seed
fold-macro value, which is the headline uncertainty. Paired comparisons use the
Wilcoxon signed-rank test on seed-averaged fold values; the number of comparisons is
stated and no multiplicity correction is implied.

## F. Radar-free reference levels

A measurement procedure needs a reference against which an estimate is judged
informative. Two are defined here and computed with the same evaluation code as the
systems.

The **template floor** is the training fold's mean R-to-R cycle, predicted for every
test window. It uses training labels only — no radar, no test data — and therefore
marks the score obtainable with zero radar-derived information about the individual.
Any beat-level result at or below this floor carries no evidence of morphology
recovered from radar, and results above it are reported with the margin, for instance
as $(\rho_{\text{model}}-\rho_{\text{template}})/(1-\rho_{\text{template}})$.

The **oracle-timing bound** places the same template at the reference R peaks. It
isolates how much of the measurand is explained by beat placement alone, with beat
shape fixed to a population average and no subject-specific morphology. Because it
consumes reference timing it is an upper bound and never an achievable score; its role
is to separate timing error from morphology error in Section VI-C.

## G. Reproducibility

Every run is registered before it starts, with a specification stating the research
question, the fold list, the budget, the metric set and the aggregation contract; the
registry records the commit that produced it and the branch it ran on. Split hashes,
per-fold configurations, validation histories, frozen checkpoints, per-window test
predictions and fold-macro summaries are retained for all 213 sealed folds, and every
analysis in Sections VI-B and VI-C is recomputed from those frozen predictions rather
than from any number transcribed by hand.
