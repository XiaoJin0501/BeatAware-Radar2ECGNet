# Adversarial Reviewer Critique for the Q1 Paper

Status: hard self-review. Purpose: decide whether the current evidence can support
a Q1 submission and what must be strengthened before writing the full manuscript.
Grounded in `docs/PAPER_CLAIM_LEDGER.md`, `docs/T1_MMECG_MORPHOLOGY_LOCKED.md`,
`docs/T3_CRRVS_EXTERNAL_AUDIT.md`, `docs/ASSEMBLY_DESIGN.md`,
`docs/Q1_MANUSCRIPT_BLUEPRINT.md`, and `docs/RESULTS_SKELETON.md`.

## Executive Verdict

The paper is viable only if framed as **supervised beat-morphology reconstruction
plus a continuous-gap discovery**, not as a complete continuous radar-to-ECG
solution. The current evidence is strong enough for a Q1 sensor or biomedical
signal processing venue if the claims are disciplined, the morphology evidence is
made clinically richer, and the comparison tiers remain explicit. It is risky for
top biomedical engineering venues such as JBHI/TBME in the current form because
the primary quantitative win is on Shape-200, an easier beat-normalized target,
while continuous ECG and clinical interval timing remain weak.

Recommended decision: **go forward**. The first substance booster has now been
completed (`docs/MORPHOLOGY_FIDUCIAL_BOOSTER.md`) without retraining. It is not a
clinical QT/QRS delineation analysis, but it does show that REM-Net improves
QRS-edge and ST/T phase-region morphology rather than only global PCC. A stricter
clinical/fiducial analysis can still be added later, but the paper no longer rests
only on a single smooth-curve metric.

## Strengths

1. **Two public radar modalities.** The study uses MMECG FMCW radar and CR_RVS CW
   radar. This is stronger than a single-dataset extension of ICONIP.

2. **Subject-independent protocols.** MMECG uses 11-fold strict LOSO, and CR_RVS
   uses verified 5-fold subject-independent splits. The paper can credibly argue
   against subject leakage and sample-level inflation.

3. **Multi-seed stability.** REM-Net achieves 0.885 +/- 0.002 on MMECG and
   0.886 +/- 0.001 on CR_RVS across three seeds. This is a real strength because
   the model is not relying on a lucky initialization.

4. **Fairness-aware comparison.** The project now separates strict-LOSO local
   results, calibrated prior results, adapted baselines, and reported-only
   literature. This will prevent a common reviewer attack: "the table mixes
   incompatible protocols."

5. **Honest negative discovery.** The route-C experiments show that morphology is
   not the only barrier; robust sub-50 ms anchor localization is the missing piece
   for continuous ECG. This gives the paper scientific content beyond a score
   table, provided it is not oversold.

## Major Reject Risks and Best Honest Responses

### Risk 1: Shape-200 is an easy proxy, not the real clinical task.

**Reviewer attack.** The paper claims radar-to-ECG progress but evaluates the main
model on beat-normalized Shape-200. This removes much of the hard timing problem.
The continuous results are weak, so the central claim may be too narrow for Q1.

**Honest response.** The paper explicitly distinguishes beat morphology from
continuous ECG. We do not claim continuous ECG is solved. Instead, we show that
morphology can be reconstructed robustly across FMCW and CW radar, then prove
through oracle assembly that the continuous gap is dominated by beat-anchor
localization precision. This is a principled decomposition of the radar-to-ECG
problem rather than a shortcut.

**Added evidence.** `docs/MORPHOLOGY_FIDUCIAL_BOOSTER.md` reports fixed
phase-region morphology metrics from existing Shape-200 outputs. On MMECG,
REM-Net reaches QRS-edge PCC 0.962 and ST/T PCC 0.787; radarODE-MTL is similarly
strong at QRS-edge (0.961) but weak in ST/T (0.157) and diastolic morphology
(0.034). This does not solve the full clinical critique, but it makes Shape-200
less "just a smooth curve metric."

### Risk 2: Model novelty is not high enough.

**Reviewer attack.** REM-Net sounds like another encoder plus auxiliary event head
plus decoder. radarODE-MTL, VMD-CBNeT, AirECG, and other works already use
physiological priors or signal decomposition. What is the algorithmic novelty?

**Honest response.** The novelty should not be sold as "we added modules." The
paper's contribution is a supervised, output-formulation-aware framework for
separating morphology learning from continuous timing. The final model is simple
but evidence-supported: representation encoder, event supervision, morphology
multi-task decoding. The discarded alignment branch is a strength of the research
process, not a module to advertise.

**Needed evidence.** The paper should include a clean ablation table showing that
the morphology decoder/event structure is the component that matters, while
alignment does not. If available, add parameter/FLOP/latency comparison to show
that REM-Net is not only accurate but also practical.

### Risk 3: The VMD-CBNeT comparison is not same-protocol.

**Reviewer attack.** The paper says CR_RVS PCC 0.886 exceeds VMD-CBNeT 0.856, but
the protocols and PCC spaces differ. This is not a valid head-to-head comparison.

**Honest response.** Correct. The paper must not present it as a same-protocol
comparison. VMD-CBNeT belongs in a reported-only block with protocol caveats. The
valid local CR_RVS comparison is REM-Net versus CW-STFT+TCN under the same 5-fold
subject-independent protocol. The VMD-CBNeT number is contextual, not decisive.

**Needed evidence.** If the paper wants a stronger external head-to-head, it must
either reproduce VMD-CBNeT under our protocol or avoid the "outperform" wording.
For the current submission path, caveated contextual comparison is acceptable but
not the strongest contribution.

### Risk 4: The qualitative overlays are not visually perfect.

**Reviewer attack.** Published ECG reconstruction papers often show nearly
identical overlaid ECG traces. In the current representative Shape-200 overlays,
the REM-Net waveform has amplitude and baseline biases, especially in the T/P
region. High PCC may hide clinically relevant differences.

**Honest response.** Some published figures are heavily selected, smoothed,
filtered, normalized, or shown over very short favorable windows. We should not
imitate that dishonestly. However, the current overlays reveal a real weakness:
REM-Net captures global beat morphology and QRS/T structure but does not perfectly
match amplitude calibration or low-amplitude regions.

**Added evidence.** Region-wise morphology metrics now explain the visual
mismatch: QRS-edge and ST/T structure are strong, but low-amplitude diastolic
regions remain weaker. The qualitative panel should still be regenerated with
transparent selection rules and optional region shading; do not cherry pick only
perfect beats.

### Risk 5: Clinical interval performance is weak.

**Reviewer attack.** CR_RVS full-window PR/QRS/QT errors are much worse than
reported VMD-CBNeT values. A biomedical reviewer may care more about RR/QT/QRS
than morphology PCC.

**Honest response.** We do not claim interval superiority. The main model is
beat-morphology reconstruction; continuous full-window interval diagnostics are
reported as limitations. The route-C analysis explains why full-window timing is
not yet solved.

**Needed evidence.** Add fiducial metrics in the normalized beat space. These do
not solve full-window timing, but they show whether the reconstructed beat shape
contains clinically relevant intra-beat structure. This is a fairer test of the
paper's stated output.

### Risk 6: The negative continuous-gap discovery may not carry enough weight.

**Reviewer attack.** A negative result is useful, but it may not be enough to make
the paper novel if the main positive result is only morphology PCC.

**Honest response.** The discovery is valuable because it is experimentally
bounded: morphology is high, oracle-anchor continuous assembly is high, and
predicted-anchor precision fails across subjects. This triangulates the hard part
of radar-to-ECG and prevents misleading claims.

**Needed evidence.** Present the discovery compactly as one mechanism figure, not
as the main result. Tie it to the paper's fair evaluation philosophy: output
formulation matters, and continuous ECG needs a different timing-localization
solution.

## Go / No-Go Assessment

**As-is for JBHI/TBME:** high risk. The reviewer will likely say the method wins
on an easier target, continuous ECG remains weak, and the model novelty is
incremental.

**As-is for IEEE Sensors / BSPC / similar Q1 signal-sensing venues:** plausible
but still needs careful framing. The dual-dataset, subject-independent,
multi-seed morphology results are meaningful, and the continuous-gap discovery is
scientifically honest.

**After the morphology-region booster:** stronger, but still not a continuous ECG
paper. The model performs well on QRS-edge and ST/T phase-region metrics within
Shape-200, so the paper can argue that it reconstructs broader beat morphology,
not merely a single normalized correlation score. It must still avoid clinical
interval superiority claims.

## Recommended Next Step

Do **not** switch to self-supervised learning for this paper; that belongs to RQ2.
Do **not** keep tuning the abandoned alignment/continuous assembly route as the
main line. The next supervised RQ1 step should be:

1. Regenerate the qualitative figure using transparent selection rules: high,
   median, and hard examples; include error bands or region highlights.
2. Add an efficiency table if quick: parameters, FLOPs, inference time, and
   preprocessing cost.
3. Only after these checks, write the full Results and Methods prose.

This path preserves the current model, strengthens the clinical credibility of
Shape-200, and keeps RQ1 cleanly supervised.
