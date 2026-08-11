# Discussion, Limitations, Conclusion (draft prose)

Obey `docs/PAPER_CLAIM_LEDGER.md`: no continuous SOTA, no identical-protocol
VMD-CBNeT superiority, no clinical-interval superiority.

## 6. Discussion

Our results support a clear separation between beat-morphology reconstruction and
continuous ECG recovery. A representation-aware, event-structured, morphology-first
model reconstructs ECG beat morphology robustly and seed-stably across two radar
modalities (FMCW and CW) under subject-independent evaluation, and does so with a
compact model. Region-wise analysis shows the improvement is structural rather than
a smooth-curve artifact: REM-Net preserves ST/T-region morphology where an adapted
ODE-based baseline collapses, indicating that the model retains broader beat
structure beyond the R/QRS edge.

At the same time, our oracle analysis shows that morphology is not the limiting
factor for continuous ECG. When beats are placed at correct positions, the learned
morphology assembles into near-perfect continuous ECG; the failure is that
predicted beat-anchor localization is not precise enough, and — critically — its
precision does not generalize across subjects, even though detection recall does.
This reframes continuous radar-to-ECG as a timing-localization problem rather than a
waveform-generation problem, and it cautions against reporting beat-normalized
morphology scores as evidence of continuous reconstruction. We believe this
decomposition is useful to the field independent of our specific model.

## 7. Limitations

Shape-200 is a beat-normalized morphology target; it deliberately removes absolute
timing and amplitude, so our main results should not be read as continuous or
clinical-interval performance. On full-window continuous ECG, interval-timing errors
(PR/QRS/QT) remain larger than reported values from methods evaluated under
different protocols, and we make no interval-superiority claim. Low-amplitude
diastolic morphology remains the weakest cycle region for all methods, and is weaker
under CW than FMCW radar. The comparison with published VMD-CBNeT numbers is
contextual and reported-only, because protocols and PCC spaces differ; we do not
reproduce it under our protocol. The adapted radarODE-MTL baseline is compared only
in morphology space; its long-term assembly is not reproduced. Finally, the headline
numbers use three seeds; the frozen-configuration confirmation run uses a single
seed and is reported only as a robustness check, not as an independent claim.

## 8. Conclusion

We presented REM-Net, a morphology-first, representation-event-morphology model for
radar-based ECG beat reconstruction. Under subject-independent evaluation, REM-Net
achieves strong and seed-stable beat-morphology reconstruction on both FMCW (MMECG)
and CW (CR_RVS) public radar datasets, improving over adapted and local baselines
and over our prior conference model, while remaining compact. Through oracle and
cross-subject analysis, we further identify beat-anchor localization precision — not
morphology capacity — as the key barrier to continuous radar-to-ECG, and we frame it
as the next open problem. We hope the explicit separation of morphology from
continuous timing, together with fairness-audited, output-formulation-aware
evaluation, supports more comparable progress in contactless ECG sensing.

## Future work

The natural next step is robust, subject-invariant sub-50 ms beat-anchor
localization from radar, which our analysis identifies as the bottleneck for
continuous assembly; improving low-amplitude (diastolic) morphology under CW radar
is a complementary direction. Both are supervised extensions of the present study
and do not require abandoning the morphology-first formulation.
