# Anchor robustness: fixed inference-only protocol

Author request, 2026-10-28. Original final A4=phase_pd seed-base42 weights, all
MMECG11 + CR_RVS5 test folds. No training/checkpoint selection; existing model
and figure artifacts unchanged. A0 is the original window-time parallel-head
context-only comparator, NOT A2.

Only shape sampling bounds change. Keep shared features, event probabilities,
shape decoder, targets and exclusive-right-endpoint grid exactly frozen.
Independent Gaussian errors on both reference R boundaries; sigma0/5/10/20/40/80ms.
Bounds remain fractional, native samples (200Hz). Each pair gets at most10
complete-pair draws to meet positive RR and60..400sample bounds. On exhaustion,
use original predicted nearest-centre refined R and median decoded predicted
RR (160samples if unavailable), even if that fallback lies outside the window;
existing border sampling is retained. No GT repair or window exclusion.
All reference bounds were checked and satisfy RR limits, including sigma0.

Five repeats, seeds0..4. Independent local np.default_rng(seed) per dataset and
sigma; common random numbers across sigma, independent two-anchor draws.
RNG streams span folds in ascending order, batch size16; post-fold states are
saved to support exact cache resume. Training RNG is not used.

Per repeat/subject: frozen per-window PCC -> Fisher-z average, same clipping
as final ablation. Then arithmetic repeat average INSIDE each subject; then
equal-subject mean. Realised error = per-window average absolute error of both
USED boundaries in ms, then windows/repeats within subject and equal subjects.
Fallback and out-of-window rates use all windows. Failed/fallback windows scored.
Bootstrap10000 paired-subject resamples using existing bootstrap_mean(seed42).
Delta versus sigma0 and versus A0 formed at subject level before bootstrapping.
CSV PCC/CI/delta use0..1; report/figure multiply100, delta in percentage points.

Predicted-anchor reference: PCC on ALL original windows (reproduced against
cached final A4 output); x-error computed on MATCHED windows only, both refined
predicted boundaries within150ms of reference and valid RR. State coverage and
subject counts; never restrict the PCC denominator to matched windows. A0 PCC
and its subject CIs come from frozen original per-subject test outputs.

Statistical crossover: paired difference A4(sigma)-A0. Sort by realised x;
first lower95%CI<=0 is loss of evidence that A4 is above A0 (includes intervals
entirely below0). If already true at sigma0, crossover=0; do not invent a positive
noise threshold. Otherwise linearly interpolate the LOWER CI boundary to0
between bracketing observed points. Report empirical mean-crossing separately.
If never crossed, print '>80ms nominal sigma' plus the measured realised-error
bound. It is incorrect to call sigma80 a realised80ms error (E|Gaussian|≈0.798sigma).

Figure3.5x1.75in, two panels, independent y axes; REM-Net color#31688E reused
from existing comparison figure; shaded subject bootstrap CI, predicted star,
context dashed gray line only. Plot script resides alongside output assets.
Dedicated source-data REPORT.md avoids overwriting any other REPORT.md.

GPU numerical preflight: original final-test subprocess uses cuDNN
deterministic=False, benchmark=False (unlike training). Match those exact eval
settings. With deterministic=False, original first-batch shapes reproduce
bitwise; True changes algorithm/numerics by about1e-4. The1e-5 shape equality
guard is retained; no threshold relaxed and no quality result inspected.
