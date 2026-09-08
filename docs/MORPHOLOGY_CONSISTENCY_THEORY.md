# Morphology consistency: mathematical scope and falsifiable claims

Research note, 2026-09-08. These are elementary properties motivating a candidate
objective, not claims of a new theorem or guarantees of journal acceptance.

## 1. Correlation leaves amplitude unidentified

For a nonconstant target y, prediction p = alpha*y + beta with alpha > 0 has
PCC(p,y) = 1. Centering removes beta and normalization cancels alpha. However,

    MSE(p,y) = (alpha-1)^2 Var(y) + ((alpha-1) mean(y)+beta)^2.

This follows by writing y = (y-mean(y))+mean(y); the cross term has zero mean.
Consequently correlation alone cannot bound amplitude error. This is a known
property, not a novel contribution. It motivates reporting amplitude metrics
and avoiding independent prediction normalization in a consistency objective.

## 2. A cycle-level error bound with explicit assumptions

Fix one annotated training interval. Let C be linear cropping/resampling to K
samples. Let a=min(C(y)), b=max(C(y))-a > epsilon and P(z)=(C(z)-a)/b.
For shape-head prediction m and normalized target s=P(y), define mean absolute
errors E_wave=||P(p)-s||_1/K, E_cons=||P(p)-m||_1/K and E_shape=||m-s||_1/K.
The triangle inequality yields E_wave <= E_cons+E_shape. Hence

    ||C(p)-C(y)||_1/K <= b*(E_cons+E_shape).

The right-hand side requires BOTH a reliable morphology estimate and agreement
with it. Lower consistency error alone does not imply lower target error.
Stop-gradient changes the optimizer's path, not this algebraic relation.

Limitations: the reference scale b varies by cycle; gradients in normalized
coordinates are scaled by 1/b. Small-scale cycles can dominate training, so
coverage, scales and gradient norms require inspection. The implementation
excludes b <= 1e-4. Reference statistics are used only during training.

## 3. The bound cannot establish full-window recovery

C is not injective: an error supported outside the selected interval has C(e)=0.
Thus p=y+e can have perfect cycle consistency while its full-window error is
arbitrarily large. Resampling can also discard within-cycle detail. Neither a
whole-window guarantee nor a generalization guarantee follows from Section 2.

This is why the controlled experiment retains the full-waveform objective and
reports waveform metrics separately. If only the chosen cycle improves, the
method has not solved continuous reconstruction. Extending to all beats would
require beat-specific morphology targets/predictions and a separate experiment;
repeating one center-cycle shape across all beats is not justified.

## 4. Timing sensitivity is distinct from morphology agreement

For y twice differentiable near t with |y''| <= M over the shifted interval,

    |y(t+delta)-y(t)-delta*y'(t)| <= M*delta^2/2.

The result is Taylor's theorem with bounded remainder. It predicts greater
first-order sensitivity in steep waveform regions, under small shifts. It does
not establish a universal 15-ms or 50-ms physiological precision threshold.
For discrete ECG, interpolation and sampling determine the approximation error.
Large shifts, sharp features and waveform-dependent PCC centering require
separate treatment.

## 5. Experiments that could reject the proposed explanation

1. Compare B1 (detached morphology target) against B2 (direct reference cycle)
   at the same weight and initialization. If B2 matches or beats B1, evidence
   supports additional local supervision rather than cross-output transfer.
2. Report both cycle and full-window errors. A cycle-only gain is insufficient.
3. Examine amplitude and baseline errors even when PCC improves.
4. Repeat the controlled comparison independently on MMECG and CR_RVS using
   development subjects, then a predeclared independent evaluation protocol.
5. Correlate predicted-shape error with waveform changes as an analysis only;
   association alone does not prove reliable shapes cause an improvement.

## Manuscript status

Do not insert a claimed benefit or a "theoretical innovation" into RQ1.tex on
the strength of these identities. They can later support a precise Methods
motivation and a limitations paragraph if the empirical candidate succeeds.
