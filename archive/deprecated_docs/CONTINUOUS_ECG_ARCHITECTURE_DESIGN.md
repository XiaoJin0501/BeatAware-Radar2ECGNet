# Continuous ECG Architecture Design (CE-Net, working name)

Design document only. No implementation, no training. Governed by
`docs/CONTINUOUS_ECG_TASK_CONTRACT.md`. The frozen REM-Net incumbent is retained
untouched as baseline B0; no existing checkpoint may be overwritten.

## 1. The constraint that drives the design

Measured 2026-09-09 on the current pipeline:

| quantity | value |
|---|---|
| WSST input | `(50, 71, 120)` per 8 s window |
| WSST span | centre 4 s only of the 8 s window |
| WSST effective rate | 30 Hz (one bin = 33.3 ms) |
| **output with no radar support** | **50 % of every window** |
| ECG target | 1600 samples @ 200 Hz |
| upsampling performed by the waveform path | **13.3x** |
| QRS width in WSST bins | **~2.4** |
| `rcg` raw radar, same H5, **unused by waveform path** | `(50, 1600)` @ 200 Hz |

Two separate problems. (i) A **severe temporal-resolution bottleneck**: ~2.4 WSST bins
per QRS makes sharp onset placement hard, though not provably impossible. (ii) A
**support asymmetry**: half of every output window has no radar input at all. (ii) is
first-order and must be settled before (i) can be attributed anything. Neither is fixed
by adding decoder capacity, which is why B2a exists as the capacity-only control.

## 2. Structure

```
              WSST (B,50,71,120)                 rcg (B,50,1600)
                     |                                  |
          [modality stem: FMCW | CW]          [modality stem: FMCW | CW]
                     |                                  |
             rhythm encoder                     morphology encoder
             (B,128,120)                          (B,128,400)
                     \                                  /
                      \----- shared temporal latent ----/
                                   (B,192,400)
                                 /            \
                     rhythm branch            morphology branch
                     (dilated TCN,            (higher-rate conv,
                      long context)            local detail)
                                 \            /
                               learnable fusion
                                      |
                          progressive upsample 400 -> 1600
                                      |
                              nn.Conv1d(...) -> LINEAR
                                      |
                              ECG (B,1,1600)
                                      |
                          aux head: R/QRS probability (B,2,1600)
```

## 3. Tensor shapes

| stage | op | out shape (MMECG) | out shape (CR_RVS) | note |
|---|---|---|---|---|
| in-A | WSST | `(B,50,71,120)` | `(B,1,F,T)` | modality-specific channel count |
| in-B | rcg | `(B,50,1600)` | `(B,1,1600)` | **new**: native-rate path |
| stem-A | Conv2d x2 + freq pool | `(B,128,120)` | `(B,128,T)` | rhythm/low-frequency |
| stem-B | strided Conv1d 1600->400 | `(B,128,400)` | `(B,128,400)` | morphology/high-frequency; strided, **not** interpolated |
| align | upsample A to 400, concat B | `(B,256,400)` | same | 400 steps = 20 ms/step |
| shared | Conv1d 256->192 | `(B,192,400)` | same | shared latent, modality-agnostic |
| rhythm | dilated TCN d=1..16 | `(B,96,400)` | same | RF 373 steps = 7.46 s |
| morph | Conv1d k=7 x3, d=1,2 | `(B,96,400)` | same | local detail |
| fuse | learnable gate | `(B,192,400)` | same | soft, event-conditioned |
| up1 | ConvTranspose1d s=2 | `(B,96,800)` | same | learned upsampling |
| up2 | ConvTranspose1d s=2 | `(B,48,1600)` | same | full rate |
| out | Conv1d 48->1, **linear** | `(B,1,1600)` | same | **no sigmoid** |
| aux | Conv1d 48->2 | `(B,2,1600)` | same | R / QRS probability |

## 4. Design rules

1. **Linear output.** No sigmoid. The target is raw ECG under a train-fold robust
   scaler, which is not bounded to [0,1].
2. **No early compress-then-interpolate.** The morphology path is strided from 1600,
   not interpolated up from 15 Hz.
3. **Modality-specific stems, shared latent and decoder.** Justified by the measured
   noise asymmetry (FMCW degrades gracefully, CW falls off a cliff): the front end
   differs, the cardiac morphology should not. CR_RVS must not trigger changes to the
   shared trunk.
4. **Event head is auxiliary and soft.** It conditions the fusion gate; it never
   produces the waveform through hard peak assembly. This is what failed as C5
   (PCC 0.186).
5. **Long-record inference** uses overlapping windows, keeps only each window's
   reliable centre region, and applies weighted overlap-add. No per-cycle template
   copying.

## 5. Loss (introduced incrementally, never all at once)

`L = L_Huber + lam_c (1 - PCC) + lam_d L_derivative + lam_f L_MRSTFT + lam_e L_event`

Each term is added only after a validation ablation shows the previous configuration
is insufficient. EGA / GradNorm balancing is a *candidate*, to be verified locally and
not assumed from the radarODE-MTL paper — whose released code has its titular ODE path
disabled (`docs/RADARODE_REPRODUCTION_AUDIT.md` A1).

## 6. Ablation ladder

B0 existing REM-Net full-window branch (frozen baseline) ->
B1 linear output + train-fold robust normalization ->
B2a dual-scale decoder, WSST only (decoder-capacity control) ->
B2b B2a + native-rate `rcg` dual path ->
B3 event-conditioned soft fusion ->
B4 multi-task gradient balancing ->
B5 final continuous model.

MMECG development folds first, small pilot. CR_RVS validated independently.

## 7. Decision rule

Per the standing rule: if B1 helps, the problem was output/normalization definition;
B2, decoder capacity and time-scale; B3, event conditioning; B4, gradient conflict.

**Amendment justified by §1:** we should not wait for B1-B4 to all fail before
re-examining the input representation. The 15 Hz WSST bound is already measured, and
the 200 Hz `rcg` is already available. B2 therefore carries the representation fix
rather than deferring it to a fallback branch.
