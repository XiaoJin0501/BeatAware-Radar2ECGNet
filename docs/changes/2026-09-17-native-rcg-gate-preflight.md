# Native-RCG Resolution Gate: Implementation and Preflight

Date: 2026-09-17

## Scientific change

The registered `q1_native_rcg_resolution_gate` compares capacity-matched
`B2a-wide + L3` against `B2b + L3` on repaired `loso_v2`. MR-STFT is locked at
0.50 with FFT/window sizes 128/256/512; event BCE remains 0.20. No dynamic loss
weighting or additional term is enabled.

## Implementation

- The training entry point permits L3 on B2a-wide/B2b only under the registered
  experiment tag and gives the two arms non-colliding C0/C1 output paths.
- Runtime guards require WSST `[B,50,F,T]`, RCG `[B,50,1600]`, a 400-step latent,
  and exact WSST/RCG temporal agreement before concatenation.
- A component-specific paired seed resets the byte-identical decoder so C0/C1
  decoder tensors start exactly equal despite variant-specific stems consuming
  different RNG streams.

## Verification

- Regression tests: 4 passed.
- Parameter counts: B2a-wide 1,487,551; B2b 1,487,667; delta 116 (0.008%).
- Decoder state keys, shapes, and initialized tensors: identical.
- CPU dummy forward/backward: output `(B,1,1600)`, event `(B,2,1600)`, all
  trainable gradients present and finite; malformed RCG length rejected.
- Real-data one-epoch smoke: both arms completed on fold 1; smoke artifacts
  archived under `archive/legacy_experiments/failed_and_trash/` and excluded
  from evidence.
- Fold 1/5/9 base/WSST counts match; RCG `(N,50,1600)`, WSST
  `(N,50,71,240)`; full-window support asserted; validation hashes distinct.
- Formal output directory absent before launch.

## Runtime state

PyTorch 2.6.0+cu124 sees one RTX 4080 SUPER and cuDNN 9.1. The training entry
point fixes the effective seed to `42 + fold`, sets
`cudnn.deterministic=True`, and disables cuDNN benchmark mode. At preflight,
another user's process occupied about 12.2 GiB, so formal training was not
launched with inadequate free memory.
