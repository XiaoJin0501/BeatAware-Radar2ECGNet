# phase_pd layer table audit — 2026-11-01

Documentation only; no model, checkpoint, preprocessing, metric or training changes.
Created `docs/paper/tab_layers.tex` from executable definitions, not architecture figures.

Sources:
- `experiments/phase_resampling_20261027/architecture.py:76–113`: phase_pd feature concatenation, grid construction and distinct interpolation branches.
- `experiments/phase_resampling_20261027/frozen_bridge.py:32–44`: exact AST import of frozen stem, TCN, dual-head and losses.
- `/tmp/rem_dual_head_formal/src/models/continuous/ce_net.py:21–42,81–98`: stem and TCN.
- `/tmp/rem_dual_head_formal/src/models/continuous/dual_head_rem.py:23–47`: heads.
- `/tmp/rem_dual_head_formal/scripts/run_dual_head_interaction.py:178–187`: BCE-with-logits.

Template corrections:
1. Split dropout, adaptive frequency pooling and squeeze so all tensor shapes are explicit.
2. TCN order is Conv-BN-GELU-Dropout-Conv-BN-GELU, then local residual addition; no global skip or post-add GELU.
3. 400→1600 interpolation belongs to the event branch and separately the phase-feature branch, with different align_corners values (False / True).
4. Event head returns logits, not Sigmoid outputs; Sigmoid is a downstream probability conversion. Shape head applies external Sigmoid after the second convolution.
5. Phase alignment returns only t_i; sampling yields 132 channels; a separate window-context interpolation yields 128 channels, concatenated to 260. There is no extra Interpolate inside or immediately before the concatenated shape-head input.
6. Replaced rounded total by exact trainable parameter count (MMECG 1,481,029; CR_RVS 1,424,581).
7. RF is 373 LATENT positions, nominal 7.46 s at 50 Hz; do not call these 373 native 200-Hz samples. The first/last position span is 7.44 s, and the full raw-input RF includes stem/interpolation.

Validation: CPU eval/no_grad forward with B=1 MMECG input, hooks checked every listed module output. Counted both models' trainable parameters. No training or GPU use. No TBD tokens/definition remain. LaTeX compiler is unavailable locally; final single-column fit should be checked in Overleaf. Table uses scriptsize and 2pt column padding to limit width; no forced scaling.
