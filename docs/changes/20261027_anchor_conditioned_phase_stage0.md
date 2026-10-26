# Anchor-conditioned phase sampling: Stage0

Author override: supplied staged architecture task, with explicit correction
that Phase Alignment outputs only sampling coordinates t_i. Stage0 only:
implement all four shape-input variants and validate without training/test access.

New code/spec/report under `experiments/phase_resampling_20261027`; tests under
`tests/test_anchor_conditioned_phase.py`. Reuse frozen original stem, TCN,
heads, decoder, reference target and loss; only new variants modify shape path.
Exclusive right endpoint verified: sample a..b−1, rather than a..b, to reproduce
the existing target exactly. No old targets/results/figures modified.

1,579 validation windows (two folds of each dataset) passed target identity;
9 unit tests passed including encoder gradients, stop-gradient, fallback,
interpolation compatibility and reference-independent eval. CPU only.
No training or checkpoint selection; no test HDF5/predictions/checkpoints read.
Stage1 waits for the author after this stage report; Stage2 remains locked.
