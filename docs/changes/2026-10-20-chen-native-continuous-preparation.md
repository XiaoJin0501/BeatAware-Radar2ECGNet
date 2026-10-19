# Chen native μ-law/autoregressive preparation — CPU only

Author requested implementation/tests/smoke only while v2 screening continues,
then explicitly selected paper-style256-class μ-law CE and autoregression.
No formal training or test access is authorised or exposed by the entry point.

New isolated worktree/branch: BeatAware-Radar2ECGNet-chen-native /
exp/mmecg-chen-native-continuous-prep. Source, spec and README are under
experiments/development/chen_native_20261020/. Outputs use the same newly named
subtree in the resource repository; old results and weights are untouched.

Paper-based implementation:640point/stride30 training windows; strictly
shifted teacher forcing, last128point CE; native3Dpositions; causal categorical
TCN with explicit512RF; deterministic single-sample autoregressive decoding.
The8s extension is implemented by variable-length conditional encoding, NOT
resampling a640-point ECG. Native centre-crop/stitch and all scoring functions
reuse actual frozen v2/A imports, checked with the existing freeze.

Paper dimensional contradictions and unspecified details (pool/up counts,
channels, spatial expansion, TCN blocks, BOS, argmax, normalisation) are
documented. This is NOT official code or an exact layer-for-layer reproduction.
AdamW1e−4/batch16 and v2 selection replace the paper's Adam1e−3/batch64 at the
author's explicit request. Startup effects are retained, not excluded.

CPU smoke uses16train samples/two epochs and one real8s val excerpt, not full
record performance. CUDA is hidden before imports, one CPU thread, nice15 and
idle I/O. No weights saved. Runtime metrics are not manuscript evidence.
The running v2 worktree, commit, metric freeze and GPU processes remain intact.
