# Last architecture screen

Author-approved27 fresh MMECG runs, folds1/2/3 × seeds42/43/44 × B/C/D.
Baseline A is the nine completed v2arm1 runs, with only append-only new regional
metrics computed from saved predictions. No changes to v2 sources/artifacts.

Input: identical8s WSST and baseline native200Hz RCG on exact shared sample
coordinates. Architecture/loss/gate are in `spec.json` and the change note.
Decision priority D>C>B>A was explicitly confirmed before training.

Parameters (all trainable):

|Dataset|Baseline total|Dual total|Added|Added / original stem+TCN|
|---|---:|---:|---:|---:|
|MMECG|1,438,789|1,533,573|94,784|6.9951%|
|CR_RVS|1,382,341|1,466,149|83,808|6.4539%|

CR_RVS is implementation/shape support only in this screen; no CR training is
launched here. The event head remains on H_tf, so the new fused waveform path
does not change the event readout topology or absolute-time BCE.

Tests:
`python -B -m pytest experiments/development/final_resolution_20261020/test_final.py -q`

Real-input engineering check: `python -B .../check.py`. Three variants ×
two epochs × one16sample training batch; one complete validation record each
epoch; no weights retained. Validation restores RNG and model training state.
These checks are not model-quality evidence. The ordinary/tolerant waveform
losses have different offsets/scoring support and must not be compared as if
their numerical training-loss scales were identical.

After clean commits and registry update, freeze sources with `lock.create()`.
`launch.py` executes three fold workers; each follows exact seed/variant order,
retains atomic last-state recovery, and generates the full36-row report only
after all27 new runs complete. Checkpoints use the unchanged v2 criterion.
No new post-result architecture is allowed. A formal continuation must match
the author's approved scope; no held-out test access or eleven-fold mode is
implemented by this screening entry point. Chen is queued later, not launched
by this entry point, and still requires the pending timing/cadence prerequisite.

All generated artifacts live in the resource repository's corresponding
`experiments/development/final_resolution_20261020/`, never in historical dirs.
