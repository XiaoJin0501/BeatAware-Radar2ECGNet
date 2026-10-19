# Chen-native validation codec diagnostic (no training)

Author request: measure a complete fold validation on GPU before Chen training,
and quantify validation mu-law round-trip error for development folds 1/2/3.

## Completed CPU-only diagnostic

Entry: `experiments/development/chen_native_20261020/quantization_check.py`.
Output (resource repository):
`experiments/development/chen_native_20261020/validation_quantization_check/results.json`.
The entry uses the existing train-only `data.statistics`, codec implementation,
raw-record references and frozen `unified_metrics.continuous_pair`. It does not
construct a model, optimizer or CUDA context and does not access test records.

Transform: z=(ECG-median_train)/IQR_train, u=z/maxabs_train; hard clip u to [-1,1],
encode with mu=255 and 256 possible codes, decode and multiply by maxabs_train,
then inverse the train median/IQR transform. The complete validation references
are checked, without window caches, cropping or temporal alignment.

| Fold | Validation records | PCC | RMSE (z space) | RMSE (original reference units) | Used codes / 256 | Clipped samples |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 0.99990909 | 0.02711381 | 0.00196794 | 182 | 0 / 71010 |
| 2 | 12 | 0.99990996 | 0.02744533 | 0.00203864 | 188 | 0 / 426060 |
| 3 | 6 | 0.99990870 | 0.02454119 | 0.00185056 | 173 | 0 / 213030 |

PCC is an equal-record Fisher-z average within each fold; RMSE is pooled over
native samples. Per-record PCC/RMSE and exact used-code indices are retained.
Original reference units are not asserted to be millivolts. These are numerical
codec diagnostics, not model reconstruction performance. The encoder still has
256 possible classes; used codes count their union across validation records.

## Pending GPU timing; no cadence change made

At the diagnostic check, `remnet-continuous-screen-v2.service` was active.
The earlier author constraint forbids competing with its GPU resources.
Therefore no concurrent GPU benchmark was launched, no GPU runtime is claimed,
and neither training nor validation frequency was changed. CPU timings in
`INFERENCE_AND_SCHEDULE.md` cannot substitute for this requested measurement.

Before the nine Chen runs, measure one entire fold's autoregressive inference
and frozen scoring with device synchronization at timer boundaries, identifying
the fold, record count, hardware, model state, data-loading scope and repetitions.
A random/untrained model must be labeled as such: waveform-dependent detector
runtime may differ from a trained model. Also separate initialization/warmup
from timed work. A complete validation time alone bounds the validation cost,
not the total training time; training cost must be explicitly accounted for in
any six-hour estimate. If a cadence revision is warranted, commit its exact
validation schedule, warmup/patience conversion and checkpoint rules before
formal training. No automatic Chen training launcher was added.
