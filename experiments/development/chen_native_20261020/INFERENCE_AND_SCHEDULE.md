# Inference, timing, quantisation and conditional training schedule

Author clarification2026-10-20. **Documentation and read-only tallying only**:
no model inference, training, GPU benchmarking, model changes or v2 rule changes
were performed for this supplement. Earlier CPU smoke remains smoke evidence.

## 1. Autoregressive state and centre crop

Current implementation **restarts generation independently for each8s window**.
`evaluate.infer_record` calls `infer_8s` separately for each approved input;
`ChenNativeAR.generate` creates a new zero causal state and BOS token on each
call. It does **not** carry predicted ECG history or decoder state along the
entire record or across adjacent overlapping inputs. There is no GT warm start.

This is an explicit adaptation for the matched8s-input/5.12s-output protocol,
not a claim that the original paper used this reset policy. The paper states
single-step autoregressive inference but does not establish its exact reset
policy in the reviewed §3.3/§4.2.3 text.

Per input,1600native200Hz samples are generated, [288,1312) is retained;
the first/last288samples(1.44s each) are discarded. Thus the first288points of
the initial transition are not scored. **Do not write that all startup effects
are guaranteed removed**: the chosen decoder RF512 spans2.56s, so224retained
points(1.12s) can still have incomplete-history startup effects. They remain in
the denominator, with no extra exclusion. The radar encoder is allowed to read
the whole8s input; causal ECG-history generation is not an online-causal radar
encoder claim. Tail handling uses the same approved uncovered-suffix rule.

## 2. Timing — what was actually measured

The existing CPU check on MMECG_0003 has:

-35505native samples(177.525s), **35** actual8s input windows;
-34929retained samples and34complete1024-point scoring windows;
- elapsed **12.975954s**, one CPU thread, nice15 and idle I/O, while v2 was
  running. The model was randomly initialised, eval-only and not trained.

The timer began immediately before `infer_record` and ended after
`score_record`/`aggregate`. It therefore includes input-window normalisation,
encoding, autoregressive decoding, stitching, NeuroKit2 and frozen metric
calculation, but excludes initial record loading, scale fitting and model
construction. **Pure model-inference time was not separately recorded.**
Do not relabel12.98s as isolated inference latency or a trained-model benchmark.
Source artifact: the resource repository's
`experiments/development/chen_native_20261020/full_record_cpu_check/COMPLETE.json`.

The read-only counts from the existing A validation inventory are:

|Scope|Records|8s forward windows|CPU whole-pipeline estimate, one seed|CPU whole-pipeline estimate,3seeds|
|---|---:|---:|---:|---:|
|Development folds1/2/3|20|700|259.52s ≈4.33min|778.56s ≈12.98min|
|All11validation folds|91|3185|1180.81s ≈19.68min|3542.44s ≈59.04min|

Formula:12.975954×(total8s windows/35). This is a single serial evaluation pass
per checkpoint, **not training time**. The9development runs correspond to the
first row×3seeds. Repeating validation each epoch multiplies this cost; record
loading, trained-model peak-detection cost, scheduling, cold caches and hardware
contention can change it. These are coarse extrapolations, not confidence bounds
or guaranteed upper bounds. No final-test files were read to produce the table.
The11fold row is capacity planning only, **not authorisation for11fold training**.

### GPU

No Chen-native GPU latency has been measured, intentionally, to leave v2's GPU
untouched. A numerical GPU estimate cannot responsibly be derived from CPU or
REM-Net/v2 timings:1600dependent generation steps create many small GPU kernel
launches and can be launch-bound, so GPU is not guaranteed faster.

If a later authorised benchmark gives a single-record generation-plus-scoring
time G seconds on the same35-window reference record, the corresponding rough
estimates are20G seconds for development1seed,60G for development3seeds,91G for
all11folds1seed and273G for3seeds. If G is **model-only** time, add CPU
NeuroKit2/scoring/I/O separately instead. Until measured, both CPU isolated
inference and GPU timings are explicitly **not measured**, not invented numbers.

## 3. Normalisation, clamp and actual training clipping rate

For each fold, use its unchanged A training-only median m and IQR q. With
filtered raw-record ECG e, compute z=(e−m)/q. Fit

    a = max(1e−6, max over unique complete TRAIN-record samples of |z|)
    u = z/a
    u_clipped = min(1, max(−1,u))
    c = sign(u_clipped) log(1+255|u_clipped|) / log(256)
    class = round(127.5(c+1)), bounded to integer[0,255]

Clamp is explicit `Tensor.clamp(-1,1)` before μ-law. This is not percentile
winsorisation, per-window min/max fitting, or a validation-derived scale.
`data.statistics` computes a on training subjects only; the same constant is
used for all train/validation windows of that fold. Restore output to the frozen
reference space by inverse μ-law followed by multiplication by a before scoring.

Read-only counts of samples with u<−1 or u>1 **before** the clamp:

|Fold|Unique train records|a|Unique train samples|Clipped samples|Clipped fraction|
|---:|---:|---:|---:|---:|---:|
|1|87|44.1660423|3088935|0|0.000000%|
|2|77|43.1570930|2733885|0|0.000000%|
|3|73|42.5120277|2591865|0|0.000000%|

Exact endpoint values±1 are not counted as clipped. Zero train clipping follows
from using the training maximum absolute value and was also checked with the
float32 conversion used by training. The overlapping640-point train windows and
their last128point supervised subsets likewise have0clipped samples, since they
are subsets of those same training records; no separate64million-sample cache
was built. The unique-record denominator avoids silently inflating n through
overlap. μ-law quantisation still has finite precision:0clipping≠0quantisation
error. The scale is sensitive to extreme training artifacts, and no tuning or
outlier-removal change was made here. Validation/test values outside the fixed
range would saturate; their clipping rates were not measured in this supplement.

## 4. Teacher forcing

**Yes, training uses teacher forcing.** A640point target q_0…q_639 is shifted to
`[BOS,q_0,…,q_638]`. Causal convolutions predict q_t using past ECG only and radar
features; CE is restricted to t512…639. A current/future target is never passed
as its own history. Unit tests verify both this shift and lack of future-label
influence. Inference instead feeds back the model's own argmax classes.

This is consistent with the paper's equation13 conditioning on x_<t and its
parallel training of128outputs with single-step autoregressive inference.
The reviewed paper text does not explicitly name “teacher forcing” or specify
BOS encoding/shift implementation, so the latter are disclosed reconstruction
choices, not claimed to be verified original source-code details. See Chen et
al., IEEE TMC23(1),270–285, DOI10.1109/TMC.2022.3214721, §3.3 and §4.2.3.

## 5. Author-approved conditional schedule — NOT STARTED

After **all36distinct v2 runs** finish and their complete report is finalised:

1. Prepare the separately registered Chen native development run using folds
   1/2/3 × exact seeds42/43/44 =9fresh runs. No reused smoke weights exist.
2. Use the same frozen v2 evaluator and harmonic checkpoint rule; original
   unshifted outputs remain the basis of strict metrics. Keep AdamW1e−4,
   batch16,patience15,max100,drop_last=True and no patience in epochs1–10.
3. Report these9runs in their own table/output subtree, including chosen epochs,
   not as arms of v2 and not as any input to v2's screening decision. Cross-model
   comparison is descriptive only, without adding statistical tests or gates.
4. v2 screening PASS is **not** an additional scheduling requirement: completion
   of36runs is the prerequisite, regardless of pass/fail. A failed/interrupted
   queue or partial count is not completion.11fold confirmation stays blocked.

This turn only records the schedule. No watcher, training process, new training
entry point, performance measurement or automatic launcher was created. The
current preparation CLI remains smoke-only; production entry-point/preflight
must be ready and committed before the scheduled training can actually start.
