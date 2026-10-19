# Chen native clarification and conditional9-run schedule

Documentation-only follow-up to author request. No model, inference, loss,
normalisation or frozen v2 metric code changed. No inference or training ran;
only existing artifact timing inspection and CPU read-only counts were used.

See experiments/development/chen_native_20261020/INFERENCE_AND_SCHEDULE.md:

- Every8s window resets autoregressive state; this is a disclosed adaptation.
  Centre crop drops288leading samples, not a guaranteed full512RF transition.
- Existing12.975954s CPU measurement includes generation/stitch/scoring, not
  isolated inference. Linear count-based estimates use20/91validation records.
  GPU timing is unmeasured; no unsupported speedup estimate is supplied.
- Explicit[-1,1] clamp and train-only max-absolute scale described. Train folds
  1/2/3 each have0clipped samples; exact denominators and scales are recorded.
- Teacher forcing is shifted past ECG, consistent with the paper's conditional
  and parallel-training description; unreported BOS details remain assumptions.
-9fresh Chen runs are scheduled conditionally after36/36v2 completion, regardless
  of v2 pass/fail. Separate descriptive reporting; never part of v2's decision.
  No executor/watcher started or added in this documentation-only turn.
