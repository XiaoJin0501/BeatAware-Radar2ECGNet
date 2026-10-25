# Table I baseline-format and source audit — 2026-10-26

Read-only audit; no table values or training/checkpoint rules changed.

## 1. Five baselines: actual endpoint implementation

**All five Table-I morphology baselines are adapted dual-head models, not native
continuous-output methods.** `DualEndpointBaseline` retains its local reproduced
backbone/latent representation, bypasses its old scored outputs, and calls a
common `_Readouts` module. Some unused native heads remain allocated in state
dicts; bypassed does not mean physically removed. This is not an exact original-
protocol reproduction, nor proof that all defining paper mechanisms are active.

|Name in table|Active representation|Scored output/adaptation|Native training mechanisms not used by this endpoint runner|
|---|---|---|---|
|Chen-TCN|local CNN/Transformer/dilated-TCN `core.represent(rcg)`|new200point shape and four-event heads|No original mu-law256-class autoregressive output/teacher forcing|
|radarODE-MTL|DCN/ResNet feature extractor and squeeze, `core.represent(wsst)`|new200point shape and four-event heads|Original shape/PPI/anchor decoders and added continuous decoder bypassed; PPI ODE is not in the scored path; no EGA multi-task backward|
|Cross-modal Conformer|local radar `core.embed(rcg)` embedding|new200point shape and four-event heads|No ECG autoencoder/deep-feature pretraining/distillation objective in this runner|
|WaveGRU-Net|local wavelet/convolution/GRU `core.represent(rcg)`|new200point shape and four-event heads|Native continuous readout bypassed|
|Attention-DCGAN|generator `core.represent(rcg)`|new200point shape and four-event heads|No discriminator or adversarial loss in this runner|

The common heads: Conv1d(k5,no bias)→BN→GELU→Conv1d(k5); interpolate latent
to200 for the sigmoid shape output,1600 for four event logits. Inputs remain
time-domain RCG for Chen/Conformer/WaveGRU/Attention-DCGAN and WSST (MMECG) or
resized log-STFT (CR_RVS) for radarODE-MTL. CR_RVS input channels=1 rather than50.

Training starts from freshly constructed models; no published pretraining or
old continuous checkpoint is loaded as an initializer in these entry points.
`last_state.pt` loading is continuation of the same fold/run, not pretraining.
All model parameters are passed to AdamW, but bypassed original heads have no
loss path/gradient. Common loss is

`L1(shape,target) +0.5*(1-mean PCC(shape,target)) +0.3*BCE(event_logits,Gaussian events)`.

No derivative, continuous-waveform, adversarial, PPI, original ODE task, EGA or
cross-modal teacher loss is added by the endpoint training loop. Consequently
the table must describe **backbone-adapted common-head baselines**, not report
their values as original published model performance.

Evidence (read-only frozen implementation):

- `/tmp/rem_dual_head_formal/src/models/baselines/dual_endpoint.py`:
  `_Readouts.forward`, `DualEndpointBaseline.represent/forward`.
- `/tmp/rem_dual_head_formal/scripts/run_dual_endpoint_baselines_mmecg.py`:
  `EndpointFold`, `train`, `evaluate`, `rank`.
- `/tmp/rem_dual_head_formal/scripts/run_dual_endpoint_baselines_crrvs.py`:
  inherited train loop; single-channel adapter; radarODE71×240 interpolation.
- `/tmp/rem_dual_head_formal/scripts/run_dual_head_interaction.py`: `losses`,
  `shape_target`, `shape_metrics`; target R-R nearest/containing window center,
  endpoint-inclusive resample200 then target min-max.
- `experiments/specs/q1_dual_endpoint_baselines_mmecg.json` in the frozen tree
  explicitly registers adapted outputs and common loss.

## 2. Are ALL current table cells from that same format?

**No. Morphology/anchor cells are, but the whole current table is mixed-source.**

|Current table block|Source|What it actually measures|
|---|---|---|
|Both datasets PCC/RMSE|seed42 `q1_dual_endpoint_baselines_*/*/test_metrics.json`; snapshot official-shape tables|200point morphology, fold macro|
|Both datasets Q-Q/R-R/S-S/T-T columns currently filled with42.2/39.0/etc for MMECG Chen|`docs/audits/q1_formal_anchor_intervals/REPORT.md`, event-position MAE table|Single Q/R/S/T position MAEs from adapted event heads, NOT successive Q-Q/R-R/S-S/T-T interval errors; current headers conflict|
|MMECG RR/QRS/QT|`docs/audits/Q1_SUBJECT_LEVEL_BENCHMARK_prominence.csv`, waveform readout; average seeds42/123/2024 within subject|Older continuous-model waveform clinical readouts; different trained checkpoints and sampling/aggregation from adapted seed42 shape heads|
|CR_RVS RR/QRS/QT|`docs/audits/Q1_CRRVS_ANCHOR_INTERVAL_COMPLETION.md`|Adapted event-head intervals: RR, Q–S peak spacing, Q–T peak spacing; latter two are NOT onset/offset clinical QRS/QT durations|

The MMECG waveform CSV exactly reproduces the current table after averaging
within subject over its available seeds and then mean/population-SD across
subjects. For example Chen32.25±32.99,5.38±3.72,20.11±7.64 and
radarODE17.36±11.63,5.53±4.06,23.48±4.78 (RR/QRS/QT ms). Thus this is a traced
source mismatch, not merely a possible naming issue. No values are changed by
this audit. REM-Net MMECG anchor report also averages seeds42/123/2024 whereas
shape rows use seed42. The current table's subject-macro blanket note does not
describe shape fold-macro aggregation on CR_RVS.

## 3. Fig4/FigS1 implementation

`scripts/figures/plot_per_beat_morphology.py` performs only visualization inference
on existing best checkpoints. All3 plotted methods use a separate8s context for
EACH reference beat, centered at its R-R midpoint, and its own200point shape head.
There is no reuse of a representative template for multiple beats. Reference
centering/placement is explicit GT assistance permitted for morphology display,
not deployable timing. Q/R/S/T markers retain absolute predicted times; GT only
identifies which local beat's predictions to display. Main trace min-max is a
display transform; its score uses the raw sigmoid200point output and exact
Table-I target construction, arithmetic mean across plotted beats.

Selection is based on the original Table-I representative-window Shape-PCC:
subject mean within scenario→scenario median→closest subject→window closest
to that subject's median. All candidate windows remain included. Fresh per-beat
sampling produces new panel scores, not unchanged aggregate test values.
Reconstruction of the16 original REM-Net fold scores from float16 saved outputs
and exact target method differs from official scores by at most about1.4e-6.

FigS1 uses a separate newly inferred shape for each PREDICTED R-R interval,
centers its contexts using predicted timing only, and places it on those anchors.
Unsupported context or uncovered regions are flat-filled and shaded gray, never
GT-filled. The reference signal is used only for display/scoring. No training,
test checkpoint selection, lag/gain fitting, smoothing or synthetic fabrication.

Sources, exact context starts, per-beat predictions, checkpoint hashes, chosen
subject/window IDs and scores are under:
`docs/paper/sections/figures/source_data/fig4_per_beat_20261026/`.

Required Fig4 caption phrase:
**beats placed on reference R-R intervals; timing evaluated separately**.
