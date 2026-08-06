# Paper Claim Ledger

This ledger maps manuscript claims to verified experimental evidence. Do not
write paper claims that cannot be traced here.

| Claim ID | Manuscript Claim | Supporting Experiments | Status | Notes |
|---|---|---|---|---|
| C01 | WSST/STFT radar representation improves beat morphology reconstruction compared with raw radar baselines. | `q1_remalign_wsst_tcn_loso` | pilot | Shape-200 only; not long continuous ECG. |
| C02 | Event supervision improves QRS morphology and beat timing. | `q1_remalign_event_lowpeak_loso` | insufficient | Needs locked ablation and timing metrics. |
| C03 | REM-AlignNet-MTL improves long-term continuous ECG reconstruction over radarODE-MTL. | none | not tested | Requires radarODE-style long ECG assembly evaluator. |
| C04 | The method generalizes across FMCW and CW radar modalities. | `q1_crrvs_remalign_mtl_5fold`, `q1_crrvs_wsst_tcn_5fold` | partial | Current evidence is Shape-200 external validation only. |
| C05 | Shape-200 results demonstrate long continuous ECG reliability. | none | forbidden | Shape-200 cannot support this claim. |
