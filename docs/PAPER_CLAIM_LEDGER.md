# Paper Claim Ledger

This ledger maps manuscript claims to verified experimental evidence. Do not
write paper claims that cannot be traced here.

| Claim ID | Manuscript Claim | Supporting Experiments | Status | Notes |
|---|---|---|---|---|
| C01 | The proposed model achieves strong ECG morphology reconstruction on MMECG. | `q1_remalign_mtl_shapefirst_loso` | pilot | Shape-200 evidence only; final table/protocol not locked. |
| C02 | The proposed model transfers to CR_RVS and remains strong on a different radar modality. | `q1_crrvs_remalign_mtl_5fold` | pilot | CR_RVS Shape-200 evidence only; final external protocol not locked. |
| C03 | The proposed model outperforms or compares favorably with most relevant published/reproduced baselines. | MMECG/CR_RVS baseline reports | partial | Needs final protocol table and source labels for each baseline. |
| C04 | Event/morphology supervision contributes to the proposed model's performance. | `q1_remalign_event_lowpeak_loso` | insufficient | Needs compact locked ablation; do not over-expand ablation set. |
| C05 | REM-AlignNet-MTL improves long-term continuous ECG reconstruction over radarODE-MTL. | none | not tested | Only claim after fair radarODE-style long ECG assembly or clearly avoid this comparison. |
| C06 | Shape-200 results demonstrate long continuous ECG reliability. | none | forbidden | Shape-200 cannot support this wording. |
