# Paper Claim Ledger

This ledger maps manuscript claims to verified experimental evidence. Do not
write paper claims that cannot be traced here.

| Claim ID | Manuscript Claim | Supporting Experiments | Status | Notes |
|---|---|---|---|---|
| C01 | The proposed model achieves strong ECG morphology reconstruction on MMECG. | `q1_remalign_mtl_shapefirst_loso` | pilot | Shape-200 evidence only; final table/protocol not locked. |
| C02 | The proposed model transfers to CR_RVS and remains strong on a different radar modality. | `q1_crrvs_remalign_mtl_5fold` | pilot | CR_RVS Shape-200 evidence only; final external protocol not locked. |
| C03 | The proposed model outperforms or compares favorably with most relevant published/reproduced baselines. | MMECG/CR_RVS baseline reports | partial | Needs final protocol table and source labels for each baseline. |
| C04 | Event/morphology supervision contributes to the proposed model's performance. | `q1_remalign_event_lowpeak_loso` | insufficient | Needs compact locked ablation; do not over-expand ablation set. |
| C05 | REM-AlignNet-MTL improves long-term continuous ECG reconstruction over radarODE-MTL. | `q1_mmecg_long_assembly_fold01_pilot`, `q1_radarode_anchor_ppi_audit_fold01` | insufficient | Fold-01 pilot only. REM shape oracle-cycle exceeds radarODE shape oracle-cycle, but REM full-window OLA is still weak. radarODE predicted/center Anchor-PPI assembly variants failed and are diagnostic only. Do not claim final long-term superiority. |
| C06 | Shape-200 results demonstrate long continuous ECG reliability. | none | forbidden | Shape-200 cannot support this wording. |
| C07 | The paper evaluates radar-to-ECG methods under output-formulation-aware fair comparisons rather than forcing all methods into `1 x 1600` full-window ECG. | `docs/Q1_PAPER_COMPARISON_STRATEGY.md` | planned | Needs literature comparison ledger and final table labels. Sample-level published results must be separated from subject-independent main comparisons. |
