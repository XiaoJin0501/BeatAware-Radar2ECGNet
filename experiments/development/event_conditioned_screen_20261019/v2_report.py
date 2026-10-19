"""Complete36-run v2 report only; no interim arm comparisons or expansion."""
import json
import environment as env
from freeze import verify,sha
from v2_metrics import screening_decision
from src.utils.atomic_state import atomic_json,atomic_csv,atomic_text

def fmt(x,d=3):return 'NA' if x is None else f'{x:.{d}f}'

def generate():
    verify()
    table=[]
    for fold in (1,2,3):
        for seed in (42,43,44):
            for arm in (1,2,3,4):
                directory=env.OUT/'runs'/f'fold_{fold:02d}'/f'seed_{seed}'/f'arm_{arm}'
                if not (directory/'COMPLETE.json').exists():raise RuntimeError('Incomplete run: '+str(directory))
                r=json.loads((directory/'result.json').read_text());v=r['validation']
                assert r['protocol_version']==v['protocol_version']=='v2'
                assert len(v['events_by_subject'])==len(v['M1']['subjects'])==1
                e,m=v['events_by_subject'][0],v['M1']['subjects'][0]
                assert e['subject_id']==m['subject_id']
                raw_product=m['pcc_fisher_mean']*m['coverage'] if m['pcc_fisher_mean'] is not None else None
                table.append(dict(protocol_version='v2',fold=fold,seed=seed,arm=arm,subject_id=e['subject_id'],
                    epochs=r['epochs'],best_epoch=r['best_epoch'],selection_score=v['selection']['score'],
                    record_optimal_PCC=v['record_optimal']['pcc']['mean'],
                    record_optimal_lag_median_ms=v['record_optimal']['lag_distribution_ms']['median'],
                    record_optimal_lag_q1_ms=v['record_optimal']['lag_distribution_ms']['q1'],
                    record_optimal_lag_q3_ms=v['record_optimal']['lag_distribution_ms']['q3'],
                    record_optimal_boundary_hits=v['record_optimal']['boundary_hits'],
                    record_count=len(v['record_optimal']['records']),M1=m['pcc_fisher_mean'],
                    M1_coverage=m['coverage'],M1_scored_coverage=m['scored_coverage'],M1_coverage_product=raw_product,
                    M1_matched=m['matched'],M1_reference=m['reference_count'],M1_scored=m['scored'],
                    M2=v['M2']['pcc']['mean'],M3=v['M3']['pcc']['mean'],
                    M3_lag_median_ms=v['M3']['lag_distribution_ms']['median'],
                    strict_beat_PCC=v['beat']['pcc']['mean'],RMSE=v['M2']['rmse']['mean'],
                    GT_RR_resampled_Shape200_PCC=v['shape']['pcc']['mean'],
                    **{k+'_mae_ms':e['fiducials'][k]['mae_ms'] for k in 'qrst'},
                    **{k+'_coverage':e['fiducials'][k]['coverage'] for k in 'qrst'},
                    head_R_F1=e['r_f1'],head_R_Se=e['r_se'],head_R_PPV=e['r_ppv'],
                    waveform_R_F1=m['waveform_r_f1'],waveform_R_Se=m['waveform_r_se'],waveform_R_PPV=m['waveform_r_ppv'],
                    predecode_order_violation=e['order_violation_rate'],
                    wave_head_consistency_mae_ms=e['consistency_mae_ms'],
                    wave_head_consistency_signed_median_ms=e['consistency_signed_median_ms'],
                    oracle=arm==4,result_path=str(directory/'result.json')))
    decision=screening_decision(table)
    atomic_csv(env.OUT/'screening_results.csv',list(table[0]),table)
    atomic_json(env.OUT/'screening_decision.json',decision)
    atomic_json(env.OUT/'summary.json',dict(protocol_version='v2',screening_only=True,
        statistical_evidence=False,all_runs=table,decision=decision,confirmation_authorized=False,test_access=False))
    lines=['# MMECG continuous screening v2','',
        '**Screening, NOT statistical evidence. Validation only; confirmation remains BLOCKED.**','',
        'All36runs are fresh v2 runs; no v1 results enter selection or gates. Record-optimal PCC uses one GT-selected ±150ms lag per record on fixed common support; it is NOT deployable native-time accuracy. M1 and waveform R-F1 use the same frozen detector and ±150ms matcher. M2/M3/strict beat PCC are report-only. Arm4 is GT-event oracle; arms1/4 are descriptive.','',
        'Checkpoint=subject-macro harmonic(max(0,M1)*matched coverage,waveform R-F1); no0.8eligibility floor; no patience in epochs1–10. Raw M1×coverage used for screening. Complete extra fields, per-record61lag curves and missingness remain in result.json/CSV.','']
    for fold in (1,2,3):
        lines += [f'## Fold{fold}','','|Seed|Arm|Record PCC±150ms|M1|Coverage|M1×coverage|M2|M3|Strict beat|RMSE|Q/R/S/T MAE ms|Head/wave R-F1|Best/final epoch|',
          '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|']
        for r in table:
            if r['fold']!=fold:continue
            mae='/'.join(fmt(r[k+'_mae_ms'],1) for k in 'qrst')
            lines.append(f"|{r['seed']}|{r['arm']}{' oracle' if r['oracle'] else ''}|{fmt(r['record_optimal_PCC'])}|{fmt(r['M1'])}|{r['M1_coverage']:.3f}|{fmt(r['M1_coverage_product'])}|{fmt(r['M2'])}|{fmt(r['M3'])}|{fmt(r['strict_beat_PCC'])}|{fmt(r['RMSE'])}|{mae}|{fmt(r['head_R_F1'])}/{fmt(r['waveform_R_F1'])}|{r['best_epoch']}/{r['epochs']}|")
        lines.append('')
    lines+=['## Locked all-conditions decision','',f"Positive record-PCC contrasts: {decision['positive_record_pcc_pairs']}/9; required≥8.",'',
        '|Subject / fold|Record-PCC seed-mean delta|M1×coverage delta|Coverage delta|Q/R/S/T MAE delta ms|',
        '|---|---:|---:|---:|---|']
    for r in decision['subject_seed_means']:
        mae='/'.join(fmt(r[k+'_mae_delta_ms'],2) for k in 'qrst')
        lines.append(f"|{r['subject_id']} / {r['fold']}|{fmt(r['record_pcc_delta'],4)}|{fmt(r['m1_coverage_product_delta'],4)}|{fmt(r['coverage_delta'],4)}|{mae}|")
    lines+=['','Subject-macro deltas: `'+json.dumps(decision['overall_subject_mean_deltas'])+'`.','',
        'Conditions: `'+json.dumps(decision['conditions'])+'`.','',
        f"**Screening {'PASS' if decision['screening_pass'] else 'FAIL'}.** No automatic expansion. No statistical superiority claim."]
    atomic_text(env.OUT/'REPORT.md','\n'.join(lines)+'\n')
    atomic_json(env.OUT/'COMPLETE.json',dict(passed=True,protocol_version='v2',runs=36,
        screening_pass=decision['screening_pass'],statistical_evidence=False,
        confirmation_11fold_authorized=False,test_access=False,metric_freeze_sha256=sha(env.OUT/'METRIC_FREEZE.json')))

if __name__=='__main__':generate()
