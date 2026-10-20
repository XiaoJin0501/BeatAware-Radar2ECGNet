"""Fixed D>C>B>A decision; no training or test access."""
import csv
import json
import math
import numpy as np
import bridge
import lock
from inputs import inventory
import regional_metrics as metrics


def baseline_regions():
    """Supplement saved arm1 only, preserving its original checkpoint selection."""
    dest=bridge.OUT/'baseline_regional.json'
    if dest.exists():return json.loads(dest.read_text())
    scores=[]
    for fold in (1,2,3):
        rows=inventory(fold);stats=bridge.old_data.statistics(fold)
        for seed in (42,43,44):
            records=[]
            source=bridge.old_environment.OUT/'runs'/f'fold_{fold:02d}'/f'seed_{seed}'/'arm_1'
            for row in rows:
                with np.load(source/'validation_predictions'/f"{row['record_id']}.npz") as z:
                    assert np.array_equal(z['starts'],row['starts'])
                    wave=bridge.old_metrics.stitch_wave(row['samples'],row['starts'],z['wave'],row['segments'])
                target=((row['_raw']['ecg']-stats['median'])/stats['iqr']).astype(np.float32)
                m1=bridge.old_metrics.supplementary_record(wave,target,row['_raw']['event_r'],288,row['samples']-288,row['subject_id'])['m1']
                records.append(dict(subject_id=row['subject_id'],regional=metrics.matched_regions(wave,target,m1)))
            scores.append(dict(fold=fold,seed=seed,regional=metrics.aggregate_regions(records),
                source=str(source/'result.json')))
    bridge.atomic_json(dest,scores)
    return scores


def table_row(result,variant,regional=None):
    v=result['validation'];m=v['M1']['subjects'][0];e=v['events_by_subject'][0]
    rg=regional or v['regional']
    return dict(fold=result['fold'],seed=result['seed'],subject_id=m['subject_id'],variant=variant,
        best_epoch=result['best_epoch'],epochs=result['epochs'],M1=m['pcc_fisher_mean'],
        coverage=m['coverage'],M1_coverage_product=m['pcc_fisher_mean']*m['coverage'] if m['pcc_fisher_mean'] is not None else None,
        qrs_pcc=rg['qrs_pcc']['mean'],t_pcc=rg['t_pcc']['mean'],amplitude_ratio=rg['amplitude_ratio']['mean'],
        M2=v['M2']['pcc']['mean'],M3=v['M3']['pcc']['mean'],strict_beat_pcc=v['beat']['pcc']['mean'],
        record_optimal_pcc=v['record_optimal']['pcc']['mean'],RMSE=v['M2']['rmse']['mean'],
        **{k+'_mae_ms':e['fiducials'][k]['mae_ms'] for k in 'qrst'},
        head_R_F1=e['r_f1'],waveform_R_F1=m['waveform_r_f1'],
        predecode_order_violation=e['order_violation_rate'],wave_head_consistency_mae_ms=e['consistency_mae_ms'])


def decision(table):
    index={(r['fold'],r['seed'],r['variant']):r for r in table}
    required={(f,s,v) for f in (1,2,3) for s in (42,43,44) for v in ('A','B','C','D')}
    if len(table)!=36 or set(index)!=required:raise ValueError('Exactly36 unique rows including9 saved baselines')
    decisions={}
    for variant in ('B','C','D'):
        pairs=[]
        for f in (1,2,3):
            for s in (42,43,44):
                a,b=index[f,s,variant],index[f,s,'A']
                pairs.append(dict(fold=f,seed=s,**{k:a[k]-b[k]
                    if a[k] is not None and b[k] is not None else None
                    for k in ('M1_coverage_product','qrs_pcc')}))
        complete=all(r[k] is not None and math.isfinite(r[k]) for r in pairs for k in ('M1_coverage_product','qrs_pcc'))
        mean={k:float(np.mean([np.mean([r[k] for r in pairs if r['fold']==f]) for f in (1,2,3)]))
            if complete else None for k in ('M1_coverage_product','qrs_pcc')}
        positive=sum(r['M1_coverage_product'] is not None and r['M1_coverage_product']>0 for r in pairs)
        decisions[variant]=dict(pairs=pairs,mean_gains=mean,positive_pairs=positive,
            passed=bool(complete and mean['M1_coverage_product']>=.01 and mean['qrs_pcc']>=.02 and positive>=7))
    selected=next((v for v in ('D','C','B') if decisions[v]['passed']),'A')
    return dict(selected=selected,priority=['D','C','B','A'],candidates=decisions,
                screening_only=True,statistical_evidence=False,new_designs_authorized=False)


def generate():
    lock.verify();extra={(r['fold'],r['seed']):r for r in baseline_regions()};table=[]
    for fold in (1,2,3):
        for seed in (42,43,44):
            base=bridge.old_environment.OUT/'runs'/f'fold_{fold:02d}'/f'seed_{seed}'/'arm_1'/'result.json'
            table.append(table_row(json.loads(base.read_text()),'A',extra[fold,seed]['regional']))
            for variant in ('B','C','D'):
                directory=bridge.OUT/'runs'/f'fold_{fold:02d}'/f'seed_{seed}'/variant
                if not (directory/'COMPLETE.json').exists():raise RuntimeError('Incomplete: '+str(directory))
                table.append(table_row(json.loads((directory/'result.json').read_text()),variant))
    verdict=decision(table)
    from src.utils.atomic_state import atomic_csv
    atomic_csv(bridge.OUT/'screening_results.csv',list(table[0]),table)
    bridge.atomic_json(bridge.OUT/'decision.json',verdict)
    bridge.atomic_json(bridge.OUT/'summary.json',dict(rows=table,decision=verdict))
    lines=['# Final resolution screen','',
        'Validation-only screening; not statistical evidence. A is reused v2 arm1. No test access.',
        'Record-optimal PCC uses GT-selected ±150ms shift and is diagnostic, not native-time reconstruction.',
        'All means in decision average seeds within subject first. Winner priority D > C > B > A.','',
        '|Fold|Seed|Variant|M1|Coverage|M1×coverage|QRS PCC|T PCC|Amplitude ratio|M2|M3|Record-optimal PCC|Strict beat PCC|RMSE|Q/R/S/T MAE ms|Head/wave R-F1|Best/final epoch|',
        '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    def fmt(v):return 'NA' if v is None else f'{v:.4f}'
    for r in table:
        fields=[r['fold'],r['seed'],r['variant']]+[fmt(r[k]) for k in ('M1','coverage','M1_coverage_product','qrs_pcc','t_pcc','amplitude_ratio','M2','M3','record_optimal_pcc','strict_beat_pcc','RMSE')]
        fields+=['/'.join(fmt(r[k+'_mae_ms']) for k in 'qrst'),fmt(r['head_R_F1'])+'/'+fmt(r['waveform_R_F1']),f"{r['best_epoch']}/{r['epochs']}"]
        lines.append('|'+'|'.join(map(str,fields))+'|')
    lines+=['',f"Selected final architecture: **{verdict['selected']}**.",'',
            'No further architecture experiment is authorized. Formal execution scope must match the author-approved specification.']
    bridge.atomic_text(bridge.OUT/'REPORT.md','\n'.join(lines)+'\n')
    bridge.atomic_json(bridge.OUT/'COMPLETE.json',dict(runs=27,baseline_rows=9,selected=verdict['selected'],test_access=False))


if __name__=='__main__':generate()
