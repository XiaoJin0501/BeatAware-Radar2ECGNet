"""Author-approved v2: selection and screen only; all v1 endpoints unchanged."""
import numpy as np
from metrics import continuous_pair, fisher_mean, distribution, stitch_wave

def checkpoint_score(m1):
    """Subject-macro harmonic mean of max(0,M1)*coverage and waveform R-F1.

    No 0.8 coverage eligibility floor or extra F1 gate: final author version.
    Missing/nonpositive conditional morphology scores zero for selection; it is
    still recorded missing and cannot satisfy the screening completeness guard.
    """
    subjects=[]
    for s in m1['subjects']:
        raw=s['pcc_fisher_mean']
        value=max(0.,float(raw)) if raw is not None and np.isfinite(raw) else 0.
        product=value*s['coverage'];f1=s['waveform_r_f1']
        harmonic=2*product*f1/(product+f1) if product+f1>0 else 0.
        subjects.append(dict(subject_id=s['subject_id'],M1=raw,coverage=s['coverage'],
            M1_coverage_product=product,waveform_R_F1=f1,score=harmonic))
    if not subjects:raise ValueError('No validation subjects')
    return dict(score=float(np.mean([s['score'] for s in subjects])),subjects=subjects,
                rule='mean_subject harmonic(max(0,M1)*matched_coverage,waveform_R_F1)')

def patience_after_epoch(epoch,improved,previous):
    # Epochs1..10 consume no patience, including when the chosen epoch is earlier.
    return 0 if improved or epoch<=10 else previous+1

def record_optimal_shift(prediction,reference,valid_start=288,valid_stop=None,max_lag=30):
    """One GT-selected lag per entire retained record, fixed common support.

    200Hz:±30samples=±150ms; default y[318:L-318], p[318+k:L-318+k].
    Positive lag = prediction late. Ties smallest |k|, then negative k.
    This is reference-assisted selection/screen information, not blind inference.
    """
    p,y=np.asarray(prediction),np.asarray(reference)
    valid_stop=len(y)-288 if valid_stop is None else valid_stop
    if p.shape!=y.shape or p.ndim!=1 or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Shared finite native record required')
    a,b=valid_start+max_lag,valid_stop-max_lag
    if b-a<2:raise ValueError('Insufficient fixed common support')
    candidates=[dict(lag_samples=k,lag_ms=k*5,pcc=continuous_pair(p[a+k:b+k],y[a:b])['pcc'])
                for k in range(-max_lag,max_lag+1)]
    best=max(candidates,key=lambda r:(r['pcc'],-abs(r['lag_samples']),-r['lag_samples']))
    return dict(**best,zero_lag_pcc_same_support=candidates[max_lag]['pcc'],
        boundary_hit=abs(best['lag_samples'])==max_lag,common_start=a,common_stop=b,
        curve=candidates,reference_assisted=True)

def record_metric(row,predictions,raw,stats):
    wave=stitch_wave(row['samples'],row['starts'],predictions,row['segments'])
    target=((raw['ecg']-stats['median'])/stats['iqr']).astype(np.float32)
    return dict(subject_id=row['subject_id'],record_id=row['record_id'],
                **record_optimal_shift(wave,target))

def aggregate_record_lags(rows):
    subjects=[]
    for sid in sorted({r['subject_id'] for r in rows}):
        group=[r for r in rows if r['subject_id']==sid]
        subjects.append(dict(subject_id=sid,records=len(group),
            pcc_fisher_mean=fisher_mean([r['pcc'] for r in group])))
    return dict(record_optimal=dict(subjects=subjects,
        pcc=distribution([s['pcc_fisher_mean'] for s in subjects]),records=rows,
        lag_distribution_ms=distribution([r['lag_ms'] for r in rows]),
        boundary_hits=sum(r['boundary_hit'] for r in rows),
        aggregation='equal-record Fisher-z within subject, equal-subject arithmetic mean',
        reference_assisted=True,max_lag_ms=150))

def screening_decision(rows):
    required={(f,s,a) for f in (1,2,3) for s in (42,43,44) for a in (1,2,3,4)}
    index={(r['fold'],r['seed'],r['arm']):r for r in rows}
    if len(rows)!=36 or set(index)!=required:raise ValueError('Exactly36 unique v2 rows required')
    if any(r.get('protocol_version')!='v2' for r in rows):raise ValueError('Old runs prohibited')
    pairs=[]
    for f in (1,2,3):
        for s in (42,43,44):
            a,b=index[f,s,3],index[f,s,2]
            if a['subject_id']!=b['subject_id']:raise ValueError('Unpaired validation subjects')
            delta=lambda key:float(a[key]-b[key]) if a[key] is not None and b[key] is not None and np.isfinite([a[key],b[key]]).all() else None
            pairs.append(dict(fold=f,seed=s,subject_id=a['subject_id'],
                record_pcc_delta=delta('record_optimal_PCC'),
                m1_coverage_product_delta=delta('M1_coverage_product'),
                coverage_delta=delta('M1_coverage'),
                **{k+'_mae_delta_ms':delta(k+'_mae_ms') for k in 'qrst'}))
    keys=('record_pcc_delta','m1_coverage_product_delta','coverage_delta',*(k+'_mae_delta_ms' for k in 'qrst'))
    means=lambda group:{k:float(np.mean([r[k] for r in group])) if all(r[k] is not None for r in group) else None for k in keys}
    subject_rows=[]
    for sid in sorted({r['subject_id'] for r in pairs}):
        g=[r for r in pairs if r['subject_id']==sid]
        if len(g)!=3:raise ValueError('Exactly three seeds within subject required')
        subject_rows.append(dict(subject_id=sid,fold=g[0]['fold'],**means(g)))
    overall=means(subject_rows)
    positive=sum(r['record_pcc_delta'] is not None and r['record_pcc_delta']>0 for r in pairs)
    guards={k:overall[k+'_mae_delta_ms'] is not None and overall[k+'_mae_delta_ms']<=5. for k in 'qrst'}
    conditions=dict(at_least8of9_record_pcc_positive=positive>=8,
        mean_record_pcc_gain_at_least_001=overall['record_pcc_delta'] is not None and overall['record_pcc_delta']>=.01,
        mean_M1_coverage_product_gain_at_least_001=overall['m1_coverage_product_delta'] is not None and overall['m1_coverage_product_delta']>=.01,
        anchor_mae_guards=guards,
        mean_coverage_degradation_at_most_002=overall['coverage_delta'] is not None and overall['coverage_delta']>=-.02)
    passed=all(v for k,v in conditions.items() if k!='anchor_mae_guards') and all(guards.values())
    return dict(protocol_version='v2',screening_only=True,statistical_evidence=False,
        pairs=pairs,subject_seed_means=subject_rows,overall_subject_mean_deltas=overall,
        positive_record_pcc_pairs=positive,conditions=conditions,screening_pass=passed,
        M2_M3_strict_beat_not_used=True,confirmation_11fold_authorized=False)
