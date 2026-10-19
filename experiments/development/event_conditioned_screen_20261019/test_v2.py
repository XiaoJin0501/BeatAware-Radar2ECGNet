import copy
import numpy as np
import pytest
from v2_metrics import (checkpoint_score,patience_after_epoch,record_optimal_shift,
                        aggregate_record_lags,screening_decision)

def test_checkpoint_harmonic_exact_no_extra_floor():
    r=checkpoint_score(dict(subjects=[dict(subject_id=1,pcc_fisher_mean=.9,coverage=.7,waveform_r_f1=.8)]))
    assert np.isclose(r['score'],2*.63*.8/(.63+.8)) and r['score']>0

@pytest.mark.parametrize('m1',[None,-.5,0.])
def test_checkpoint_missing_negative_zero(m1):
    assert checkpoint_score(dict(subjects=[dict(subject_id=1,pcc_fisher_mean=m1,coverage=1.,waveform_r_f1=1.)]))['score']==0

def test_checkpoint_averages_subject_scores_not_products():
    sub=[dict(subject_id=1,pcc_fisher_mean=.9,coverage=1.,waveform_r_f1=.9),dict(subject_id=2,pcc_fisher_mean=.5,coverage=0.,waveform_r_f1=.5)]
    assert np.isclose(checkpoint_score(dict(subjects=sub))['score'],.45)

def test_no_patience_in_first10():
    wait=0
    for epoch in range(1,11):
        wait=patience_after_epoch(epoch,False,wait)
        assert wait==0
    assert patience_after_epoch(11,False,wait)==1
    assert patience_after_epoch(20,True,9)==0

@pytest.mark.parametrize('lag',[-30,-12,0,4,30])
def test_record_lag_exact_and_fixed_support(lag):
    rng=np.random.default_rng(123);y=rng.normal(size=2400)
    p=np.roll(y,lag);r=record_optimal_shift(p,y)
    assert r['lag_samples']==lag and abs(r['pcc']-1)<1e-10
    assert len(r['curve'])==61 and r['common_stop']-r['common_start']==len(y)-636

def test_record_zero_variance_tie_zero_lag():
    r=record_optimal_shift(np.zeros(2400),np.ones(2400))
    assert r['lag_samples']==0 and r['pcc']==0

def test_record_aggregation_fisher_per_subject():
    rows=[dict(subject_id=1,record_id='a',pcc=.2,lag_ms=0,boundary_hit=False),dict(subject_id=1,record_id='b',pcc=.8,lag_ms=5,boundary_hit=False)]
    x=aggregate_record_lags(rows)['record_optimal']
    assert np.isclose(x['pcc']['mean'],np.tanh(np.mean(np.arctanh([.2,.8]))))

def make_rows():
    return [dict(protocol_version='v2',fold=f,subject_id=f+10,seed=s,arm=a,
        record_optimal_PCC=.4+.02*(a==3),M1=.9,M1_coverage=.9,
        M1_coverage_product=.81+.02*(a==3),M2=-.1-2*(a==3),M3=-1,strict_beat_PCC=-1,
        **{k+'_mae_ms':20.+4*(a==3) for k in 'qrst'})
        for f in (1,2,3) for s in (42,43,44) for a in (1,2,3,4)]

def test_approved_gates_reportonly_metrics_cannot_change_decision():
    rows=make_rows();r=screening_decision(rows)
    assert r['screening_pass'] and r['positive_record_pcc_pairs']==9
    assert not r['statistical_evidence'] and not r['confirmation_11fold_authorized']
    for x in rows:x['M2'],x['M3'],x['strict_beat_PCC']=100,-100,100
    assert screening_decision(rows)==r

@pytest.mark.parametrize('field,value',[('record_optimal_PCC',.39),('M1_coverage_product',.80),('q_mae_ms',100.),('M1_coverage',.6)])
def test_each_gate_rejects_failure(field,value):
    rows=make_rows()
    for r in rows:
        if r['arm']==3:r[field]=value
    assert not screening_decision(rows)['screening_pass']

def test_missing_endpoint_not_silently_dropped():
    rows=make_rows();rows[2]['q_mae_ms']=None
    assert not screening_decision(rows)['screening_pass']

def test_seven_positive_insufficient():
    rows=make_rows()
    for r in rows:
        if r['arm']==3 and r['fold']==1 and r['seed'] in (42,43):r['record_optimal_PCC']=.399
    decision=screening_decision(rows)
    assert decision['positive_record_pcc_pairs']==7 and not decision['screening_pass']

def test_v1_and_incomplete_not_accepted():
    rows=make_rows()
    with pytest.raises(ValueError):screening_decision(rows[:-1])
    rows[0]['protocol_version']='v1'
    with pytest.raises(ValueError):screening_decision(rows)

def test_complete_report_and_selected_epochs(tmp_path,monkeypatch):
    import v2_report as report
    from src.utils.atomic_state import atomic_json
    import json
    monkeypatch.setattr(report,'verify',lambda:None)
    monkeypatch.setattr(report.env,'OUT',tmp_path)
    atomic_json(tmp_path/'METRIC_FREEZE.json',{'synthetic':True})
    for r in make_rows():
        sid=r['subject_id'];cov=r['M1_coverage']
        m=dict(subject_id=sid,pcc_fisher_mean=r['M1'],coverage=cov,scored_coverage=cov,
            matched=90,reference_count=100,scored=90,waveform_r_f1=.9,waveform_r_se=.9,waveform_r_ppv=.9)
        e=dict(subject_id=sid,fiducials={k:dict(mae_ms=r[k+'_mae_ms'],coverage=.9) for k in 'qrst'},
            r_f1=.9,r_se=.9,r_ppv=.9,order_violation_rate=0.,consistency_mae_ms=2.,consistency_signed_median_ms=0.)
        v=dict(protocol_version='v2',events_by_subject=[e],M1={'subjects':[m]},selection={'score':.8},
            record_optimal=dict(pcc={'mean':r['record_optimal_PCC']},lag_distribution_ms={'median':0.,'q1':0.,'q3':0.},boundary_hits=0,records=[{}]),
            M2=dict(pcc={'mean':r['M2']},rmse={'mean':.1}),M3=dict(pcc={'mean':r['M3']},lag_distribution_ms={'median':0.}),
            beat={'pcc':{'mean':r['strict_beat_PCC']}},shape={'pcc':{'mean':.5}})
        path=tmp_path/'runs'/f"fold_{r['fold']:02d}"/f"seed_{r['seed']}"/f"arm_{r['arm']}"
        atomic_json(path/'COMPLETE.json',{'passed':True})
        atomic_json(path/'result.json',dict(protocol_version='v2',epochs=25,best_epoch=10,validation=v))
    report.generate()
    import csv
    with (tmp_path/'screening_results.csv').open() as f:table=list(csv.DictReader(f))
    assert len(table)==36 and all(r['best_epoch']=='10' for r in table)
    # M1 product is constructed from actual M1 and coverage, not synthetic gate-helper field.
    result=json.loads((tmp_path/'screening_decision.json').read_text())
    assert not result['screening_pass'] and result['M2_M3_strict_beat_not_used']
