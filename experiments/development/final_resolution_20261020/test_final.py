import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
import bridge
import numpy as np
import pytest
import torch
from architecture import FinalREM, counts, waveform_loss, objective
from regional_metrics import matched_regions, aggregate_regions
from inputs import aligned_td

torch.set_num_threads(1)


def test_common_parameters_match_baseline():
    for seed in (42,43,44):
        base=bridge.old_model.initialize_arm(1,seed).state_dict()
        for variant in ('A','B','C','D'):
            model=FinalREM(variant,seed)
            assert all(torch.equal(model.state_dict()[k],v) for k,v in base.items())


def test_budget_and_CR_shape():
    for channels in (50,1):
        model=FinalREM('D',42,channels).eval()
        assert counts(model)['added_fraction_of_backbone']<.15
        with torch.no_grad():
            out=model(torch.randn(1,channels,8,12),torch.randn(1,channels,1600))
        assert out['wave'].shape==(1,1,1600)
        assert out['anchor_logits'].shape==(1,4,1600)


def test_event_path_unchanged():
    a,b=FinalREM('B').eval(),FinalREM('D').eval()
    x=torch.randn(1,50,8,12)
    with torch.no_grad():
        left=a(x);right=b(x,torch.randn(1,50,1600))
    assert torch.equal(left['anchor_logits'],right['anchor_logits'])


def test_fusion_gradients():
    model=FinalREM('D').eval()
    out=model(torch.randn(1,50,8,12),torch.randn(1,50,1600))
    loss,_=objective(out,torch.randn(1,1,1600),torch.rand(1,4,1600),'D')
    loss.backward()
    for module in (model.time_path,model.fusion_gate,model.time_projection,model.global_path,model.anchor_head):
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())


def test_shift_50ms_tolerance():
    x=torch.arange(1600,dtype=torch.float32)
    y=sum(torch.exp(-((x-p)/width)**2) for p,width in ((300,5),(650,9),(1000,6),(1300,8)))[None,None]
    shifted=torch.zeros_like(y);shifted[...,10:]=y[...,:-10]
    original=waveform_loss(y,y,True);delayed=waveform_loss(y,shifted,True)
    # A finite soft-min lag set is approximately, not exactly, shift invariant:
    # translating changes its tail candidates even when the exact match remains.
    assert abs(float(original-delayed))<1e-3
    assert waveform_loss(y,shifted,False)>waveform_loss(y,y,False)+.05


def test_softmin_matches_enumeration():
    from architecture import per_sample_loss
    p,y=torch.randn(2,1,1600),torch.randn(2,1,1600)
    values=torch.stack([per_sample_loss(p[...,20:1580],y[...,20+lag:1580+lag]) for lag in range(-20,21)],1)
    expected=(-.1*torch.logsumexp(-values/.1,1)).mean()
    assert torch.allclose(waveform_loss(p,y,True),expected,atol=1e-6)


def test_ordinary_objective_exact():
    output=dict(wave=torch.randn(2,1,1600),anchor_logits=torch.randn(2,4,1600))
    y,e=torch.randn(2,1,1600),torch.rand(2,4,1600)
    assert torch.equal(objective(output,y,e,'C')[0],bridge.old_model.objective(output,y,e)[0])


def test_regional_identity_gain_and_short_RR():
    y=np.random.default_rng(8).normal(size=2000)
    m1=dict(rows=[dict(subject_id=1,gt_r=800.,predicted_r=800.,rr_reference_samples=100.)])
    rows=matched_regions(y*2,y,m1)
    assert rows[0]['qrs_pcc']==pytest.approx(1.)
    assert rows[0]['t_pcc']==pytest.approx(1.)
    assert rows[0]['amplitude_ratio']==pytest.approx(2.)
    assert rows[0]['t_scale']==pytest.approx(75/90)
    flat=matched_regions(np.zeros_like(y),y,m1)
    assert flat[0]['qrs_pcc']==0
    summary=aggregate_regions([dict(subject_id=1,regional=rows)])
    assert summary['qrs_pcc']['mean']==pytest.approx(1.,abs=1e-6)


def test_native_support_no_phase_shift():
    radar=np.tile(np.arange(4000,dtype=np.float32),(50,1))
    result=aligned_td('MMECG',radar,500,2100)
    expected=bridge.radar_window('MMECG',radar,500)
    assert np.array_equal(result,expected)
    with pytest.raises(ValueError):aligned_td('MMECG',radar,500,2099)


def test_missing_radar_rejected():
    with pytest.raises(ValueError):
        FinalREM('D')(torch.randn(1,50,8,12))


def test_decision_priority_and_fallback():
    from report_final import decision
    rows=[dict(fold=f,seed=s,variant=v,M1_coverage_product=.7+(.02 if v in ('B','C') else 0),
        qrs_pcc=.7+(.03 if v in ('B','C') else 0))
        for f in (1,2,3) for s in (42,43,44) for v in ('A','B','C','D')]
    assert decision(rows)['selected']=='C'
    for row in rows:
        if row['variant']=='D':row['M1_coverage_product']=.73;row['qrs_pcc']=.74
    assert decision(rows)['selected']=='D'
    for row in rows:row['M1_coverage_product']=.7
    assert decision(rows)['selected']=='A'


def test_decision_no_missing_or_duplicate_runs():
    from report_final import decision
    with pytest.raises(ValueError):decision([])
