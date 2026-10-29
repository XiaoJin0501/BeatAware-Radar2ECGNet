import importlib.util
from pathlib import Path
import numpy as np
import torch
from experiments.phase_resampling_20261027 import frozen_bridge as frozen
from experiments.phase_resampling_20261027.architecture import soft_refine

p=Path(__file__).resolve().parents[1]/'docs/paper/sections/figures/plot_phase_pd_bland_altman_v2.py'
s=importlib.util.spec_from_file_location('ba_v2',p)
v2=importlib.util.module_from_spec(s);s.loader.exec_module(v2)

def test_hr_attribution_exclusive_and_overlapping_categories():
    match=frozen.clinical.match_beats
    assert v2.counts([100,300,500],[100,500],match)==(1,0,'missed_only')
    assert v2.counts([100,300],[100,300,500],match)==(0,1,'extra_only')
    assert v2.counts([100,300],[100,500],match)==(1,1,'both')
    assert v2.counts([100,300],[105,305],match)==(0,0,'neither')

def test_timing_mismatch_can_be_both_without_net_count_change():
    assert v2.counts([100,300],[150,350],frozen.clinical.match_beats)==(2,2,'both')

def test_probability_inverse_recovers_native_refinement():
    logits=torch.linspace(-3,2,1600)
    p=np.tile(logits.sigmoid().numpy(),(4,1))
    actual=v2.refined_r(p,lambda p:{'r':np.array([100.,700.])})
    expected=np.array([float(soft_refine(logits,r)) for r in (100,700)])
    assert np.allclose(actual,expected,atol=5e-5,rtol=0)

def test_subject_slope_and_bootstrap_fit_displayed_points():
    result=v2.subject_slope([1,2,3,4],[5,7,9,11],repeats=1000)
    assert np.isclose(result['subject_mean_slope'],2)
    assert np.isclose(result['subject_mean_intercept'],3)
    assert np.isclose(result['subject_mean_slope_ci_low'],2)
    assert np.isclose(result['subject_mean_slope_ci_high'],2)
    assert result['regression_line_drawn']
    assert result['subject_mean_bootstrap_valid']<=1000

def test_subject_slope_zero_does_not_draw_line():
    result=v2.subject_slope([1,2,3,4],[0,0,0,0],repeats=1000)
    assert result['subject_mean_slope']==0
    assert not result['regression_line_drawn']

def test_added_subject_slope_keeps_original_beat_statistics():
    row=dict(dataset='mmecg',metric='rr',slope='123',slope_ci_low='100',slope_ci_high='140',bias='9')
    pairs=[]
    for subject,n,x,y in [('1',100,1,2),('2',2,2,1),('3',2,3,3)]:
        pairs.extend(dict(dataset='mmecg',metric='rr',subject=subject,ref=x-y/2,pred=x+y/2) for _ in range(n))
    out=v2.add_slopes(pairs,[row])[0]
    assert all(out[k]==v for k,v in row.items())
    assert out['beat_slope']=='123'
    assert np.isclose(out['subject_mean_slope'],.5)

def test_clear_lower_annotation_is_preferred_without_rescaling():
    fig=v2.plt.figure(figsize=(4,3));ax=fig.add_axes([.15,.2,.6,.7]);side=fig.add_axes([.78,.2,.15,.7],sharey=ax)
    ax.set_xlim(0,1);ax.set_ylim(-30,30)
    limit,position,y=v2.place_annotation(fig,ax,side,'Bias 0\nLoA [−10, 10]',[.5],[0],dict(loa_low=-10,loa_high=10))
    assert limit==30 and position=='lower_left'
    v2.plt.close(fig)
