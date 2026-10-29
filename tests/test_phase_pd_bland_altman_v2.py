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
