import importlib.util
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('anchor_robustness',ROOT/'docs/paper/sections/figures/plot_anchor_robustness.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_sigma_zero_is_exact_reference_and_no_fallback():
    ref=np.array([[600.,800.],[799.,1000.]])
    used,failed,attempts=m.jitter_pairs(ref,0,np.random.default_rng(0),np.zeros_like(ref))
    assert np.array_equal(used,ref) and not failed.any() and np.all(attempts==1)

def test_independent_gaussian_boundaries_and_reproducibility():
    ref=np.tile([600.,800.],(10000,1))
    used,failed,_=m.jitter_pairs(ref,20,np.random.default_rng(3),ref)
    again,_,_=m.jitter_pairs(ref,20,np.random.default_rng(3),ref)
    errors=used-ref
    assert np.array_equal(used,again) and not failed.any()
    assert abs(np.corrcoef(errors.T)[0,1])<.03
    assert np.allclose(errors.std(0),4,rtol=.03)
    assert abs(abs(errors).mean()*5-20*np.sqrt(2/np.pi))<.4

def test_exhaustion_uses_exact_supplied_predicted_fallback():
    ref=np.array([[100.,110.]])
    fallback=np.array([[1550.,1710.]])
    used,failed,attempts=m.jitter_pairs(ref,0,np.random.default_rng(0),fallback)
    assert failed.item() and attempts.item()==10 and np.array_equal(used,fallback)

def test_crossover_and_repeat_fisher_conventions():
    assert m.crossover([0,10,20],[2,1,-1])==15
    assert m.crossover([0,10],[-1,-2])==0
    assert m.crossover([0,10],[1,2]) is None
    assert np.isclose(m.fisher(np.array([.5,.9])),np.tanh(np.mean(np.arctanh([.5,.9]))))
