import sys
from pathlib import Path
import numpy as np

FOLDER=Path(__file__).resolve().parents[1]/'docs/paper/sections/figures'
sys.path.insert(0,str(FOLDER))
import plot_anchor_robustness_delta as m

def test_common_and_differential_error_decomposition():
    v=m.error_components(np.array([[20.,20.],[20.,-20.],[-5.,-10.]]))
    assert np.array_equal(v['rr_span_error_ms'],[0,40,5])
    assert np.array_equal(v['anchor_error_ms'],[20,20,7.5])
    assert np.array_equal(v['common_abs_ms'],[20,0,7.5])
    assert np.array_equal(v['differential_abs_ms'],[0,20,2.5])
    assert np.array_equal(v['same_sign'],[1,0,1])

def test_paired_ci_is_not_difference_of_marginal_intervals():
    context=np.linspace(.5,.95,30);pred=context+.01
    mean,lo,hi=m.old.bootstrap_mean(pred-context)
    assert np.allclose([mean,lo,hi],[.01,.01,.01])
    assert (m.old.bootstrap_mean(context)[2]-m.old.bootstrap_mean(context)[1])>.01

def test_subject_balanced_median_is_not_pooled_window_median():
    arrays=[np.array([0.]),np.full(9,100.)]
    assert m.subject_balanced_median(arrays)==0
    assert np.median(np.concatenate(arrays))==100
