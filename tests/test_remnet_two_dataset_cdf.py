import importlib.util
import sys
from pathlib import Path
import numpy as np

folder=Path(__file__).resolve().parents[1]/'scripts/figures'
sys.path.insert(0,str(folder))
spec=importlib.util.spec_from_file_location('two_dataset_cdf',folder/'plot_remnet_two_dataset_cdf.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_equal_subject_weights_not_pooled_beats():
    subjects={'one':{'r':[0,0,0,0]},'two':{'r':[100]}}
    x,F,ns,ne=m.empirical(subjects,'r')
    assert np.array_equal(x,[0,100])
    assert np.array_equal(F,[.5,1]) and ns==2 and ne==5
    assert m.quantile(x,F,.5)==0 and m.quantile(x,F,.9)==100


def test_observed_fractional_nodes_and_missing_subjects():
    x,F,ns,ne=m.empirical({'one':{'rr':[2.5,5,10]},'two':{'rr':[]}},'rr')
    assert np.array_equal(x,[0,2.5,5,10]) and ns==1 and ne==3
    assert F[0]==0 and F[-1]==1 and np.all(np.diff(F)>=0)
    assert m.quantile(x,F,.5)==5


def test_quantile_decimal_tie_and_tail():
    assert m.quantile(np.array([0,10,15]),np.array([.3,.9-1e-16,1]),.9)==10
    x,F,_,_=m.empirical({'s':{'rr':[0,200]}},'rr')
    assert m.quantile(x,F,.9)==200
