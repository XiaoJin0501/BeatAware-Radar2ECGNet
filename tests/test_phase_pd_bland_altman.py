import importlib.util
from pathlib import Path
import numpy as np

path=Path(__file__).resolve().parents[1]/'docs/paper/sections/figures/plot_phase_pd_bland_altman.py'
spec=importlib.util.spec_from_file_location('phase_pd_ba',path)
ba=importlib.util.module_from_spec(spec);spec.loader.exec_module(ba)

def test_repeated_loa_balanced_anova():
    d=np.array([-2.,0.,2.,8.,10.,12.]);s=np.array(['a']*3+['b']*3)
    bias,low,high,within,between=ba.repeated_loa(d,s)
    assert bias==5 and np.isclose(within,4)
    assert np.isclose(between,(150-4)/3)
    assert np.isclose(high-low,2*1.96*np.sqrt(within+between))

def test_roundtrip_exclusive_end_and_no_replication():
    x=np.r_[np.linspace(0,1,100),np.linspace(1,0,100),np.zeros(20)]
    y,covered=ba.roundtrip_signal(x,np.array([0,100,200]))
    assert np.allclose(y[:100],x[:100])
    assert np.allclose(y[100:200],x[100:200])
    assert covered[:200].all() and not covered[200:].any()

def test_endpoint_match_requires_both_endpoints_within_tolerance():
    p={k:np.array([400.]) for k in ba.FIELDS};p['r']=np.array([450.]);p['qrs_on']=np.array([440.]);p['qrs_off']=np.array([460.])
    q={k:v.copy() for k,v in p.items()};q['qrs_off']+=40
    rows=ba.make_rows('mmecg',1,1,'a',0,p,q,['qrs'])
    assert len(rows)==1 and np.isnan(rows[0]['pred'])

def test_dedup_never_selects_based_on_prediction_success():
    common=dict(dataset='mmecg',subject='1',record='a',ref_r=500,metric='rr',ref=800.)
    a=dict(**common,start=0,centre_distance=5.,pred=np.nan)
    b=dict(**common,start=10,centre_distance=15.,pred=800.)
    assert np.isnan(ba.dedup([b,a])[0]['pred'])
