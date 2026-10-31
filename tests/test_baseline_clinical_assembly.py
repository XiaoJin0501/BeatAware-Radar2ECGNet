"""Regression for complete8s synthesis before central clinical scoring."""
import importlib.util
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('baseline_clinical_eval',Path(__file__).resolve().parents[1]/'scripts/figures/evaluate_baseline_clinical_intervals.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)

def test_synthesis_keeps_pairs_entirely_outside_central_crop():
 r=np.arange(80,1600,160,dtype=float);start=1600
 lookup={int(round(start+(a+b)/2-800)):np.ones(200) for a,b in zip(r[:-1],r[1:])}
 y,covered,bad=e.assemble_native_pairs(r,start,lookup)
 assert bad==0
 assert np.all(y[80:1520]==1) and covered[80:1520].all()
 # Both intervals excluded by the erroneous early-centre crop must be present.
 assert y[120]==1 and y[1400]==1
 assert np.all(y[:80]==0) and np.all(y[1520:]==0)

def test_missing_or_invalid_pairs_keep_fixed_zero_fallback():
 r=np.array([100.,260.,900.]);start=1600
 y,covered,bad=e.assemble_native_pairs(r,start,{})
 assert bad==1 and not covered.any() and np.all(y==0)
