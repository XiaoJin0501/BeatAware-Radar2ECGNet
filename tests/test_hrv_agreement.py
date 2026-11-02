"""Frozen HRV conventions and prediction-only stitching; no checkpoints required."""
import importlib.util
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parents[1]/'scripts/evaluate_hrv_agreement.py'
spec=importlib.util.spec_from_file_location('hrv_analysis',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_constant_rr_metrics_and_sample_sd():
 out=m.rr_metrics(np.ones(20))
 assert out==dict(hr=60.,sdnn=0.,rmssd=0.)
 x=np.array([.8,1.,1.2]);out=m.rr_metrics(x)
 assert np.isclose(out['sdnn'],200)
 assert np.isclose(out['rmssd'],200)
 assert out['hr']==60

def test_onepass_correction_and_inclusive_thresholds():
 rr=np.array([1.,1.,1.,1.2,1.,1.,1.3,1.,.2,1.,2.1,1.])
 r=np.r_[0,np.cumsum(rr)*200]
 actual,keep,physio,median=m.correct_rr(r)
 assert keep[3]  # Exactly20% is retained.
 assert not keep[6]  # In-range but >20% from local median.
 assert not physio[8] and not physio[10]
 assert np.all(median[physio]>0)
 assert np.isclose(actual[3],1.2)

def test_eighty_percent_boundary():
 r=np.arange(11)*200.
 keep=np.array([True]*8+[False]*2)
 prep=(np.ones(10),keep,np.ones(10,bool),np.ones(10))
 assert m.signal_segment(r,prep,0,2400)['acceptable']
 prep=(prep[0],np.array([True]*7+[False]*3),prep[2],prep[3])
 assert not m.signal_segment(r,prep,0,2400)['acceptable']

def test_stitch_ownership_tie_and_no_reference_needed():
 r,intervals=m.stitch([0,400],[np.array([999.,1000.]),np.array([599.,600.])])
 assert np.array_equal(r,[999.,1000.])
 assert m.covered(0,2000,intervals)
 assert not m.covered(0,4000,list(zip(*m.ownership([0,2400]))))

def test_segment_half_open_and_no_cross_boundary_rr():
 r=np.array([11800.,12000.,12200.,12400.]);prep=m.correct_rr(r)
 first=m.signal_segment(r,prep,0,12000)
 second=m.signal_segment(r,prep,12000,24000)
 assert first['n_rr']==0
 assert second['n_rr']==2
 assert second['hr']==60.

def test_corrupt_and_missing_predictions_are_not_silently_dropped():
 meta=dict(dataset='x',fold=1,subject=1,record='r',method='rem_net')
 starts=np.arange(0,12000-1600+1,400)
 rs=[np.array([]) for _ in starts]
 ref=np.arange(0,12000,200.)
 rows,inv=m.paired_rows(meta,12000,starts,rs,ref)
 assert len(rows)==1 and not rows[0]['used']
 assert 'pred_no_rr' in rows[0]['discard_reason']
 assert rows[0]['ref_acceptable']

def test_keep_rejected_interval_gap_convention():
 rr=np.array([1.,1.1,.9])
 assert np.isclose(m.rr_metrics(rr)['rmssd'],np.sqrt((.1**2+.2**2)/2)*1000)
