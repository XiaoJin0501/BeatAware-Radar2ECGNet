import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.phase_resampling_20261027.train_stage1 import gate_decision

def case(base=.9,p=.9,pd=.91,oracle=.92,r=20):
    def m(pc,rr):return dict(shape=dict(pcc=pc),direct=dict(r=dict(mae_ms=rr)))
    return dict(interp=m(base,20),phase_p=dict(validation=m(p,r),oracle=m(oracle,20)),
                phase_pd=dict(validation=m(pd,20),oracle=m(oracle,20)))

def test_phase_p_priority():
    assert gate_decision([case()]*3)['decision']=='phase_p'

def test_oracle_stop():
    assert gate_decision([case(oracle=.89)]*3)['decision'].startswith('Plan A')

def test_fallback_to_pd_and_equal_case_average():
    assert gate_decision([case(r=26)]*3)['decision']=='phase_pd'
    result=gate_decision([case(p=.87),case(p=.915),case(p=.915)])
    assert abs(result['variants']['phase_p']['pcc']-.9)<1e-9
    assert result['decision']=='phase_p'
