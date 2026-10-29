import csv
import numpy as np
from scripts.figures import plot_examples_v3 as ex

def test_selection_uses_subject_then_window_medians_with_fixed_ties(tmp_path,monkeypatch):
    monkeypatch.setattr(ex,'CACHE',tmp_path)
    monkeypatch.setattr(ex.old,'SCENES',{'mmecg':[('NB','Normal breathing')]})
    rows=[]
    for subject,values in [(1,[.25,.25]),(2,[.75,.5]),(3,[.875,.875])]:
        for index,value in enumerate(values):
            rows.append(dict(dataset='mmecg',scene='NB',subject=subject,window_pcc=value,fold=1,index=index,
                record_id='a' if index==0 else 'b',start=index*400,stop=index*400+1600))
    with (tmp_path/'selection_candidates.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    chosen,_=ex.select()
    assert chosen[0]['subject']==2 and chosen[0]['record_id']=='a'
    assert chosen[0]['scenario_median_pcc']==.625

def test_per_beat_score_uses_individual_correlations_and_retains_flat_failures():
    ref=np.tile(np.linspace(0,1,200,dtype=np.float32),(3,1))
    pred=np.stack([ref[0],-ref[1],np.zeros(200,dtype=np.float32)])
    scores=ex.score_shapes(pred,ref)
    assert np.allclose(scores,[1,-1,0],atol=1e-6)
