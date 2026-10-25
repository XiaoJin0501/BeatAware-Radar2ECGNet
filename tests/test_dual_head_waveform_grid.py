import importlib.util
from pathlib import Path
import numpy as np

path=Path(__file__).resolve().parents[1]/'scripts/figures/plot_dual_head_waveform_grid.py'
spec=importlib.util.spec_from_file_location('waveform_grid',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_assembly_uses_only_prediction():
    shape=np.linspace(0,1,200)
    r=np.array([100,300,500,700,900,1100,1300,1500])
    wave=m.assemble(shape,r)
    assert np.allclose(wave[100:300],shape)
    assert wave.shape==(1600,) and np.isfinite(wave).all()
    assert np.array_equal(m.assemble(shape,[]),np.zeros(1600))


def test_reference_display_beatwise_minmax():
    wave=np.r_[np.arange(100),np.arange(100)*4+30]
    y=m.reference_display(wave,[0,100])
    assert np.allclose(y[:100],y[100:])
    assert np.isclose(m.pcc(y,y),1)
    assert m.pcc(np.zeros(100),np.arange(100))==0


def test_fixed_median_selection_not_best():
    rows=[]
    for ds,scenes in m.SCENES.items():
        for scene,_ in scenes:
            for subject,center in ((1,.1),(2,.5),(3,.9)):
                for i,value in enumerate((center-.05,center,center+.05)):
                    rows.append(dict(dataset=ds,scene=scene,subject=subject,record_id='r',
                        start=i*200,stop=i*200+1600,index=i,fold=1,window_pcc=value,predicted_r_count=8))
    selections,_=m.select(rows)
    assert len(selections)==7
    assert all(r['subject']==2 and r['index']==1 for r in selections)


def test_archived_decoder_order_and_native_sample_coordinates():
    prob=np.zeros((4,1600))
    for k,points in enumerate(([180,580],[200,600],[220,620],[270,670])):
        prob[k,points]=1.
    points=m.decoder()(prob)
    assert np.array_equal(points['r'],[200,600])
    assert all(np.all(points[a]<points[b]) for a,b in zip('qrs','rst'))
