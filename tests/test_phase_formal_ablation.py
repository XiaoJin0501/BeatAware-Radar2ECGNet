import sys
from pathlib import Path
import numpy as np
import pytest
import torch
import json

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.phase_resampling_20261027.architecture import AnchorConditionedREM
from experiments.phase_resampling_20261027 import formal_ablation as run
from experiments.phase_resampling_20261027 import ablation_statistics as stats

torch.set_num_threads(1)

@pytest.mark.parametrize('variant,channels',[('window_p',132),('phase_d',256)])
def test_new_factorial_paths_and_gradients(variant,channels):
    model=AnchorConditionedREM(variant,in_bins=1).eval()
    h=torch.randn(2,128,400,requires_grad=True);logits=torch.randn(2,4,1600,requires_grad=True)
    out=model.decode_features(h,logits)
    assert model.shape_head[0].in_channels==channels and out['shape'].shape==(2,1,200)
    out['shape'].square().mean().backward()
    assert h.grad is not None and h.grad.abs().sum()>0 and logits.grad is None
    if variant=='window_p':assert 't_i' not in out
    else:assert out['t_i'].shape==(2,200)

def test_single_task_losses_and_selection():
    vals=[torch.tensor(10.),torch.tensor(2.),torch.tensor(3.)]
    assert run.effective_loss('S1',vals)==2
    assert torch.isclose(run.effective_loss('S2',vals),torch.tensor(.9))
    assert run.effective_loss('A4',vals)==10
    r=dict(shape=dict(pcc=.8,rmse=.1,r2=.5),loss=dict(anchor=.2),independent=dict(macro_f1_100ms=.7))
    assert run.selection('S1',r)==(.8,-.1,.5)
    assert run.selection('S2',r)==(.7,-.2,0.)

@pytest.mark.parametrize('arm,inactive',[('S1','anchor_head'),('S2','shape_head')])
def test_single_task_inactive_head_has_no_gradient(arm,inactive):
    m=run.model_for('crrvs',arm);out=m(torch.randn(1,1,8,24))
    batch=dict(shape=torch.rand_like(out['shape']),anchor=torch.rand_like(out['anchor_logits']))
    run.effective_loss(arm,run.frozen.losses(out,batch)).backward()
    assert all(p.grad is None or p.grad.abs().sum()==0 for p in getattr(m,inactive).parameters())
    assert m.stem.net[0].weight.grad.abs().sum()>0

def test_factorial_formula_subject_bootstrap_and_rng():
    subjects={a:{str(i):dict(pcc=b+i*.001) for i in range(11)} for a,b in [('A0',.8),('A1',.81),('A2',.82),('A4',.85)]}
    d=stats.contrast(subjects,{'A0':-.5,'A1':-.5,'A2':.5,'A4':.5},'pcc')
    assert np.allclose(d,.03)
    np.random.seed(13);before=np.random.get_state();ci=stats.bootstrap_mean(d);after=np.random.get_state()
    assert np.array_equal(before[1],after[1]) and np.allclose(ci,(.03,.03,.03))
    assert stats.signed_rank(np.zeros(11))==1
    assert np.allclose(stats.holm([.01,.04,.03,.2]),[.04,.09,.09,.2])

def test_test_barrier_prevents_early_access(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'OUT',tmp_path)
    with pytest.raises(RuntimeError,match='all112'):run.assert_test_barrier()

def test_missing_subjects_not_silently_dropped():
    with pytest.raises(ValueError):stats.contrast({'a':{'1':dict(pcc=.5)},'b':{'2':dict(pcc=.6)}},{'a':1,'b':-1},'pcc')

def test_complete_synthetic_table_exports(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'OUT',tmp_path)
    for ds,nfold in [('mmecg',11),('crrvs',5)]:
        for fold in range(1,nfold+1):
            for ai,arm in enumerate(run.ARMS):
                d=run.run_dir(ds,fold,arm);d.mkdir(parents=True)
                subjects={}
                ids=[fold] if ds=='mmecg' else [(fold-1)*6+j for j in range(1,7)]
                for sid in ids:
                    row=dict(pcc=.80+.005*ai+.001*sid if arm!='S2' else None,rmse=.1 if arm!='S2' else None,n_windows=10,
                        fallback_rate=.01 if arm in {'A2','A3','A4'} else None,valid_unmatched_rate=.1 if arm in {'A2','A3','A4'} else None)
                    for event in 'qrst':
                        row[event+'_mae_ms']=20+ai if arm!='S1' else None
                        row[event+'_eligible']=100 if arm!='S1' else None
                        row[event+'_evaluable']=90 if arm!='S1' else None
                        row[event+'_coverage']=.9 if arm!='S1' else None
                    subjects[str(sid)]=row
                (d/'test_subjects.json').write_text(json.dumps(subjects))
                (d/'selected.json').write_text(json.dumps(dict(selected_epoch=10,reused=False,parameters_allocated=100,parameters_active=90)))
    run.collect()
    for name in ['Table_modules.tex','Table_factorial.tex','Table_multitask.tex','Table_S1.tex','per_subject.csv','REPORT.md']:
        assert (tmp_path/name).exists()
    assert r'\multicolumn{11}' in (tmp_path/'Table_multitask.tex').read_text()
    assert 'phase' in (tmp_path/'table2_factorial.csv').read_text()
