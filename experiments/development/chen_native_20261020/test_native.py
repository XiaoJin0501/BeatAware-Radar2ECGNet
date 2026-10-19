import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import inspect
import bridge
import numpy as np
import torch
import pytest
from model import ChenNativeAR,Decoder,mulaw_encode,mulaw_decode,objective,BOS
from evaluate import score_record,aggregate,infer_record

torch.set_num_threads(1)

def test_mulaw_monotone_fullrange_and_finite():
    x=torch.linspace(-1,1,10001);q=mulaw_encode(x);back=mulaw_decode(q)
    assert q[0]==0 and q[-1]==255 and torch.all(q[1:]>=q[:-1])
    assert torch.max(abs(back-x))<.022 and torch.isfinite(back).all()
    assert mulaw_encode(torch.tensor([-5.,5.])).tolist()==[0,255]

def test_cached_decoder_is_parallel_teacher_forced_decoder():
    torch.manual_seed(7);model=Decoder().eval()
    previous=torch.randint(0,256,(2,640));previous[:,0]=BOS
    cond=torch.randn(2,4,640)
    with torch.no_grad():
        parallel=model(previous,cond)
        state=model.new_state(2,'cpu',torch.float32)
        cached=torch.stack([model.step(previous[:,i],cond[:,:,i],state) for i in range(640)],-1)
    assert torch.allclose(cached,parallel,atol=2e-6,rtol=1e-5)
    assert model.receptive_field==512

def test_decoder_future_history_cannot_affect_past():
    model=Decoder().eval();codes=torch.zeros(1,640,dtype=torch.long);cond=torch.randn(1,4,640)
    with torch.no_grad():
        before=model(codes,cond);codes[:,400:]=128;after=model(codes,cond)
    assert torch.equal(before[:,:,:400],after[:,:,:400])

def test_model_shapes_real640_and_native1600():
    m=ChenNativeAR().eval();positions=torch.zeros(1,50,3)
    with torch.no_grad():
        for t in (640,1600):assert m.encoder(torch.zeros(1,50,t),positions).shape==(1,4,t)
        assert m(torch.zeros(1,50,640),positions,torch.zeros(1,640,dtype=torch.long)).shape==(1,256,640)

def test_target_not_fed_at_current_sample():
    m=ChenNativeAR().eval();x=torch.zeros(1,50,640);p=torch.zeros(1,50,3)
    codes=torch.zeros(1,640,dtype=torch.long)
    with torch.no_grad():
        a=m(x,p,codes);codes[:,550:]=200;b=m(x,p,codes)
    assert torch.equal(a[:,:,:551],b[:,:,:551])

def test_cross_entropy_only_last128():
    logits=torch.randn(2,256,640,requires_grad=True);codes=torch.zeros(2,640,dtype=torch.long)
    objective(logits,codes).backward()
    assert torch.count_nonzero(logits.grad[:,:,:512])==0
    assert torch.count_nonzero(logits.grad[:,:,512:])>0

def test_generate_is_eval_only_and_no_GT_interface():
    assert list(inspect.signature(ChenNativeAR.generate).parameters)==['self','radar','positions']
    m=ChenNativeAR().train()
    with pytest.raises(RuntimeError):m.generate(torch.zeros(1,50,8),torch.zeros(1,50,3))

def test_actual_autoregressive_feedback_reaches_next_step(monkeypatch):
    model=ChenNativeAR().eval();seen=[]
    monkeypatch.setattr(model.encoder,'forward',lambda x,p:torch.zeros(1,4,8))
    def step(previous,condition,state):
        seen.append(int(previous[0]));o=torch.zeros(1,256);o[:,len(seen)]=1;return o
    monkeypatch.setattr(model.decoder,'step',step)
    q=model.generate(torch.zeros(1,50,8),torch.zeros(1,50,3))
    assert seen==[BOS,1,2,3,4,5,6,7] and q.tolist()==[[1,2,3,4,5,6,7,8]]

def synthetic():
    n=6000;t=np.arange(n);rs=np.arange(500,5500,200);y=np.zeros(n)
    for r in rs:
        for off,amp,sigma in [(-6,-.2,2),(0,1.,2),(7,-.2,2),(50,.3,8)]:
            y+=amp*np.exp(-.5*((t-r-off)/sigma)**2)
    raw=dict(ecg=y,**{'event_'+k:rs+off for k,off in zip('qrst',(-6,0,7,50))})
    return y,raw

def test_identical_wave_uses_frozen_metric_and_f1_exactly():
    y,raw=synthetic();stats={'median':0.,'iqr':1.}
    result=aggregate([score_record(y,raw,stats,1,'synthetic')])
    assert result['M1']['pcc']['mean']>.999999
    assert result['M1']['coverage']['mean']==1
    assert result['M2']['pcc']['mean']>.999999
    assert result['M3']['pcc']['mean']>.999999
    assert result['record_optimal']['pcc']['mean']>.999999
    assert result['strict_beat']['pcc']['mean']>.999999
    assert result['waveform_events']['r']['f1']['mean']==1
    assert result['events_are_waveform_not_head']

@pytest.mark.parametrize('n',[1600,2624,3000,35505])
def test_approved_stitch_is_sample_aligned(n):
    seg=bridge.approved_layout(n);starts=sorted({s['input_start'] for s in seg})
    outputs=np.array([np.arange(s,s+1600) for s in starts])
    wave=bridge.frozen_metrics.stitch_wave(n,starts,outputs,seg)
    np.testing.assert_array_equal(wave[288:n-288],np.arange(288,n-288))

def test_frozen_contract_and_GPU_is_never_initialized():
    assert bridge.verify_frozen()
    assert not torch.cuda.is_available() and not torch.cuda.is_initialized()

def test_infer_record_wrapper_is_aligned_including_tail(monkeypatch):
    import evaluate
    import data
    n=3000
    radar=np.tile(np.arange(n,dtype=np.float32),(50,1))
    dummy=torch.nn.Linear(1,1).eval()
    monkeypatch.setattr(data,'normalize_radar',lambda x:x)
    monkeypatch.setattr(evaluate,'infer_8s',lambda m,x,p,s:x[:,0].numpy())
    wave=evaluate.infer_record(dummy,radar,np.zeros((50,3),np.float32),1.)
    np.testing.assert_array_equal(wave[288:n-288],np.arange(288,n-288))

def test_train_only_scale_and_split_do_not_read_val_amplitude(monkeypatch):
    import data
    from types import SimpleNamespace
    reads=[]
    fake_stats={'median':0.,'iqr':2.,'train_subjects':[7]}
    monkeypatch.setattr(data.json,'loads',lambda s:fake_stats)
    # Real statistics JSON is read, but values are replaced for this unit test.
    monkeypatch.setattr(data.bridge,'split_rows',lambda ds,fold,split:[dict(record_id='tr' if split=='train' else 'va',subject_id=7 if split=='train' else 8)])
    def raw(rid):
        reads.append(rid);assert rid=='tr'
        return {'ecg':np.array([-8.,2.])}
    monkeypatch.setattr(data,'reference',raw)
    stats=data.statistics(1)
    assert reads==['tr'] and stats['mu_scale']==4. and stats['validation_subjects']==[8]
