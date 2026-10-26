import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.phase_resampling_20261027.architecture import (
    AnchorConditionedREM,phase_alignment,phase_resample,predicted_pairs,choose_bounds,mix_fraction)
from experiments.phase_resampling_20261027.frozen_bridge import FrozenDualHeadREM,shape_target,losses

torch.set_num_threads(1)

def logits_at(peaks):
    x=torch.full((1,4,1600),-30.)
    for p in peaks:x[0,1,p]=30
    return x

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_identity_matches_existing_exclusive_target(dtype):
    x=np.sin(np.arange(1600)*.032).astype(np.float32)
    y=shape_target(x,np.array([700,900]))
    t=phase_alignment(torch.tensor([[700.,900.]],dtype=dtype))
    assert t.shape==(1,200) and t[0,0]==700 and t[0,-1]==899
    z=phase_resample(torch.tensor(x,dtype=dtype)[None,None],t)[0,0]
    z=(z-z.min())/(z.max()-z.min())
    assert np.max(abs(z.numpy()-y))<1e-3

def test_sampling_grid_and_probability_gradients_blocked():
    bounds=torch.tensor([[700.,900.]],requires_grad=True)
    h=torch.randn(1,128,1600,requires_grad=True)
    logits=torch.randn(1,4,1600,requires_grad=True)
    t=phase_alignment(bounds)
    (phase_resample(h,t).sum()+phase_resample(logits.sigmoid().detach(),t).sum()).backward()
    assert h.grad is not None and h.grad.abs().sum()>0
    assert bounds.grad is None and logits.grad is None and not t.requires_grad

@pytest.mark.parametrize('variant,channels',[('phase',128),('phase_p',132),('phase_pd',260)])
def test_shapes_and_real_encoder_gradient(variant,channels):
    torch.manual_seed(42)
    model=AnchorConditionedREM(variant,in_bins=1)
    refs=torch.tensor([[650.,850.]],requires_grad=True)
    out=model(torch.randn(1,1,8,24),refs,epoch_index=0)
    assert model.shape_head[0].in_channels==channels
    assert out['shape'].shape==(1,1,200) and out['anchor_logits'].shape==(1,4,1600)
    assert out['t_i'].dtype==torch.float32 and out['t_i'].device.type=='cpu'
    out['anchor_logits'].retain_grad();out['shape'].square().mean().backward()
    assert model.stem.net[0].weight.grad.abs().sum()>0
    assert model.backbone.blocks[0][0].weight.grad.abs().sum()>0
    assert all(p.grad is None for p in model.anchor_head.parameters())
    assert refs.grad is None and out['anchor_logits'].grad is None
    model.zero_grad(set_to_none=True)
    out=model(torch.randn(1,1,8,24),refs,epoch_index=0)
    batch=dict(shape=torch.rand_like(out['shape']),anchor=torch.rand_like(out['anchor_logits']))
    loss,_,_=losses(out,batch);loss.backward()
    assert model.anchor_head[0].weight.grad.abs().sum()>0

def test_interp_bitwise_and_checkpoint_compatibility():
    torch.manual_seed(42);old=FrozenDualHeadREM(in_bins=1).eval()
    new=AnchorConditionedREM('interp',in_bins=1).eval()
    new.load_state_dict(old.state_dict(),strict=True)
    x=torch.randn(2,1,8,24)
    with torch.no_grad():a=old(x);b=new(x)
    assert torch.equal(a['shape'],b['shape']) and torch.equal(a['anchor_logits'],b['anchor_logits'])

def test_predicted_pair_status_and_no_gt_fallback():
    x=logits_at([650,850,1050])
    a,v,status=predicted_pairs(x,torch.tensor([[850.,1050.]]))
    assert v.item() and status==['matched'] and torch.allclose(a,torch.tensor([[850.,1050.]]),atol=1e-3)
    b,_,status=predicted_pairs(x,torch.tensor([[650.,850.]]))
    assert torch.equal(a,b) and status==['valid_unmatched']
    p,v,status=predicted_pairs(logits_at([]))
    assert not v.item() and status==['fallback'] and torch.equal(p,torch.tensor([[800.,960.]]))
    p,v,status=predicted_pairs(logits_at([1500]))
    assert not v.item() and float(p[0,1])>1599
    z=phase_resample(torch.arange(1600.)[None,None],phase_alignment(p))
    assert torch.isfinite(z).all() and z.shape==(1,1,200) and z[0,0,-1]==1599

def test_schedule_and_eval_no_reference_leak():
    pred=torch.tensor([[800.,1000.],[800.,960.]])
    ref=torch.tensor([[700.,900.],[700.,900.]])
    valid=torch.tensor([True,False])
    assert mix_fraction(0,100)==0 and mix_fraction(15,100)==0
    assert np.isclose(mix_fraction(30,100),.5) and mix_fraction(45,100)==1
    b,a=choose_bounds(pred,valid,ref,True,30,100)
    assert torch.equal(b[0],torch.tensor([750.,950.])) and torch.equal(b[1],ref[1])
    b,a=choose_bounds(pred,valid,ref,False,0,100)
    assert torch.equal(b,pred)
    model=AnchorConditionedREM('phase_p',in_bins=1).eval()
    h=torch.randn(2,128,400);logits=torch.cat([logits_at([])]*2)
    with torch.no_grad():
        a=model.decode_features(h,logits,ref)
        b=model.decode_features(h,logits,ref+100)
    assert torch.equal(a['shape'],b['shape']) and torch.equal(a['t_i'],b['t_i'])
    with pytest.raises(ValueError):choose_bounds(pred,valid,ref,True,0,100,oracle=True)
