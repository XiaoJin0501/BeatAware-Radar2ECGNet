"""Stage0: anchor-conditioned phase sampling; no training/test entry point."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from experiments.phase_resampling_20261027.frozen_bridge import FrozenDualHeadREM,structured_decode

WIN=1600
BEAT=200

def phase_alignment(bounds):
    """ONLY returns t_i [B,200], in native sample coordinates.

    Existing target slices ECG[a:b], exclusive b. Hence last sample is b-1.
    Interpolation includes both endpoints of that SLICE, not boundary b itself.
    """
    bounds=bounds.detach()
    if bounds.ndim!=2 or bounds.shape[1]!=2:raise ValueError('bounds must be [B,2]')
    if not torch.isfinite(bounds).all() or not torch.all(bounds[:,1]-bounds[:,0]>1):
        raise ValueError('non-finite or non-increasing exclusive beat bounds')
    phi=torch.linspace(0,1,BEAT,device=bounds.device,dtype=bounds.dtype)
    return bounds[:,:1]+phi*(bounds[:,1:]-bounds[:,:1]-1)

def phase_resample(signal,t_i):
    """Independent Time-to-Phase Warping, with border padding for fallback."""
    if signal.ndim!=3 or signal.shape[-1]!=WIN:raise ValueError('signal must be [B,C,1600]')
    if t_i.shape!=(len(signal),BEAT):raise ValueError('t_i must be [B,200]')
    t_i=t_i.detach().to(device=signal.device,dtype=signal.dtype)
    g=2*t_i/(WIN-1)-1
    grid=torch.stack((g,torch.zeros_like(g)),dim=-1)[:,None]
    return F.grid_sample(signal[:,:,None,:],grid,mode='bilinear',
                         padding_mode='border',align_corners=True)[:,:,0]

def soft_refine(r_logits,r):
    r=int(r);lo=max(0,r-10);hi=min(WIN,r+11)
    positions=torch.arange(lo,hi,device=r_logits.device,dtype=r_logits.dtype)
    # tau=1, inclusive ±10 samples, detached from shape supervision.
    return (torch.softmax(r_logits.detach()[lo:hi],dim=0)*positions).sum().detach()

def predicted_pairs(logits,reference_bounds=None):
    """Decoder, nearest-centre R + next R, fixed soft argmax and GT-free fallback.

    Reference bounds only label the status; NEVER change pair generation.
    """
    detached=logits.detach();prob=torch.sigmoid(detached).cpu().numpy()
    pairs=[];valid=[];statuses=[]
    for i,p in enumerate(prob):
        r=structured_decode(p)['r'];rr=np.diff(r)
        k=int(np.argmin(abs(r-800))) if len(r) else None
        first=soft_refine(detached[i,1],r[k]) if k is not None else detached.new_tensor(800.)
        second=soft_refine(detached[i,1],r[k+1]) if k is not None and k+1<len(r) else None
        good=second is not None and bool(torch.isfinite(first)) and bool(torch.isfinite(second)) and 60<=float(second-first)<=400
        if not good:
            duration=float(np.median(rr)) if len(rr) else 160.
            second=first+duration
        pair=torch.stack((first,second)).detach();pairs.append(pair);valid.append(good)
        matched=good and reference_bounds is not None and bool(torch.all(abs(pair-reference_bounds[i].detach().to(pair))<=30))
        statuses.append('matched' if matched else ('valid_unmatched' if good else 'fallback'))
    return torch.stack(pairs),torch.tensor(valid,device=logits.device,dtype=torch.bool),statuses

def mix_fraction(epoch_index,total_epochs):
    if total_epochs<=0 or epoch_index<0:raise ValueError('epochs must be nonnegative; E>0')
    return float(np.clip((epoch_index/total_epochs-.15)/.30,0,1))

def choose_bounds(pred,valid,reference,training,epoch_index,total_epochs,oracle=False):
    if oracle:
        if training:raise ValueError('oracle is evaluation-only')
        if reference is None:raise ValueError('oracle needs reference bounds')
        return reference.detach(),1.
    if not training:return pred.detach(),1.
    if reference is None:raise ValueError('training needs target-construction reference bounds')
    ref=reference.detach().to(pred);alpha=mix_fraction(epoch_index,total_epochs)
    safe=torch.where(valid[:,None],pred.detach(),ref)
    return ((1-alpha)*ref+alpha*safe).detach(),alpha

class AnchorConditionedREM(FrozenDualHeadREM):
    def __init__(self,shape_input='phase_p',in_bins=50,width=128,head_width=64,dropout=.1):
        if shape_input not in {'interp','phase','phase_p','phase_pd'}:raise ValueError(shape_input)
        super().__init__('qrst',in_bins,width,head_width,dropout)
        self.shape_input=shape_input
        channels={'interp':width,'phase':width,'phase_p':width+4,'phase_pd':2*width+4}[shape_input]
        if channels!=width:
            self.shape_head[0]=nn.Conv1d(channels,head_width,5,padding=2,bias=False)

    def decode_features(self,latent,logits,reference_bounds=None,epoch_index=0,total_epochs=100,oracle=False):
        if self.shape_input=='interp':
            return dict(shape=torch.sigmoid(self.shape_head(F.interpolate(latent,size=200,mode='linear',align_corners=False))),
                        anchor_logits=logits)
        pred,valid,status=predicted_pairs(logits,reference_bounds)
        bounds,alpha=choose_bounds(pred,valid,reference_bounds,self.training,epoch_index,total_epochs,oracle)
        t_i=phase_alignment(bounds)
        full_h=F.interpolate(latent,size=1600,mode='linear',align_corners=True)
        h_beat=phase_resample(full_h,t_i)
        p_det=torch.sigmoid(logits).detach()
        p_beat=phase_resample(p_det,t_i)
        components=[h_beat]
        if self.shape_input in {'phase_p','phase_pd'}:components.append(p_beat)
        if self.shape_input=='phase_pd':components.append(F.interpolate(latent,size=200,mode='linear',align_corners=False))
        return dict(shape=torch.sigmoid(self.shape_head(torch.cat(components,dim=1))),anchor_logits=logits,
                    t_i=t_i,anchor_bounds=bounds,predicted_bounds=pred,anchor_valid=valid,anchor_status=status,
                    anchor_mix_alpha=alpha,outside_grid_fraction=((t_i<0)|(t_i>1599)).float().mean(1),oracle=oracle)

    def forward(self,wsst,reference_bounds=None,epoch_index=0,total_epochs=100,oracle=False):
        if wsst.ndim!=4 or wsst.shape[1]!=self.in_bins:raise ValueError('input channel/shape mismatch')
        latent=self.backbone(self.stem(wsst))
        # Event path is EXACTLY the incumbent path: align_corners=False.
        logits=self.anchor_head(F.interpolate(latent,size=1600,mode='linear',align_corners=False))
        return self.decode_features(latent,logits,reference_bounds,epoch_index,total_epochs,oracle)
