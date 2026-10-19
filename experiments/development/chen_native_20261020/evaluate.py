"""Native waveform metrics reuse actual frozen v2 and A modules unchanged."""
import bridge
import numpy as np
import torch
from model import mulaw_decode

def event_readout(wave,raw,start=288,stop=None):
    stop=len(wave)-288 if stop is None else stop
    detected=bridge.frozen_metrics.waveform_events(wave[start:stop])
    pred={k:np.asarray(detected[k],float)+start for k in 'qrst'}
    ref={k:np.asarray(raw['event_'+k],float) for k in 'qrst'}
    pred={k:x[np.isfinite(x)&(x>=start)&(x<stop)] for k,x in pred.items()}
    ref={k:x[np.isfinite(x)&(x>=start)&(x<stop)] for k,x in ref.items()}
    return pred,bridge.event_metrics(pred,ref)

def score_record(wave,raw,stats,subject_id,record_id):
    bridge.verify_frozen()
    target=((raw['ecg']-stats['median'])/stats['iqr']).astype(np.float32)
    if np.shape(wave)!=np.shape(target):raise ValueError('Record-native coordinates required')
    n=len(target);detected,events=event_readout(wave,raw)
    supplement=bridge.frozen_metrics.supplementary_record(wave,target,raw['event_r'],288,n-288,subject_id,detected['r'])
    beats=[dict(subject_id=subject_id,**r) for r in bridge.frozen_metrics.beat_metrics(wave,target,raw['event_r'],288,n-288)]
    record=dict(subject_id=subject_id,record_id=record_id,**bridge.frozen_v2.record_optimal_shift(wave,target))
    return dict(subject_id=subject_id,supplement=supplement,beats=beats,
        events=dict(subject_id=subject_id,events=events),record_optimal=record)

def aggregate(records):
    result=bridge.frozen_metrics.aggregate_supplement([r['supplement'] for r in records])
    result['strict_beat']=bridge.frozen_metrics.summarize_subjects([b for r in records for b in r['beats']])
    result.update(bridge.frozen_v2.aggregate_record_lags([r['record_optimal'] for r in records]))
    result['waveform_events']=bridge.summarize_events([r['events'] for r in records])
    result['selection']=bridge.frozen_v2.checkpoint_score(result['M1'])
    result['events_are_waveform_not_head']=True
    return result

@torch.inference_mode()
def infer_8s(model,radar,positions,scale):
    if radar.shape[-1]!=1600:raise ValueError('Exactly8s radar context')
    codes=model.generate(radar,positions)
    return (mulaw_decode(codes)*scale).cpu().numpy()

@torch.inference_mode()
def infer_record(model,radar,positions,scale):
    """No ECG argument, no GT initialization; state resets at each8s window."""
    from data import normalize_radar
    n=radar.shape[-1];segments=bridge.approved_layout(n)
    starts=sorted({r['input_start'] for r in segments});waves=[]
    device=next(model.parameters()).device
    for start in starts:
        x=torch.from_numpy(normalize_radar(radar[:,start:start+1600])[None]).to(device)
        pos=torch.from_numpy(np.asarray(positions,np.float32)[None]).to(device)
        waves.append(infer_8s(model,x,pos,scale)[0])
    return bridge.frozen_metrics.stitch_wave(n,starts,np.asarray(waves),segments)
