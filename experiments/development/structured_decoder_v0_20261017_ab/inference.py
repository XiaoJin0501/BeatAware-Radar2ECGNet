"""Frozen-weight inference on new native grids, with bounded-memory WSST."""
from __future__ import annotations
import concurrent.futures
import hashlib
import json
import os
import time

import bootstrap
core=bootstrap.activate()
import numpy as np
import h5py
import torch
import torch.nn.functional as F
from src.models.continuous.dual_head_rem import DualHeadREM
from src.models.baselines.dual_endpoint import DualEndpointBaseline
from src.utils.atomic_state import atomic_npz,atomic_json
from rawio import record_data,radar_window,cw,load_module,split_path,MM,text
from layout import approved_layout,dense_starts
from structured import representative_r

HERE,ROOT=bootstrap.HERE,bootstrap.ROOT
NAMES=('rem_net','mmecg_tcn','radarode_mtl','cross_modal_conformer','wavegru_net','attention_dcgan')
_legacy=None

def one_wsst(x):
    global _legacy
    if _legacy is None:
        _legacy=load_module(HERE/'precompute_wsst8s_mmecg.py','frozen_wsst')
    return np.stack([_legacy._wsst_full_one_bin(v,200,16).astype(np.float16) for v in x])

def record_features(row,starts,rcg,pool):
    """Reuse only exact key/input matches, never approximate neighboring frames."""
    if row['dataset']=='CR_RVS':
        return np.stack([cw._stft_map(v[0],200,128,112) for v in rcg]),dict(reused=0,computed=len(starts))
    pool_path=MM/f"processed/wsst8s_subject_pool/subject_{row['subject_id']:02d}.h5"
    features=np.empty((len(starts),50,71,240),np.float16); reused=[]
    with h5py.File(split_path('MMECG',row['fold'],'val'),'r') as native,h5py.File(pool_path,'r') as cache:
        assert cache.attrs['support']=='full_window_samples_0_1600'
        assert int(cache.attrs['fs'])==200 and int(cache.attrs['nv'])==16
        stored={(text(r),int(s)):i for i,(r,s) in enumerate(zip(cache['record_id'][:],cache['start_idx'][:]))}
        original={(text(r),int(s)):i for i,(r,s) in enumerate(zip(native['record_id'][:],native['start_idx'][:]))}
        for i,start in enumerate(starts):
            key=row['record_id'],int(start)
            if key not in stored or key not in original: continue
            assert np.array_equal(rcg[i],native['rcg'][original[key]]), 'Refusing inexact cached radar input'
            assert int(cache['end_idx'][stored[key]])==int(start)+1600
            features[i]=cache['wsst'][stored[key]]; reused.append(i)
    if reused:
        # Full 50-channel independent bitwise check for each record, before reuse.
        assert np.array_equal(one_wsst(rcg[reused[0]]),features[reused[0]]), 'WSST cache provenance changed'
    reused_set=set(reused); missing=[i for i in range(len(starts)) if i not in reused_set]
    for i,feature in zip(missing,pool.map(one_wsst,(rcg[j] for j in missing),chunksize=1)):
        features[i]=feature
    return features,dict(reused=len(reused),computed=len(missing),
                         cached_radar_checked_bitwise=True,cached_feature_checked_bitwise=bool(reused))

def checkpoint_path(dataset,fold,name):
    if name=='rem_net' and dataset=='MMECG':
        return ROOT/f'experiments_mmecg/q1_dual_head_balanced_full_loso_seed42/A1/fold_{fold:02d}/checkpoints/best.pt'
    tag=('q1_dual_endpoint_remnet_crrvs' if name=='rem_net' else
         'q1_dual_endpoint_baselines_'+('mmecg' if dataset=='MMECG' else 'crrvs'))
    return ROOT/f'experiments_mmecg/{tag}/seed_42/{name}/fold_{fold:02d}/checkpoints/best.pt'

def make_models(dataset,fold,device):
    models={}; provenance={}
    for name in NAMES:
        model=(DualHeadREM('qrst',in_bins=50 if dataset=='MMECG' else 1) if name=='rem_net'
               else DualEndpointBaseline(name,in_bins=50 if dataset=='MMECG' else 1))
        path=checkpoint_path(dataset,fold,name)
        saved=torch.load(path,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model'],strict=True)
        model.to(device).eval().requires_grad_(False); models[name]=model
        provenance[name]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            epoch=saved.get('epoch'),strict_load=True,parameters=sum(p.numel() for p in model.parameters()))
    return models,provenance

def prepare_record(row,pool):
    ds,rid=row['dataset'],row['record_id']; started=time.perf_counter()
    data=record_data(ds,rid,True); radar=data['radar']; n=radar.shape[-1]
    coarse=sorted({s['input_start'] for s in approved_layout(n)})
    dense=dense_starts(n); starts=sorted(set(coarse)|set(dense))
    rcg=np.stack([radar_window(ds,radar,s) for s in starts])
    # Each record is ~0.35 GiB; the runner holds at most current+one prefetched.
    features,feature_provenance=record_features(row,starts,rcg,pool)
    return n,coarse,dense,starts,rcg,features,feature_provenance,started


def infer_record(row,models,device,pool,batch_size=8,prepared=None):
    ds,rid=row['dataset'],row['record_id']
    n,coarse,dense,starts,rcg,features,feature_provenance,started=(
        prepare_record(row,pool) if prepared is None else prepared)
    predictions={name:dict(shape=[],representative=[],coarse_prob=[]) for name in NAMES}
    coarse_set=set(coarse)
    with torch.inference_mode():
        for a in range(0,len(starts),batch_size):
            b=min(a+batch_size,len(starts)); x=torch.from_numpy(features[a:b].astype(np.float32)).to(device)
            r=torch.from_numpy(rcg[a:b]).to(device)
            for name,model in models.items():
                if name=='rem_net': output=model(x)
                else:
                    z=F.interpolate(x,size=(71,240),mode='bilinear',align_corners=False) if ds=='CR_RVS' and name=='radarode_mtl' else x
                    output=model(wsst=z,rcg=r)
                shape=output['shape'][:,0].cpu().numpy()
                probs=torch.sigmoid(output['anchor_logits']).cpu().numpy()
                assert shape.shape==(b-a,200) and probs.shape==(b-a,4,1600)
                assert np.isfinite(shape).all() and np.isfinite(probs).all()
                p=predictions[name]; p['shape'].extend(shape)
                for offset,prob in enumerate(probs):
                    representative=representative_r(core.structured_decode(prob))
                    p['representative'].append(np.nan if representative is None else representative)
                    if starts[a+offset] in coarse_set: p['coarse_prob'].append(prob)
    out=HERE/'inference_results'/ds/f"fold_{row['fold']:02d}"/rid
    for name,p in predictions.items():
        atomic_npz(out/f'{name}.npz',dict(starts=np.asarray(starts,np.int32),
            dense_starts=np.asarray(dense,np.int32),coarse_starts=np.asarray(coarse,np.int32),
            shape=np.asarray(p['shape'],np.float32),
            representative_r_local=np.asarray(p['representative'],np.float32),
            coarse_prob=np.asarray(p['coarse_prob'],np.float32)))
    meta=dict(**row,length=n,coarse_windows=len(coarse),dense_windows=len(dense),
        union_windows=len(starts),elapsed_seconds=time.perf_counter()-started,
        radar_shape=list(rcg.shape),feature_shape=list(features.shape),
        features_persisted=False,model_training=False,test_access=False)
    meta['feature_provenance']=feature_provenance
    atomic_json(out/'COMPLETE.json',meta)
    return meta

def run_inference(inventory,workers=12,device='cuda:0'):
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    # spawn avoids fork-after-CUDA and child processes do not create CUDA contexts.
    import multiprocessing
    context=multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=context) as pool:
        pending=[(i,row) for i,row in enumerate(inventory) if not
            (HERE/'inference_results'/row['dataset']/f"fold_{row['fold']:02d}"/row['record_id']/'COMPLETE.json').exists()]
        if not pending:return
        # One-record bounded prefetch overlaps CPU transform with frozen GPU
        # forward passes. No model/metric/RNG operation is run in the CPU thread.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as prefetch:
            prepared=prefetch.submit(prepare_record,pending[0][1],pool)
            models={}; previous=None
            for j,(i,row) in enumerate(pending):
                out=HERE/'inference_results'/row['dataset']/f"fold_{row['fold']:02d}"/row['record_id']
                ready=prepared.result()
                if j+1<len(pending): prepared=prefetch.submit(prepare_record,pending[j+1][1],pool)
                key=(row['dataset'],row['fold'])
                if key!=previous:
                    models.clear(); torch.cuda.empty_cache()
                    models,provenance=make_models(*key,device)
                    atomic_json(out.parent/'checkpoint_manifest.json',provenance); previous=key
                meta=infer_record(row,models,device,pool,prepared=ready)
                del ready
                print(f"INFERENCE {i+1}/{len(inventory)} {key} {row['record_id']} "
                      f"windows={meta['union_windows']} seconds={meta['elapsed_seconds']:.1f}",flush=True)
                atomic_json(HERE/f"inference_results/progress_{row['dataset']}.json",dict(completed=i+1,total=len(inventory),last=meta))
