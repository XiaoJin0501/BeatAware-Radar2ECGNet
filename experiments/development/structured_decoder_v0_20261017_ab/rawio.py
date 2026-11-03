"""Raw-record reference and unchanged radar preprocessing; train/val only."""
from __future__ import annotations
import csv
import importlib.util
import json
from pathlib import Path

import bootstrap
core=bootstrap.activate()
import h5py
import numpy as np
from scipy import signal
from src.utils.atomic_state import atomic_npz, atomic_json
from unified_metrics import waveform_events

HERE,ROOT=bootstrap.HERE,bootstrap.ROOT
MM=Path('/home/qhh2237/Datasets/MMECG')
CR=Path('/home/qhh2237/Datasets/CR_RVS')

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

cw=load_module(bootstrap.FROZEN/'scripts/build_crrvs_wsst_shape_h5.py','cw_frozen_builder')
mm=load_module(MM/'04B_preprocess_records.py','mm_original_filter')
with (CR/'metadata/crrvs_metadata.csv').open() as handle:
    CW_META={x['record_id']:x for x in csv.DictReader(handle)}

def split_path(dataset,fold,split):
    if split not in ('train','val'): raise PermissionError('No test access in A/B')
    root=MM/'processed/loso_v2' if dataset=='MMECG' else CR/'processed/q1_continuous_kfold'
    return root/f'fold_{fold:02d}'/f'{split}.h5'

def text(x): return x.decode() if isinstance(x,bytes) else str(x)

def split_rows(dataset,fold,split):
    with h5py.File(split_path(dataset,fold,split),'r') as h:
        return [dict(record_id=text(r),subject_id=int(s),start=int(a),stop=int(b))
                for r,s,a,b in zip(h['record_id'][:],h['subject_id'][:],h['start_idx'][:],h['end_idx'][:])]

def validation_inventory():
    out=[]
    for ds,n in [('MMECG',11),('CR_RVS',5)]:
        for fold in range(1,n+1):
            records={x['record_id']:x for x in split_rows(ds,fold,'val')}
            tr={x['subject_id'] for x in split_rows(ds,fold,'train')}
            assert not tr & {x['subject_id'] for x in records.values()}
            out.extend(dict(dataset=ds,fold=fold,record_id=k,subject_id=x['subject_id'])
                       for k,x in sorted(records.items()))
    return out

def record_data(dataset,rid,with_radar=False):
    """Regenerate ECG from raw. Cache ONLY newly generated references HERE."""
    cache=HERE/'raw_reference'/dataset/f'{rid}.npz'
    if cache.exists() and not with_radar:
        with np.load(cache) as z: return {k:z[k] for k in z.files}
    if dataset=='MMECG':
        with h5py.File(MM/'processed/raw_mmecg_archive.h5','r') as h:
            ecg=np.asarray(h[f'records/{rid}/ECG'],np.float32)
        if not np.isfinite(ecg).all(): ecg=mm.linear_interp_nan(ecg)
        ecg=mm.butter_bandpass(ecg,.5,40.,200,4)[:,0]
        radar=None
        if with_radar:
            with h5py.File(MM/'processed/clean_mmecg_archive.h5','r') as h:
                radar=np.asarray(h[f'records/{rid}/RCG'],np.float32).T
    else:
        row=CW_META[rid]; data=cw._load_mat(Path(row['file_path']))
        phase=signal.detrend(np.unwrap(np.angle(data['radar_i']+1j*data['radar_q']))).astype(np.float32)
        phase=cw._downsample(phase,data['fs_radar'],200)
        ecg=cw._downsample(data['ecg'],data['fs_ecg'],200)
        n=min(len(phase),len(ecg)); ecg=cw._bandpass(ecg[:n],200,.5,40.)
        radar=cw._bandpass(phase[:n],200,.6,12.)[None] if with_radar else None
    if cache.exists():
        with np.load(cache) as z:
            assert np.array_equal(ecg,z['ecg']), 'Reference changed during inference'
            result={k:z[k] for k in z.files}
    else:
        points=waveform_events(ecg)
        result=dict(ecg=ecg,**{f'event_{k}':v for k,v in points.items()})
        atomic_npz(cache,result)
    if with_radar:
        assert radar.shape[-1]==len(ecg)
        result['radar']=radar
    return result

def fit_training(dataset,fold):
    path=HERE/'training_statistics'/dataset/f'fold_{fold:02d}.json'
    template_path=path.with_suffix('.npz')
    if path.exists():
        result=json.loads(path.read_text())
        with np.load(template_path) as z: result['mean_shape']=z['mean_shape']
        return result
    rows=split_rows(dataset,fold,'train'); val=split_rows(dataset,fold,'val')
    assert not {r['subject_id'] for r in rows}&{r['subject_id'] for r in val}
    records={r['record_id']:record_data(dataset,r['record_id']) for r in rows}
    samples=[]; minima=[]; ranges=[]; templates=[]
    for row in rows:
        data=records[row['record_id']]; a,b=row['start'],row['stop']; x=data['ecg'][a:b]
        if len(x)!=1600: raise AssertionError('Training index mismatch')
        samples.append(x)
        r=data['event_r']; local=r[(r>=a)&(r<b)]-a
        left,right=core._center_cycle_bounds_from_rpeaks(local,1600)
        cycle=x[left:right]
        target=np.interp(np.linspace(0,len(cycle)-1,200),np.arange(len(cycle)),cycle).astype(np.float32)
        lo,hi=float(target.min()),float(target.max())
        minima.append(lo); ranges.append(hi-lo)
        templates.append(np.zeros(200,np.float32) if hi-lo<1e-8 else (target-lo)/(hi-lo))
    values=np.concatenate(samples); q1,median,q3=np.percentile(values,[25,50,75])
    scale=max(float(q3-q1),1e-8)
    result=dict(dataset=dataset,fold=fold,median=float(median),iqr=scale,
        gain=float(np.median(ranges)/scale),baseline=float((np.median(minima)-median)/scale),
        train_subjects=sorted({r['subject_id'] for r in rows}),
        validation_subjects=sorted({r['subject_id'] for r in val}),
        train_window_count=len(rows),train_samples=len(values),
        fit_weighting='Each original train-window sample once, including historical overlap multiplicity',
        template_rule='Training raw-record R detection; representative central R-R per historical train window')
    mean=np.mean(templates,axis=0).astype(np.float32)
    atomic_npz(template_path,{'mean_shape':mean})
    atomic_json(path,result); result['mean_shape']=mean
    return result

def radar_window(dataset,radar,start):
    x=radar[:,start:start+1600].astype(np.float32)
    if x.shape[-1]!=1600: raise ValueError('No short/padded radar input')
    if dataset=='MMECG':
        return ((x-x.mean(axis=1,keepdims=True))/(x.std(axis=1,keepdims=True)+1e-9)).astype(np.float32)
    return cw._robust_z(x[0])[None]
