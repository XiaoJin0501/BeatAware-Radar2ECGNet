"""640-point native radar/ECG samples; no Shape or anchor labels."""
import bridge
import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

MM='/home/qhh2237/Datasets/MMECG/processed/raw_mmecg_archive.h5'

def reference(rid):
    with np.load(bridge.frozen_environment.A/'raw_reference/MMECG'/f'{rid}.npz') as z:
        return {k:z[k] for k in z.files}

def statistics(fold):
    stats=json.loads((bridge.frozen_environment.A/'training_statistics/MMECG'/f'fold_{fold:02d}.json').read_text())
    records=sorted({r['record_id'] for r in bridge.split_rows('MMECG',fold,'train')})
    train_subjects={r['subject_id'] for r in bridge.split_rows('MMECG',fold,'train')}
    val_subjects={r['subject_id'] for r in bridge.split_rows('MMECG',fold,'val')}
    assert not train_subjects&val_subjects
    assert sorted(train_subjects)==stats['train_subjects']
    max_abs=0.
    for rid in records:
        x=(reference(rid)['ecg']-stats['median'])/stats['iqr']
        max_abs=max(max_abs,float(np.max(np.abs(x))))
    return dict(median=stats['median'],iqr=stats['iqr'],mu_scale=max(max_abs,1e-6),
        train_records=records,train_subjects=sorted(train_subjects),validation_subjects=sorted(val_subjects),
        scale_fit='unique complete TRAIN records only, no validation/test amplitude')

def radar_record(rid):
    with h5py.File(MM,'r') as h:
        g=h['records/'+rid];radar=np.asarray(g['RCG'],np.float32).T;positions=np.asarray(g['posXYZ'],np.float32)
    if positions.shape==(3,50):positions=positions.T
    if radar.shape[0]!=50 or positions.shape!=(50,3):raise ValueError('Original50clusters + measuredpositions required')
    if not np.isfinite(radar).all() or not np.isfinite(positions).all():raise ValueError('Nonfinite raw radar; no unapproved imputation')
    return radar,positions

def normalize_radar(x):
    return ((x-x.mean(-1,keepdims=True))/(x.std(-1,keepdims=True)+1e-9)).astype(np.float32)

class Train640(Dataset):
    def __init__(self,fold,limit=None):
        self.stats=statistics(fold);self.rows=[]
        with h5py.File(MM,'r') as h:
            for rid in self.stats['train_records']:
                n=len(h['records/'+rid+'/ECG'])
                self.rows.extend((rid,start) for start in range(0,n-640+1,30))
        if limit is not None:
            indices=np.linspace(0,len(self.rows)-1,limit,dtype=int);self.rows=[self.rows[i] for i in indices]
        self.cache={}
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        rid,start=self.rows[i]
        if rid not in self.cache:self.cache[rid]=(*radar_record(rid),reference(rid))
        radar,pos,raw=self.cache[rid]
        y=((raw['ecg'][start:start+640]-self.stats['median'])/self.stats['iqr']).astype(np.float32)
        x=normalize_radar(radar[:,start:start+640])
        from model import mulaw_encode
        return dict(radar=torch.from_numpy(x),positions=torch.from_numpy(pos.copy()),
            codes=mulaw_encode(torch.from_numpy(y/self.stats['mu_scale'])),
            reference=torch.from_numpy(y),record_id=rid,start=start)
