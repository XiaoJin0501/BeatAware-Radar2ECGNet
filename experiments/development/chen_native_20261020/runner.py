"""Only CPU smoke is executable. Formal training is deliberately not exposed."""
import os
# Hide CUDA before importing torch or the frozen evaluator.
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import json
import time
import subprocess
import bridge
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import ChenNativeAR,mulaw_encode,objective
from data import Train640,reference,radar_record,normalize_radar
from evaluate import infer_8s,score_record,aggregate
from src.utils.atomic_state import capture_rng_state,restore_rng_state

def train_epoch(model,loader,optimizer):
    model.train();total=seen=0
    for batch in loader:
        if batch['radar'].device.type!='cpu':raise RuntimeError('Preparation is CPU-only')
        optimizer.zero_grad(set_to_none=True)
        logits=model(batch['radar'],batch['positions'],batch['codes'])
        loss=objective(logits,batch['codes']);loss.backward()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite CE')
        optimizer.step();total+=loss.detach().item()*len(batch['radar']);seen+=len(batch['radar'])
    if not seen:raise ValueError('drop_last produced no complete batches')
    return total/seen

def new_optimizer(model):
    return torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)

def validation_metrics(model,radar,positions,raw,stats,subject_id,record_id):
    """Actual AR output only. Restore train/eval and all RNG states."""
    state=capture_rng_state();was=model.training
    try:
        model.eval()
        with torch.no_grad():
            wave=infer_8s(model,radar,positions,stats['mu_scale'])[0]
            return aggregate([score_record(wave,raw,stats,subject_id,record_id)])
    finally:
        model.train(was);restore_rng_state(state)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['smoke'],required=True)
    parser.add_argument('--check-id',default='cpu_integration')
    args=parser.parse_args()
    if not args.check_id.replace('_','').isalnum():raise ValueError('Simple smoke identifier required')
    out=bridge.OUT/args.check_id
    if (out/'COMPLETE.json').exists():raise RuntimeError('Do not overwrite completed smoke')
    torch.set_num_threads(1)
    assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
    torch.manual_seed(42);np.random.seed(42)
    lock=bridge.verify_frozen()
    model=ChenNativeAR().cpu()
    data=Train640(1,limit=16)
    loader=DataLoader(data,batch_size=16,shuffle=True,drop_last=True,num_workers=0,
                      generator=torch.Generator().manual_seed(42))
    optimizer=new_optimizer(model)
    row=bridge.split_rows('MMECG',1,'val')[0];rid=row['record_id']
    raw=reference(rid);radar,pos=radar_record(rid)
    # Smoke covers one actual8s validation excerpt; never claim whole-record results.
    excerpt=dict(ecg=raw['ecg'][:1600],**{f'event_{k}':raw[f'event_{k}'][(raw['event_r']>=0)&(raw['event_r']<1600)] for k in 'qrst'})
    x=torch.from_numpy(normalize_radar(radar[:,:1600])[None]);positions=torch.from_numpy(pos[None])
    logs=[];best=-float('inf');wait=0;selected=None
    started=time.perf_counter()
    for epoch in (1,2):
        train_ce=train_epoch(model,loader,optimizer)
        before=capture_rng_state()
        metrics=validation_metrics(model,x,positions,excerpt,data.stats,row['subject_id'],rid+'[0:1600]')
        after=capture_rng_state()
        assert model.training and before['python']==after['python']
        assert all(np.array_equal(a,b) for a,b in zip(before['numpy'],after['numpy']))
        assert torch.equal(before['torch_cpu'],after['torch_cpu'])
        score=metrics['selection']['score'];improved=score>best
        if improved:best=score;selected=epoch
        wait=bridge.frozen_v2.patience_after_epoch(epoch,improved,wait)
        assert wait==0
        logs.append(dict(epoch=epoch,train_CE=train_ce,val=metrics,checkpoint_score=score,
            selected_epoch_so_far=selected,patience=wait,validation_is_autoregressive=True,
            validation_rng_unchanged=True,smoke_only=True))
        print(f'CPU SMOKE epoch={epoch} CE={train_ce:.5f} select={score:.5f} patience={wait}',flush=True)
    bridge.verify_frozen()
    bridge.atomic_text(out/'epoch_log.jsonl',''.join(json.dumps(r,allow_nan=False)+'\n' for r in logs))
    bridge.atomic_json(out/'COMPLETE.json',dict(passed=True,formal_training=False,epochs=2,fold=1,
        train_windows=16,batch=16,validation_excerpt_samples=1600,retained_samples=1024,
        native_record_evaluation_smoke=False,CPU_only=True,cuda_initialized=torch.cuda.is_initialized(),
        weights_saved=False,parameters=sum(p.numel() for p in model.parameters()),
        training_target='256-class mu-law; CE over last128 of640; shifted teacher forcing',
        inference='1600 actual single-sample autoregressive steps per validation excerpt',
        frozen_v2_metric_sha256=lock,seconds=time.perf_counter()-started,
        source_sha256={p.name:bridge.freeze.sha(p) for p in sorted(bridge.HERE.glob('*.py'))},
        compile_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=bridge.HERE.parents[2],text=True).strip(),
        normalization=data.stats,performance_evidence=False))
    print('COMPLETE: CPU-only integration; no weights saved; formal training remains blocked',flush=True)

if __name__=='__main__':main()
