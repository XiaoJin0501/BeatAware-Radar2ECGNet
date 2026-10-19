"""Two-epoch tiny integration check, no persisted weights or screening evidence."""
import json
import time
import environment as env
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import TrainWindows
from model import initialize_arm,objective,parameter_counts
from run import validate,preload_validation
from v2_metrics import patience_after_epoch
from src.utils.atomic_state import capture_rng_state,atomic_json,atomic_text

def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    data=TrainWindows(1,limit=16).preload()
    inventory=json.loads((env.V1_OUT/'data/fold_01/validation_inventory.json').read_text())[:1]
    preload_validation(inventory,data.stats)
    model=initialize_arm(3,42).to('cuda:0').train()
    env.core.set_seed(42)
    generator=torch.Generator().manual_seed(42)
    loader=DataLoader(data,batch_size=16,shuffle=True,drop_last=True,generator=generator,num_workers=0)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    logs=[];wait=0
    for epoch in (1,2):
        started=time.perf_counter()
        for batch in loader:
            x,y,events=[batch[k].to('cuda:0') for k in ('wsst','target','events')]
            optimizer.zero_grad(set_to_none=True)
            out=model(x);loss,pieces=objective(out,y,events)
            loss.backward();optimizer.step()
        before=capture_rng_state()
        metrics=validate(model,inventory,data.stats,'cuda:0')
        after=capture_rng_state()
        assert model.training
        assert before['python']==after['python']
        assert all(np.array_equal(x,y) for x,y in zip(before['numpy'],after['numpy']))
        assert torch.equal(before['torch_cpu'],after['torch_cpu'])
        assert all(torch.equal(x,y) for x,y in zip(before['torch_cuda'],after['torch_cuda']))
        assert metrics['protocol_version']=='v2' and metrics['record_optimal']['max_lag_ms']==150
        assert np.isfinite(metrics['selection']['score'])
        wait=patience_after_epoch(epoch,False,wait);assert wait==0
        logs.append(dict(epoch=epoch,train_loss=float(loss.detach().cpu()),val_loss=metrics['losses']['loss'],
            selection=metrics['selection'],record_optimal_PCC=metrics['record_optimal']['pcc'],
            M1=metrics['M1'],M2=metrics['M2'],M3=metrics['M3'],
            strict_beat=metrics['beat'],events=metrics['events_by_subject'],patience=wait,
            seconds=time.perf_counter()-started,rng_preserved=True,model_training_restored=True,
            smoke_only=True,publishable=False))
    counts={str(a):parameter_counts(initialize_arm(a,42)) for a in (1,2,3,4)}
    path=env.OUT/'integration_check'
    atomic_text(path/'epoch_log.jsonl',''.join(json.dumps(r,allow_nan=False)+'\n' for r in logs))
    atomic_json(path/'COMPLETE.json',dict(passed=True,epochs=2,fold=1,train_windows=16,val_records=1,
        parameter_counts=counts,rng_preserved=True,weights_saved=False,screening_evidence=False))
    print('PASS v2 integration:2epochs; native metrics and RNG checked; no saved weights',flush=True)

if __name__=='__main__':main()
