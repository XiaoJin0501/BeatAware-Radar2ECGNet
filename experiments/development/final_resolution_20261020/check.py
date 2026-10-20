"""Real-input GPU engineering check only: three variants, two one-batch epochs."""
import json
import time
import bridge
import numpy as np
import torch
from torch.utils.data import DataLoader
from architecture import FinalREM, objective, counts
from inputs import TrainWindows, inventory
from train import validate


def main():
    out=bridge.OUT/'engineering_check'
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device='cuda:0'
    assert torch.cuda.is_available()
    data=TrainWindows(1,dual=True,limit=16).preload()
    rows=inventory(1)[:1]
    results=[]
    for variant in ('B','C','D'):
        model=FinalREM(variant).to(device)
        optim=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
        loader=DataLoader(data,batch_size=16,drop_last=True,shuffle=False)
        for epoch in (1,2):
            started=time.perf_counter();model.train()
            batch=next(iter(loader));optim.zero_grad(set_to_none=True)
            x,y,e=[batch[k].to(device) for k in ('wsst','target','events')]
            td=batch['td'].to(device) if variant in ('C','D') else None
            loss,_=objective(model(x,td),y,e,variant);loss.backward();optim.step()
            rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
            metrics=validate(model,rows,data.stats,device)
            assert model.training and torch.equal(rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state())
            results.append(dict(variant=variant,epoch=epoch,loss=float(loss.detach()),
                val_selection=metrics['selection'],regional=metrics['regional'],
                seconds=time.perf_counter()-started,parameters=counts(model)))
            print(f'ENGINEERING CHECK {variant} epoch{epoch} finished',flush=True)
        del model,optim
        torch.cuda.empty_cache()
    bridge.atomic_json(out/'COMPLETE.json',dict(passed=True,training_batches_per_epoch=1,epochs_per_variant=2,
        full_validation_records=1,variants=['B','C','D'],weights_saved=False,publishable=False,
        cuda_device=torch.cuda.get_device_name(),results=results,validation_preserves_rng_and_train_mode=True))


if __name__=='__main__':main()
