"""27 author-approved validation-only runs. No test or formal expansion mode."""
import argparse
import json
import sys
import time
import hashlib
import bridge
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from architecture import FinalREM, counts, objective
from inputs import TrainWindows, inventory
import regional_metrics as metrics
import lock


@torch.no_grad()
def validate(model, rows, stats, device, save=None):
    rng, training = bridge.capture_rng_state(), model.training
    model.eval()
    result, sums, count = [], {}, 0
    try:
        for row in rows:
            waves, probs = [], []
            features = row['_features']
            for a in tqdm(range(0,len(features),16),desc='val',dynamic_ncols=True,
                          leave=False,mininterval=1.,disable=not sys.stderr.isatty()):
                x = torch.as_tensor(features[a:a+16].astype(np.float32),device=device)
                td = torch.as_tensor(row['_td'][a:a+16],device=device) if model.variant in ('C','D') else None
                y = torch.as_tensor(row['_targets'][a:a+16],device=device)
                ev = torch.as_tensor(row['_events'][a:a+16],device=device)
                output = model(x,td)
                _, pieces = objective(output,y,ev,model.variant)
                for k,v in pieces.items():
                    sums[k]=sums.get(k,0.)+v.detach().item()*len(x)
                count+=len(x)
                waves.extend(output['wave'][:,0].cpu().numpy())
                probs.extend(torch.sigmoid(output['anchor_logits']).cpu().numpy())
            wave_array, prob_array = np.asarray(waves), np.asarray(probs)
            record = metrics.native_record(row,wave_array,prob_array,row['_raw'],stats)
            record['record_optimal']=bridge.v2_metrics.record_metric(row,wave_array,row['_raw'],stats)
            result.append(record)
            if save is not None:
                bridge.atomic_npz(save/f"{row['record_id']}.npz",dict(starts=np.asarray(row['starts']),
                    wave=wave_array,anchor_probability=prob_array))
        output=metrics.aggregate(result)
        output['losses']={k:v/count for k,v in sums.items()}
        return output
    finally:
        model.train(training)
        bridge.restore_rng_state(rng)


def one(fold,seed,variant,data,rows,device):
    frozen=lock.verify()
    directory=bridge.OUT/'runs'/f'fold_{fold:02d}'/f'seed_{seed}'/variant
    if (directory/'COMPLETE.json').exists():
        return
    directory.mkdir(parents=True,exist_ok=True)
    bridge.old_environment.core.set_seed(seed)
    model=FinalREM(variant,seed).to(device)
    assert counts(model)['added_fraction_of_backbone']<=.15
    bridge.old_environment.core.set_seed(seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(data,batch_size=16,shuffle=True,drop_last=True,num_workers=0,
                      generator=generator,pin_memory=device.startswith('cuda'))
    meta=dict(fold=fold,seed=seed,variant=variant,commit=frozen['commit'],normalization=data.stats,
        parameters=counts(model),training_windows=len(data),validation_records=len(rows),
        training_subjects=sorted({r['subject_id'] for r in data.rows}),
        validation_subjects=sorted({r['subject_id'] for r in rows}),
        spec=json.loads((bridge.HERE/'spec.json').read_text()),test_access=False,screening_only=True)
    assert not set(meta['training_subjects']) & set(meta['validation_subjects'])
    bridge.atomic_json(directory/'meta.json',meta)
    history,best,wait,start=[], -float('inf'),0,1
    last=directory/'last_state.pt'
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=False)
        assert saved['commit']==frozen['commit']
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        history,best,wait=saved['history'],saved['best_val'],saved['patience']
        bridge.restore_rng_state(saved['rng'],generator)
        start=saved['epoch']+1
    for epoch in range(start,101):
        if wait>=15:break
        started=time.perf_counter();model.train();seen=0;norms=0.;sums={};order=[]
        progress=tqdm(loader,desc=f'f{fold}s{seed}{variant} ep{epoch}',dynamic_ncols=True,
                      leave=False,mininterval=1.,disable=not sys.stderr.isatty())
        for step,batch in enumerate(progress,1):
            x,y,ev=[batch[k].to(device,non_blocking=True) for k in ('wsst','target','events')]
            td=batch['td'].to(device,non_blocking=True) if variant in ('C','D') else None
            optimizer.zero_grad(set_to_none=True)
            output=model(x,td);loss,pieces=objective(output,y,ev,variant)
            loss.backward()
            norm=torch.stack([p.grad.detach().square().sum() for p in model.parameters() if p.grad is not None]).sum().sqrt()
            if not torch.isfinite(loss) or not torch.isfinite(norm):raise FloatingPointError('Nonfinite training')
            optimizer.step()
            for k,v in pieces.items():sums[k]=sums.get(k,0.)+v.detach().item()*len(x)
            seen+=len(x);norms+=norm.detach().item();order.extend(batch['index'].tolist())
            if step%20==0:progress.set_postfix(loss=sums['loss']/seen,anchor=sums['anchor']/seen,
                gnorm=norms/step,lr=optimizer.param_groups[0]['lr'])
        validation=validate(model,rows,data.stats,device)
        score=validation['selection']['score']
        if score is None or not np.isfinite(score):raise FloatingPointError('Invalid checkpoint score')
        improved=score>best
        if improved:
            best=score
            bridge.atomic_torch_save(directory/'best.pt',dict(model=model.state_dict(),epoch=epoch,validation=validation))
        wait=bridge.v2_metrics.patience_after_epoch(epoch,improved,wait)
        entry=dict(epoch=epoch,train={k:v/seen for k,v in sums.items()},val=validation,
            seconds=time.perf_counter()-started,best=improved,patience_used=wait,
            average_gnorm=norms/len(loader),lr=optimizer.param_groups[0]['lr'],
            training_order_sha256=hashlib.sha256(np.asarray(order,np.int32).tobytes()).hexdigest())
        history.append(entry)
        bridge.atomic_text(directory/'epoch_log.jsonl',''.join(json.dumps(r,allow_nan=False)+'\n' for r in history))
        bridge.atomic_torch_save(last,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
            history=history,best_val=best,patience=wait,epoch=epoch,commit=frozen['commit'],rng=bridge.capture_rng_state(generator)))
        print(f"f={fold} seed={seed} variant={variant} ep={epoch} loss={sums['loss']/seen:.5f} select={score:.5f} wait={wait} {'★' if improved else ''}",flush=True)
    checkpoint=torch.load(directory/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model'])
    validation=validate(model,rows,data.stats,device,directory/'validation_predictions')
    assert abs(validation['selection']['score']-checkpoint['validation']['selection']['score'])<1e-8
    lock.verify()
    bridge.atomic_json(directory/'result.json',dict(**meta,validation=validation,epochs=len(history),best_epoch=checkpoint['epoch']))
    bridge.atomic_json(directory/'COMPLETE.json',dict(completed=True,fold=fold,seed=seed,variant=variant,test_access=False))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fold',type=int,choices=(1,2,3),required=True)
    parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    lock.verify();torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=TrainWindows(args.fold,dual=True).preload();rows=inventory(args.fold)
    for seed in (42,43,44):
        for variant in ('B','C','D'):one(args.fold,seed,variant,data,rows,args.device)


if __name__=='__main__':main()
