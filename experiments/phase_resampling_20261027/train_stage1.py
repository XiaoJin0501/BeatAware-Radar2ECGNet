"""Validation-only six-run Stage1. No test API and no automatic Stage2."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys
import time
import json
import csv
import argparse
import subprocess
import importlib.util
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.phase_resampling_20261027.architecture import AnchorConditionedREM,mix_fraction
from experiments.phase_resampling_20261027 import frozen_bridge as frozen

PROJECT=Path(os.environ.get('REM_PHASE_PROJECT_ROOT','/home/qhh2237/Projects/BeatAware-Radar2ECGNet'))
OUT=PROJECT/'experiments/phase_resampling_20261027/stage1'
SPEC=Path(__file__).with_name('stage1_spec.json')
sys.path.insert(0,str(frozen.FROZEN))
import scripts.train_continuous_ecg_pilot as data_core
from src.utils.atomic_state import atomic_json,atomic_torch_save,atomic_npz,capture_rng_state,restore_rng_state
from scripts.run_dual_endpoint_baselines_mmecg import set_seed,rank

def labels(ecg):
    p=frozen.extract_points(ecg)
    return p,frozen.shape_target(ecg,p['r']),frozen.center_bounds(p['r'],1600)

class PhaseFold(Dataset):
    def __init__(self,ds,fold,split,limit=None):
        if split not in {'train','val'}:raise ValueError('Stage1 forbids test data')
        root='loso_v2' if ds=='mmecg' else 'q1_continuous_kfold'
        data_core._set_split_root(root,ds)
        self.base=data_core.Fold(fold,split,'B1',data_core.fit_scaler(fold),limit=limit,
            allow_test=False,wsst_source='subject_pool' if ds=='mmecg' else 'per_split',needs_rcg=False)
        cache=OUT/'label_cache'/ds/f'fold_{fold:02d}'/(split+('' if limit is None else f'_limit{limit}')+'.npz')
        if cache.exists():
            with np.load(cache,allow_pickle=True) as saved:
                self.points=list(saved['points']);self.shapes=saved['shapes'];self.bounds=saved['bounds']
        else:
            with ProcessPoolExecutor(max_workers=8) as pool:
                result=list(tqdm(pool.map(labels,self.base.ecg,chunksize=16),total=len(self.base),
                    desc=f'{ds} fold{fold} {split} labels',disable=not sys.stderr.isatty(),leave=False,mininterval=1))
            self.points=[x[0] for x in result];self.shapes=np.stack([x[1] for x in result]);self.bounds=np.stack([x[2] for x in result])
            atomic_npz(cache,dict(points=np.array(self.points,object),shapes=self.shapes,bounds=self.bounds))
        if len(self.shapes)!=len(self.base):raise RuntimeError('cache size mismatch')
        import h5py
        with h5py.File(self.base.h5p,'r') as h:
            self.subjects=h['subject_id'][:len(self.base)]
            self.scenes=[str(x.decode() if isinstance(x,bytes) else x) for x in h['state_code'][:len(self.base)]] if 'state_code' in h else ['unknown']*len(self.base)

    def __len__(self):return len(self.base)
    def __getitem__(self,i):
        # Original loader owns input preprocessing; frozen Gaussian target owns labels.
        item=self.base[i]
        anchor=np.stack([frozen.gaussian_mask(np.asarray(self.points[i][p])[np.isfinite(self.points[i][p])].astype(int),1600,4.) for p in 'qrst'])
        return dict(wsst=item['wsst'],shape=torch.from_numpy(self.shapes[i][None]),
            anchor=torch.from_numpy(anchor),bounds=torch.as_tensor(self.bounds[i],dtype=torch.float32),index=i)

@torch.no_grad()
def evaluate(model,data,device,oracle=False):
    previous=model.training;model.eval();shapes=[];probs=[];statuses=[];totals=np.zeros(3);n=0
    try:
        loader=DataLoader(data,batch_size=16,shuffle=False,num_workers=0)
        for batch in tqdm(loader,desc='val oracle' if oracle else 'val',disable=not sys.stderr.isatty(),
                          dynamic_ncols=True,leave=False,mininterval=1):
            batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
            out=model(batch['wsst'],batch['bounds'],oracle=oracle)
            losses=frozen.losses(out,batch);size=len(batch['shape'])
            totals+=np.array([x.detach().item() for x in losses])*size;n+=size
            shapes.append(out['shape'][:,0].cpu().numpy());probs.append(out['anchor_logits'].sigmoid().cpu().numpy())
            statuses.extend(out.get('anchor_status',['not_applicable']*size))
    finally:model.train(previous)
    shape=np.concatenate(shapes);prob=np.concatenate(probs)
    result=dict(shape=frozen.shape_metrics(shape,data.shapes),independent=frozen.score_events(prob,data.points,False),
        structured=frozen.score_events(prob,data.points,True),loss=dict(zip(('total','shape','anchor'),(totals/n).tolist())),
        fallback_rate=statuses.count('fallback')/n,valid_unmatched_rate=statuses.count('valid_unmatched')/n,
        matched_rate=statuses.count('matched')/n,oracle=oracle,n_windows=n)
    # Existing checkpoint scorer exposes MEDIAN error, not MAE. Use the frozen
    # formal R-pair accumulator for true arithmetic MAE, without changing rank().
    bucket=frozen.empty_bucket()
    for p,actual in zip(prob,data.points):frozen.accumulate_window(bucket,actual,frozen.structured_decode(p))
    result['direct']=frozen.finalise(bucket)
    for event in 'qrst':
        errors=bucket[event]['errors_ms']
        result['direct'][event]['p90_ms']=float(np.percentile(errors,90)) if errors else None
    pc=frozen.pcc_per_row(shape,data.shapes)
    result['subject_fisher_pcc']={str(s):float(np.tanh(np.mean(np.arctanh(np.clip(pc[data.subjects==s],-1+1e-7,1-1e-7))))) for s in np.unique(data.subjects)}
    result['scenario_anchor_status']={scene:{st:sum(statuses[i]==st for i,s in enumerate(data.scenes) if s==scene)/sum(s==scene for s in data.scenes)
        for st in ('matched','valid_unmatched','fallback')} for scene in sorted(set(data.scenes))}
    return result,shape,prob,statuses

def incumbent_checkpoint(ds,fold):
    if ds=='mmecg':return PROJECT/f'experiments_mmecg/q1_dual_head_balanced_full_loso_seed42/A1/fold_{fold:02d}/checkpoints/best.pt'
    return PROJECT/f'experiments_mmecg/q1_dual_endpoint_remnet_crrvs/seed_42/rem_net/fold_{fold:02d}/checkpoints/best.pt'

def curves(history,path):
    fig,axes=plt.subplots(1,3,figsize=(10,2.6));ep=[h['epoch'] for h in history]
    for label,values in [('train',[h['train']['total'] for h in history]),('val',[h['validation']['loss']['total'] for h in history])]:axes[0].plot(ep,values,label=label)
    axes[0].set_title('Loss');axes[0].legend()
    axes[1].plot(ep,[h['validation']['shape']['pcc'] for h in history]);axes[1].set_title('Validation shape PCC')
    axes[2].plot(ep,[h['validation']['fallback_rate'] for h in history]);axes[2].set_title('Fallback rate')
    for ax in axes:ax.set_xlabel('Epoch');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)

def run(ds,fold,variant,device,smoke=False):
    if (ds,fold) not in {('mmecg',1),('mmecg',2),('crrvs',1)}:raise ValueError('unapproved development fold')
    directory=OUT/('engineering_check_v2' if smoke else 'runs')/ds/f'fold_{fold:02d}'/variant
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'COMPLETE.json').exists():return
    set_seed(42+fold);model=AnchorConditionedREM(variant,in_bins=50 if ds=='mmecg' else 1).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    train=PhaseFold(ds,fold,'train',limit=16 if smoke else None);val=PhaseFold(ds,fold,'val',limit=16 if smoke else None)
    metadata=dict(dataset=ds,physical_fold=fold,logical_fold=fold-1,variant=variant,seed_base=42,actual_seed=42+fold,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),spec=json.loads(SPEC.read_text()),
        implementation_files=frozen.SOURCES,test_accessed=False,train_split=str(train.base.h5p),val_split=str(val.base.h5p),
        train_windows=len(train),val_windows=len(val),parameter_count=sum(p.numel() for p in model.parameters()),smoke=smoke)
    atomic_json(directory/'meta.json',metadata)
    generator=torch.Generator().manual_seed(42+fold)
    loader=DataLoader(train,batch_size=16,shuffle=True,num_workers=0,drop_last=ds=='crrvs',generator=generator)
    max_epochs=1 if smoke else 100
    start=1;wait=0;best_rank=(-np.inf,-np.inf,-np.inf);history=[]
    if (directory/'last.pt').exists():
        saved=torch.load(directory/'last.pt',map_location=device,weights_only=False)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        start=saved['epoch']+1;wait=saved['wait'];best_rank=tuple(saved['best_rank']);history=saved['history']
        restore_rng_state(saved['rng'],generator)
    for epoch in range(start,max_epochs+1):
        begin=time.monotonic();model.train();total=np.zeros(3);count=0;train_pcc_sum=0.;train_samples=0
        progress=tqdm(loader,desc=f'{ds} f{fold} {variant} ep{epoch}',dynamic_ncols=True,leave=False,
                      mininterval=1.,disable=not sys.stderr.isatty())
        for step,batch in enumerate(progress,1):
            batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
            out=model(batch['wsst'],batch['bounds'],epoch_index=epoch-1,total_epochs=100)
            loss,shape,anchor=frozen.losses(out,batch)
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            total+=np.array([x.detach().item() for x in (loss,shape,anchor)]);count+=1
            pc=frozen.pcc_per_row(out['shape'][:,0].detach().cpu().numpy(),batch['shape'][:,0].detach().cpu().numpy())
            train_pcc_sum+=float(pc.sum());train_samples+=len(pc)
            if step%20==0:progress.set_postfix(loss=f'{total[0]/count:.4f}',shape=f'{total[1]/count:.4f}',anchor=f'{total[2]/count:.4f}')
        result,_,_,_=evaluate(model,val,device);current=rank(result);improved=current>best_rank
        if improved:
            best_rank=current;wait=0
            atomic_torch_save(directory/'best.pt',dict(model=model.state_dict(),epoch=epoch,validation=result))
        else:wait+=1
        row=dict(epoch=epoch,train=dict(zip(('total','shape','anchor'),(total/max(count,1)).tolist())),validation=result,
                 anchor_mix_alpha=mix_fraction(epoch-1,100),lr=optimizer.param_groups[0]['lr'],seconds=time.monotonic()-begin,best=improved)
        row['train']['shape_pcc']=train_pcc_sum/max(train_samples,1)
        history.append(row);atomic_json(directory/'history.json',history)
        with (directory/'epoch_log.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        atomic_torch_save(directory/'last.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
            wait=wait,best_rank=best_rank,history=history,rng=capture_rng_state(generator)))
        r=result['direct']['r']
        tqdm.write(f"{ds} f{fold} {variant} ep{epoch:03d} L={total[0]/count:.4f} valPCC={result['shape']['pcc']:.4f} "
            f"R_MAE={r['mae_ms']} cov={r['coverage']} fallback={result['fallback_rate']:.3f} a={row['anchor_mix_alpha']:.3f} "
            f"sec={row['seconds']:.1f} {'BEST' if improved else ''}")
        if wait>=15:break
    checkpoint=torch.load(directory/'best.pt',map_location=device,weights_only=False);model.load_state_dict(checkpoint['model'])
    normal,shape,prob,status=evaluate(model,val,device)
    oracle,oracle_shape,_,_=evaluate(model,val,device,oracle=True)
    atomic_npz(directory/'validation_predictions.npz',dict(shape=shape,oracle_shape=oracle_shape,anchor_probability=prob,
        anchor_status=np.array(status),subject_id=val.subjects))
    atomic_json(directory/'COMPLETE.json',dict(selected_epoch=checkpoint['epoch'],validation=normal,oracle=oracle,test_accessed=False))
    curves(history,directory/'training_curves.png')

def interp(ds,fold,device):
    path=OUT/'interp'/ds/f'fold_{fold:02d}'/'validation.json'
    if path.exists():return
    data=PhaseFold(ds,fold,'val');model=AnchorConditionedREM('interp',in_bins=50 if ds=='mmecg' else 1).to(device)
    checkpoint_path=incumbent_checkpoint(ds,fold)
    checkpoint=torch.load(checkpoint_path,map_location=device,weights_only=False)
    model.load_state_dict(checkpoint['model'],strict=True)
    result,shape,prob,_=evaluate(model,data,device)
    atomic_json(path,dict(validation=result,checkpoint=str(checkpoint_path),selected_epoch=checkpoint['epoch'],test_accessed=False))

def gate_decision(cases):
    base=np.mean([x['interp']['shape']['pcc'] for x in cases])
    base_r=np.mean([x['interp']['direct']['r']['mae_ms'] if x['interp']['direct']['r']['mae_ms'] is not None else np.inf for x in cases])
    means={}
    for variant in ('phase_p','phase_pd'):
        means[variant]=dict(pcc=float(np.mean([x[variant]['validation']['shape']['pcc'] for x in cases])),
            oracle_pcc=float(np.mean([x[variant]['oracle']['shape']['pcc'] for x in cases])),
            r_mae_ms=float(np.mean([x[variant]['validation']['direct']['r']['mae_ms']
                if x[variant]['validation']['direct']['r']['mae_ms'] is not None else np.inf for x in cases])))
    both_oracles_fail=all(x['oracle_pcc']<=base for x in means.values())
    chosen=None
    if not both_oracles_fail:
        for variant in ('phase_p','phase_pd'):
            x=means[variant]
            if x['pcc']>=base-.005 and np.isfinite(base_r) and np.isfinite(x['r_mae_ms']) and x['r_mae_ms']<=base_r+5:
                chosen=variant;break
    return dict(aggregation='three development folds equally weighted',interp_pcc=float(base),interp_r_mae_ms=float(base_r),
        variants=means,both_oracles_no_better_than_interp=both_oracles_fail,decision=chosen or 'Plan A — keep original dual-head',
        stage2_authorized=False,test_accessed=False)

def summarize():
    cases=[];rows=[]
    for ds,fold in (('mmecg',1),('mmecg',2),('crrvs',1)):
        base=json.loads((OUT/'interp'/ds/f'fold_{fold:02d}'/'validation.json').read_text())
        case=dict(dataset=ds,fold=fold,interp=base['validation'])
        evaluations=[('interp',base['selected_epoch'],base['validation'])]
        for variant in ('phase_p','phase_pd'):
            done=json.loads((OUT/'runs'/ds/f'fold_{fold:02d}'/variant/'COMPLETE.json').read_text());case[variant]=done
            evaluations.extend([(variant,done['selected_epoch'],done['validation']),(variant+'_oracle',done['selected_epoch'],done['oracle'])])
        cases.append(case)
        for name,epoch,result in evaluations:
            row=dict(dataset=ds,fold=fold,variant=name,selected_epoch=epoch,shape_pcc=result['shape']['pcc'],shape_rmse=result['shape']['rmse'],
                subject_fisher_pcc_mean=float(np.mean(list(result['subject_fisher_pcc'].values()))),fallback_rate=result['fallback_rate'],valid_unmatched_rate=result['valid_unmatched_rate'])
            for event in 'qrst':
                row[event+'_mae_ms']=result['direct'][event]['mae_ms'];row[event+'_coverage']=result['direct'][event]['coverage']
            rows.append(row)
    with (OUT/'validation_comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    decision=gate_decision(cases);atomic_json(OUT/'gate_decision.json',decision)
    print(json.dumps(decision,ensure_ascii=False,indent=2),flush=True)

def queue(device):
    spec=json.loads(SPEC.read_text());assert spec['training_authorized'] and not spec['test_access_authorized']
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise RuntimeError('training requires clean worktree')
    if not json.loads((PROJECT/'experiments/phase_resampling_20261027/stage0/results.json').read_text())['passed']:raise RuntimeError('Stage0 did not pass')
    atomic_json(OUT/'queue_state.json',dict(stage='RUNNING',tasks=spec['tasks'],test_accessed=False))
    for ds,fold,variant in spec['tasks']:
        cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'--dataset',ds,'--fold',str(fold),'--variant',variant,'--device',str(device)]
        atomic_json(OUT/'queue_state.json',dict(stage='RUNNING',current=[ds,fold,variant],test_accessed=False))
        subprocess.run(cmd,cwd=ROOT,check=True)
    for ds,fold in (('mmecg',1),('mmecg',2),('crrvs',1)):
        subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--dataset',ds,'--fold',str(fold),'--interp','--device',str(device)],cwd=ROOT,check=True)
    summarize()
    atomic_json(OUT/'queue_state.json',dict(stage='COMPLETE_VALIDATION_ONLY',test_accessed=False,stage2_authorized=False))
    print('Stage1 six trainings complete. STOP. Stage2 remains locked.',flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--queue',action='store_true');parser.add_argument('--dataset',choices=('mmecg','crrvs'))
    parser.add_argument('--fold',type=int);parser.add_argument('--variant',choices=('phase_p','phase_pd'))
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--smoke',action='store_true');parser.add_argument('--interp',action='store_true')
    args=parser.parse_args();torch.set_num_threads(2)
    if args.queue:queue(torch.device(args.device))
    elif args.interp:interp(args.dataset,args.fold,torch.device(args.device))
    else:run(args.dataset,args.fold,args.variant,torch.device(args.device),args.smoke)

if __name__=='__main__':main()
