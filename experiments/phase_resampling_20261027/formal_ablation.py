"""Author-approved option1: complete A0-A4/S1/S2; no other experiment routes."""
import os
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys
import csv
import gzip
import json
import time
import hashlib
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset,DataLoader
from tqdm.auto import tqdm

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.phase_resampling_20261027 import train_stage1 as core
from experiments.phase_resampling_20261027.architecture import AnchorConditionedREM,mix_fraction
from experiments.phase_resampling_20261027 import ablation_statistics as stats
frozen=core.frozen
PROJECT=core.PROJECT
OUT=PROJECT/'experiments/phase_resampling_20261027/formal_ablation_20261027'
SPEC=Path(__file__).with_name('formal_spec.json')
core.OUT=OUT
ARMS={'A0':'interp','A1':'window_p','A2':'phase_d','A3':'phase_p','A4':'phase_pd','S1':'interp','S2':'interp'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def model_for(ds,arm):return AnchorConditionedREM(ARMS[arm],in_bins=50 if ds=='mmecg' else 1)

def run_dir(ds,fold,arm):return OUT/'runs'/ds/f'fold_{fold:02d}'/arm

def assert_test_barrier():
    spec=json.loads(SPEC.read_text())
    if not spec['formal_test_authorized']:raise RuntimeError('final test not authorized')
    for ds,n in spec['datasets'].items():
        for fold in range(1,n+1):
            for arm in ARMS:
                if not (run_dir(ds,fold,arm)/'selected.json').exists():raise RuntimeError('all112 checkpoints must be frozen before ANY final test')

def reused_checkpoint(ds,fold,arm):
    if arm=='A0':return core.incumbent_checkpoint(ds,fold)
    if arm in {'A3','A4'} and (ds,fold) in {('mmecg',1),('mmecg',2),('crrvs',1)}:
        return PROJECT/'experiments/phase_resampling_20261027/stage1/runs'/ds/f'fold_{fold:02d}'/ARMS[arm]/'best.pt'
    if arm=='S1' and ds=='mmecg' and fold in (1,5,9):
        return PROJECT/'experiments_mmecg/q1_dual_head_component_ablation_seed42/shape_only'/f'fold_{fold:02d}/checkpoints/best.pt'
    return None

def effective_loss(arm,losses):
    total,shape,anchor=losses
    return shape if arm=='S1' else (.3*anchor if arm=='S2' else total)

def selection(arm,result):
    if arm=='S1':
        s=result['shape'];return s['pcc'],-s['rmse'],s['r2']
    if arm=='S2':return result['independent']['macro_f1_100ms'],-result['loss']['anchor'],0.
    return core.rank(result)

def save_selected(ds,fold,arm,path,reused):
    d=run_dir(ds,fold,arm);d.mkdir(parents=True,exist_ok=True)
    saved=torch.load(path,map_location='cpu',weights_only=False)
    m=model_for(ds,arm);m.load_state_dict(saved['model'],strict=True)
    total=sum(p.numel() for p in m.parameters())
    inactive=sum(p.numel() for p in (m.anchor_head if arm=='S1' else m.shape_head).parameters()) if arm in {'S1','S2'} else 0
    core.atomic_json(d/'selected.json',dict(dataset=ds,fold=fold,arm=arm,checkpoint=str(path),checkpoint_sha256=sha(path),
        selected_epoch=saved['epoch'],reused=reused,parameters_allocated=total,parameters_active=total-inactive,
        selection='validation only',test_evaluated_this_protocol=False))

def preflight():
    spec=json.loads(SPEC.read_text());assert spec['training_authorized'] and spec['formal_test_authorized']
    passed=json.loads((PROJECT/'experiments/phase_resampling_20261027/stage0/results.json').read_text())['passed']
    assert passed
    rows=[]
    for ds,count in spec['datasets'].items():
        for fold in range(1,count+1):
            for arm in ARMS:
                path=reused_checkpoint(ds,fold,arm)
                if path is not None:
                    if not path.exists():raise FileNotFoundError(path)
                    saved=torch.load(path,map_location='cpu',weights_only=False)
                    model_for(ds,arm).load_state_dict(saved['model'],strict=True)
                rows.append(dict(dataset=ds,fold=fold,arm=arm,reuse=path is not None,checkpoint=str(path) if path else None,
                                 sha256=sha(path) if path else None))
    new=sum(not r['reuse'] for r in rows)
    if new!=spec['new_trainings_expected']:raise RuntimeError(f'Expected87 new trainings, got {new}')
    core.atomic_json(OUT/'reuse_audit.json',dict(rows=rows,new_trainings=new,reused=len(rows)-new,
        original_shape_only_audit='MM folds1/5/9: exact state keys/shapes, same seed42+fold, data, loss, optimizer, target and val PCC/-RMSE/R2 selection'))
    print(f'PASS preflight: {len(rows)} variant-folds, {new} new trainings, {len(rows)-new} reused',flush=True)

def train(ds,fold,arm,device,engineering=False):
    core.set_seed(42+fold)
    if reused_checkpoint(ds,fold,arm) is not None and not engineering:
        save_selected(ds,fold,arm,reused_checkpoint(ds,fold,arm),True);return
    d=run_dir(ds,fold,arm) if not engineering else OUT/'engineering'/ds/f'fold_{fold:02d}'/arm
    if (d/'selected.json').exists():return
    d.mkdir(parents=True,exist_ok=True)
    model=model_for(ds,arm).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    tr=core.PhaseFold(ds,fold,'train',limit=16 if engineering else None)
    va=core.PhaseFold(ds,fold,'val',limit=16 if engineering else None)
    generator=torch.Generator().manual_seed(42+fold)
    loader=DataLoader(tr,batch_size=16,shuffle=True,num_workers=0,drop_last=ds=='crrvs',generator=generator)
    start=1;wait=0;best=(-np.inf,-np.inf,-np.inf);history=[]
    if (d/'last.pt').exists():
        ck=torch.load(d/'last.pt',map_location=device,weights_only=False)
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);start=ck['epoch']+1
        wait=ck['wait'];best=tuple(ck['best_rank']);history=ck['history'];core.restore_rng_state(ck['rng'],generator)
    core.atomic_json(d/'meta.json',dict(dataset=ds,fold=fold,arm=arm,seed_base=42,actual_seed=42+fold,
        config=json.loads(SPEC.read_text()),implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        frozen_sources=frozen.SOURCES,test_accessed=False,engineering=engineering,train_split=str(tr.base.h5p),val_split=str(va.base.h5p)))
    for epoch in range(start,(1 if engineering else 100)+1):
        begin=time.monotonic();model.train();sums=np.zeros(4);n=0
        bar=tqdm(loader,desc=f'{ds} f{fold} {arm} ep{epoch}',leave=False,dynamic_ncols=True,mininterval=1,disable=not sys.stderr.isatty())
        for step,b in enumerate(bar,1):
            b={k:v.to(device) if torch.is_tensor(v) else v for k,v in b.items()}
            out=model(b['wsst'],b['bounds'],epoch_index=epoch-1,total_epochs=100)
            losses=frozen.losses(out,b);loss=effective_loss(arm,losses)
            if not torch.isfinite(loss):raise RuntimeError('non-finite loss; stop rather than skip windows')
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            sums+=np.array([x.detach().item() for x in (loss,*losses)]);n+=1
            if step%20==0:bar.set_postfix(loss=f'{sums[0]/n:.4f}')
        result,_,_,_=core.evaluate(model,va,device);current=selection(arm,result);improved=current>best
        if improved:
            best=current;wait=0;core.atomic_torch_save(d/'best.pt',dict(model=model.state_dict(),epoch=epoch,validation=result))
        else:wait+=1
        row=dict(epoch=epoch,train_effective_loss=float(sums[0]/n),train_shape_loss=float(sums[2]/n),train_anchor_loss=float(sums[3]/n),
            validation=result,validation_effective_loss=float(effective_loss(arm,[result['loss'][k] for k in ('total','shape','anchor')])),
            anchor_mix_alpha=mix_fraction(epoch-1,100) if arm in {'A2','A3','A4'} else None,lr=1e-4,best=improved,seconds=time.monotonic()-begin)
        history.append(row);core.atomic_json(d/'history.json',history)
        with (d/'epoch_log.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        core.atomic_torch_save(d/'last.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
            wait=wait,best_rank=best,history=history,rng=core.capture_rng_state(generator)))
        pc='NA' if arm=='S2' else f"{result['shape']['pcc']:.4f}"
        event='NA' if arm=='S1' else f"{result['independent']['macro_f1_100ms']:.4f}"
        print(f"{ds} f{fold:02d} {arm} ep{epoch:03d} loss={sums[0]/n:.4f} valPCC={pc} "
              f"eventF1={event} sec={row['seconds']:.1f} {'BEST' if improved else ''}",flush=True)
        if wait>=15:break
    core.curves([dict(epoch=h['epoch'],train=dict(total=h['train_effective_loss']),validation=h['validation']) for h in history],d/'training_curves.png')
    if not engineering:save_selected(ds,fold,arm,d/'best.pt',False)
    else:core.atomic_json(d/'ENGINEERING_COMPLETE.json',dict(training_check_only=True,not_for_results=True))

class TestFold(Dataset):
    """Separate final-test class; never weaken the Stage1 test guard."""
    def __init__(self,ds,fold,selected):
        if not selected or selected['selection']!='validation only':raise RuntimeError('checkpoint not frozen')
        root='loso_v2' if ds=='mmecg' else 'q1_continuous_kfold';core.data_core._set_split_root(root,ds)
        self.base=core.data_core.Fold(fold,'test','B1',core.data_core.fit_scaler(fold),allow_test=True,
            wsst_source='subject_pool' if ds=='mmecg' else 'per_split',needs_rcg=False)
        cache=OUT/'label_cache'/ds/f'fold_{fold:02d}/test.npz'
        if cache.exists():
            with np.load(cache,allow_pickle=True) as x:self.points=list(x['points']);self.shapes=x['shapes'];self.bounds=x['bounds']
        else:
            with ProcessPoolExecutor(max_workers=8) as pool:labels=list(pool.map(core.labels,self.base.ecg,chunksize=16))
            self.points=[x[0] for x in labels];self.shapes=np.stack([x[1] for x in labels]);self.bounds=np.stack([x[2] for x in labels])
            core.atomic_npz(cache,dict(points=np.array(self.points,object),shapes=self.shapes,bounds=self.bounds))
        with h5py.File(self.base.h5p,'r') as h:
            self.subjects=h['subject_id'][:]
            decode=lambda x:x.decode() if isinstance(x,bytes) else str(x)
            self.records=np.array([decode(x) for x in h['record_id'][:]])
            self.starts=h['start_idx'][:]
            self.scenes=[decode(x) for x in h['state_code'][:]] if 'state_code' in h else ['unknown']*len(self.base)
    def __len__(self):return len(self.base)
    __getitem__=core.PhaseFold.__getitem__

def final_test(ds,fold,arm,device):
    assert_test_barrier()
    d=run_dir(ds,fold,arm)
    if (d/'test_subjects.json').exists():return
    selected=json.loads((d/'selected.json').read_text())
    if sha(selected['checkpoint'])!=selected['checkpoint_sha256']:raise RuntimeError('selected weights changed')
    core.atomic_json(d/'test_state.json',dict(stage='EVALUATING',checkpoint_sha256=selected['checkpoint_sha256']))
    m=model_for(ds,arm).to(device)
    m.load_state_dict(torch.load(selected['checkpoint'],map_location=device,weights_only=False)['model']);m.eval()
    data=TestFold(ds,fold,selected)
    _,shape,prob,status=core.evaluate(m,data,device)
    per_window=frozen.pcc_per_row(shape,data.shapes);rmse=np.sqrt(np.mean((shape-data.shapes)**2,axis=1))
    subjects={}
    for sid in np.unique(data.subjects):
        indices=np.flatnonzero(data.subjects==sid);bucket=frozen.empty_bucket()
        if arm!='S1':
            for i in indices:frozen.accumulate_window(bucket,data.points[i],frozen.structured_decode(prob[i]))
        row=dict(pcc=float(np.tanh(np.mean(np.arctanh(np.clip(per_window[indices],-1+1e-7,1-1e-7))))) if arm!='S2' else None,
            pcc_window_arithmetic=float(per_window[indices].mean()) if arm!='S2' else None,
            rmse=float(rmse[indices].mean()) if arm!='S2' else None,n_windows=len(indices),
            fallback_rate=sum(status[i]=='fallback' for i in indices)/len(indices) if arm in {'A2','A3','A4'} else None,
            valid_unmatched_rate=sum(status[i]=='valid_unmatched' for i in indices)/len(indices) if arm in {'A2','A3','A4'} else None)
        direct=frozen.finalise(bucket)
        for event in 'qrst':
            row[event+'_mae_ms']=direct[event]['mae_ms'] if arm!='S1' else None
            row[event+'_eligible']=direct[event]['eligible'] if arm!='S1' else None
            row[event+'_evaluable']=direct[event]['evaluable'] if arm!='S1' else None
            row[event+'_coverage']=direct[event]['coverage'] if arm!='S1' else None
        subjects[str(int(sid))]=row
    with gzip.open(d/'test_windows.csv.gz','wt',newline='') as f:
        fields=['dataset','fold','arm','subject','record','start','scene','pcc','rmse','anchor_status']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for i in range(len(data)):
            w.writerow(dict(dataset=ds,fold=fold,arm=arm,subject=int(data.subjects[i]),record=data.records[i],start=int(data.starts[i]),
                scene=data.scenes[i],pcc=float(per_window[i]) if arm!='S2' else None,rmse=float(rmse[i]) if arm!='S2' else None,
                anchor_status=status[i] if arm in {'A2','A3','A4'} else 'not_applicable'))
    core.atomic_npz(d/'test_predictions.npz',dict(shape=shape,anchor_probability=prob,subject_id=data.subjects,
        record_id=data.records,start_idx=data.starts,scene=np.array(data.scenes),anchor_status=np.array(status),
        pcc=per_window,rmse=rmse,shape_assessable=np.array(arm!='S2'),events_assessable=np.array(arm!='S1')))
    core.atomic_json(d/'test_subjects.json',subjects)
    core.atomic_json(d/'test_state.json',dict(stage='COMPLETE',test_evaluated_this_protocol=True,checkpoint_sha256=selected['checkpoint_sha256']))
    print('FINAL TEST COMPLETE',ds,fold,arm,flush=True)

def write_csv(name,rows):
    path=OUT/name
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def collect():
    spec=json.loads(SPEC.read_text());table1=[];table2=[];table3=[];supp=[];supp_summary=[];subject_rows=[]
    for ds,nfold in spec['datasets'].items():
        values={arm:{} for arm in ARMS}
        for fold in range(1,nfold+1):
            for arm in ARMS:
                d=run_dir(ds,fold,arm);selected=json.loads((d/'selected.json').read_text())
                subjects=json.loads((d/'test_subjects.json').read_text())
                for sid,row in subjects.items():
                    if sid in values[arm]:raise RuntimeError('test subject appears in multiple folds')
                    values[arm][sid]=row;subject_rows.append(dict(dataset=ds,fold=fold,arm=arm,subject=sid,**row))
                supp.append(dict(dataset=ds,fold=fold,arm=arm,selected_epoch=selected['selected_epoch'],reused=selected['reused'],
                    parameters_allocated=selected['parameters_allocated'],parameters_active=selected['parameters_active'],
                    **{k:float(np.mean([r[k] for r in subjects.values() if r[k] is not None])) if any(r[k] is not None for r in subjects.values()) else None
                       for k in ['fallback_rate','valid_unmatched_rate',*[p+'_coverage' for p in 'qrst']]}))
        ids=set(values['A4'])
        if len(ids)!=(11 if ds=='mmecg' else 30):raise RuntimeError('unexpected test subject count')
        if any(set(x)!=ids for x in values.values()):raise RuntimeError('unpaired subjects')
        for arm in ARMS:
            v=list(values[arm].values());records=[r for r in supp if r['dataset']==ds and r['arm']==arm]
            mean=lambda k:float(np.mean([x[k] for x in v if x[k] is not None])) if any(x[k] is not None for x in v) else None
            entry=dict(dataset=ds,arm=arm,n_subjects=len(v),parameters_allocated=records[0]['parameters_allocated'],
                parameters_active=records[0]['parameters_active'],reused_folds=sum(r['reused'] for r in records),
                fallback_rate=mean('fallback_rate'),valid_unmatched_rate=mean('valid_unmatched_rate'),
                r_subjects_assessable=sum(x['r_mae_ms'] is not None for x in v))
            for event in 'qrst':
                entry[event+'_coverage']=mean(event+'_coverage')
                eligible=sum(x[event+'_eligible'] or 0 for x in v);evaluable=sum(x[event+'_evaluable'] or 0 for x in v)
                entry[event+'_eligible']=eligible if arm!='S1' else None;entry[event+'_evaluable']=evaluable if arm!='S1' else None
                entry[event+'_coverage_pooled']=evaluable/eligible if eligible else None
            supp_summary.append(entry)
        pvals=[];start=len(table1)
        for arm in ('A0','A1','A2','A3','A4'):
            v=values[arm];pc=np.array([v[s]['pcc'] for s in sorted(ids)])*100
            if arm=='A4':delta=lo=hi=p=None
            else:
                differences=stats.contrast(values,{arm:1,'A4':-1},'pcc')*100
                delta,lo,hi=stats.bootstrap_mean(differences);p=stats.signed_rank(differences);pvals.append(p)
            r=[x['r_mae_ms'] for x in v.values() if x['r_mae_ms'] is not None]
            table1.append(dict(dataset=ds,arm=arm,pcc_percent=float(pc.mean()),rmse=float(np.mean([x['rmse'] for x in v.values()])),
                r_mae_ms=float(np.mean(r)) if r else None,r_subjects_assessable=len(r),delta_pp=delta,ci_low_pp=lo,ci_high_pp=hi,p_raw=p,p_holm=None,star=''))
        for row,p in zip(table1[start:start+4],stats.holm(pvals)):row['p_holm']=float(p);row['star']='*' if p<.05 else ''
        pvals=[];start=len(table2)
        for effect,weights in spec['factorial_contrasts'].items():
            differences=stats.contrast(values,weights,'pcc')*100
            delta,lo,hi=stats.bootstrap_mean(differences);p=stats.signed_rank(differences);pvals.append(p)
            r_ids=[s for s in sorted(ids) if all(values[a][s]['r_mae_ms'] is not None for a in weights)]
            rdiff=[sum(w*values[a][s]['r_mae_ms'] for a,w in weights.items()) for s in r_ids]
            table2.append(dict(dataset=ds,effect=effect,delta_pp=delta,ci_low_pp=lo,ci_high_pp=hi,p_raw=p,p_holm=None,star='',
                delta_r_mae_ms=float(np.mean(rdiff)) if rdiff else None,r_complete_subjects=len(r_ids),pcc_subjects=len(ids)))
        for row,p in zip(table2[start:],stats.holm(pvals)):row['p_holm']=float(p);row['star']='*' if p<.05 else ''
        for arm in ('S1','S2','A0','A4'):
            v=list(values[arm].values())
            mean=lambda k:float(np.mean([x[k] for x in v if x[k] is not None])) if any(x[k] is not None for x in v) else None
            table3.append(dict(dataset=ds,arm=arm,pcc_percent=100*mean('pcc') if mean('pcc') is not None else None,rmse=mean('rmse'),
                **{p+'_mae_ms':mean(p+'_mae_ms') for p in 'qrst'},**{p+'_coverage':mean(p+'_coverage') for p in 'qrst'},
                overall_coverage_pct=100*float(np.mean([sum(x[p+'_evaluable'] or 0 for p in 'qrst')/max(sum(x[p+'_eligible'] or 0 for p in 'qrst'),1) for x in v])) if arm!='S1' else None))
    for name,rows in [('table1_modules.csv',table1),('table2_factorial.csv',table2),('table3_multitask.csv',table3),('supplement_S1_per_fold.csv',supp),('supplement_S1.csv',supp_summary),('per_subject.csv',subject_rows)]:write_csv(name,rows)
    render_tables(table1,table2,table3,supp_summary)

def render_tables(t1,t2,t3,supp):
    fmt=lambda x,dec=2:'--' if x is None else f'{x:.{dec}f}'
    lines=['# Complete original-protocol architecture ablation','',
        'Test metrics only after all validation-selected weights were frozen. Subject Fisher-z PCC, then equal-subject means.',
        'Single seed-base42 (actual42+fold). This is not multi-seed evidence. Unequal head capacities are explicitly retained.',
        'Q/R/S/T are peaks. Coupling means forward conditioning with stop-gradient, not bidirectional gradients.','']
    for ds in ('mmecg','crrvs'):
        lines.extend([f'## {ds.upper()} — module ablation','','| Arm | PCC (%) | ΔPCC [95% CI] pp | RMSE | R MAE ms |','|---|---:|---|---:|---:|'])
        for r in t1:
            if r['dataset']==ds:
                delta='—' if r['delta_pp'] is None else f"{r['delta_pp']:+.2f} [{r['ci_low_pp']:+.2f}, {r['ci_high_pp']:+.2f}]{r['star']}"
                lines.append(f"| {r['arm']} | {fmt(r['pcc_percent'])} | {delta} | {fmt(r['rmse'],3)} | {fmt(r['r_mae_ms'])} |")
        lines.extend(['','### Factorial contrasts','','| Effect | ΔPCC [95% CI] pp | ΔR MAE ms |','|---|---|---:|'])
        for r in t2:
            if r['dataset']==ds:lines.append(f"| {r['effect']} | {r['delta_pp']:+.2f} [{r['ci_low_pp']:+.2f}, {r['ci_high_pp']:+.2f}]{r['star']} | {fmt(r['delta_r_mae_ms'])} |")
        lines.extend(['','### Multi-task ablation','','| Arm | PCC (%) | RMSE | Q MAE | R MAE | S MAE | T MAE | Coverage (%) |','|---|---:|---:|---:|---:|---:|---:|---:|'])
        for r in t3:
            if r['dataset']==ds:lines.append('| '+r['arm']+' | '+' | '.join(fmt(r[k],3 if k=='rmse' else 2) for k in ['pcc_percent','rmse','q_mae_ms','r_mae_ms','s_mae_ms','t_mae_ms','overall_coverage_pct'])+' |')
        lines.append('')
    lines.extend(['## Statistical note','',
        'ΔPCC = variant minus A4, percentage points; paired-subject 10000-resample percentile bootstrap 95% CI.',
        'Two-sided Wilcoxon with Holm: four module comparisons per dataset; four factorial PCC contrasts form a separate planned family per dataset. * denotes adjusted p<0.05, including significant inferiority.',
        'Factorial contrasts include architecture/capacity changes, not strictly capacity-controlled causal effects. R-MAE contrasts are descriptive and negative favors the added component.',
        'Coverage, fallback, valid-unmatched rates, assessor counts and allocated/active parameter counts: supplement_S1.csv / per_subject.csv.',
        'Unassessable R errors are NA, never zero; R mean uses assessable subjects with counts explicitly exported. Primary PCC contrasts retain every subject.',
        'Table3 is descriptive. No new baseline training, significance cherry-picking or extra experiment routes.',
        'Existing manuscript and historical figures/results are unchanged.'])
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    # IEEE two-column module table, directly insertable but saved separately.
    index={(r['dataset'],r['arm']):r for r in t1}
    tex=[r'\begin{table*}[t]',r'\centering\footnotesize',r'\caption{Architecture ablation on MMECG and CR\_RVS.}',r'\label{tab:phase_module_ablation}',
         r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{llccc cccc ccc}',r'\toprule',
         r'ID & Variant & Phase & $P$ & Context & \multicolumn{4}{c}{MMECG} & \multicolumn{3}{c}{CR\_RVS} \\',
         r' & & & & & PCC (\%) & $\Delta$PCC [95\% CI] & RMSE & R-MAE & PCC (\%) & $\Delta$PCC [95\% CI] & RMSE \\',r'\midrule']
    names={'A0':'Window-time','A1':'+ event priors','A2':'Phase + context','A3':'Phase + priors','A4':'REM-Net (full)'}
    flags={'A0':('-', '-', '+'),'A1':('-', '+', '+'),'A2':('+','-','+'),'A3':('+','+','-'),'A4':('+','+','+')}
    for arm in ('A0','A1','A2','A3','A4'):
        cells=[arm,names[arm],*flags[arm]]
        for ds in ('mmecg','crrvs'):
            r=index[ds,arm];delta='--' if r['delta_pp'] is None else f"{r['delta_pp']:+.2f} [{r['ci_low_pp']:+.2f},{r['ci_high_pp']:+.2f}]"+(r'$^{*}$' if r['star'] else '')
            cells.extend([fmt(r['pcc_percent']),delta,fmt(r['rmse'],3)])
            if ds=='mmecg':cells.append(fmt(r['r_mae_ms']))
        tex.append(' & '.join(cells)+r' \\')
    tex.extend([r'\bottomrule',r'\end{tabular}',r'\par\smallskip\begin{minipage}{\textwidth}\scriptsize',
        r'$\Delta$PCC is relative to A4 in percentage points. CIs use paired-subject bootstrap; * indicates two-sided Wilcoxon with Holm-adjusted $p<0.05$. Coverage, fallback and parameter counts are in Table S1. Capacities are not matched.',
        r'\end{minipage}',r'\end{table*}'])
    (OUT/'Table_modules.tex').write_text('\n'.join(tex)+'\n')
    ci=lambda r:f"{r['delta_pp']:+.2f} [{r['ci_low_pp']:+.2f},{r['ci_high_pp']:+.2f}]"+(r'$^{*}$' if r['star'] else '')
    idx={(r['dataset'],r['effect']):r for r in t2}
    output=[r'\begin{table*}[t]\centering\footnotesize',r'\caption{Preplanned architecture contrasts.}',r'\label{tab:phase_factorial}',
        r'\setlength{\tabcolsep}{3pt}\begin{tabular}{llcccc}\toprule',
        r'Effect & Contrast & MMECG $\Delta$PCC [95\% CI] & CR\_RVS $\Delta$PCC [95\% CI] & MMECG $\Delta$R-MAE & CR\_RVS $\Delta$R-MAE \\ \midrule']
    formulas={'phase':('Phase',r'$\frac12[(A2-A0)+(A4-A1)]$'),
        'event_probability':('Event priors',r'$\frac12[(A1-A0)+(A4-A2)]$'),
        'interaction':('Interaction',r'$(A4-A1)-(A2-A0)$'),
        'window_context':('Window context',r'$A4-A3$')}
    for effect,(name,formula) in formulas.items():
        mm=idx['mmecg',effect];cw=idx['crrvs',effect]
        output.append(' & '.join([name,formula,ci(mm),ci(cw),fmt(mm['delta_r_mae_ms']),fmt(cw['delta_r_mae_ms'])])+r' \\')
    output.extend([r'\bottomrule\end{tabular}',
        r'\par\smallskip\begin{minipage}{\textwidth}\scriptsize PCC differences are percentage points; R-MAE differences are ms (negative is better). CIs resample paired subjects. * denotes Holm-adjusted $p<0.05$ for the four PCC contrasts per dataset. These architecture contrasts are not capacity-controlled causal estimates.\end{minipage}',r'\end{table*}'])
    (OUT/'Table_factorial.tex').write_text('\n'.join(output)+'\n')
    output=[r'\begin{table*}[t]\centering\footnotesize',r'\caption{Multi-task ablation under window-time and conditioned reconstruction.}',r'\label{tab:phase_multitask}',
        r'\setlength{\tabcolsep}{3pt}\begin{tabular}{llccccccccc}\toprule',
        r'ID & Variant & Shape & Event & PCC (\%) & RMSE & Q-MAE & R-MAE & S-MAE & T-MAE & Coverage (\%) \\']
    task_names={'S1':'Shape-only','S2':'Event-only','A0':'Parallel heads','A4':'REM-Net (full)'}
    for ds in ('mmecg','crrvs'):
        output.append(r'\midrule\multicolumn{11}{c}{'+ds.upper().replace('_',r'\_')+r'} \\ \midrule')
        for r in t3:
            if r['dataset']!=ds:continue
            arm=r['arm'];shape=r'$\times$' if arm=='S2' else r'$\checkmark$';event=r'$\times$' if arm=='S1' else r'$\checkmark$'
            if arm=='A4':event+=r' (cond.)'
            output.append(' & '.join([arm,task_names[arm],shape,event]+[fmt(r[k],3 if k=='rmse' else 2) for k in ['pcc_percent','rmse','q_mae_ms','r_mae_ms','s_mae_ms','t_mae_ms','overall_coverage_pct']])+r' \\')
    output.extend([r'\bottomrule\end{tabular}',
        r'\par\smallskip\begin{minipage}{\textwidth}\scriptsize MAEs are ms. Conditional forward coupling uses stop-gradient on anchors and event probabilities. Single-task inactive heads are retained but not assessed. Coverage is the subject-macro fraction of eligible reference Q/R/S/T endpoints with matched predictions. Table S1 gives endpoint-specific coverage.\end{minipage}',r'\end{table*}'])
    (OUT/'Table_multitask.tex').write_text('\n'.join(output)+'\n')
    fields=['dataset','arm','parameters_allocated','parameters_active','fallback_rate','valid_unmatched_rate','q_coverage','r_coverage','s_coverage','t_coverage']
    output=[r'\begin{table*}[t]\centering\scriptsize',r'\caption{Coverage, fallback and allocated/active parameters. Rates are percentages; non-applicable entries are --.}',r'\label{tab:phase_supplement_s1}',
        r'\begin{tabular}{llrrrrrrrr}\toprule',r'Dataset & Arm & Params & Active & Fallback & Unmatched & Q cov. & R cov. & S cov. & T cov. \\ \midrule']
    for r in supp:
        cells=[r['dataset'].upper().replace('_',r'\_'),r['arm'],str(r['parameters_allocated']),str(r['parameters_active'])]
        cells.extend('--' if r[k] is None else f'{100*r[k]:.1f}' for k in fields[4:])
        output.append(' & '.join(cells)+r' \\')
    output.extend([r'\bottomrule\end{tabular}',r'\end{table*}']);(OUT/'Table_S1.tex').write_text('\n'.join(output)+'\n')

def queue(device):
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise RuntimeError('requires clean committed worktree')
    preflight();spec=json.loads(SPEC.read_text())
    tasks=[(ds,fold,a) for ds,n in spec['datasets'].items() for fold in range(1,n+1) for a in ARMS]
    for ds,fold,arm in tasks:
        core.atomic_json(OUT/'queue_state.json',dict(stage='TRAINING_OR_FREEZING',current=[ds,fold,arm],test_started=False))
        subprocess.run([sys.executable,'-u','-B',str(Path(__file__).resolve()),'--action','train','--dataset',ds,'--fold',str(fold),'--arm',arm,'--device',str(device)],cwd=ROOT,check=True)
    for ds,fold,arm in tasks:
        core.atomic_json(OUT/'queue_state.json',dict(stage='FINAL_TEST',current=[ds,fold,arm],all_training_frozen=True))
        subprocess.run([sys.executable,'-u','-B',str(Path(__file__).resolve()),'--action','test','--dataset',ds,'--fold',str(fold),'--arm',arm,'--device',str(device)],cwd=ROOT,check=True)
    collect();core.atomic_json(OUT/'queue_state.json',dict(stage='COMPLETE',new_experiment_routes_authorized=False))
    print('Complete ablation tables exported. STOP.',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--action',choices=('preflight','queue','train','test','collect','engineering'),required=True)
    p.add_argument('--dataset',choices=('mmecg','crrvs'));p.add_argument('--fold',type=int);p.add_argument('--arm',choices=tuple(ARMS));p.add_argument('--device',default='cuda:0')
    a=p.parse_args();torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);device=torch.device(a.device)
    if a.action in {'train','test','engineering'}:
        spec=json.loads(SPEC.read_text())
        if a.dataset not in spec['datasets'] or a.fold is None or not 1<=a.fold<=spec['datasets'][a.dataset] or a.arm is None:raise ValueError('unapproved fold/variant')
    if a.action=='preflight':preflight()
    elif a.action=='queue':
        try:queue(device)
        except Exception as error:
            core.atomic_json(OUT/'queue_state.json',dict(stage='FAILED',error=repr(error),no_automatic_route_change=True))
            raise
    elif a.action=='collect':collect()
    elif a.action in {'train','engineering'}:train(a.dataset,a.fold,a.arm,device,a.action=='engineering')
    else:final_test(a.dataset,a.fold,a.arm,device)

if __name__=='__main__':main()
