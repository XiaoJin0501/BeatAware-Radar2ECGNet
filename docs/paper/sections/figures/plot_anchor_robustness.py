"""Frozen A4 anchor robustness, no training. Run --infer, then default plot/report."""
import os
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys
import csv
import json
import argparse
import subprocess
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator,FuncFormatter
from tqdm.auto import tqdm

ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from experiments.phase_resampling_20261027 import formal_ablation as original
from experiments.phase_resampling_20261027.architecture import phase_alignment,phase_resample,predicted_pairs
from experiments.phase_resampling_20261027.ablation_statistics import bootstrap_mean

FIGURES=ROOT/'docs/paper/sections/figures'
ASSETS=FIGURES/'source_data/anchor_robustness_20261028'
SIGMAS=(0,5,10,20,40,80)
COLOR='#31688E'

def jitter_pairs(reference,sigma_ms,rng,fallback,max_tries=10):
    """Reject entire pairs, independent boundary noise, no clipping/boundary repair."""
    reference=np.asarray(reference,float);used=reference.copy();pending=np.arange(len(reference));attempts=np.zeros(len(reference),int)
    for _ in range(max_tries):
        if not len(pending):break
        proposal=reference[pending]+rng.normal(0,sigma_ms/5,size=(len(pending),2))
        attempts[pending]+=1;rr=proposal[:,1]-proposal[:,0]
        valid=np.isfinite(proposal).all(1)&(rr>=60)&(rr<=400)
        used[pending[valid]]=proposal[valid];pending=pending[~valid]
    failed=np.zeros(len(reference),bool);failed[pending]=True;used[pending]=np.asarray(fallback)[pending]
    return used,failed,attempts

def crossover(x,lower):
    order=np.argsort(x);x=np.asarray(x)[order];lower=np.asarray(lower)[order]
    indices=np.flatnonzero(lower<=0)
    if not len(indices):return None
    i=int(indices[0])
    if i==0:return float(x[0])
    return float(x[i-1]+(x[i]-x[i-1])*lower[i-1]/(lower[i-1]-lower[i]))

def save_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

@torch.no_grad()
def infer(device):
    ASSETS.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    # Match the ORIGINAL final-test subprocess defaults (not training flags).
    torch.backends.cudnn.deterministic=False;torch.backends.cudnn.benchmark=False
    for ds,nfold in (('mmecg',11),('crrvs',5)):
        rngs={(s,r):np.random.default_rng(r) for s in SIGMAS for r in range(5)}
        for fold in range(1,nfold+1):
            path=ASSETS/f'{ds}_fold_{fold:02d}.npz';state=path.with_suffix('.json')
            directory=original.run_dir(ds,fold,'A4');selected=json.loads((directory/'selected.json').read_text())
            if path.exists() and state.exists():
                cached=json.loads(state.read_text())
                if cached['checkpoint_sha256']!=original.sha(selected['checkpoint']):raise RuntimeError('cached weights changed')
                for key,rng in rngs.items():rng.bit_generator.state=cached['rng_state_after'][f'{key[0]}_{key[1]}']
                continue
            if original.sha(selected['checkpoint'])!=selected['checkpoint_sha256']:raise RuntimeError('frozen checkpoint changed')
            data=original.TestFold(ds,fold,selected)
            reference=np.asarray(data.bounds,float)
            if np.any((np.diff(reference,axis=1)<60)|(np.diff(reference,axis=1)>400)):raise RuntimeError('sigma0 would not be pure oracle')
            model=original.model_for(ds,'A4').to(device)
            model.load_state_dict(torch.load(selected['checkpoint'],map_location=device,weights_only=False)['model']);model.eval()
            with np.load(directory/'test_predictions.npz') as z:incumbent=z['shape']
            n=len(data);pcs=np.empty((6,5,n),np.float32);errors=np.empty((6,5,n),np.float64)
            failures=np.zeros((6,5,n),bool);outside=np.zeros((6,5,n),np.float32)
            pred_pc=np.empty(n,np.float32);pred_error=np.empty(n,np.float64);pred_matched=np.zeros(n,bool);pred_fallback=np.zeros(n,bool)
            max_pred_difference=0.;max_oracle_difference=0.
            for begin in tqdm(range(0,n,16),desc=f'{ds} fold{fold} jitter',disable=not sys.stderr.isatty(),mininterval=1):
                end=min(begin+16,n)
                # Exactly original input preprocessing, no model/BN/RNG updates.
                inputs=torch.stack([data.base[i]['wsst'] for i in range(begin,end)]).to(device)
                refs=torch.as_tensor(reference[begin:end],device=device,dtype=inputs.dtype)
                h=model.backbone(model.stem(inputs));logits=model.anchor_head(F.interpolate(h,size=1600,mode='linear',align_corners=False))
                h_full=F.interpolate(h,size=1600,mode='linear',align_corners=True)
                h_window=F.interpolate(h,size=200,mode='linear',align_corners=False);p_full=logits.sigmoid().detach()
                def shape_at(bounds):
                    ti=phase_alignment(bounds)
                    features=torch.cat((phase_resample(h_full,ti),phase_resample(p_full,ti),h_window),dim=1)
                    return model.shape_head(features).sigmoid()[:,0],ti
                bounds,valid,status=predicted_pairs(logits,refs)
                pred,ti=shape_at(bounds)
                pred_numpy=pred.cpu().numpy();max_pred_difference=max(max_pred_difference,float(abs(pred_numpy-incumbent[begin:end]).max()))
                pred_pc[begin:end]=original.frozen.pcc_per_row(pred_numpy,data.shapes[begin:end])
                pred_error[begin:end]=abs(bounds.cpu().numpy()-reference[begin:end]).mean(1)*5
                pred_matched[begin:end]=[s=='matched' for s in status];pred_fallback[begin:end]=~valid.cpu().numpy()
                fallback=bounds.cpu().numpy().copy()
                for j,p in enumerate(p_full.cpu().numpy()):
                    rs=original.frozen.structured_decode(p)['r'];rr=np.diff(rs)
                    fallback[j,1]=fallback[j,0]+(float(np.median(rr)) if len(rr) else 160.)
                if begin==0:
                    direct=model.decode_features(h,logits,refs,oracle=True)['shape'][:,0]
                    ref_pred,_=shape_at(refs);max_oracle_difference=float(abs(direct-ref_pred).max().cpu())
                    if max_oracle_difference>1e-6:raise RuntimeError('oracle injection differs from existing decode_features')
                for si,sigma in enumerate(SIGMAS):
                    for repeat in range(5):
                        used,failed,_=jitter_pairs(reference[begin:end],sigma,rngs[sigma,repeat],fallback)
                        used_tensor=torch.as_tensor(used,dtype=inputs.dtype,device=device)
                        y,ti=shape_at(used_tensor)
                        pcs[si,repeat,begin:end]=original.frozen.pcc_per_row(y.cpu().numpy(),data.shapes[begin:end])
                        errors[si,repeat,begin:end]=abs(used_tensor.cpu().numpy()-reference[begin:end]).mean(1)*5
                        failures[si,repeat,begin:end]=failed
                        outside[si,repeat,begin:end]=((ti<0)|(ti>1599)).float().mean(1).cpu().numpy()
            if max_pred_difference>1e-5:raise RuntimeError(f'predicted shape changed: {max_pred_difference}')
            original.core.atomic_npz(path,dict(pcc=pcs,realised_error_ms=errors,fallback=failures,outside_grid_fraction=outside,
                predicted_pcc=pred_pc,predicted_error_ms=pred_error,predicted_matched=pred_matched,predicted_fallback=pred_fallback,
                subject=data.subjects,record=data.records,start=data.starts))
            original.core.atomic_json(state,dict(dataset=ds,fold=fold,checkpoint=selected['checkpoint'],checkpoint_sha256=selected['checkpoint_sha256'],
                max_predicted_shape_difference=max_pred_difference,max_oracle_injection_difference=max_oracle_difference,
                rng_state_after={f'{k[0]}_{k[1]}':rng.bit_generator.state for k,rng in rngs.items()}))
            print('completed inference',ds,fold,'of',nfold,'windows',n,flush=True)

def fisher(pc):return float(np.tanh(np.mean(np.arctanh(np.clip(pc,-1+1e-7,1-1e-7)))))

def aggregate():
    rows=[];details=[];repeat_rows=[];points=[];crossings={};datasets={}
    for ds,nfold in (('mmecg',11),('crrvs',5)):
        subjects={};context={};old_pred={}
        for fold in range(1,nfold+1):
            with np.load(ASSETS/f'{ds}_fold_{fold:02d}.npz') as z:a={k:z[k] for k in z.files}
            for sid in np.unique(a['subject']):
                sid=str(int(sid));idx=a['subject']==int(sid)
                if sid in subjects:raise RuntimeError('subject appears in more than one test fold')
                subjects[sid]={}
                for si,sigma in enumerate(SIGMAS):
                    rep=[]
                    for r in range(5):
                        entry=dict(dataset=ds,subject=sid,sigma_ms=sigma,repeat=r,pcc=fisher(a['pcc'][si,r,idx]),
                            realised_err_ms=float(a['realised_error_ms'][si,r,idx].mean()),fallback_rate=float(a['fallback'][si,r,idx].mean()),
                            outside_grid_fraction=float(a['outside_grid_fraction'][si,r,idx].mean()))
                        repeat_rows.append(entry);rep.append(entry)
                    subjects[sid][sigma]={k:float(np.mean([v[k] for v in rep])) for k in ('pcc','realised_err_ms','fallback_rate','outside_grid_fraction')}
                match=a['predicted_matched'][idx]
                subjects[sid]['predicted']=dict(pcc=fisher(a['predicted_pcc'][idx]),matched_windows=int(match.sum()),n_windows=int(idx.sum()),
                    realised_err_ms=float(a['predicted_error_ms'][idx][match].mean()) if match.any() else None,
                    fallback_rate=float(a['predicted_fallback'][idx].mean()))
            baseline=json.loads((original.run_dir(ds,fold,'A0')/'test_subjects.json').read_text())
            reference=json.loads((original.run_dir(ds,fold,'A4')/'test_subjects.json').read_text())
            context.update({s:v['pcc'] for s,v in baseline.items()});old_pred.update({s:v['pcc'] for s,v in reference.items()})
        ids=sorted(subjects);oracle=np.array([subjects[s][0]['pcc'] for s in ids]);base=np.array([context[s] for s in ids])
        dataset_rows=[];lowers=[]
        for sigma in SIGMAS:
            vals=np.array([subjects[s][sigma]['pcc'] for s in ids]);mean,lo,hi=bootstrap_mean(vals)
            delta,dl,dh=bootstrap_mean(vals-oracle);difference,cl,ch=bootstrap_mean(vals-base)
            row=dict(dataset=ds,sigma_ms=sigma,realised_err_ms=float(np.mean([subjects[s][sigma]['realised_err_ms'] for s in ids])),
                pcc_mean=mean,ci_low=lo,ci_high=hi,delta_vs_oracle=delta,fallback_rate=float(np.mean([subjects[s][sigma]['fallback_rate'] for s in ids])))
            rows.append(row);dataset_rows.append(row);lowers.append(cl)
            details.append(dict(**row,delta_ci_low=dl,delta_ci_high=dh,delta_vs_context=difference,context_diff_ci_low=cl,context_diff_ci_high=ch,
                n_subjects=len(ids),outside_grid_fraction=float(np.mean([subjects[s][sigma]['outside_grid_fraction'] for s in ids]))))
        pred=np.array([subjects[s]['predicted']['pcc'] for s in ids]);reproduction=float(np.max(abs(pred-np.array([old_pred[s] for s in ids]))))
        if reproduction>1e-6:raise RuntimeError(f'predicted subject PCC changed: {reproduction}')
        for name,vals in [('predicted',pred),('context',base)]:
            mean,lo,hi=bootstrap_mean(vals)
            matched=[subjects[s]['predicted']['realised_err_ms'] for s in ids if subjects[s]['predicted']['realised_err_ms'] is not None]
            points.append(dict(dataset=ds,point=name,pcc_mean=mean,ci_low=lo,ci_high=hi,
                realised_err_ms=float(np.mean(matched)) if name=='predicted' and matched else None,
                subjects_with_anchor_matches=len(matched) if name=='predicted' else None,n_subjects=len(ids),
                matched_windows=sum(subjects[s]['predicted']['matched_windows'] for s in ids) if name=='predicted' else None,
                all_windows=sum(subjects[s]['predicted']['n_windows'] for s in ids),
                fallback_rate=float(np.mean([subjects[s]['predicted']['fallback_rate'] for s in ids])) if name=='predicted' else None))
        x=[r['realised_err_ms'] for r in dataset_rows]
        cross=crossover(x,lowers);mean_cross=crossover(x,[r['pcc_mean']-base.mean() for r in dataset_rows])
        crossings[ds]=dict(statistical_crossover_realised_ms=cross,mean_crossover_realised_ms=mean_cross,
            never_crossed=cross is None,largest_nominal_sigma_ms=80,largest_realised_ms=max(x),
            oracle_already_not_above_context=lowers[0]<=0,predicted_subject_reproduction_max_abs=reproduction)
        datasets[ds]=subjects
    save_csv(FIGURES/'anchor_robustness.csv',rows);save_csv(ASSETS/'anchor_robustness_statistics.csv',details)
    save_csv(ASSETS/'per_subject_repeat.csv',repeat_rows);save_csv(ASSETS/'reference_points.csv',points)
    (ASSETS/'crossovers.json').write_text(json.dumps(crossings,indent=2)+'\n')
    return rows,details,points,crossings

def plot(rows,points):
    plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':7,'axes.labelsize':8,
        'xtick.labelsize':7,'ytick.labelsize':7,'axes.titlesize':8,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.linewidth':.6,'xtick.direction':'out','ytick.direction':'out','xtick.major.size':2.5,'ytick.major.size':2.5})
    fig,axes=plt.subplots(1,2,figsize=(3.5,1.75))
    for ax,ds,letter,title in zip(axes,('mmecg','crrvs'),('a','b'),('MMECG (77-GHz FMCW)','CR_RVS (24-GHz CW)')):
        r=[v for v in rows if v['dataset']==ds];r.sort(key=lambda v:v['realised_err_ms'])
        x=np.array([v['realised_err_ms'] for v in r]);y=100*np.array([v['pcc_mean'] for v in r])
        lo=100*np.array([v['ci_low'] for v in r]);hi=100*np.array([v['ci_high'] for v in r])
        pred=next(p for p in points if p['dataset']==ds and p['point']=='predicted')
        ctx=next(p for p in points if p['dataset']==ds and p['point']=='context')
        ax.fill_between(x,lo,hi,color=COLOR,alpha=.18,lw=0)
        ax.plot(x,y,'-o',color=COLOR,lw=1.2,ms=3.5)
        ax.axhline(ctx['pcc_mean']*100,color='#7F7F7F',ls='--',lw=1)
        ax.errorbar(pred['realised_err_ms'],100*pred['pcc_mean'],yerr=[[100*(pred['pcc_mean']-pred['ci_low'])],[100*(pred['ci_high']-pred['pcc_mean'])]],
            marker='*',ms=7,mfc=COLOR,mec='black',mew=.6,color=COLOR,elinewidth=.7,capsize=2,lw=0,zorder=5)
        bottom=min(lo.min(),100*pred['ci_low'],100*ctx['pcc_mean']);top=max(hi.max(),100*pred['ci_high'],100*ctx['pcc_mean'])
        pad=.1*max(top-bottom,1);ax.set_ylim(bottom-pad,top+pad);ax.set_xlim(0,1.06*max(x.max(),pred['realised_err_ms']))
        ticks=[v['realised_err_ms'] for v in r if v['sigma_ms'] in (0,10,20,40,80)]
        ax.set_xticks(ticks);ax.set_xticklabels([f'{v:.0f}' for v in ticks]);ax.yaxis.set_major_locator(MaxNLocator(4,steps=[1,2,2.5,5,10]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'))
        ax.set(xlabel='Anchor error (ms)',ylabel='PCC (%)',title=title)
        ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True);ax.grid(axis='y',color='#E6E6E6',lw=.5)
        ax.text(-.22,1.12,f'({letter})',transform=ax.transAxes,fontweight='bold',fontsize=8,va='bottom')
        if ds=='mmecg':
            ax.annotate('reference anchors',xy=(x[0],y[0]),xytext=(4,8),textcoords='offset points',fontsize=5.5,
                        arrowprops=dict(arrowstyle='-',color=COLOR,lw=.45,shrinkA=1,shrinkB=3),color=COLOR)
    handles=[Line2D([0],[0],color=COLOR,ls='-',marker='o',ms=3,lw=1.2,label='REM-Net (jittered)'),
        Line2D([0],[0],color=COLOR,ls='none',marker='*',ms=7,mec='black',mew=.6,label='REM-Net (predicted anchors)'),
        Line2D([0],[0],color='#7F7F7F',ls='--',lw=1,label='Context only')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,-.01),ncol=3,frameon=False,fontsize=6.5,
               handlelength=1.2,columnspacing=.6,handletextpad=.35)
    fig.subplots_adjust(left=.13,right=.985,bottom=.32,top=.79,wspace=.55)
    for ext in ('pdf','svg','png'):fig.savefig(FIGURES/f'fig_anchor_robustness.{ext}',dpi=600)
    plt.close(fig)

def report(rows,details,points,crossings):
    lines=['# REM-Net anchor robustness — inference-only oracle diagnostic','',
        'All16 final A4 phase_pd seed-base42 checkpoints; original Shape-200 targets and subject Fisher-z aggregation. No training, test-driven tuning, old figure edits or checkpoint changes.',
        'CSV PCC/CI/delta are fractions0..1; tables show PCC percent and delta percentage points. CI uses10000 paired-subject bootstrap draws.',
        'Five repeats are averaged WITHIN subjects before dataset means/CIs. Error is the realised mean absolute error of both used boundaries, in ms.','']
    for ds in ('mmecg','crrvs'):
        lines.extend([f'## {ds.upper()}','','| Sigma ms | Realised error ms | PCC % [95% CI] | Δ vs oracle pp [95% CI] | Fallback % |',
            '|---:|---:|---|---|---:|'])
        for r in details:
            if r['dataset']==ds:lines.append(f"| {r['sigma_ms']} | {r['realised_err_ms']:.2f} | {100*r['pcc_mean']:.2f} [{100*r['ci_low']:.2f}, {100*r['ci_high']:.2f}] | {100*r['delta_vs_oracle']:+.3f} [{100*r['delta_ci_low']:+.3f}, {100*r['delta_ci_high']:+.3f}] | {100*r['fallback_rate']:.3f} |")
        for p in points:
            if p['dataset']!=ds:continue
            lines.extend(['',f"{p['point']}: PCC={100*p['pcc_mean']:.2f}% [{100*p['ci_low']:.2f}, {100*p['ci_high']:.2f}]."])
            if p['point']=='predicted':lines.append(f"Matched-window realised error={p['realised_err_ms']:.2f}ms; matched windows={p['matched_windows']}/{p['all_windows']}; subjects with matches={p['subjects_with_anchor_matches']}/{p['n_subjects']}; fallback={100*p['fallback_rate']:.2f}%. PCC includes ALL windows, not only matches.")
        c=crossings[ds]
        label=f"{c['statistical_crossover_realised_ms']:.2f}ms" if c['statistical_crossover_realised_ms'] is not None else f">80ms nominal sigma (realised >{c['largest_realised_ms']:.2f}ms within the tested range)"
        lines.extend(['',f'Statistical crossover: **{label}**.'])
        if c['oracle_already_not_above_context']:lines.append('At sigma0 the oracle is already not demonstrably above Context only; no positive noise-tolerance threshold is supported.')
        central=f"{c['mean_crossover_realised_ms']:.2f}ms" if c['mean_crossover_realised_ms'] is not None else f"not observed through sigma80ms (realised {c['largest_realised_ms']:.2f}ms)"
        lines.append(f'Central mean crossing (separate descriptive quantity): {central}.')
    lines.extend(['','## Definitions, limitations and deviations','',
        '* Jitter is GT-assisted diagnostic, not deployable reconstruction performance. Event probabilities remain model predictions, not GT probability tracks.',
        '* The predicted star uses conditional matched-window anchor error on x, but unconditional all-window PCC on y, as requested; it is not perfectly equivalent to homogeneous Gaussian jitter.',
        '* Reference-anchor oracle is an intervention on a decoder trained with a reference-to-predicted curriculum, not a guaranteed numerical upper bound. Predicted-anchor PCC can exceed this reference-anchor point.',
        '* Crossover interpolates the lower CI boundary of the SUBJECT-PAIRED difference versus A0. It is not the intersection of separate marginal CI bands.',
        '* A statistical crossover at0 is retained; no manipulation to create a favorable curve. >80 refers to nominal sigma, not measured x; both bounds are reported.',
        '* All reference RR bounds satisfy constraints. Fractional anchors and exclusive-right endpoint convention preserved. Border sampling handles out-of-window coordinates; rates are in companion statistics CSV.',
        '* Noise streams use local fixed repeat seeds0–4, reset for each dataset/sigma (common random numbers across sigma); independent boundary draws. Redraw max10, then GT-free standard predicted fallback.',
        '* Color#31688E is reused from existing REM-Net comparison figures. Dedicated source-data/REPORT.md avoids generic-report filename collisions.',
        '* Oracle annotation is shown with a short leader in MMECG only; omitted in CR_RVS to avoid collision with the nearby predicted-star CI whisker in the single-column layout.',
        '* Original final-test cuDNN evaluation defaults (deterministic=False, benchmark=False) are preserved. The initial mismatch under training-style flags was fixed without relaxing the replay threshold.',
        '* Weight hashes and exact predicted/oracle injection equivalence checks are in16 fold JSON manifests; per-window diagnostic arrays and per-subject repeat results are preserved.',
        '* Existing figures, targets, model code and original evaluation artifacts are unchanged.'])
    (ASSETS/'REPORT.md').write_text('\n'.join(lines)+'\n')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--infer',action='store_true');parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    if args.infer:infer(torch.device(args.device))
    rows,details,points,crossings=aggregate();plot(rows,points);report(rows,details,points,crossings)
    manifest=dict(inference_only=True,training=False,seed_base=42,repeats=list(range(5)),bootstrap_subject_resamples=10000,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        protocol_sha256=original.sha(ASSETS/'PROTOCOL.md'),script_sha256=original.sha(__file__),
        model_code_sha256=original.sha(ROOT/'experiments/phase_resampling_20261027/architecture.py'),
        bootstrap_code_sha256=original.sha(ROOT/'experiments/phase_resampling_20261027/ablation_statistics.py'),
        folds=[json.loads(p.read_text()) for p in sorted(ASSETS.glob('*_fold_*.json'))],
        output_sha256={str(p.relative_to(ROOT)):original.sha(p) for p in [FIGURES/'anchor_robustness.csv',*(FIGURES/f'fig_anchor_robustness.{e}' for e in ('pdf','svg','png'))]})
    (ASSETS/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Anchor robustness inference/exports complete; no training.',flush=True)

if __name__=='__main__':main()
