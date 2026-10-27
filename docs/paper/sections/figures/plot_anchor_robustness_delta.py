"""Paired ΔPCC revision; reuses frozen old observations, extends inference only."""
import sys
import csv
import json
import shutil
import argparse
from pathlib import Path
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator,FuncFormatter,Formatter
import plot_anchor_robustness as old

FIGURES=old.FIGURES
ASSETS=FIGURES/'source_data/anchor_robustness_delta_20261028'
EXTENSION=(128,256,512,1024)

def subject_balanced_median(arrays):
    arrays=[np.sort(np.asarray(a,float)) for a in arrays if len(a)]
    x=np.unique(np.concatenate(arrays))
    cdf=np.mean([np.searchsorted(a,x,side='right')/len(a) for a in arrays],axis=0)
    return float(x[np.flatnonzero(cdf>=.5-1e-12)[0]])

def error_components(signed):
    signed=np.asarray(signed,float);common=signed.mean(1);differential=(signed[:,1]-signed[:,0])/2
    return dict(anchor_error_ms=abs(signed).mean(1),rr_span_error_ms=abs(signed[:,1]-signed[:,0]),
        common_abs_ms=abs(common),differential_abs_ms=abs(differential),same_sign=(signed[:,0]*signed[:,1]>0).astype(float),
        common_energy=common**2,differential_energy=differential**2)

def build(extended):
    # Old repeated-subject estimates are immutable; this also avoids recomputing
    # old sigma conditions when extending the grid.
    with (old.ASSETS/'per_subject_repeat.csv').open() as f:original=list(csv.DictReader(f))
    subjects={ds:{} for ds in ('mmecg','crrvs')}
    for r in original:
        ds=r['dataset'];s=r['subject'];sigma=int(r['sigma_ms'])
        subjects[ds].setdefault(s,{}).setdefault(sigma,[]).append({k:float(r[k]) for k in ('pcc','realised_err_ms','fallback_rate')})
    baselines={};predicted={};components=[];median_arrays={}
    for ds,nfold in (('mmecg',11),('crrvs',5)):
        baseline={};predicted[ds]={}
        for fold in range(1,nfold+1):
            base=json.loads((old.original.run_dir(ds,fold,'A0')/'test_subjects.json').read_text())
            baseline.update({s:r['pcc'] for s,r in base.items()})
            for sigma in extended:
                with np.load(ASSETS/f'sigma_{sigma}'/f'{ds}_fold_{fold:02d}.npz') as z:
                    for sid in np.unique(z['subject']):
                        s=str(int(sid));idx=z['subject']==sid
                        if sigma in subjects[ds][s]:raise RuntimeError('duplicate sigma')
                        subjects[ds][s][sigma]=[dict(pcc=old.fisher(z['pcc'][0,r,idx]),
                            realised_err_ms=float(z['realised_error_ms'][0,r,idx].mean()),fallback_rate=float(z['fallback'][0,r,idx].mean())) for r in range(5)]
            # Anchor component diagnostics only require the first extension.
            with np.load(ASSETS/f'sigma_{extended[0]}'/f'{ds}_fold_{fold:02d}.npz') as z:
                with np.load(old.ASSETS/f'{ds}_fold_{fold:02d}.npz') as frozen:
                    assert np.array_equal(z['predicted_pcc'],frozen['predicted_pcc'])
                    assert np.array_equal(z['predicted_matched'],frozen['predicted_matched'])
                for sid in np.unique(z['subject']):
                    s=str(int(sid));idx=z['subject']==sid
                    predicted[ds][s]=dict(pcc=old.fisher(z['predicted_pcc'][idx]),
                        realised_err_ms=float(z['predicted_error_ms'][idx][z['predicted_matched'][idx]].mean()) if z['predicted_matched'][idx].any() else None)
                    for group,mask in [('matched',idx&z['predicted_matched']),('valid',idx&z['predicted_valid']),('all',idx)]:
                        if not mask.any():continue
                        values=error_components(z['predicted_signed_errors_ms'][mask])
                        target=median_arrays.setdefault((ds,group),dict(r=[],rr=[]))
                        target['r'].append(abs(z['predicted_signed_errors_ms'][mask,0]))
                        target['rr'].append(values['rr_span_error_ms'])
                        energy=values['common_energy'].mean()+values['differential_energy'].mean()
                        components.append(dict(dataset=ds,subject=s,group=group,n_windows=int(mask.sum()),
                            **{k:float(v.mean()) for k,v in values.items()},common_energy_share=float(values['common_energy'].mean()/energy) if energy>0 else None))
        baselines[ds]=baseline
    rows=[];per_subject=[];points=[];crossings={};summaries=[]
    for ds in subjects:
        ids=sorted(subjects[ds]);ctx=np.array([baselines[ds][s] for s in ids]);all_sigmas=(*old.SIGMAS,*extended)
        for sigma in all_sigmas:
            means={s:{k:float(np.mean([r[k] for r in subjects[ds][s][sigma]])) for k in ('pcc','realised_err_ms','fallback_rate')} for s in ids}
            d=np.array([means[s]['pcc']-baselines[ds][s] for s in ids]);mean,lo,hi=old.bootstrap_mean(d)
            rows.append(dict(dataset=ds,sigma_ms=sigma,realised_err_ms=float(np.mean([means[s]['realised_err_ms'] for s in ids])),
                delta_context_pp=100*mean,ci_low_pp=100*lo,ci_high_pp=100*hi,pcc_absolute=float(np.mean([means[s]['pcc'] for s in ids])),
                fallback_rate=float(np.mean([means[s]['fallback_rate'] for s in ids]))))
            per_subject.extend(dict(dataset=ds,subject=s,sigma_ms=sigma,**means[s],context_pcc=baselines[ds][s],delta_context_pp=100*(means[s]['pcc']-baselines[ds][s])) for s in ids)
        pred=np.array([predicted[ds][s]['pcc'] for s in ids]);d=pred-ctx;mean,lo,hi=old.bootstrap_mean(d)
        anchor_errors=[predicted[ds][s]['realised_err_ms'] for s in ids if predicted[ds][s]['realised_err_ms'] is not None]
        points.append(dict(dataset=ds,realised_err_ms=float(np.mean(anchor_errors)),delta_context_pp=100*mean,
            ci_low_pp=100*lo,ci_high_pp=100*hi,pcc_absolute=float(pred.mean()),context_pcc_absolute=float(ctx.mean()),
            n_subjects=len(ids),n_subjects_with_anchor_matches=len(anchor_errors)))
        rr=[r for r in rows if r['dataset']==ds];x=[r['realised_err_ms'] for r in rr]
        crossings[ds]=dict(crossover_realised_ms=old.crossover(x,[r['ci_low_pp'] for r in rr]),
            mean_crossover_realised_ms=old.crossover(x,[r['delta_context_pp'] for r in rr]),
            max_sigma_ms=max(all_sigmas),max_realised_err_ms=max(x))
        for group in ('matched','valid','all'):
            selected=[r for r in components if r['dataset']==ds and r['group']==group]
            mean,lo,hi=old.bootstrap_mean(np.array([r['rr_span_error_ms'] for r in selected]))
            summaries.append(dict(dataset=ds,group=group,n_subjects=len(selected),n_windows=sum(r['n_windows'] for r in selected),
                rr_span_error_ms=mean,rr_span_ci_low_ms=lo,rr_span_ci_high_ms=hi,
                median_r_error_ms=subject_balanced_median(median_arrays[ds,group]['r']),
                median_rr_error_ms=subject_balanced_median(median_arrays[ds,group]['rr']),
                pooled_median_r_error_ms=float(np.median(np.concatenate(median_arrays[ds,group]['r']))),
                pooled_median_rr_error_ms=float(np.median(np.concatenate(median_arrays[ds,group]['rr']))),
                **{k:float(np.mean([r[k] for r in selected])) for k in ('anchor_error_ms','common_abs_ms','differential_abs_ms','same_sign','common_energy_share')}))
    return rows,per_subject,points,crossings,components,summaries

def plot(rows,points):
    plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':7,'axes.labelsize':8,
        'xtick.labelsize':7,'ytick.labelsize':7,'axes.titlesize':8,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.linewidth':.6,'axes.unicode_minus':True,'xtick.direction':'out','ytick.direction':'out','xtick.major.size':2.5,'ytick.major.size':2.5})
    ranges=[]
    for ds in ('mmecg','crrvs'):
        items=[r for r in rows if r['dataset']==ds]+[p for p in points if p['dataset']==ds]
        lo=min(0,min(r['ci_low_pp'] for r in items));hi=max(0,max(r['ci_high_pp'] for r in items));ranges.append((lo,hi))
    shared=max(hi-lo for lo,hi in ranges)/max(min(hi-lo for lo,hi in ranges),1e-12)<=2
    fig,axes=plt.subplots(1,2,figsize=(3.5,1.75))
    for ax,ds,letter,title,limits in zip(axes,('mmecg','crrvs'),('a','b'),('MMECG (77-GHz FMCW)','CR_RVS (24-GHz CW)'),ranges):
        r=sorted([v for v in rows if v['dataset']==ds],key=lambda v:v['realised_err_ms']);p=next(v for v in points if v['dataset']==ds)
        x=np.array([v['realised_err_ms'] for v in r]);y=np.array([v['delta_context_pp'] for v in r])
        ax.fill_between(x,[v['ci_low_pp'] for v in r],[v['ci_high_pp'] for v in r],color=old.COLOR,alpha=.18,lw=0)
        ax.plot(x,y,'-o',color=old.COLOR,lw=1.2,ms=3,mec='white',mew=.4)
        ax.axhline(0,color='#7F7F7F',ls='--',lw=1,label='Context only')
        ax.errorbar(p['realised_err_ms'],p['delta_context_pp'],yerr=[[p['delta_context_pp']-p['ci_low_pp']],[p['ci_high_pp']-p['delta_context_pp']]],
            marker='*',ms=7,mfc='#D55E00',mec='black',mew=.6,color='#D55E00',elinewidth=.7,capsize=2,lw=0,zorder=5)
        lo=min(t[0] for t in ranges) if shared else limits[0];hi=max(t[1] for t in ranges) if shared else limits[1]
        pad=.1*max(hi-lo,1);ax.set_ylim(lo-pad,hi+pad);ax.set_xlim(0,1.06*max(x.max(),p['realised_err_ms']))
        ticks=[0,20,40,60,80,100]
        ax.set_xticks(ticks);ax.set_xticklabels([f'{v:.0f}' for v in ticks]);ax.yaxis.set_major_locator(MaxNLocator(5,steps=[1,2,2.5,5,10]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:Formatter.fix_minus(f'{v:.1f}')));ax.set(xlabel='Anchor error (ms)',ylabel=r'$\Delta$PCC (pp)' if ds=='mmecg' else '',title=title)
        ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True);ax.grid(False)
        ax.text(-.22,1.12,f'({letter})',transform=ax.transAxes,fontweight='bold',fontsize=8,va='bottom')
    handles=[Line2D([0],[0],color=old.COLOR,ls='-',marker='o',ms=3,mec='white',mew=.4,lw=1.2,label='Jittered reference anchors'),
        Line2D([0],[0],color='#D55E00',ls='none',marker='*',ms=7,mec='black',mew=.6,label='Predicted anchors'),
        Line2D([0],[0],color='#7F7F7F',ls='--',lw=1,label='Context only (Δ = 0)')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.005),ncol=3,frameon=False,fontsize=6.5,handlelength=.9,columnspacing=.6,handletextpad=.25)
    fig.subplots_adjust(left=.13,right=.985,bottom=.31,top=.79,wspace=.55)
    previous=ASSETS/'previous_absolute_figure';previous.mkdir(exist_ok=True)
    for ext in ('pdf','svg','png'):
        target=FIGURES/f'fig_anchor_robustness.{ext}'
        if target.exists() and not (previous/target.name).exists():shutil.copy2(target,previous/target.name)
        fig.savefig(target,dpi=600)
    plt.close(fig)
    return shared

def report(rows,points,crossings,summaries,shared):
    lines=['# Paired ΔPCC anchor robustness revision','',
        'Inference only, all16 frozen A4 checkpoints. Original sigma0–80 observations are reused unchanged. Y is a paired subject-level difference, in percentage points, with10000 paired-subject bootstrap CI.',
        f'Shared y-limits: {shared} (predefined CI-span ratio threshold2). Original absolute figure archived under previous_absolute_figure/.','']
    for ds in ('mmecg','crrvs'):
        lines.extend([f'## {ds.upper()}','','| Sigma ms | Realised anchor error ms | ΔPCC vs Context [95% CI] pp | Fallback % |',
            '|---:|---:|---|---:|'])
        for r in rows:
            if r['dataset']==ds:lines.append(f"| {r['sigma_ms']} | {r['realised_err_ms']:.2f} | {r['delta_context_pp']:+.3f} [{r['ci_low_pp']:+.3f}, {r['ci_high_pp']:+.3f}] | {100*r['fallback_rate']:.3f} |")
        p=next(x for x in points if x['dataset']==ds);c=crossings[ds]
        statistical=f"{c['crossover_realised_ms']:.2f}ms realised" if c['crossover_realised_ms'] is not None else f"not observed through sigma{c['max_sigma_ms']}ms"
        central=f"{c['mean_crossover_realised_ms']:.2f}ms realised" if c['mean_crossover_realised_ms'] is not None else f"not observed through sigma{c['max_sigma_ms']}ms"
        lines.extend(['',f"Predicted-anchor star: error={p['realised_err_ms']:.2f}ms on matched central pairs; ALL-window ΔPCC={p['delta_context_pp']:+.3f} [{p['ci_low_pp']:+.3f}, {p['ci_high_pp']:+.3f}]pp.",
            f"Statistical crossover, interpolating paired lowerCI to0: {statistical}. Mean crossing: {central}.",'',
            '| Predicted-window subset | N windows | Anchor MAE ms | RR-span MAE ms [95% CI] | Common MAE ms | Differential MAE ms | Same-sign % | Common-energy % |',
            '|---|---:|---:|---|---:|---:|---:|---:|'])
        for a in summaries:
            if a['dataset']==ds:lines.append(f"| {a['group']} | {a['n_windows']} | {a['anchor_error_ms']:.2f} | {a['rr_span_error_ms']:.2f} [{a['rr_span_ci_low_ms']:.2f}, {a['rr_span_ci_high_ms']:.2f}] | {a['common_abs_ms']:.2f} | {a['differential_abs_ms']:.2f} | {100*a['same_sign']:.1f} | {100*a['common_energy_share']:.1f} |")
        lines.extend(['','### Median absolute predicted-anchor errors','',
            'R error means the FIRST R boundary error e0, not the mean error of both boundaries. RR error is e1−e0. Primary medians use an equal-subject empirical mixture CDF; pooled-window medians are supplied separately.','',
            '| Subset | Median absolute R error (ms) | Median absolute RR error (ms) | Pooled median absolute R error (ms) | Pooled median absolute RR error (ms) |',
            '|---|---:|---:|---:|---:|'])
        for a in summaries:
            if a['dataset']==ds:lines.append(f"| {a['group']} | {a['median_r_error_ms']:.2f} | {a['median_rr_error_ms']:.2f} | {a['pooled_median_r_error_ms']:.2f} | {a['pooled_median_rr_error_ms']:.2f} |")
        a=next(a for a in summaries if a['dataset']==ds and a['group']=='matched')
        support=a['common_abs_ms']>a['differential_abs_ms'] and a['same_sign']>.5
        lines.extend(['','Matched-pair diagnostics '+('support common-mode dominance within paired endpoints.' if support else 'do NOT support a claim of mostly common-mode errors.')])
    lines.extend(['','## Interpretation safeguards','',
        'c=(e0+e1)/2, d=(e1−e0)/2, RR-span error=2|d|. Common-energy share is per-subject mean(c²)/(mean(c²)+mean(d²)), then subject macro.',
        'Primary star error and span diagnosis use the SAME matched subset; report valid/all subsets separately. Star y/CI always retains ALL windows.',
        'Do not replace ALL-window errors by matched-subset errors. Valid/all rows include unmatched cycle choices and fallback; common-mode energy alone does not demonstrate correct beat correspondence or a small physiological latency.',
        'These are errors of the SPECIFIC central phase-grid pair against the target central pair, without rematching to another cycle. They are not the usual per-beat event-head R-MAE or all-beat RR-interval MAE.',
        'Same-sign rate and larger common component can support common-mode dominance WITHIN PAIRS, not a constant per-record offset or acquisition synchronization explanation.',
        'Independent Gaussian perturbations differ from the empirical correlated predicted-error distribution; the star need not lie on the jitter curve.',
        'Crossover is loss of paired statistical support above Context, not necessarily a mean intersection. No new significance-based model or checkpoint selection.',
        'CI bands are pointwise, not simultaneous; the interpolated crossover is a grid-dependent diagnostic, not a95% confidence interval for the threshold itself.',
        'Shortened legend labels use one shared row; orange predicted-anchor stars and whiskers, white-edged 3-point markers, no annotation or horizontal grid, and a left-panel-only y label. Axis labels are8pt and ticks7pt; data coordinates and curve statistics are unchanged.',
        'All old data and reports remain unchanged. Only the three requested figure files are replaced, after archival. The new delta source is anchor_robustness_delta.csv.'])
    (ASSETS/'REPORT.md').write_text('\n'.join(lines)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--infer',action='store_true');p.add_argument('--device',default='cuda:0');args=p.parse_args()
    ASSETS.mkdir(parents=True,exist_ok=True);extended=[]
    for sigma in EXTENSION:
        if args.infer:old.infer(torch.device(args.device),sigmas=(sigma,),assets=ASSETS/f'sigma_{sigma}')
        if not all((ASSETS/f'sigma_{sigma}'/f'{ds}_fold_{fold:02d}.npz').exists() for ds,n in (('mmecg',11),('crrvs',5)) for fold in range(1,n+1)):
            if not extended:raise RuntimeError('run with --infer to add sigma128')
            break
        extended.append(sigma);result=build(extended)
        if all(c['crossover_realised_ms'] is not None for c in result[3].values()):break
    rows,subjects,points,crossings,components,summaries=result
    old.save_csv(FIGURES/'anchor_robustness_delta.csv',rows);old.save_csv(ASSETS/'per_subject.csv',subjects)
    old.save_csv(ASSETS/'predicted_points.csv',points);old.save_csv(ASSETS/'predicted_anchor_components_subject.csv',components)
    old.save_csv(ASSETS/'predicted_anchor_components.csv',summaries)
    (ASSETS/'crossovers.json').write_text(json.dumps(crossings,indent=2)+'\n')
    shared=plot(rows,points);report(rows,points,crossings,summaries,shared)
    matched={s['dataset']:s for s in summaries if s['group']=='matched'}
    (ASSETS/'caption.tex').write_text(
        r'\caption{Anchor robustness of REM-Net relative to the window-time parallel-head model (Context only). '
        r'The ordinate is the subject-paired PCC difference in percentage points; shading and star whiskers are pointwise subject-bootstrap 95\% confidence intervals. '
        r'The first curve point uses reference anchors. The horizontal axis is realised mean absolute error of the two phase-grid boundaries, not nominal jitter sigma. '
        r'The star uses predicted anchors, with horizontal position measured on matched central R--R pairs and PCC assessed on all windows. '
        f"On matched pairs, RR-span MAE is {matched['mmecg']['rr_span_error_ms']:.2f}/{matched['crrvs']['rr_span_error_ms']:.2f} ms for MMECG/CR\\_RVS, "
        f"versus endpoint MAE {matched['mmecg']['anchor_error_ms']:.2f}/{matched['crrvs']['anchor_error_ms']:.2f} ms, consistent with a stronger common-mode component within these pairs. "
        r'This subset-specific result does not establish a record-constant synchronization offset.}'+'\n')
    import subprocess
    (ASSETS/'manifest.json').write_text(json.dumps(dict(inference_only=True,training=False,extra_sigmas=extended,shared_y_limits=shared,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=old.ROOT,text=True).strip(),script_sha256=old.original.sha(__file__),
        original_source_csv_sha256=old.original.sha(FIGURES/'anchor_robustness.csv'),protocol_sha256=old.original.sha(ASSETS/'PROTOCOL.md'),
        output_sha256={p.name:old.original.sha(p) for p in [FIGURES/'anchor_robustness_delta.csv',*(FIGURES/f'fig_anchor_robustness.{ext}' for ext in ('pdf','svg','png'))]},
        extra_fold_manifests=[json.loads(p.read_text()) for p in sorted(ASSETS.glob('sigma_*/*_fold_*.json'))]),indent=2)+'\n')
    print('Paired-delta figure/report complete; no training.',flush=True)

if __name__=='__main__':main()
