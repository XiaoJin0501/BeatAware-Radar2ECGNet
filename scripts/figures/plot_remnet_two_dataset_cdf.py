"""REM-Net-only matched-event CDFs; both datasets, one endpoint definition.

Does NOT rename Q/S/T peak spacings as clinical onset/offset intervals.
No new model inference, training or modification of original benchmark results.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import csv
import hashlib
import importlib.util
import json
import shutil
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.interpolate import PchipInterpolator
import plot_dual_head_waveform_grid as shared

ROOT=shared.ROOT
SOURCE=shared.SOURCE
ASSETS=ROOT/'docs/paper/sections/figures/source_data/remnet_two_dataset_cdf_20261026'
OUTPUT=ROOT/'docs/paper/sections/figures/fig_mmecg_clinical_error_cdf'
METRICS=('q','r','s','t','rr','qs_peak','qt_peak')
FROZEN_CLINICAL=Path('/tmp/rem_dual_head_formal/src/utils/clinical_endpoints.py')
spec=importlib.util.spec_from_file_location('cdf_frozen_clinical',FROZEN_CLINICAL)
clinical=importlib.util.module_from_spec(spec);spec.loader.exec_module(clinical)


def reference_points(ecg):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return clinical.delineate_signal(ecg,fs=200,method='prominence')


def build():
    cache=ASSETS/'deduplicated_errors.json'
    if cache.exists():return json.loads(cache.read_text())
    decode=shared.decoder();raw={};counts={};sources=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for ds,nfold in (('mmecg',11),('crrvs',5)):
            raw[ds]={};counts[ds]={}
            for fold in range(1,nfold+1):
                refpath=SOURCE/'references'/ds/f'fold_{fold:02d}.npz'
                predpath=SOURCE/'predictions'/ds/'rem_net/seed_42'/f'fold_{fold:02d}/test_predictions.npz'
                ref=shared.load_npz(refpath);pred=shared.load_npz(predpath)
                sources.extend([dict(path=str(p),sha256=shared.sha(p)) for p in (refpath,predpath)])
                labels=list(pool.map(reference_points,ref['ecg'],chunksize=16))
                for wi,(actual,prob) in enumerate(zip(labels,pred['anchor_probability'])):
                    subject=str(int(ref['subject_id'][wi]));rid=str(ref['record_id'][wi]);start=int(ref['start_idx'][wi])
                    group=raw[ds].setdefault(subject,{k:{} for k in METRICS})
                    totals=counts[ds].setdefault(subject,{k:dict(eligible=0,matched=0) for k in METRICS})
                    predicted=decode(prob.astype(float))
                    pairs=clinical.match_beats(actual['r'],predicted['r'],30)
                    for i,r in enumerate(actual['r']):
                        key=f'{rid}:{start+int(r)}'
                        j=pairs.get(i)
                        for name in 'qrst':
                            if i>=len(actual[name]) or not np.isfinite(actual[name][i]):continue
                            totals[name]['eligible']+=1
                            if j is None or j>=len(predicted[name]) or not np.isfinite(predicted[name][j]):continue
                            error=abs(float(predicted[name][j]-actual[name][i]))*5
                            group[name].setdefault(key,[]).append(error);totals[name]['matched']+=1
                        if i>0:
                            totals['rr']['eligible']+=1
                            # Both ref and predicted beats must be adjacent, not merely matched.
                            if j is not None and j>0 and pairs.get(i-1)==j-1:
                                error=abs(float((predicted['r'][j]-predicted['r'][j-1])-(r-actual['r'][i-1])))*5
                                interval_key=f'{rid}:{start+int(actual["r"][i-1])}:{start+int(r)}'
                                group['rr'].setdefault(interval_key,[]).append(error);totals['rr']['matched']+=1
                        for name,end in (('qs_peak','s'),('qt_peak','t')):
                            if not (np.isfinite(actual['q'][i]) and np.isfinite(actual[end][i])):continue
                            interval=actual[end][i]-actual['q'][i]
                            if interval<=0:continue
                            totals[name]['eligible']+=1
                            if j is None:continue
                            estimated=predicted[end][j]-predicted['q'][j]
                            if not np.isfinite(estimated) or estimated<=0:continue
                            group[name].setdefault(key,[]).append(abs(float(estimated-interval))*5)
                            totals[name]['matched']+=1
                print(f'REM-Net CDF source complete: {ds} fold{fold}/{nfold}',flush=True)
    result=dict(version='remnet_dual_head_subject_macro_cdf_v1',seed=42,model='rem_net',
        errors={ds:{s:{k:[float(np.median(v)) for key,v in sorted(metrics[k].items())] for k in METRICS}
                for s,metrics in subjects.items()} for ds,subjects in raw.items()},
        window_event_counts=counts,sources=sources,reference_method='frozen NeuroKit2 prominence on original per-window ECG',
        pairing='R-based order-preserving one-to-one match, maximum cardinality then minimum error;150ms; same pair for Q/S/T',
        deduplication='median error over overlapping-window observations per subject/record/global reference R; RR key includes both adjacent reference R',
        cdf='equal-weight empirical CDF per subject with nonempty matched observations; then equal-subject mean',
        missingness='unmatched events excluded from conditional error CDF, retained in eligible/matched counts; CDF is NOT detection probability',
        intervals='RR, Q-to-S peak spacing, Q-to-T peak spacing; NOT clinical QRS/QT onset/offset durations',
        PR='not included: current main table has no PR column and model has no P-boundary head',
        frozen_reference_source_sha256=shared.sha(FROZEN_CLINICAL),
        baseline_included=False,legacy_continuous_included=False)
    cache.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def empirical(subjects,metric):
    arrays=[np.asarray(m[metric],float) for m in subjects.values() if len(m[metric])]
    if not arrays:raise ValueError('No errors: '+metric)
    x=np.unique(np.r_[0.,np.concatenate(arrays)])
    F=np.mean([np.searchsorted(np.sort(a),x,side='right')/len(a) for a in arrays],axis=0)
    return x,F,len(arrays),sum(len(a) for a in arrays)


def quantile(x,F,p):
    # Numerical tolerance only: preserve exact inverse-CDF nodes at decimal ties.
    indices=np.flatnonzero(F>=p-1e-12)
    return float(x[int(indices[0]) if len(indices) else len(x)-1])


def plot(result):
    plt.rcParams.update({'font.family':'Times New Roman','font.size':7,'axes.linewidth':.6,
        'xtick.direction':'out','ytick.direction':'out','xtick.major.size':3,'ytick.major.size':3,
        'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,4,figsize=(7.16,4.15),sharey=True,
        gridspec_kw=dict(wspace=.35,hspace=.95))
    summary={};nodes=[]
    colors={'q':'#D9822B','r':'#2A9D8F','s':'#D17DB8','t':'#E0B03A'}
    for row,ds in enumerate(('mmecg','crrvs')):
        subjects=result['errors'][ds];summary[ds]={}
        panels=(('qrst','Fiducial timing'),(('rr',),'RR interval'),(('qs_peak',),'Q–S peak interval'),(('qt_peak',),'Q–T peak interval'))
        interval_color='#3D6B91' if ds=='mmecg' else '#AF6549'
        for col,(metrics,title) in enumerate(panels):
            ax=axes[row,col];xmax=150.
            for metric in metrics:
                x,F,ns,ne=empirical(subjects,metric)
                for xx,ff in zip(x,F):nodes.append(dict(dataset=ds,metric=metric,error_ms=float(xx),subject_macro_cdf=float(ff)))
                color=colors[metric] if col==0 else interval_color
                limit=x<=xmax
                xx,ff=x[limit],F[limit]
                # Connect observed nodes; at the plotting boundary evaluate the
                # actual CDF, not the value of a smoothing/extrapolation model.
                if col==0 and len(xx)>1:
                    grid=np.linspace(xx[0],xx[-1],500)
                    ax.plot(grid,PchipInterpolator(xx,ff)(grid),color=color,lw=1,label=metric.upper())
                else:ax.plot(xx,ff,color=color,lw=1.1,marker='o',ms=1.8,mec='white',mew=.25)
                if xx[-1]<xmax:
                    ax.plot([xx[-1],xmax],[ff[-1]]*2,color=color,lw=1.1)
                if F[0]>0:ax.plot([0,0],[0,F[0]],color=color,lw=1)
                p50,p90=quantile(x,F,.5),quantile(x,F,.9)
                eligible=sum(s[metric]['eligible'] for s in result['window_event_counts'][ds].values())
                matched=sum(s[metric]['matched'] for s in result['window_event_counts'][ds].values())
                summary[ds][metric]=dict(p50_ms=p50,p90_ms=p90,subjects=ns,unique_matched_errors=ne,
                    window_events_eligible=eligible,window_events_matched=matched,
                    window_event_coverage=matched/eligible if eligible else None,
                    CDF_at_150ms=float(ff[-1]),missing_subjects=[s for s,m in subjects.items() if not m[metric]])
                if col:
                    text='\n'.join(f'P{int(p*100)} = {v:g} ms' if v<=xmax else f'P{int(p*100)} > {xmax:g} ms' for p,v in ((.5,p50),(.9,p90)))
                    ax.text(.96,.08,text,transform=ax.transAxes,ha='right',va='bottom',fontsize=6,color='#555555')
            if col==0:ax.legend(ncol=2,loc='lower right',frameon=False,fontsize=6,handlelength=1.3,columnspacing=.7)
            else:
                for p in (.5,.9):ax.axhline(p,color='#AAAAAA',ls=(0,(2,2)),lw=.45,zorder=0)
            ax.set(title=title,xlim=(0,xmax),ylim=(0,1.025),xlabel='Absolute error (ms)',ylabel='CDF')
            ax.set_xticks([0,50,100,150]);ax.set_yticks([0,.2,.4,.6,.8,1])
            ax.tick_params(labelleft=True,labelsize=6.5)
            ax.grid(color='#E7E7E7',lw=.4)
            ax.text(.5,-.38,f'({chr(97+row*4+col)})',transform=ax.transAxes,ha='center',va='top',fontsize=8,fontweight='bold')
    fig.subplots_adjust(left=.07,right=.98,bottom=.16,top=.86)
    for row,ds in enumerate(('MMECG','CR_RVS')):
        bounds=axes[row,0].get_position()
        n=len(result['errors'][ds.lower() if ds=='MMECG' else 'crrvs'])
        fig.text(.07,bounds.y1+.075,f'{ds} · REM-Net · {n} subjects',fontsize=8.5,fontweight='bold',ha='left')
    for ext in ('pdf','svg','png'):fig.savefig(OUTPUT.with_suffix('.'+ext),dpi=600,facecolor='white')
    plt.close(fig)
    (ASSETS/'quantiles_and_coverage.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    with (ASSETS/'exact_cdf_nodes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(nodes[0]));writer.writeheader();writer.writerows(nodes)


def main():
    ASSETS.mkdir(parents=True,exist_ok=True)
    result=build()
    for ext in ('png','pdf','svg'):
        old=OUTPUT.with_suffix('.'+ext);backup=ASSETS/('previous_'+old.name)
        if old.exists() and not backup.exists():shutil.copy2(old,backup)
    plot(result)
    print('Completed REM-Net-only two-dataset CDF; PR absent from table; no legacy continuous QRS/QT mixed in',flush=True)


if __name__=='__main__':main()
