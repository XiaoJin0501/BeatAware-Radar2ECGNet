#!/usr/bin/env python3
"""Fig4: median-selected, matched dual-head assembly views, not native regressors."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
import h5py
import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'docs/paper/source_data/2026-10-16_existing_remnet_v1'
OUTPUT = ROOT/'docs/paper/sections/figures/fig_mmecg_waveform_examples'
FS, LEFT, RIGHT = 200, 288, 1312
METHODS = ('rem_net', 'mmecg_tcn', 'radarode_mtl')
SCENES = {'mmecg': [('NB','Normal breathing'),('IB','Irregular breathing'),('SP','Sleep'),('PE','Post-exercise')],
          'crrvs': [('NB','Resting'),('IB','Valsalva'),('SP','Apnea')]}
COLORS = dict(q='#D9822B', r='#2A9D8F', s='#D17DB8', t='#E0B03A')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decoder(source=SOURCE):
    """Reuse the exact archived pure decoder without importing training modules."""
    path=source/'code_snapshot/scripts/run_dual_head_interaction.py'
    tree=ast.parse(path.read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='structured_decode')
    module=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
    namespace=dict(np=np,find_peaks=find_peaks,WIN=1600,FIDUCIALS=('q','r','s','t'))
    exec(compile(module,str(path),'exec'),namespace)
    return namespace['structured_decode']


def pcc(p,y):
    p,y=np.asarray(p,float),np.asarray(y,float)
    a,b=p-p.mean(),y-y.mean()
    denominator=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/denominator) if denominator>1e-12 else 0.


def minmax(x):
    x=np.asarray(x,float)
    return (x-x.min())/(x.max()-x.min()) if np.ptp(x)>1e-12 else np.zeros_like(x)


def reference_display(ecg,r):
    """GT-only display normalization at its OWN R-R boundaries, no resampling."""
    n=len(ecg)
    boundaries=np.unique(np.r_[0,np.asarray(r,int),n])
    boundaries=boundaries[(boundaries>=0)&(boundaries<=n)]
    y=np.zeros(n,float)
    for a,b in zip(boundaries[:-1],boundaries[1:]):
        y[a:b]=minmax(ecg[a:b])
    return y


def assemble(shape,r,n=1600):
    """Uniform inverse-RR resampling of representative shape using predicted R only.

    Fewer than2 predicted R gives an explicit flat failure. Outside first/last
    R, use nearest predicted RR to extend the phase; no GT, lag or peak fitting.
    """
    r=np.asarray(r,float)
    if len(r)<2:return np.zeros(n,float)
    shape=minmax(shape)
    nodes=r.copy()
    while nodes[0]>0:nodes=np.r_[nodes[0]-(r[1]-r[0]),nodes]
    while nodes[-1]<n:nodes=np.r_[nodes,nodes[-1]+(r[-1]-r[-2])]
    y=np.zeros(n,float)
    native=np.arange(n)
    for a,b in zip(nodes[:-1],nodes[1:]):
        indices=native[(native>=a)&(native<b)]
        # Original target uses endpoint-inclusive interpolation over [R,nextR).
        grid=(indices-a)/max(b-a-1.,1.)*199
        y[indices]=np.interp(grid,np.arange(200),shape)
    return y


def refs_at(ref,k,index):
    return ref[k+'_values'][ref[k+'_offsets'][index]:ref[k+'_offsets'][index+1]]


def load_npz(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}


def candidates(source=SOURCE):
    decode=decoder(source);rows=[]
    for ds,nfold in (('mmecg',11),('crrvs',5)):
        for fold in range(1,nfold+1):
            ref=load_npz(source/'references'/ds/f'fold_{fold:02d}.npz')
            pred=load_npz(source/'predictions'/ds/'rem_net/seed_42'/f'fold_{fold:02d}/test_predictions.npz')
            assert len(pred['shape'])==len(ref['ecg'])
            for i,(shape,prob) in enumerate(zip(pred['shape'],pred['anchor_probability'])):
                points=decode(prob.astype(float))
                wave=assemble(shape,points['r'])
                gt=reference_display(ref['ecg'][i],refs_at(ref,'r',i))
                rows.append(dict(dataset=ds,fold=fold,index=i,subject=int(ref['subject_id'][i]),
                    scene=str(ref['state_code'][i]),record_id=str(ref['record_id'][i]),
                    start=int(ref['start_idx'][i]),stop=int(ref['end_idx'][i]),
                    window_pcc=pcc(wave[LEFT:RIGHT],gt[LEFT:RIGHT]),predicted_r_count=len(points['r'])))
            print(f'Candidate scores: {ds} fold{fold} ({len(ref["ecg"])} windows)',flush=True)
    return rows


def select(rows):
    """All windows, including flat failures; deterministic ties by subject/record/start."""
    selections=[];subjects=[]
    for ds,scenes in SCENES.items():
        for scene,label in scenes:
            group=[r for r in rows if r['dataset']==ds and r['scene']==scene]
            scores=[]
            for subject in sorted({r['subject'] for r in group}):
                values=[r['window_pcc'] for r in group if r['subject']==subject]
                score=float(np.tanh(np.mean(np.arctanh(np.clip(values,-1+1e-7,1-1e-7)))))
                scores.append(dict(dataset=ds,scene=scene,subject=subject,pcc=score,n_windows=len(values)))
            if not scores:raise ValueError(f'No data: {ds}/{scene}')
            scene_median=float(np.median([r['pcc'] for r in scores]))
            chosen=min(scores,key=lambda r:(abs(r['pcc']-scene_median),r['subject']))
            subset=[r for r in group if r['subject']==chosen['subject']]
            window_median=float(np.median([r['window_pcc'] for r in subset]))
            window=min(subset,key=lambda r:(abs(r['window_pcc']-window_median),r['record_id'],r['start'],r['index']))
            selections.append(dict(**window,scene_label=label,scenario_subject_median=scene_median,
                subject_pcc=chosen['pcc'],subject_window_median=window_median,
                valid_start_global=window['start']+LEFT,valid_stop_global=window['start']+RIGHT))
            subjects.extend(scores)
    return selections,subjects


def radar_input(row):
    base=Path('/home/qhh2237/Datasets')
    split=(base/'MMECG/processed/loso_v2' if row['dataset']=='mmecg' else base/'CR_RVS/processed/q1_continuous_kfold')/f"fold_{row['fold']:02d}/test.h5"
    with h5py.File(split,'r') as h:
        i=row['index']
        rid=h['record_id'][i];rid=rid.decode() if isinstance(rid,bytes) else str(rid)
        assert rid==row['record_id'] and int(h['start_idx'][i])==row['start']
        x=np.asarray(h['rcg'][i],float).reshape(-1,1600)
    filtered=sosfiltfilt(butter(4,[.8,3.],fs=FS,btype='bandpass',output='sos'),x,axis=-1)
    energy=np.mean(filtered**2,axis=-1)/np.maximum(np.mean((x-x.mean(-1,keepdims=True))**2,axis=-1),1e-12)
    channel=int(np.argmax(energy))  # Independent of ECG or model outputs.
    return filtered[channel,LEFT:RIGHT],dict(path=str(split),channel_zero_based=channel,
        cardiac_band_hz=[.8,3.],selection='maximum cardiac-band/total AC energy over entire8s; no ECG',
        energy_fraction=float(energy[channel]))


def selected_data(row,source=SOURCE):
    refpath=source/'references'/row['dataset']/f"fold_{row['fold']:02d}.npz"
    ref=load_npz(refpath);i=row['index'];decode=decoder(source)
    gt=reference_display(ref['ecg'][i],refs_at(ref,'r',i))
    result=dict(gt=gt[LEFT:RIGHT],gt_r=refs_at(ref,'r',i)-LEFT)
    provenance=[dict(path=str(refpath),sha256=sha(refpath))]
    scores={}
    for method in METHODS:
        path=source/'predictions'/row['dataset']/method/'seed_42'/f"fold_{row['fold']:02d}/test_predictions.npz"
        pred=load_npz(path)
        assert pred['shape'].shape[0]==len(ref['ecg'])
        points=decode(pred['anchor_probability'][i].astype(float))
        wave=assemble(pred['shape'][i],points['r'])
        result[method]=wave[LEFT:RIGHT]
        scores[method]=pcc(result[method],result['gt'])
        if method=='rem_net':
            for k in 'qrst':result['pred_'+k]=points[k]-LEFT
        provenance.append(dict(path=str(path),sha256=sha(path)))
    assert abs(scores['rem_net']-row['window_pcc'])<1e-10
    result['radar'],radar=radar_input(row)
    return result,dict(**row,window_pcc_by_method=scores,radar=radar,sources=provenance)


def grid(ax,radar=False,duration=5.12):
    ax.set_facecolor('#EAF4EA' if radar else '#FBEAEA')
    ax.set_xlim(0,duration)
    ax.xaxis.set_major_locator(MultipleLocator(.2));ax.xaxis.set_minor_locator(MultipleLocator(.04))
    ax.yaxis.set_major_locator(MultipleLocator(.5));ax.yaxis.set_minor_locator(MultipleLocator(.1))
    ax.grid(which='major',color='#B5C6B7' if radar else '#D8B8BA',lw=.30,alpha=.7)
    ax.grid(which='minor',color='#D2DFD3' if radar else '#EDD5D7',lw=.18,alpha=.65)
    ax.tick_params(which='both',bottom=False,left=False,labelbottom=False,labelleft=False)
    for spine in ax.spines.values():spine.set_linewidth(.4);spine.set_color('#777777')


def plot(cases,output):
    plt.rcParams.update({'font.family':'Times New Roman','font.size':7,'axes.linewidth':.4,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','path.simplify':False})
    fig=plt.figure(figsize=(7.16,6.65))
    outer=fig.add_gridspec(2,1,left=.125,right=.995,bottom=.055,top=.925,hspace=.32)
    labels=['Radar input\n(a.u.)','GT ECG','REM-Net','Chen-TCN','radarODE-MTL']
    time=np.arange(1024)/FS
    for block,ds in enumerate(('mmecg','crrvs')):
        gs=outer[block].subgridspec(5,4,wspace=.045,hspace=.12)
        data=[c for c in cases if c[0]['dataset']==ds]
        axes=[]
        for col,(meta,arrays) in enumerate(data):
            for row,key in enumerate(('radar','gt',*METHODS)):
                ax=fig.add_subplot(gs[row,col]);axes.append(ax);grid(ax,radar=row==0)
                y=arrays[key]
                if row==0:
                    # Radar amplitude is untouched; symmetric limits are purely visual.
                    limit=max(float(np.max(abs(y)))*1.13,1e-6);ax.set_ylim(-limit,limit)
                else:
                    ax.set_ylim(-.13,1.23)
                    for r in arrays['gt_r']:
                        if 0<=r<1024:ax.axvline(r/FS,color='#888888',ls=(0,(2,2)),lw=.45,alpha=.5,zorder=2)
                ax.plot(time,y,color='#262626',lw=.55,zorder=4)
                if row>0:
                    score=1. if key=='gt' else meta['window_pcc_by_method'][key]
                    ax.text(.98,.95,f'PCC = {score:.2f}',ha='right',va='top',transform=ax.transAxes,
                        fontsize=5.7,bbox=dict(facecolor=ax.get_facecolor(),edgecolor='none',pad=.15),zorder=7)
                if key=='rem_net':
                    for k,color in COLORS.items():
                        p=arrays['pred_'+k];p=p[(p>=0)&(p<1024)]
                        ax.scatter(p/FS,np.interp(p,np.arange(1024),y),s=5,color=color,
                            edgecolor='white',linewidth=.15,zorder=6)
                if row==0:ax.set_title(meta['scene_label'],fontsize=7.1,fontweight='bold',pad=3)
                if col==0:
                    ax.text(-.06,.5,labels[row],transform=ax.transAxes,ha='right',va='center',
                        fontsize=7.0,fontweight='bold' if row==2 else 'normal')
        bounds=outer[block].get_position(fig)
        fig.text(.125,bounds.y1+.047,f"({'ab'[block]}) {'MMECG (77-GHz FMCW)' if ds=='mmecg' else 'CR_RVS (24-GHz CW)'}",
                 fontsize=8.6,fontweight='bold',ha='left')
        # Shared true 1s scale, same physical length for both5.12s blocks.
        bar=axes[4]
        bar.plot([.65,1.65],[-.37,-.37],transform=bar.get_xaxis_transform(),color='#333333',lw=1.,clip_on=False)
        bar.text(1.15,-.44,'1 s',transform=bar.get_xaxis_transform(),ha='center',va='top',fontsize=6.5)
        if ds=='crrvs':
            meta,arrays=data[0]  # Fixed Resting example; no second quality selection.
            r=arrays['pred_r']
            # One predicted-RR-duration crop, R at quarter phase to expose QRS/T.
            # Both traces use identical absolute indices: NO independent R alignment.
            pairs=[]
            for r0,r1 in zip(r[:-1],r[1:]):
                rr=r1-r0;a=int(round(r0-.25*rr));b=int(round(r0+.75*rr))
                if a>=0 and b<1024:pairs.append((a,b,float(r0)))
            if not pairs:raise ValueError('Chosen Resting window has no complete predicted-RR-duration crop; no silent replacement')
            a,b,r0=min(pairs,key=lambda p:(abs(p[2]-512),p[2]))
            ax=fig.add_subplot(gs[1:,3]);grid(ax,duration=(b-a)/FS);ax.set_ylim(-.13,1.23)
            x=np.arange(a,b)/FS-a/FS
            ax.plot(x,arrays['gt'][a:b],color='#C95252',lw=.8,label='GT')
            ax.plot(x,arrays['rem_net'][a:b],color='#425D78',lw=.8,ls='--',label='REM-Net')
            for k,color in COLORS.items():
                points=arrays['pred_'+k];points=points[(points>=a)&(points<b)]
                vals=np.interp(points,np.arange(1024),arrays['rem_net'])
                ax.scatter((points-a)/FS,vals,s=11,color=color,edgecolor='white',linewidth=.25,zorder=7)
                for point,value in zip(points,vals):
                    dx={'q':-6,'r':0,'s':6,'t':0}[k]
                    ax.annotate(k.upper(),((point-a)/FS,value),xytext=(dx,6),textcoords='offset points',ha='center',fontsize=6,color=color)
            ax.legend(loc='lower center',ncol=1,fontsize=6,frameon=False,handlelength=1.8)
            ax.text(.98,.98,f"PCC = {pcc(arrays['rem_net'][a:b],arrays['gt'][a:b]):.2f}",transform=ax.transAxes,ha='right',va='top',fontsize=5.8)
            head=fig.add_subplot(gs[0,3]);head.axis('off')
            head.text(.5,1.04,'Single-beat zoom (1 RR)',ha='center',va='bottom',transform=head.transAxes,fontsize=7.1,fontweight='bold')
            head.text(.5,.45,f"Resting zoom\n{(b-a)/FS:.2f} s; no time alignment",ha='center',va='center',fontsize=6)
            meta['zoom']=dict(source_scene='Resting',local_start=a,local_stop=b,
                predicted_r=r0,rule='one predicted-RR-duration, -.25RR to +.75RR about predicted R nearest window center; same absolute crop for both traces; no alignment')
    legend=[Line2D([],[],marker='o',color=COLORS[k],lw=0,markersize=3,label=k.upper()) for k in 'qrst']
    fig.legend(handles=legend,loc='upper right',bbox_to_anchor=(.994,.985),ncol=4,
               frameon=False,fontsize=6.5,handlelength=.6,columnspacing=.8)
    fig.text(.65,.985,'Predicted fiducials:',ha='right',va='top',fontsize=6.5,color='#555555')
    for ext in ('pdf','svg','png'):
        fig.savefig(output.with_suffix('.'+ext),dpi=600,facecolor='white')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--output',type=Path,default=OUTPUT);args=parser.parse_args()
    output=args.output;output.parent.mkdir(parents=True,exist_ok=True)
    assets=output.parent/'source_data/fig4_dual_head_20261026';assets.mkdir(parents=True,exist_ok=True)
    cache=assets/'candidate_windows.csv'
    if cache.exists():
        with cache.open() as f:
            rows=list(csv.DictReader(f))
        for r in rows:
            for k in ('fold','index','subject','start','stop','predicted_r_count'):r[k]=int(r[k])
            r['window_pcc']=float(r['window_pcc'])
    else:
        rows=candidates(args.source)
        with cache.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    selections,subjects=select(rows);cases=[]
    for row in selections:
        arrays,meta=selected_data(row,args.source);cases.append((meta,arrays))
        np.savez_compressed(assets/f"{row['dataset']}_{row['scene']}.npz",**arrays)
    # Preserve the user's former files before the requested replacement.
    for ext in ('png','pdf','svg'):
        old=output.with_suffix('.'+ext);backup=assets/f'previous_{old.name}'
        if old.exists() and not backup.exists():shutil.copy2(old,backup)
    plot(cases,output)
    manifest=dict(figure_width_inches=7.16,seed=42,methods=list(METHODS),
        protocol='dual-head representative-shape/predicted-R assembly; no SG, GT warp, lag, amplitude fitting or sample selection by baseline',
        selection='scenario median of subject Fisher-z means of central1024 displayed-waveform PCC; subject nearest median; window nearest its median; all failures retained',
        pcc='Pearson on displayed independently beat-wise min-max normalized signals; no alignment; not Shape-200 or population native reconstruction scores',
        zoom='only panel with a different time scale, explicitly labeled',
        decoder_source_sha256=sha(args.source/'code_snapshot/scripts/run_dual_head_interaction.py'),
        cases=[meta for meta,_ in cases],subject_scenario_scores=subjects)
    (assets/'selection_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(json.dumps([dict(dataset=m['dataset'],scene=m['scene'],subject=m['subject'],record=m['record_id'],start=m['start'],pcc=m['window_pcc_by_method']) for m,_ in cases],indent=2))


if __name__=='__main__':main()
