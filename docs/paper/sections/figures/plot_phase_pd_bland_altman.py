"""Frozen seed-base42 A4, roundtrip gate BEFORE inference; never trains.

All individual paired values are exported. No medians substituted for pairs.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MPLBACKEND','Agg')
import sys,json,csv,warnings,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from torch.nn import functional as F
import neurokit2 as nk
import h5py
from matplotlib import pyplot as plt
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from experiments.phase_resampling_20261027 import formal_ablation as formal
from experiments.phase_resampling_20261027.architecture import phase_alignment,phase_resample,soft_refine
clinical=formal.frozen.clinical
FIG=ROOT/'docs/paper/sections/figures'
OUT=FIG/'source_data/phase_pd_bland_altman_20261029'
RAW=ROOT/'experiments/development/structured_decoder_v0_20261017_ab/raw_reference'
DATASETS=(('mmecg','MMECG',11),('crrvs','CR_RVS',5))
FIELDS=clinical.FIELDS
INTERVALS={'qrs':('qrs_on','qrs_off'),'qt':('qrs_on','t_off'),
           'qs_peak':('q','s'),'qt_peak':('q','t')}

def save_csv(path,rows):
    if not rows:raise ValueError('Empty export '+str(path))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def delineate(x,r=None):
    """Exact frozen clean/delineate parameters; optional supplied R only change."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if r is None:return clinical.delineate_signal(x,fs=200,method='prominence')
        r=np.asarray(r,float);out={k:np.full(len(r),np.nan) for k in FIELDS};out['r']=r
        try:
            clean=nk.ecg_clean(np.asarray(x,float),sampling_rate=200)
            _,waves=nk.ecg_delineate(clean,rpeaks=np.rint(r).astype(int),sampling_rate=200,method='prominence')
        except (ValueError,IndexError,ZeroDivisionError):return out
        for k,name in FIELDS.items():
            a=np.asarray(waves.get(name,[]),float)
            if len(a)==len(r):out[k]=np.where((a>=0)&(a<len(x)),a,np.nan)
        return out

def roundtrip_signal(x,r):
    y=np.zeros(len(x),float);covered=np.zeros(len(x),bool)
    for a,b in zip(np.asarray(r,int)[:-1],np.asarray(r,int)[1:]):
        if b-a<2:continue
        v=np.interp(np.linspace(a,b-1,200),np.arange(a,b),x[a:b])
        span=np.ptp(v);v=(v-v.min())/span if span>1e-8 else np.zeros(200)
        y[a:b]=np.interp(np.linspace(0,199,b-a),np.arange(200),v);covered[a:b]=True
    return y,covered

def make_rows(ds,fold,subject,record,start,ref,pred,metrics):
    matches=clinical.match_beats(ref['r'],pred['r'],30);rows=[]
    for i,r in enumerate(ref['r']):
        if not 288<=r<1312:continue
        j=matches.get(i)
        for metric in metrics:
            if metric=='rr':
                if i+1>=len(ref['r']):continue
                actual=(ref['r'][i+1]-r)*5
                ok=j is not None and j+1<len(pred['r']) and matches.get(i+1)==j+1
                estimate=(pred['r'][j+1]-pred['r'][j])*5 if ok else np.nan
            else:
                a,b=INTERVALS[metric];ra,rb=ref[a][i],ref[b][i]
                if not np.isfinite([ra,rb]).all() or rb<=ra:continue
                actual=(rb-ra)*5;estimate=np.nan
                if j is not None:
                    pa,pb=pred[a][j],pred[b][j]
                    if np.isfinite([pa,pb]).all() and pb>pa and abs(pa-ra)<=30 and abs(pb-rb)<=30:
                        estimate=(pb-pa)*5
            rows.append(dict(dataset=ds,fold=fold,subject=str(subject),record=record,start=int(start),
                ref_r=int(start+r),metric=metric,ref=float(actual),pred=float(estimate),
                centre_distance=float(abs(r-800))))
    return rows

def repeated_loa(d,subjects):
    """Bland/Altman2007 §3 unequal-count one-way ANOVA on paired differences."""
    groups=[d[subjects==s] for s in np.unique(subjects)];counts=np.array([len(g) for g in groups]);N=len(d);K=len(groups)
    bias=float(np.mean(d));within=sum(np.sum((g-g.mean())**2) for g in groups)/max(N-K,1)
    ms_between=sum(len(g)*(g.mean()-bias)**2 for g in groups)/max(K-1,1)
    divisor=(N-np.sum(counts**2)/N)/max(K-1,1)
    between=max(0.,(ms_between-within)/divisor) if K>1 and divisor>0 else 0.
    half=1.96*np.sqrt(within+between)
    return bias,float(bias-half),float(bias+half),float(within),float(between)

def agreement(rows,bootstrap=10000):
    valid=[r for r in rows if np.isfinite(r['pred'])];n=len(valid)
    base=dict(n_eligible=len(rows),n_pairs=n,n_subjects=len(set(r['subject'] for r in valid)),
        coverage=n/len(rows) if rows else 0.,detection_failure_rate=1-n/len(rows) if rows else 1.)
    if n<2:return {**base,**{k:np.nan for k in ('mae','bias','bias_ci_low','bias_ci_high','loa_low','loa_high','naive_loa_low','naive_loa_high','within_variance','between_variance','slope','slope_ci_low','slope_ci_high','icc_a1')}}
    pred=np.array([r['pred'] for r in valid]);ref=np.array([r['ref'] for r in valid]);s=np.array([r['subject'] for r in valid]);d=pred-ref;x=(pred+ref)/2
    bias,lo,hi,within,between=repeated_loa(d,s)
    unique=np.unique(s)
    sums=np.array([[len(d[s==k]),d[s==k].sum(),x[s==k].sum(),(x[s==k]**2).sum(),(x[s==k]*d[s==k]).sum()] for k in unique])
    rng=np.random.default_rng(42);tot=sums[rng.integers(0,len(sums),(bootstrap,len(sums)))].sum(1)
    nb,sd,sx,sxx,sxd=tot.T;den=sxx-sx*sx/nb
    slopes=np.divide(sxd-sx*sd/nb,den,out=np.full(bootstrap,np.nan),where=abs(den)>1e-12)
    biasci=np.percentile(sd/nb,[2.5,97.5]);slopeci=np.nanpercentile(slopes,[2.5,97.5])
    slope=float(np.sum((x-x.mean())*(d-d.mean()))/np.sum((x-x.mean())**2)) if np.ptp(x)>0 else np.nan
    a=np.c_[pred,ref];grand=a.mean();rm=a.mean(1);cm=a.mean(0)
    msr=2*np.sum((rm-grand)**2)/(n-1);msc=n*np.sum((cm-grand)**2);mse=np.sum((a-rm[:,None]-cm+grand)**2)/(n-1)
    denicc=msr+mse+2*(msc-mse)/n
    return dict(**base,mae=float(abs(d).mean()),bias=bias,bias_ci_low=float(biasci[0]),bias_ci_high=float(biasci[1]),
        loa_low=lo,loa_high=hi,naive_loa_low=float(bias-1.96*d.std(ddof=1)),naive_loa_high=float(bias+1.96*d.std(ddof=1)),
        within_variance=within,between_variance=between,slope=slope,slope_ci_low=float(slopeci[0]),slope_ci_high=float(slopeci[1]),
        icc_a1=float((msr-mse)/denicc) if abs(denicc)>1e-12 else np.nan)

def dedup(rows):
    chosen={}
    for row in rows:
        key=(row['dataset'],row['subject'],row['record'],row['metric'],row['ref_r'])
        if key not in chosen or (row['centre_distance'],row['start'])<(chosen[key]['centre_distance'],chosen[key]['start']):chosen[key]=row
    return list(chosen.values())

def inventory():
    rows=[]
    for ds,name,n in DATASETS:
        root=Path('/home/qhh2237/Datasets')/('MMECG/processed/loso_v2' if ds=='mmecg' else 'CR_RVS/processed/q1_continuous_kfold')
        for fold in range(1,n+1):
            with h5py.File(root/f'fold_{fold:02d}/test.h5') as h:
                for i,(rid,subject,start) in enumerate(zip(h['record_id'][:],h['subject_id'][:],h['start_idx'][:])):
                    rid=rid.decode() if isinstance(rid,bytes) else str(rid)
                    rows.append(dict(dataset=ds,name=name,fold=fold,index=i,record=rid,subject=int(subject),start=int(start)))
    return rows

def roundtrip_record(args):
    group=args;first=group[0]
    with np.load(RAW/first['name']/(first['record']+'.npz')) as z:record=z['ecg']
    rows=[];audit=[]
    for item in group:
        start=item['start'];x=record[start:start+1600]
        if len(x)!=1600:raise ValueError('Raw alignment/length mismatch')
        ref=delineate(x);y,covered=roundtrip_signal(x,ref['r']);pred=delineate(y,ref['r'])
        rows+=make_rows(item['dataset'],item['fold'],item['subject'],item['record'],start,ref,pred,('qrs','qt'))
        audit.append(dict(**item,reference_r=ref['r'].tolist(),roundtrip_uncovered_fraction=float(1-covered[288:1312].mean())))
    return rows,audit

def run_roundtrip(items):
    path=OUT/'roundtrip_pairs.csv'
    if path.exists():return list(csv.DictReader(path.open()))
    groups={}
    for item in items:groups.setdefault((item['dataset'],item['record']),[]).append(item)
    raw=[];audit=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for k,(r,a) in enumerate(pool.map(roundtrip_record,groups.values()),1):
            raw+=r;audit+=a
            if k%10==0:print(f'Roundtrip {k}/{len(groups)} records',flush=True)
    rows=dedup(raw);save_csv(path,rows);(OUT/'roundtrip_window_audit.json').write_text(json.dumps(audit))
    return rows

def numeric(rows):
    for row in rows:
        for k in ('ref','pred','centre_distance'):row[k]=float(row[k])
    return rows

def interval_inference(items,clinical_ok):
    torch.set_num_threads(2);torch.backends.cudnn.deterministic=False;torch.backends.cudnn.benchmark=False
    fallback=[];rows=[]
    for ds,name,nfold in DATASETS:
        for fold in range(1,nfold+1):
            cache=OUT/f'{ds}_fold_{fold:02d}_pairs.csv';fcache=cache.with_suffix('.json')
            if cache.exists():rows+=numeric(list(csv.DictReader(cache.open())));fallback+=json.loads(fcache.read_text());continue
            selected=json.loads((formal.run_dir(ds,fold,'A4')/'selected.json').read_text())
            if formal.sha(selected['checkpoint'])!=selected['checkpoint_sha256']:raise RuntimeError('Checkpoint changed')
            data=formal.TestFold(ds,fold,selected);model=formal.model_for(ds,'A4').to('cuda');model.load_state_dict(torch.load(selected['checkpoint'],map_location='cuda',weights_only=False)['model']);model.eval()
            foldrows=[];foldfb=[];records={}
            with torch.no_grad():
                for begin in range(0,len(data),16):
                    end=min(begin+16,len(data));inputs=torch.stack([data.base[i]['wsst'] for i in range(begin,end)]).to('cuda')
                    h=model.backbone(model.stem(inputs));logits=model.anchor_head(F.interpolate(h,size=1600,mode='linear',align_corners=False));probs=logits.sigmoid()
                    full=F.interpolate(h,size=1600,mode='linear',align_corners=True);context=F.interpolate(h,size=200,mode='linear',align_corners=False)
                    for b,i in enumerate(range(begin,end)):
                        rid=str(data.records[i]);start=int(data.starts[i]);subject=str(int(data.subjects[i]))
                        if rid not in records:
                            with np.load(RAW/name/(rid+'.npz')) as z:records[rid]=z['ecg']
                        x=records[rid][start:start+1600];ref=delineate(x)
                        anchors=formal.frozen.structured_decode(probs[b].cpu().numpy())
                        # Same frozen soft-argmax as phase_pd's native phase-grid path.
                        r=np.array([float(soft_refine(logits[b,1],v)) for v in anchors['r']])
                        anchors['r']=r
                        bounds=np.c_[r[:-1],r[1:]] if len(r)>1 else np.empty((0,2))
                        good=(np.diff(bounds,axis=1)[:,0]>=60)&(np.diff(bounds,axis=1)[:,0]<=400) if len(bounds) else np.zeros(0,bool)
                        y=np.zeros(1600,float);covered=np.zeros(1600,bool)
                        if np.any(good):
                            bb=torch.as_tensor(bounds[good],device='cuda',dtype=h.dtype);ti=phase_alignment(bb);nb=len(bb)
                            features=torch.cat((phase_resample(full[b:b+1].expand(nb,-1,-1),ti),phase_resample(probs[b:b+1].expand(nb,-1,-1),ti),context[b:b+1].expand(nb,-1,-1)),1)
                            shapes=model.shape_head(features).sigmoid()[:,0].cpu().numpy()
                            for (a,c),shape in zip(bounds[good],shapes):
                                indices=np.arange(max(0,int(np.ceil(a))),min(1600,int(np.ceil(c))))
                                y[indices]=np.interp(indices,np.linspace(a,c-1,200),shape);covered[indices]=True
                        # Existing structured.synthesize uncovered-region fallback: fixed baseline=0.
                        foldfb.append(dict(dataset=ds,fold=fold,subject=subject,record=rid,start=start,
                            fallback_fraction=float(1-covered[288:1312].mean()),invalid_pairs=int((~good).sum()),pairs=len(bounds),
                            reference_central_beats=int(np.sum((ref['r']>=288)&(ref['r']<1312)))))
                        metrics=['rr','qs_peak','qt_peak']+(['qrs','qt'] if clinical_ok else [])
                        pred=delineate(y,r) if clinical_ok else anchors
                        foldrows+=make_rows(ds,fold,subject,rid,start,ref,pred,metrics)
                        rr=r[(r>=288)&(r<1312)];gt=ref['r'][(ref['r']>=288)&(ref['r']<1312)]
                        actual=12000/np.diff(gt).mean() if len(gt)>1 else np.nan
                        estimate=12000/np.diff(rr).mean() if len(rr)>1 else np.nan
                        if np.isfinite(actual):foldrows.append(dict(dataset=ds,fold=fold,subject=subject,record=rid,start=start,ref_r=start,metric='hr',ref=float(actual),pred=float(estimate),centre_distance=0.))
                    if begin%512==0:print(f'Inference {ds} fold{fold} {end}/{len(data)}',flush=True)
            save_csv(cache,foldrows);fcache.write_text(json.dumps(foldfb));rows+=foldrows;fallback+=foldfb
            del model,data;torch.cuda.empty_cache()
    return dedup(rows),fallback

def plot(rows,stats,clinical_ok,stem):
    plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':7,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':True})
    metrics=['hr','rr','qrs','qt'] if clinical_ok else ['hr','rr','qs_peak','qt_peak'];titles=['HR (bpm)','RR (ms)','QRS duration (ms)','QT interval (ms)'] if clinical_ok else ['HR (bpm)','RR (ms)','Q–S peak interval (ms)','Q–T peak interval (ms)']
    fig,axes=plt.subplots(2,4,figsize=(7.16,3.4));clipped=[]
    for row,(ds,name,_) in enumerate(DATASETS):
        for col,(metric,title) in enumerate(zip(metrics,titles)):
            ax=axes[row,col];st=next(v for v in stats if v['dataset']==ds and v['metric']==metric)
            subset=[r for r in rows if r['dataset']==ds and r['metric']==metric and np.isfinite(r['pred'])]
            a=np.array([r['ref'] for r in subset]);b=np.array([r['pred'] for r in subset]);x=(a+b)/2;d=b-a
            limit=1.28*max(abs(st['loa_low']),abs(st['loa_high']),1.);inside=abs(d)<=limit
            if len(x)>20000:ax.hexbin(x[inside],d[inside],gridsize=35,mincnt=1,cmap='Greys',linewidths=0,rasterized=True)
            else:ax.scatter(x[inside],d[inside],s=1.5,c='#9E9E9E',alpha=.15,rasterized=True)
            for mask,sign,marker in ((d>limit,1,'^'),(d<-limit,-1,'v')):
                ax.scatter(x[mask],np.full(mask.sum(),sign*limit),s=5,c='#777777',marker=marker,clip_on=False,rasterized=True)
            clipped.append(dict(dataset=ds,metric=metric,n_above=int((d>limit).sum()),n_below=int((d<-limit).sum())))
            ax.axhline(0,color='#BDBDBD',lw=.5)
            for v,ls in ((st['bias'],'-'),(st['loa_low'],'--'),(st['loa_high'],'--')):ax.axhline(v,color='#1F4E79',ls=ls,lw=.8)
            if len(x):
                xx=np.array([x.min(),x.max()]);ax.plot(xx,st['bias']+st['slope']*(xx-x.mean()),':',color='#D55E00',lw=.8)
            ax.set_ylim(-limit,limit);ax.set_xlabel('Mean '+title);ax.set_title(title,fontsize=8)
            if col==0:ax.set_ylabel(name+'\nPredicted − reference')
            ax.text(-.15,1.1,f'({chr(97+row*4+col)})',transform=ax.transAxes,fontweight='bold',fontsize=8)
            ax.text(.98,.98,f"bias {st['bias']:.1f} [{st['bias_ci_low']:.1f}, {st['bias_ci_high']:.1f}]\nLoA [{st['loa_low']:.1f}, {st['loa_high']:.1f}]",transform=ax.transAxes,ha='right',va='top',fontsize=6.3,bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=.7))
            ax.spines[['top','right']].set_visible(False);ax.grid(False)
    fig.subplots_adjust(left=.09,right=.99,bottom=.12,top=.89,wspace=.48,hspace=.85)
    for ext in ('pdf','svg','png'):fig.savefig(FIG/(stem+'.'+ext),dpi=600)
    plt.close(fig);return clipped

def main():
    OUT.mkdir(parents=True,exist_ok=True);items=inventory()
    rt=numeric(run_roundtrip(items));rtstats=[]
    for ds,_,_ in DATASETS:
        for metric in ('qrs','qt'):
            st=agreement([r for r in rt if r['dataset']==ds and r['metric']==metric]);st.update(dataset=ds,metric=metric)
            st['passed']=bool(st['n_pairs']>0 and abs(st['bias'])<=5 and (st['loa_high']-st['loa_low'])/2<=20)
            rtstats.append(st)
    save_csv(FIG/'roundtrip_check.csv',rtstats);ok=all(r['passed'] for r in rtstats)
    print('ROUNDTRIP GATE:',ok,rtstats,flush=True)
    (OUT/'gate.json').write_text(json.dumps(dict(passed=ok,criteria='abs(bias)<=5 ms AND repeated-measures LoA half-width<=20 ms, every dataset/metric',results=rtstats),indent=2))
    rows,fb=interval_inference(items,ok);save_csv(OUT/'paired_values.csv',rows);save_csv(OUT/'fallback.csv',fb)
    stats=[]
    for ds,_,_ in DATASETS:
        for metric in ['hr','rr','qs_peak','qt_peak']+(['qrs','qt'] if ok else []):
            st=agreement([r for r in rows if r['dataset']==ds and r['metric']==metric]);stats.append(dict(dataset=ds,metric=metric,**st))
    save_csv(FIG/'bland_altman_stats.csv',stats)
    clips=plot(rows,stats,ok,'fig_bland_altman');plot(rows,stats,False,'fig_bland_altman_peak_intervals');save_csv(OUT/'clipped_points.csv',clips)
    report=['# REM-Net phase_pd clinical-interval inference, 2026-10-29','',
        'All16 test folds; frozen A4 seed-base42 checkpoints (actual historical initialization seed42+fold). No training, selection or parameter tuning.',
        '## Roundtrip gate', '', 'PASS' if ok else '**临床间期在该流程下不可靠。** Clinical route stopped; main figure uses peak intervals.',
        '', '|Dataset|Metric|MAE ms|Bias ms|Repeated LoA ms|Coverage|Pass|','|---|---|---:|---:|---|---:|---|']
    for s in rtstats:report.append(f"|{s['dataset']}|{s['metric']}|{s['mae']:.2f}|{s['bias']:.2f}|[{s['loa_low']:.2f}, {s['loa_high']:.2f}]|{s['coverage']:.3f}|{s['passed']}|")
    report+=['','## Paired agreement','', '|Dataset|Metric|N|Subjects|MAE|Bias [bootstrap CI]|Repeated LoA|Naive LoA|Coverage|ICC(A,1)|Slope [CI]|','|---|---|---:|---:|---:|---|---|---|---:|---:|---|']
    for s in stats:report.append(f"|{s['dataset']}|{s['metric']}|{s['n_pairs']}|{s['n_subjects']}|{s['mae']:.2f}|{s['bias']:.2f} [{s['bias_ci_low']:.2f}, {s['bias_ci_high']:.2f}]|[{s['loa_low']:.2f}, {s['loa_high']:.2f}]|[{s['naive_loa_low']:.2f}, {s['naive_loa_high']:.2f}]|{s['coverage']:.3f}|{s['icc_a1']:.3f}|{s['slope']:.3f} [{s['slope_ci_low']:.3f}, {s['slope_ci_high']:.3f}]|")
    report+=['','## Frozen protocol and deviations',
        '- Reference: raw-record regenerated 200Hz ECG, existing raw_reference caches; crop by original record/start indices. No continuous-model predictions used.',
        '- Roundtrip and inference delineation use the same ecg_clean and ecg_delineate(method=prominence) default parameters as frozen clinical_endpoints.py; supplied R replaces only internal R detection. Reference R is independently detected. Endpoint matching requires the same R match and both endpoints within150ms. No normal-range filters.',
        '- Each adjacent ordered-decoder R pair gets its own phase_pd feature resampling and shape-head forward, not a repeated template. Native soft_refine (softmax of R logits over integer peak±10samples, tau1) defines both fractional phase-grid boundaries and supplied R. NeuroKit accepts integer R, so supplied fractional R is rounded only at delineation. Exclusive-right target convention uses linspace(a,b−1,200); inverse maps at actual integer time samples, no integer-boundary truncation.',
        '- Fixed zero baseline for uncovered/invalid RR regions, as existing structured.synthesize; no GT or subject amplitude enters inference. RR bounds0.3–2s; no event-node warp. Per-window8s contexts retain[288:1312]. No model-side smoothing or GT alignment.',
        '- Overlapping heartbeats deduplicate by dataset/subject/record/referenceR/metric, selecting the window with referenceR closest to its centre, then earliest start, independently of predictions. Actual pairs remain individual beat pairs. HR is per available central window, 60/mean adjacentRR; windows lacking two predicted R remain failures.',
        '- LoA: Bland & Altman2007 §3 (changing true value), unequal-count one-way ANOVA on paired differences: within=MSE; between=max(0,(MS_subject−MSE)/((N−sum(n_i²)/N)/(K−1))). Bias is mean of individual differences. Naive LoA also supplied. Bootstrap resamples subjects with all their observations,10000 repetitions, local seed42. ICC(A,1) is pooled absolute-agreement ICC; regression is pooled paired difference on paired mean with subject-bootstrap CI.',
        '- Repeated-measures LoA still assumes homogeneous within-subject variance and independent within-subject residuals. Serial correlation and overlapping HR windows remain limitations; bootstrap preserves subject clusters but does not remove those model assumptions.',
        '- Peak fallback uses direct ordered-head Q/S/T peak spacings, not clinical boundaries; these are explicitly labelled peak intervals. HR/RR always use predicted R. No Q–S/Q–T relabelling as clinical QRS/QT.',
        '- Existing manuscript/Table III untouched as requested; new Table_III_clinical_addendum.tex is a reviewable addendum. If roundtrip fails, clinical columns remain unavailable, not populated with unreliable estimates.',
        '- Baseline clinical entries unavailable because no baseline continuous assembly was performed in this task; baseline shape outputs are not intrinsically incapable of assembly.',
        '', 'Reference: https://doi.org/10.1080/10543400701329422', '', '## Fallback and detection failures']
    for ds,_,_ in DATASETS:
        a=[r for r in fb if r['dataset']==ds];report.append(f"- {ds}: mean central fallback fraction {np.mean([r['fallback_fraction'] for r in a]):.4%}; {len(a)} windows. Detection failure per metric=1−coverage in bland_altman_stats.csv; reference-unassessable beats are not eligible.")
    report+=['','## Display clipping (retained in all statistics)']+[str(c) for c in clips]
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n')
    if not (FIG/'REPORT.md').exists():(FIG/'REPORT.md').write_text('\n'.join(report)+'\n')
    tex=['% New addendum only; existing Table III untouched.','\\begin{tabular}{lllrrrrr}','Dataset & Method & Metric & MAE & Bias & 95\\% LoA & ICC(A,1) & Coverage \\\\','\\hline']
    for ds,name,_ in DATASETS:
        for metric in ('qrs','qt'):
            s=next((r for r in stats if r['dataset']==ds and r['metric']==metric),None)
            values=f"{s['mae']:.2f} & {s['bias']:.2f} & [{s['loa_low']:.2f},{s['loa_high']:.2f}] & {s['icc_a1']:.3f} & {100*s['coverage']:.1f}\\%" if s else '--- & --- & --- & --- & ---'
            tex.append(name.replace('_',r'\_')+f' & REM-Net & {metric.upper()} & '+values+r' \\')
        for method in ('Chen-TCN','radarODE-MTL','Conformer','WaveGRU-Net','Attention-DCGAN'):
            for metric in ('QRS','QT'):tex.append(name.replace('_',r'\_')+f' & {method} & {metric} & --- & --- & --- & --- & ---'+r' \\')
    tex+=['\\end{tabular}','% Baselines: single-beat output; no continuous assembly evaluated in this task.',
        '% If roundtrip gate fails, REM-Net clinical values are unavailable too. Units: ms; coverage percent.']
    (OUT/'Table_III_clinical_addendum.tex').write_text('\n'.join(tex)+'\n')
    manifest=dict(training=False,gate_passed=ok,neurokit_version=nk.__version__,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),frozen_sources=formal.frozen.SOURCES,
        checkpoints=[json.loads((formal.run_dir(ds,f,'A4')/'selected.json').read_text()) for ds,_,n in DATASETS for f in range(1,n+1)])
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('DONE',OUT,flush=True)

if __name__=='__main__':main()
