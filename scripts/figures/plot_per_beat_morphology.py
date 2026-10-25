"""Reference-timed per-beat morphology Fig4 and predicted-timing FigS1.

Visualization-only frozen-weight inference. No training, template replication,
checkpoint selection or mutation of existing benchmark files.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('SSQ_PARALLEL','0')
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
ASSETS=ROOT/'docs/paper/sections/figures/source_data/fig4_per_beat_20261026'
os.environ['MPLCONFIGDIR']=str(ASSETS/'.cache/matplotlib')
os.environ['NUMBA_CACHE_DIR']=str(ASSETS/'.cache/numba')
sys.path[:0]=[str(HERE),'/tmp/rem_dual_head_formal',str(ROOT/'experiments/development/structured_decoder_v0_20261017_ab')]
import json,csv,warnings,importlib.util
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import h5py
import torch
from torch.nn import functional as F
import plot_dual_head_waveform_grid as style
import scripts.run_dual_head_interaction as metric
from src.models.continuous.dual_head_rem import DualHeadREM
from src.models.baselines.dual_endpoint import DualEndpointBaseline
from src.utils.atomic_state import atomic_json,atomic_npz
import rawio
import inference as frozen_inference


def target_for_window(ecg):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return metric.shape_target(ecg,metric.extract_points(ecg)['r'])


def selection_candidates():
    path=ASSETS/'selection_candidates.csv'
    if path.exists():
        rows=list(csv.DictReader(path.open()))
        for r in rows:
            for k in ('fold','index','subject','start','stop'):r[k]=int(r[k])
            r['window_pcc']=float(r['window_pcc'])
        return rows
    rows=[];audits=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for ds,nfold in (('mmecg',11),('crrvs',5)):
            for fold in range(1,nfold+1):
                ref=style.load_npz(style.SOURCE/'references'/ds/f'fold_{fold:02d}.npz')
                pred=style.load_npz(style.SOURCE/'predictions'/ds/'rem_net/seed_42'/f'fold_{fold:02d}/test_predictions.npz')
                targets=np.stack(list(pool.map(target_for_window,ref['ecg'],chunksize=16)))
                scores=metric.pcc_per_row(pred['shape'].astype(float),targets)
                frozen=style.load_npz(style.SOURCE/'predictions'/ds/'rem_net/seed_42'/f'fold_{fold:02d}/test_predictions.npz')
                official=json.loads((style.SOURCE/'predictions'/ds/'rem_net/seed_42'/f'fold_{fold:02d}/test_metrics.json').read_text())['shape']['pcc']
                audits.append(dict(dataset=ds,fold=fold,recomputed_mean=float(scores.mean()),official_pcc=official,
                    absolute_difference=abs(float(scores.mean())-official)))
                if abs(float(scores.mean())-official)>.002:
                    raise ValueError(f'Table morphology target mismatch {ds} fold{fold}: {audits[-1]}')
                for i,score in enumerate(scores):
                    rows.append(dict(dataset=ds,fold=fold,index=i,subject=int(ref['subject_id'][i]),
                        scene=str(ref['state_code'][i]),record_id=str(ref['record_id'][i]),
                        start=int(ref['start_idx'][i]),stop=int(ref['end_idx'][i]),window_pcc=float(score)))
                print(f'Table-target audit {ds} f{fold}: delta={audits[-1]["absolute_difference"]:.6g}',flush=True)
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic_json(ASSETS/'table_target_audit.json',audits)
    return rows


def record_radar(ds,rid):
    # Read existing reference cache only. No writes to the old A directories.
    if not (rawio.HERE/'raw_reference'/ds/f'{rid}.npz').exists():raise FileNotFoundError(rid)
    return rawio.record_data(ds,rid,with_radar=True)


def get_models(ds,fold):
    models={};provenance={}
    for name in style.METHODS:
        path=frozen_inference.checkpoint_path(ds,fold,name)
        model=DualHeadREM('qrst',in_bins=50 if ds=='MMECG' else 1) if name=='rem_net' else DualEndpointBaseline(name,in_bins=50 if ds=='MMECG' else 1)
        saved=torch.load(path,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model'],strict=True)
        models[name]=model.to('cuda:0').eval().requires_grad_(False)
        provenance[name]=dict(path=str(path),sha256=style.sha(path),epoch=saved.get('epoch'))
    return models,provenance


def transform_task(task):
    ds,x=task
    if ds=='MMECG':return frozen_inference.one_wsst(x)
    return rawio.cw._stft_map(x[0],200,128,112).astype(np.float16)


@torch.inference_mode()
def predict_contexts(ds,radar,starts,models):
    rcg=np.stack([rawio.radar_window(ds,radar,int(s)) for s in starts])
    # Fork CPU work BEFORE models have ever initialized CUDA is unsafe here;
    # use spawn so workers never inherit a live CUDA context.
    import multiprocessing
    context=multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=4,mp_context=context) as pool:
        features=np.stack(list(pool.map(transform_task,[(ds,x) for x in rcg],chunksize=1)))
    out={name:dict(shape=[],prob=[]) for name in models}
    for a in range(0,len(starts),4):
        x=torch.from_numpy(features[a:a+4].astype(np.float32)).to('cuda:0')
        r=torch.from_numpy(rcg[a:a+4]).to('cuda:0')
        for name,model in models.items():
            if name=='rem_net':z=model(x)
            else:
                tf=F.interpolate(x,size=(71,240),mode='bilinear',align_corners=False) if ds=='CR_RVS' and name=='radarode_mtl' else x
                z=model(wsst=tf,rcg=r)
            out[name]['shape'].extend(z['shape'][:,0].cpu().numpy())
            out[name]['prob'].extend(torch.sigmoid(z['anchor_logits']).cpu().numpy())
    return {name:{k:np.asarray(v) for k,v in values.items()} for name,values in out.items()}


def per_case(row):
    ds='MMECG' if row['dataset']=='mmecg' else 'CR_RVS'
    cache=ASSETS/f"{row['dataset']}_{row['scene']}_inference.npz"
    meta_path=cache.with_suffix('.json')
    snapshot=style.load_npz(style.SOURCE/'references'/row['dataset']/f"fold_{row['fold']:02d}.npz")
    ecg=snapshot['ecg'][row['index']]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');ref_r=metric.extract_points(ecg)['r'].astype(int)
    bounds=[(int(a),int(b)) for a,b in zip(ref_r[:-1],ref_r[1:]) if b>style.LEFT and a<style.RIGHT]
    if not bounds:raise ValueError('No reference beats; do not replace the selected example')
    base_path=style.SOURCE/'predictions'/row['dataset']/'rem_net/seed_42'/f"fold_{row['fold']:02d}/test_predictions.npz"
    coarse=style.decoder()(style.load_npz(base_path)['anchor_probability'][row['index']].astype(float))
    pred_bounds=[(int(a),int(b)) for a,b in zip(coarse['r'][:-1],coarse['r'][1:]) if b>style.LEFT and a<style.RIGHT]
    origins=[int(round(row['start']+(a+b)/2-800)) for a,b in bounds]
    pred_origins=[int(round(row['start']+(a+b)/2-800)) for a,b in pred_bounds]
    if cache.exists():
        saved=style.load_npz(cache);provenance=json.loads(meta_path.read_text())
        assert np.array_equal(saved['reference_origins'],origins)
        assert np.array_equal(saved['predicted_origins'],pred_origins)
    else:
        raw=record_radar(ds,row['record_id']);n=raw['radar'].shape[-1]
        # No clamping to a neighboring context and no template substitution.
        if any(s<0 or s+1600>n for s in origins):raise ValueError('Chosen morphology beat lacks full8s centered context')
        valid_pred=[s for s in pred_origins if 0<=s and s+1600<=n]
        starts=sorted(set(origins+valid_pred))
        models,provenance=get_models(ds,row['fold'])
        prediction=predict_contexts(ds,raw['radar'],starts,models)
        saved=dict(starts=np.asarray(starts),reference_origins=np.asarray(origins),predicted_origins=np.asarray(pred_origins))
        for name,values in prediction.items():
            saved[name+'_shape']=values['shape'];saved[name+'_prob']=values['prob']
        atomic_npz(cache,saved);atomic_json(meta_path,provenance)
        del models;torch.cuda.empty_cache()
    lookup={int(s):i for i,s in enumerate(saved['starts'])}
    gt=style.reference_display(ecg,ref_r)
    arrays=dict(gt=gt[style.LEFT:style.RIGHT],gt_r=ref_r-style.LEFT,
                beat_bounds=np.asarray(bounds)-style.LEFT)
    targets=np.stack([metric.shape_target(ecg,np.array([a,b])) for a,b in bounds])
    arrays['reference_shapes']=targets
    scores={};main_events={k:[] for k in 'qrst'};beat_logs=[]
    for name in style.METHODS:
        wave=np.zeros(1600);shapes=np.stack([saved[name+'_shape'][lookup[s]] for s in origins])
        # Table PCC definition is arithmetic mean of per-beat200sample Pearson.
        scores[name]=float(metric.pcc_per_row(shapes,targets).mean())
        arrays[name+'_shapes']=shapes
        for k,(a,b) in enumerate(bounds):
            wave[a:b]=np.interp(np.linspace(0,199,b-a),np.arange(200),style.minmax(shapes[k]))
            if name=='rem_net':
                points=style.decoder()(saved[name+'_prob'][lookup[origins[k]]])
                local_reference_r=row['start']+a-origins[k]
                if len(points['r']):
                    j=int(np.argmin(abs(points['r']-local_reference_r)))
                    for event in 'qrst':main_events[event].append(float(points[event][j]+origins[k]-row['start']-style.LEFT))
        arrays[name]=wave[style.LEFT:style.RIGHT]
    for k,v in main_events.items():arrays['pred_'+k]=np.unique(v)
    arrays['radar'],radar=style.radar_input(row)
    # GT-free supplement: own newly inferred morphology per predicted interval.
    supplement=np.zeros(1600);covered=np.zeros(1600,bool);fallback_reasons=[]
    for (a,b),origin in zip(pred_bounds,pred_origins):
        if origin not in lookup:
            fallback_reasons.append(dict(start=a,stop=b,reason='full centered radar context unavailable'));continue
        shape=saved['rem_net_shape'][lookup[origin]]
        supplement[a:b]=np.interp(np.linspace(0,199,b-a),np.arange(200),style.minmax(shape))
        covered[a:b]=True
    arrays['assembled']=supplement[style.LEFT:style.RIGHT]
    arrays['fallback']=~covered[style.LEFT:style.RIGHT]
    arrays['assembled_pred_r']=coarse['r']-style.LEFT
    meta=dict(**row,window_pcc_by_method=scores,metric='arithmetic mean per-beat200sample PCC; same formula/target construction as Table I, new sliding-window sampling',
        radar=radar,checkpoints=provenance,reference_beat_count=len(bounds),no_replication=True,
        reference_beat_global_bounds=[[row['start']+a,row['start']+b] for a,b in bounds],
        reference_context_starts=origins,predicted_context_starts=pred_origins,
        assembled_window_pcc=style.pcc(arrays['assembled'],arrays['gt']),fallback_samples=int(arrays['fallback'].sum()),
        fallback_reasons=fallback_reasons)
    atomic_npz(ASSETS/f"{row['dataset']}_{row['scene']}_plotted.npz",arrays)
    return meta,arrays


def supplementary(cases,path):
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(7.16,4.5));outer=fig.add_gridspec(2,1,left=.125,right=.995,bottom=.09,top=.9,hspace=.45)
    for block,ds in enumerate(('mmecg','crrvs')):
        gs=outer[block].subgridspec(3,4,hspace=.12,wspace=.045)
        group=[c for c in cases if c[0]['dataset']==ds]
        for col,(meta,a) in enumerate(group):
            for row,key in enumerate(('radar','gt','assembled')):
                ax=fig.add_subplot(gs[row,col]);style.grid(ax,radar=row==0)
                ax.plot(np.arange(1024)/200,a[key],color='#262626',lw=.55)
                if row:ax.set_ylim(-.13,1.23)
                if key=='assembled':
                    mask=a['fallback'];edges=np.diff(np.r_[False,mask,False].astype(int))
                    for left,right in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
                        ax.axvspan(left/200,right/200,color='#AAAAAA',alpha=.45,zorder=5)
                    ax.text(.98,.95,f"PCC = {meta['assembled_window_pcc']:.2f}",transform=ax.transAxes,ha='right',va='top',fontsize=5.5)
                if col==0:ax.text(-.06,.5,('Radar input\n(a.u.)','GT ECG','REM-Net')[row],transform=ax.transAxes,ha='right',va='center',fontsize=7,fontweight='bold' if row==2 else 'normal')
                if row==0:ax.set_title(meta['scene_label'],fontsize=7,fontweight='bold',pad=3)
                if col==0 and row==2:
                    ax.plot([.6,1.6],[-.34,-.34],transform=ax.get_xaxis_transform(),clip_on=False,color='k',lw=1)
                    ax.text(1.1,-.40,'1 s',transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=6)
        b=outer[block].get_position(fig)
        fig.text(.125,b.y1+.065,f"({'ab'[block]}) {'MMECG (77-GHz FMCW)' if ds=='mmecg' else 'CR_RVS (24-GHz CW)'}",fontsize=8.5,fontweight='bold')
        if ds=='crrvs':
            ax=fig.add_subplot(gs[:,3]);ax.axis('off');ax.text(.5,.5,'Predicted R–R assembly\nGray: fallback / uncovered\nFlat fill; never GT-filled',ha='center',va='center',fontsize=7)
    for ext in ('pdf','svg','png'):fig.savefig(path.with_suffix('.'+ext),dpi=600,facecolor='white')
    plt.close(fig)


def main():
    ASSETS.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows=selection_candidates()
    # Table uses arithmetic subject means, not Fisher-z; fix that for selection.
    selected=[];subject_scores=[]
    for ds,scenes in style.SCENES.items():
        for scene,label in scenes:
            group=[r for r in rows if r['dataset']==ds and r['scene']==scene]
            subjects=[dict(subject=s,pcc=float(np.mean([r['window_pcc'] for r in group if r['subject']==s]))) for s in sorted({r['subject'] for r in group})]
            center=float(np.median([s['pcc'] for s in subjects]))
            subject=min(subjects,key=lambda s:(abs(s['pcc']-center),s['subject']))
            subset=[r for r in group if r['subject']==subject['subject']];median=float(np.median([r['window_pcc'] for r in subset]))
            row=min(subset,key=lambda r:(abs(r['window_pcc']-median),r['record_id'],r['start']))
            selected.append(dict(**row,scene_label=label,subject_morphology_pcc=subject['pcc'],scenario_subject_median=center,subject_window_median=median))
            subject_scores.extend([dict(dataset=ds,scene=scene,**s) for s in subjects])
    atomic_json(ASSETS/'selection_before_inference.json',dict(rows=selected,subjects=subject_scores,rule='Table-I formula; arithmetic subject mean, scenario median, within-subject window median; no baseline-guided selection'))
    cases=[]
    for row in selected:
        print(f"Per-beat inference {row['dataset']} {row['scene']} {row['record_id']} start{row['start']}",flush=True)
        cases.append(per_case(row))
    import shutil
    for ext in ('pdf','png','svg'):
        old=style.OUTPUT.with_suffix('.'+ext);backup=ASSETS/('previous_'+old.name)
        if old.exists() and not backup.exists():shutil.copy2(old,backup)
    style.plot(cases,style.OUTPUT,metric_label='Shape-200 PCC',morphology=True)
    supplementary(cases,style.OUTPUT.parent/'fig_s1_predicted_anchor_assembly')
    atomic_json(ASSETS/'manifest.json',dict(cases=[m for m,_ in cases],seed=42,model_training=False,
        caption_required='beats placed on reference R-R intervals; timing evaluated separately',
        main='reference-assisted morphology visualization, not deployable continuous reconstruction',
        supplement='no GT anchors/shapes/amplitudes used to generate predictions; unknown support flat-filled and shaded gray',
        scoring='Table I per-beat target construction and arithmetic-PCC definition; new beat-centered test sampling, not unchanged aggregate test numbers'))
    print('Fig4 and FigS1 complete; no training',flush=True)


if __name__=='__main__':main()
