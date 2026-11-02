"""Frozen main-table phase_pd and adapted-baseline 60s HR/HRV agreement.

No training/GT event matching. Frozen protocol: docs/paper/hrv_agreement_20261103/spec.json.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,csv,json,hashlib,argparse,subprocess,importlib.util
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
FS=200
SEGMENT=12000
METHODS=('rem_net','mmecg_tcn','radarode_mtl','cross_modal_conformer','wavegru_net','attention_dcgan')
NAMES=dict(zip(METHODS,('REM-Net','Chen-TCN','radarODE-MTL','Conformer','WaveGRU-Net','Attention-DCGAN')))
OUT=ROOT/'docs/paper/hrv_agreement_20261103'
BASE=ROOT/'docs/paper/sections/figures/source_data/baseline_clinical_20261101'
RAW=ROOT/'experiments/development/structured_decoder_v0_20261017_ab/raw_reference'
FORMAL=ROOT/'experiments/phase_resampling_20261027/formal_ablation_20261027'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False));tmp.replace(p)
def save_csv(p,rows):
 if not rows:raise ValueError('No rows '+str(p))
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def correct_rr(r):
 """One-pass physiology + centered5 original-position median, no interpolation."""
 r=np.asarray(r,float)
 if np.any(~np.isfinite(r)) or np.any(np.diff(r)<=0):raise ValueError('R must be finite and strictly increasing')
 rr=np.diff(r)/FS
 physiologic=(rr>=.3)&(rr<=2.)
 med=np.full(len(rr),np.nan)
 for i in range(len(rr)):
  a,b=max(0,i-2),min(len(rr),i+3)
  values=rr[a:b][physiologic[a:b]]
  if len(values):med[i]=np.median(values)
 keep=physiologic&np.isfinite(med)&(np.abs(rr-med)<=.2*med+1e-12)
 return rr,keep,physiologic,med

def rr_metrics(rr):
 rr=np.asarray(rr,float)
 if len(rr)<2:return dict(hr=float('nan'),sdnn=float('nan'),rmssd=float('nan'))
 return dict(hr=float(60/np.mean(rr)),sdnn=float(np.std(rr,ddof=1)*1000),rmssd=float(np.sqrt(np.mean(np.diff(rr)**2))*1000))

def ownership(starts):
 starts=np.asarray(starts,int)
 if len(starts)==0 or np.any(np.diff(starts)<=0):raise ValueError('Unique increasing native window starts required')
 centers=starts+800
 mids=(centers[:-1]+centers[1:])/2
 left=np.maximum(starts,np.r_[starts[0],mids])
 right=np.minimum(starts+1600,np.r_[mids,starts[-1]+1600])
 return left,right

def stitch(starts,rs):
 """Use each native window only in its closest-center time ownership region."""
 left,right=ownership(starts);events=[]
 if len(rs)!=len(starts):raise ValueError('Window/anchor cache count differs')
 for s,r,a,b in zip(starts,rs,left,right):
  r=np.asarray(r,float)+s
  events.extend(r[(r>=a)&(r<b)])
 events=np.asarray(events,float)
 if len(events)>1 and np.any(np.diff(events)<=0):raise ValueError('Duplicate/out-of-order stitched R')
 return events, list(zip(left,right))

def covered(a,b,intervals):
 cursor=float(a)
 for x,y in intervals:
  if y<=cursor:continue
  if x>cursor+1e-8:return False
  cursor=max(cursor,y)
  if cursor>=b:return True
 return False

def signal_segment(r,prepared,a,b):
 rr,keep,physio,med=prepared
 indices=np.flatnonzero((r[:-1]>=a)&(r[1:]<b))
 raw=rr[indices];accepted=keep[indices];kept=raw[accepted]
 n=len(raw);nk=len(kept)
 fraction=nk/n if n else float('nan')
 acceptable=n>0 and fraction>=.8 and nk>=2
 return dict(n_rr=n,n_kept=nk,n_removed=n-nk,n_out_of_range=int((~physio[indices]).sum()),
             n_local_outlier=int((physio[indices]&~accepted).sum()),retained_fraction=fraction,
             acceptable=bool(acceptable),**rr_metrics(kept))

def paired_rows(meta,length,starts,rs,ref):
 pred,intervals=stitch(starts,rs);predprep=correct_rr(pred);refprep=correct_rr(ref);out=[]
 for index,a in enumerate(range(0,length-SEGMENT+1,SEGMENT)):
  b=a+SEGMENT;supported=covered(a,b,intervals)
  p=signal_segment(pred,predprep,a,b);y=signal_segment(ref,refprep,a,b)
  usable=supported and p['acceptable'] and y['acceptable']
  reasons=[]
  if not supported:reasons.append('unavailable_test_window_support')
  for name,x in [('pred',p),('ref',y)]:
   if not x['n_rr']:reasons.append(name+'_no_rr')
   elif x['retained_fraction']<.8:reasons.append(name+'_retained_below_80pct')
   elif x['n_kept']<2:reasons.append(name+'_insufficient_rr')
  row=dict(**meta,segment_index=index,start_sample=a,end_sample=b,start_s=a/FS,end_s=b/FS,
           supported=supported,used=bool(usable),discard_reason=';'.join(reasons))
  for prefix,x in [('pred',p),('ref',y)]:
   row.update({prefix+'_'+k:v for k,v in x.items()})
  for metric in ('hr','sdnn','rmssd'):
   row[metric+'_error']=p[metric]-y[metric] if usable else float('nan')
  out.append(row)
 return out,dict(**meta,n_native_windows=len(starts),n_stitched_r=len(pred),n_reference_r=len(ref),
     n_complete_60s_segments=length//SEGMENT,tail_samples=length%SEGMENT,record_samples=length,
     n_short_stitched_rr=int((np.diff(pred)<60).sum()))

def load_shared():
 sys.path.insert(0,str(ROOT))
 p=ROOT/'docs/paper/sections/figures/plot_phase_pd_bland_altman.py'
 spec=importlib.util.spec_from_file_location('hrv_shared_clinical',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 return m

class NativeInput:
 """Exactly the frozen main-table subject-pool/per-split TF tensors."""
 def __init__(self,ds,fold,items):
  import h5py
  self.ds=ds;self.files={};self.maps={}
  if ds=='crrvs':self.files[0]=h5py.File(Path('/home/qhh2237/Datasets/CR_RVS/processed/q1_continuous_kfold')/f'fold_{fold:02d}/test_wsst8s.h5')
  else:
   for sid in sorted(set(x['subject'] for x in items)):
    h=h5py.File(Path('/home/qhh2237/Datasets/MMECG/processed/wsst8s_subject_pool')/f'subject_{sid:02d}.h5');self.files[sid]=h
    text=lambda v:v.decode() if isinstance(v,bytes) else str(v)
    self.maps[sid]={(text(r),int(s)):i for i,(r,s) in enumerate(zip(h['record_id'][:],h['start_idx'][:]))}
 def get(self,x):
  if self.ds=='crrvs':return np.asarray(self.files[0]['wsst'][x['index']],np.float32)
  return np.asarray(self.files[x['subject']]['wsst'][self.maps[x['subject']][x['record'],x['start']]],np.float32)
 def close(self):
  for h in self.files.values():h.close()

def rem_anchors(shared,ds,fold,items,device,source_manifest,provenance):
 import torch
 from torch.nn import functional as F
 selected=FORMAL/f'runs/{ds}/fold_{fold:02d}/A4/selected.json';sel=json.loads(selected.read_text());cp=Path(sel['checkpoint'])
 fingerprint=sha(cp)
 assert fingerprint==sel['checkpoint_sha256']
 expected=next(r['checkpoint_sha256'] for r in source_manifest['checkpoints'] if r['dataset']==ds and r['fold']==fold)
 assert fingerprint==expected,'Main clinical checkpoint fingerprint differs'
 provenance[str(cp)]=fingerprint
 path=OUT/f'anchor_cache/{ds}/fold_{fold:02d}/rem_net.npz';path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():
  with np.load(path,allow_pickle=True) as z:
   assert str(z['checkpoint_sha256'])==fingerprint
   assert np.array_equal(z['starts'],[x['start'] for x in items])
   assert np.array_equal(z['records'].astype(str),[x['record'] for x in items])
   return list(z['anchors'])
 model=shared.formal.model_for(ds,'A4').to(device)
 model.load_state_dict(torch.load(cp,map_location=device,weights_only=False)['model'],strict=True);model.eval()
 native=NativeInput(ds,fold,items);rs=[]
 try:
  with torch.no_grad():
   for a in range(0,len(items),8):
    x=torch.from_numpy(np.stack([native.get(i) for i in items[a:a+8]])).to(device)
    h=model.backbone(model.stem(x));logits=model.anchor_head(F.interpolate(h,size=1600,mode='linear',align_corners=False))
    for row,prob in zip(logits,logits.sigmoid().cpu().numpy()):
     decoded=shared.formal.frozen.structured_decode(prob)
     rs.append(np.array([float(shared.soft_refine(row[1],v)) for v in decoded['r']],float))
    if a%512==0:print(f'REM event inference {ds} fold{fold}: {min(a+8,len(items))}/{len(items)}',flush=True)
 finally:native.close()
 np.savez_compressed(path,anchors=np.array(rs,object),checkpoint_sha256=fingerprint,starts=np.array([x['start'] for x in items]),records=np.array([x['record'] for x in items]))
 del model
 if device.startswith('cuda'):torch.cuda.empty_cache()
 return rs

def dense_cr_anchors(shared, items, baseline_cp, provenance, device):
 """Fill sparse CR test-window gaps, same frozen models and native4s grid."""
 import torch, h5py
 from torch.nn import functional as F
 from scripts.figures import evaluate_baseline_clinical_intervals as native_eval
 baseline_class=native_eval.DualEndpointBaseline
 result={};audit=[]
 for fold in range(1,6):
  fi=[r for r in items if r['dataset']=='crrvs' and r['fold']==fold]
  models={};fingerprints={}
  for method in METHODS:
   if method=='rem_net':
    sel=json.loads((FORMAL/f'runs/crrvs/fold_{fold:02d}/A4/selected.json').read_text());cp=Path(sel['checkpoint']);fp=sel['checkpoint_sha256']
   else:
    sel=baseline_cp[f'crrvs:{fold}:{method}'];cp=Path(sel['path']);fp=sel['sha256']
   assert sha(cp)==fp
   provenance[str(cp)]=fp;fingerprints[method]=fp
   model=shared.formal.model_for('crrvs','A4') if method=='rem_net' else baseline_class(method,in_bins=1)
   model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['model'],strict=True)
   models[method]=model.to(device).eval()
  native=NativeInput('crrvs',fold,fi)
  base_path=Path('/home/qhh2237/Datasets/CR_RVS/processed/q1_continuous_kfold')/f'fold_{fold:02d}/test.h5'
  try:
   for rid in sorted(set(x['record'] for x in fi)):
    group=[x for x in fi if x['record']==rid]
    with np.load(RAW/'CR_RVS'/(rid+'.npz')) as z:length=len(z['ecg'])
    old_starts=set(x['start'] for x in group)
    starts=sorted(set(range(0,length-1600+1,800))-old_starts)
    path=OUT/f'anchor_cache/crrvs/fold_{fold:02d}/{rid}_extra.npz'
    if path.exists():
     with np.load(path,allow_pickle=True) as z:
      assert np.array_equal(z['starts'],starts)
      assert json.loads(str(z['checkpoint_sha256']))==fingerprints
      result[fold,rid]=(np.array(starts),{method:list(z[method]) for method in METHODS})
     continue
    if not starts:
     result[fold,rid]=(np.array([],int),{method:[] for method in METHODS});continue
    raw=native_eval.helper.rawio.record_data('CR_RVS',rid,with_radar=True)
    radar=raw['radar']
    # Record-only preprocessing fidelity check, not a result-based adjustment.
    check=group[len(group)//2]
    generated=native_eval.helper.rawio.radar_window('CR_RVS',radar,check['start'])
    with h5py.File(base_path) as h:stored=np.asarray(h['rcg'][check['index']],np.float32)
    assert np.allclose(generated,stored,atol=2e-5,rtol=2e-5),'RCG preprocessing changed'
    tf=native_eval.tf_worker(('crrvs',generated))
    assert np.array_equal(tf,native.get(check).astype(np.float16)),'STFT preprocessing changed'
    anchors={method:[] for method in METHODS}
    with torch.no_grad():
     for begin in range(0,len(starts),8):
      x=np.stack([native_eval.helper.rawio.radar_window('CR_RVS',radar,start) for start in starts[begin:begin+8]])
      rcg=torch.from_numpy(x).to(device)
      tf=np.stack([native_eval.tf_worker(('crrvs',xx)) for xx in x]).astype(np.float32)
      tf=torch.from_numpy(tf).to(device)
      for method,model in models.items():
       if method=='rem_net':
        h=model.backbone(model.stem(tf));logits=model.anchor_head(F.interpolate(h,size=1600,mode='linear',align_corners=False))
       elif method=='radarode_mtl':
        logits=model(wsst=F.interpolate(tf,size=(71,240),mode='bilinear',align_corners=False))['anchor_logits']
       else:logits=model(rcg=rcg)['anchor_logits']
       for row,prob in zip(logits,logits.sigmoid().cpu().numpy()):
        decoded=shared.formal.frozen.structured_decode(prob)
        anchors[method].append(np.array([float(shared.soft_refine(row[1],v)) for v in decoded['r']],float))
    np.savez_compressed(path,starts=np.array(starts),checkpoint_sha256=json.dumps(fingerprints),**{k:np.array(v,object) for k,v in anchors.items()})
    result[fold,rid]=(np.array(starts),anchors)
    audit.append(dict(fold=fold,record=rid,original_windows=len(group),added_windows=len(starts),rcg_match=True,stft_bitwise_match=True))
    print(f'CR contiguous completion f{fold} {rid}: +{len(starts)} windows x6',flush=True)
  finally:
   native.close();models.clear()
   if device.startswith('cuda'):torch.cuda.empty_cache()
 save_csv(OUT/'dense_preprocessing_audit.csv',audit) if audit else None
 return result

def summarize(rows,shared):
 from scipy.stats import pearsonr
 result=[]
 for ds in ('mmecg','crrvs'):
  for method in METHODS:
   allrows=[r for r in rows if r['dataset']==ds and r['method']==method]
   supported=[r for r in allrows if r['supported']];used=[r for r in allrows if r['used']]
   n=len(allrows);ns=len(supported)
   counts=dict(n_segments_total=n,n_segments_supported=ns,n_segments_used=len(used),
      n_segments_discarded=n-len(used),segments_discarded_pct=100*(n-len(used))/n if n else float('nan'),
      n_unavailable_support=n-ns,artefact_discarded_pct_supported=100*(ns-len(used))/ns if ns else float('nan'),
      pred_segments_failed_correction=sum(not r['pred_acceptable'] for r in supported),
      ref_segments_failed_correction=sum(not r['ref_acceptable'] for r in supported),
      n_subjects=len(set(r['subject'] for r in used)))
   for signal in ('pred','ref'):
    for key in ('n_rr','n_kept','n_removed','n_out_of_range','n_local_outlier'):
     counts[signal+'_'+key]=sum(r[signal+'_'+key] for r in supported)
    counts[signal+'_rr_removed_pct']=100*counts[signal+'_n_removed']/counts[signal+'_n_rr'] if counts[signal+'_n_rr'] else float('nan')
   for metric,unit in [('hr','bpm'),('sdnn','ms'),('rmssd','ms')]:
    ref=np.array([r['ref_'+metric] for r in used]);pred=np.array([r['pred_'+metric] for r in used]);diff=pred-ref
    subjects=np.array([r['subject'] for r in used])
    if len(used)>1 and len(np.unique(subjects))>1:
     bias,low,high,within,between=shared.repeated_loa(diff,subjects)
    else:bias=float(diff.mean()) if len(diff) else float('nan');low=high=within=between=float('nan')
    correlation=float(pearsonr(ref,pred).statistic) if len(ref)>1 and np.ptp(ref)>0 and np.ptp(pred)>0 else float('nan')
    result.append(dict(dataset=ds,method=method,method_label=NAMES[method],metric=metric,unit=unit,
       mae=float(abs(diff).mean()) if len(diff) else float('nan'),bias=bias,loa_low=low,loa_high=high,
       pearson_r=correlation,within_variance=within,between_variance=between,**counts))
 return result

def anchor_consistency(items,sources):
 """Check native cached REM R against frozen clinical central-window HR."""
 audit=[]
 for ds,nf in [('mmecg',11),('crrvs',5)]:
  for fold in range(1,nf+1):
   path=ROOT/f'docs/paper/sections/figures/source_data/phase_pd_bland_altman_20261029/{ds}_fold_{fold:02d}_pairs.csv'
   expected={(x['record'],int(x['start'])):float(x['pred']) for x in csv.DictReader(path.open()) if x['metric']=='hr'}
   errors=[];disagree=0
   for x in items:
    if x['dataset']!=ds or x['fold']!=fold:continue
    old=expected.get((x['record'],x['start']))
    if old is None:continue
    r=sources[(ds,fold,'rem_net',x['index'])];r=r[(r>=288)&(r<1312)]
    new=12000/np.diff(r).mean() if len(r)>1 else np.nan
    if np.isfinite(old) and np.isfinite(new):errors.append(abs(old-new))
    else:disagree+=int(np.isfinite(old)!=np.isfinite(new))
   assert not disagree and max(errors,default=0)<.001,'Frozen refined anchors differ'
   audit.append(dict(dataset=ds,fold=fold,finite_pairs=len(errors),max_hr_difference_bpm=max(errors,default=0),finite_status_disagreements=disagree))
 write_json(OUT/'refined_anchor_consistency.json',audit)
 return audit

def report(rows,summary,records,provenance,shared):
 retention=[]
 for ds in ('mmecg','crrvs'):
  for method in METHODS:
   subset=[r for r in rows if r['dataset']==ds and r['method']==method]
   for sid in sorted(set(r['subject'] for r in subset)):
    a=[r for r in subset if r['subject']==sid]
    retention.append(dict(dataset=ds,method=method,subject=sid,n_segments=len(a),n_used=sum(r['used'] for r in a),n_pred_correction_failed=sum(not r['pred_acceptable'] for r in a),n_ref_correction_failed=sum(not r['ref_acceptable'] for r in a)))
 save_csv(OUT/'subject_retention.csv',retention)
 assert all(r['supported'] for r in rows),'Incomplete60s support remains'
 assert len({(r['dataset'],r['record'],r['method'],r['segment_index']) for r in rows})==len(rows)
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 script_sha=sha(Path(__file__))
 spec_path=OUT/'spec.json'
 frozen_source=ROOT/'docs/paper/sections/figures/source_data/phase_pd_bland_altman_20261029/manifest.json'
 info=dict(complete=True,training=False,date='2026-11-03',producing_commit=commit,script='scripts/evaluate_hrv_agreement.py',
           script_sha256=script_sha,spec_sha256=sha(spec_path),shared_repeated_loa_sha256=sha(ROOT/'docs/paper/sections/figures/plot_phase_pd_bland_altman.py'),
           checkpoint_sha256=provenance,outputs={p.name:sha(p) for p in [OUT/'hrv_segments.csv',OUT/'hrv_summary.csv',OUT/'record_inventory.csv',OUT/'subject_retention.csv',OUT/'dense_preprocessing_audit.csv',OUT/'refined_anchor_consistency.json']},
           reference_manifest_sha256=sha(frozen_source),reference_source='frozen record-level event_r',n_records=len(records)//len(METHODS))
 write_json(OUT/'manifest.json',info)
 lines=['# HR/HRV agreement — final phase_pd and adapted baselines','',
        'Date:2026-11-03. Inference only; no retraining, tuning, or test-based selection.',
        f'Producing script: `scripts/evaluate_hrv_agreement.py`; commit `{commit}`; script SHA256 `{script_sha}`.',
        'Frozen specification: `docs/paper/hrv_agreement_20261103/spec.json`. All numbers below are copied from hrv_summary.csv; full individual accepted AND discarded segments are in hrv_segments.csv.',
        '', '## Interpretation and denominators',
        '- Short60s time-domain estimates, not standard long-duration clinical HRV validation. ECG R events are automated references, not adjudicated normal/sinus beats; corrected RR are not independently confirmed NN intervals.',
        '- Original CR test windows were sparse. The first sparse-only pass is archived and NOT the final agreement analysis. Frozen-checkpoint inference fills missing4s-grid native radar windows for the same80 formal-test records; no ECG gating or threshold tuning. Per-record RCG/STFT fidelity audited in dense_preprocessing_audit.csv.',
        '- Recording-time-zero nonoverlapping60s segments; incomplete tail excluded and listed by record. All methods use the SAME native window ownership, no reference matching or lag correction. Native full8s-window edge anchors are retained at recording edges; this R-event-only analysis does not use the clinical waveform central5.12s cropping.',
        '- Prediction-only nearest-center native window ownership prevents duplicate overlapping-window events. No clustering or reference-driven beat assignment. Reference R from full record, never generated from window-normalized ECG.',
        '- Both signals corrected independently once. Local5-index median includes current RR and up to2 neighbors each side, excluding physiologically invalid entries from median; no iterative filtering. Remove relative deviations strictly>20%. Segment quality threshold>=80% separately for BOTH signals. At least2 keptRR required.',
        '- HR=60/meanRR, SDNN=sampleSD, RMSSD=difference of consecutive retained RR (without interpolation); no extra gap-filling or synthetic beats.',
        '- Segments used are paired intersection per method. Total discard percentage includes unavailable native test-window support; artifact-only discard percentage uses supported segments. RR-removal rates pool original intervals over ALL supported segments, including segments ultimately discarded; separate prediction/reference counts exported. Method-dependent accepted sets can bias ranking; this is descriptive agreement, no model-selection or comparative significance claim.',
        '- Repeated-measures LoA reuses the frozen unequal-count subject-ANOVA function (Bland & Altman2007, doi:10.1080/10543400701329422). MAE/bias/r pool paired segments, not subject means. Homogeneous within-subject variance and independent residual assumptions remain; subject clustering does not remove serial dependence.',
        '- Local outlier correction cannot reliably distinguish missed beats, ectopy or arrhythmia; missed/double beats may still corrupt short-term HRV. Retained fraction is not R detection coverage.',
        '- CRRVS_0024 has no windows in the formal test splits and no model-output coverage; it is not silently synthesized. Evaluated records are exactly the main-table test inventory (91 MMECG,80 CR_RVS).',
        '', '## Agreement summary',
        '|Dataset|Method|Metric|MAE|Bias|Repeated95%LoA|Pearson r|Used/total|Discard%|RR removed pred/ref%|',
        '|---|---|---|---:|---:|---|---:|---|---:|---|']
 for r in summary:
  lines.append(f"|{r['dataset']}|{r['method_label']}|{r['metric']} ({r['unit']})|{r['mae']:.6f}|{r['bias']:.6f}|[{r['loa_low']:.6f}, {r['loa_high']:.6f}]|{r['pearson_r']:.6f}|{r['n_segments_used']}/{r['n_segments_total']}|{r['segments_discarded_pct']:.6f}|{r['pred_rr_removed_pct']:.6f}/{r['ref_rr_removed_pct']:.6f}|")
 lines+=['','## Complete numeric fields', 'Values below include variance components, subject counts, artifact/support denominators and individual failure counts.']
 cols=list(summary[0]);lines+=['| '+' | '.join(cols)+' |','|'+'---|'*len(cols)]
 lines+=['| '+' | '.join(str(r[k]) for k in cols)+' |' for r in summary]
 lines+=['','## Record inventory','`record_inventory.csv` supplies native window counts, complete segments, tails, stitched short-RR counts per method/record; these remain in artifact denominators.']
 for ds in ('mmecg','crrvs'):
  rr=[r for r in records if r['dataset']==ds and r['method']=='rem_net']
  lines.append(f"- {ds}: records={len(rr)}, complete segments={sum(r['n_complete_60s_segments'] for r in rr)}, discarded incomplete-tail seconds={sum(r['tail_samples'] for r in rr)/FS:.6f}.")
 lines+=['','## Subjects with zero retained segments','All candidates remain in the discard denominator; these subjects are not silently assigned zero error.','|Dataset|Method|Subject|Candidate segments|Pred failed|Ref failed|','|---|---|---|---:|---:|---:|']
 for r in retention:
  if r['n_used']==0:lines.append(f"|{r['dataset']}|{NAMES[r['method']]}|{r['subject']}|{r['n_segments']}|{r['n_pred_correction_failed']}|{r['n_ref_correction_failed']}|")
 lines+=['','## Consistency audit','All16 original-window REM anchor sets reproduce the frozen clinical central-window HR (max difference<0.001bpm, floating-point implementation tolerance), with identical finite/unavailable status. Source: refined_anchor_consistency.json. Added contexts use exactly matching native RCG and bitwise float16 STFT on audited original windows; see dense_preprocessing_audit.csv.']
 text='\n'.join(lines)+'\n';(OUT/'REPORT.md').write_text(text)
 target=ROOT/'docs/paper/RESULTS_MANIFEST.md'
 marker='<!-- HRV_20261103 -->'
 original=target.read_text() if target.exists() else '# Results manifest\n\n'
 if marker in original:original=original.split(marker)[0]
 target.write_text(original+marker+'\n\n'+text)


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
 import torch
 torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
 shared=load_shared();items=shared.inventory();provenance={};sources={}
 main_manifest=json.loads((ROOT/'docs/paper/sections/figures/source_data/phase_pd_bland_altman_20261029/manifest.json').read_text())
 baseline_manifest=json.loads((BASE/'manifest.json').read_text());assert baseline_manifest['complete']
 baseline_cp=json.loads((BASE/'checkpoints.json').read_text())
 main_table_manifest=json.loads((ROOT/'docs/paper/sections/results_verified_20261101/manifest.json').read_text())
 print('Source manifest keys',list(main_manifest),flush=True)
 for ds,_,nf in shared.DATASETS:
  for fold in range(1,nf+1):
   fi=[r for r in items if r['dataset']==ds and r['fold']==fold]
   phase_prediction=FORMAL/f'runs/{ds}/fold_{fold:02d}/A4/test_predictions.npz'
   assert sha(phase_prediction)==main_table_manifest['source_sha256'][str(phase_prediction)], 'Main table REM outputs changed'
   rs=rem_anchors(shared,ds,fold,fi,args.device,main_manifest,provenance)
   for item,r in zip(fi,rs):sources[(ds,fold,'rem_net',item['index'])]=r
   print(f'Cached refined REM anchors {ds}/{fold}',flush=True)
 anchor_consistency(items,sources)
 dense=dense_cr_anchors(shared,items,baseline_cp,provenance,args.device)
 rows=[];records=[]
 for ds,dsname,nf in shared.DATASETS:
  for fold in range(1,nf+1):
   fi=[r for r in items if r['dataset']==ds and r['fold']==fold]
   for rid in sorted(set(r['record'] for r in fi)):
    group=[r for r in fi if r['record']==rid];starts=[r['start'] for r in group];order=np.argsort(starts);starts=np.array(starts)[order]
    reference=RAW/dsname/(rid+'.npz')
    assert sha(reference)==main_manifest['raw_reference_sha256'][str(reference)]
    with np.load(reference) as z:ref=z['event_r'].astype(float);length=len(z['ecg'])
    for method in METHODS:
     if method=='rem_net':rs=[sources[(ds,fold,method,r['index'])] for r in group]
     else:
      key=f'{ds}:{fold}:{method}';cp=baseline_cp[key];cpath=Path(cp['path'])
      if str(cpath) not in provenance:
       fingerprint=sha(cpath);assert fingerprint==cp['sha256'];provenance[str(cpath)]=fingerprint
      path=BASE/f'records/{ds}/fold_{fold:02d}/{method}/{rid}_shape_cache.npz'
      with np.load(path,allow_pickle=True) as z:
       assert str(z['checkpoint_sha256'])==cp['sha256'];rs=list(z['anchors'])
      assert len(rs)==len(group)
     rs=[rs[int(i)] for i in order]
     final_starts=starts
     if ds=='crrvs':
      extra_starts,extra=dense[fold,rid]
      combined=sorted(zip(list(starts)+list(extra_starts),rs+extra[method]),key=lambda pair:pair[0])
      final_starts=np.array([x[0] for x in combined]);rs=[x[1] for x in combined]
     meta=dict(dataset=ds,fold=fold,subject=group[0]['subject'],record=rid,method=method)
     rr,inv=paired_rows(meta,length,final_starts,rs,ref);rows+=rr;records.append(inv)
   print(f'Segments computed {ds}/{fold}',flush=True)
 summary=summarize(rows,shared)
 save_csv(OUT/'hrv_segments.csv',rows);save_csv(OUT/'hrv_summary.csv',summary);save_csv(OUT/'record_inventory.csv',records)
 report(rows,summary,records,provenance,shared)
 print('COMPLETE',len(rows),'segment rows',len(summary),'summary rows',flush=True)

if __name__=='__main__':main()
