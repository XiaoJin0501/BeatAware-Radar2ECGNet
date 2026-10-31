"""Author-authorized frozen baseline clinical QRS/QT evaluation; never trains.

Same phase_pd clinical detector, reference, pairing, dedup and aggregation. Native
adapted baselines lack a phase-conditioned head: each predicted RR gets its OWN
8s radar context centered on that predicted interval and its own shape output.
No representative-shape copying and no GT used to synthesize. ALL valid RR pairs
in the FULL1600 are synthesized before central scoring. Same fixed-zero
uncovered fallback, refined R anchors,1600->central1024 scoring and RR bounds.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['SSQ_PARALLEL']='0';os.environ.setdefault('MPLBACKEND','Agg')
import sys,csv,json,time,warnings,hashlib,importlib.util,multiprocessing,argparse,gc
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch,h5py
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/figures'),str(ROOT),str(ROOT/'experiments/development/structured_decoder_v0_20261017_ab')]
from scripts.figures import plot_per_beat_morphology as helper
# Frozen core baseline classes and preprocessing already pinned by helper.
from src.models.baselines.dual_endpoint import DualEndpointBaseline
spec=importlib.util.spec_from_file_location('baseline_clinical_shared',ROOT/'docs/paper/sections/figures/plot_phase_pd_bland_altman.py')
shared=importlib.util.module_from_spec(spec);spec.loader.exec_module(shared)
OUT=ROOT/'docs/paper/sections/figures/source_data/baseline_clinical_20261101'
METHODS=('mmecg_tcn','radarode_mtl','cross_modal_conformer','wavegru_net','attention_dcgan')
META=ROOT/'docs/paper/source_data/2026-10-16_existing_remnet_v1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,x):
 tmp=Path(str(p)+'.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)
def save(p,rows):
 if not rows:raise ValueError('No rows '+str(p))
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def state(**kw):write_json(OUT/'progress.json',dict(time=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw))

def tf_worker(task):
 ds,x=task
 if ds=='crrvs':return helper.rawio.cw._stft_map(x[0],200,128,112).astype(np.float16)
 from ssqueezepy import ssq_cwt
 from scipy.signal import resample
 # Same independently normalized transforms; vectorize over range bins ONLY.
 tx,*_=ssq_cwt(x.astype(np.float64),wavelet='morlet',fs=200,nv=16)
 mag=np.abs(tx).astype(np.float32);offset=mag.shape[1]//2 if mag.shape[1]>=142 else 0
 mag=mag[:,offset:offset+71]
 if mag.shape[1]<71:mag=np.concatenate([mag,np.zeros((len(x),71-mag.shape[1],mag.shape[2]),np.float32)],axis=1)
 mag=resample(mag,240,axis=-1).astype(np.float32);mag=np.clip(mag,0,None)
 lo=mag.min((1,2),keepdims=True);span=mag.max((1,2),keepdims=True)-lo
 return np.divide(mag-lo,span,out=np.zeros_like(mag),where=span>=1e-8).astype(np.float16)

def reference_worker(task):
 ds,fold,sid,rid,start,x=task
 return shared.delineate(x)

def clinical_worker(task):
 ds,fold,sid,rid,start,reference,y,r=task
 pred=shared.delineate(y,r)
 return shared.make_rows(ds,fold,sid,rid,start,reference,pred,('qrs','qt'))

class NativeTF:
 def __init__(self,ds,fold,items):
  self.ds=ds;self.files={};self.maps={}
  if ds=='crrvs':
   self.files[0]=h5py.File(Path('/home/qhh2237/Datasets/CR_RVS/processed/q1_continuous_kfold')/f'fold_{fold:02d}/test_wsst8s.h5')
  else:
   for sid in sorted(set(x['subject'] for x in items)):
    h=h5py.File(Path('/home/qhh2237/Datasets/MMECG/processed/wsst8s_subject_pool')/f'subject_{sid:02d}.h5');self.files[sid]=h
    decode=lambda v:v.decode() if isinstance(v,bytes) else str(v)
    self.maps[sid]={(decode(r),int(s)):i for i,(r,s) in enumerate(zip(h['record_id'][:],h['start_idx'][:]))}
 def get(self,item):
  if self.ds=='crrvs':return np.asarray(self.files[0]['wsst'][item['index']],np.float32)
  return np.asarray(self.files[item['subject']]['wsst'][self.maps[item['subject']][item['record'],item['start']]],np.float32)
 def close(self):
  for f in self.files.values():f.close()

@torch.no_grad()
def coarse_anchors(ds,model,items,radar,native,chunk=8):
 anchors=[]
 for begin in range(0,len(items),chunk):
  batch=items[begin:begin+chunk]
  if model.name=='radarode_mtl':
   x=torch.from_numpy(np.stack([native.get(item) for item in batch])).to('cuda')
   if ds=='crrvs':x=F.interpolate(x,size=(71,240),mode='bilinear',align_corners=False)
   out=model(wsst=x)
  else:
   x=np.stack([helper.rawio.radar_window('MMECG' if ds=='mmecg' else 'CR_RVS',radar,item['start']) for item in batch])
   out=model(rcg=torch.from_numpy(x).to('cuda'))
  logits=out['anchor_logits']
  for row,prob in zip(logits,logits.sigmoid().cpu().numpy()):
   decoded=shared.formal.frozen.structured_decode(prob)
   anchors.append(np.array([float(shared.soft_refine(row[1],r)) for r in decoded['r']],float))
 return anchors

@torch.no_grad()
def shapes_for(ds,model,radar,starts,pool,chunk=32):
 result={};name='MMECG' if ds=='mmecg' else 'CR_RVS';count=len(starts)
 for begin in range(0,count,chunk):
  ss=starts[begin:begin+chunk];rcg=np.stack([helper.rawio.radar_window(name,radar,s) for s in ss])
  features=np.stack(list(pool.map(tf_worker,[(ds,x) for x in rcg],chunksize=1))) if model.name=='radarode_mtl' else None
  for offset in range(0,len(ss),8):
   subset=ss[offset:offset+8]
   if features is not None:
    x=torch.from_numpy(features[offset:offset+8].astype(np.float32)).to('cuda')
    if ds=='crrvs':x=F.interpolate(x,size=(71,240),mode='bilinear',align_corners=False)
    latent=model.represent(wsst=x,rcg=None)
   else:latent=model.represent(wsst=None,rcg=torch.from_numpy(rcg[offset:offset+8]).to('cuda'))
   shapes=model.readouts.shape(F.interpolate(latent,size=200,mode='linear',align_corners=False)).sigmoid()[:,0].cpu().numpy()
   for start,shape in zip(subset,shapes):result[start]=shape
  if begin%256==0:
   print('Contexts',ds,model.name,begin+len(ss),'/',count,flush=True)
   state(stage='CONTEXT_INFERENCE',dataset=ds,model=model.name,contexts_completed=begin+len(ss),contexts_total=count)
 return result

def assemble_native_pairs(r,start,lookup):
 """Predictions ONLY. Synthesize full8s, never crop before clinical delineation."""
 y=np.zeros(1600,float);covered=np.zeros(1600,bool);invalid=0
 for a,b in zip(r[:-1],r[1:]):
  if not 60<=b-a<=400:invalid+=1;continue
  origin=int(round(start+(a+b)/2-800))
  if origin not in lookup:continue
  indices=np.arange(max(0,int(np.ceil(a))),min(1600,int(np.ceil(b))))
  y[indices]=np.interp(indices,np.linspace(a,b-1,200),lookup[origin]);covered[indices]=True
 return y,covered,invalid


def process_record(ds,fold,model,items,native,pool):
 rid=items[0]['record'];directory=OUT/'records'/ds/f'fold_{fold:02d}'/model.name;directory.mkdir(parents=True,exist_ok=True)
 path=directory/(rid+'_pairs.csv');meta_path=directory/(rid+'_meta.json')
 if path.exists() and meta_path.exists() and json.loads(meta_path.read_text()).get('full_window_synthesis',False):return shared.numeric(list(csv.DictReader(path.open()))),json.loads(meta_path.read_text())
 name='MMECG' if ds=='mmecg' else 'CR_RVS';raw=helper.rawio.record_data(name,rid,with_radar=True);radar=raw['radar'];ecg=raw['ecg'];length=len(ecg)
 # Native RCG preprocessing consistency, without ECG-based channel selection.
 base=Path('/home/qhh2237/Datasets')/('MMECG/processed/loso_v2' if ds=='mmecg' else 'CR_RVS/processed/q1_continuous_kfold')/f'fold_{fold:02d}/test.h5'
 with h5py.File(base) as h:
  item=items[len(items)//2];stored=np.asarray(h['rcg'][item['index']],np.float32)
  generated=helper.rawio.radar_window(name,radar,item['start'])
  if not np.allclose(stored,generated,atol=2e-5,rtol=2e-5):raise RuntimeError(f'Native RCG mismatch {ds}/{rid}: {abs(stored-generated).max()}')
 if model.name=='radarode_mtl':
  generated_tf=tf_worker((ds,generated));native_tf=native.get(item).astype(np.float16)
  if not np.array_equal(generated_tf,native_tf):raise RuntimeError(f'TF preprocessing mismatch {ds}/{rid}; do not alter evaluation inputs')
 with torch.no_grad():
  if model.name=='radarode_mtl':
   t=torch.from_numpy(native.get(item)[None]).to('cuda')
   if ds=='crrvs':t=F.interpolate(t,size=(71,240),mode='bilinear',align_corners=False)
   expected_shape=model(wsst=t)['shape'];h=model.represent(t,None)
  else:
   t=torch.from_numpy(generated[None]).to('cuda');expected_shape=model(rcg=t)['shape'];h=model.represent(None,t)
  native_shape=model.readouts.shape(F.interpolate(h,size=200,mode='linear',align_corners=False)).sigmoid()
  assert torch.allclose(expected_shape,native_shape,atol=1e-7,rtol=1e-6),'Shape-only branch differs from native forward'
 refcache=OUT/'references'/ds/(rid+'.npz');refcache.parent.mkdir(parents=True,exist_ok=True)
 if refcache.exists():
  with np.load(refcache,allow_pickle=True) as z:references=list(z['points']);cache_starts=z['starts']
  assert np.array_equal(cache_starts,[r['start'] for r in items])
 else:
  tasks=[(ds,fold,x['subject'],rid,x['start'],ecg[x['start']:x['start']+1600]) for x in items]
  references=list(pool.map(reference_worker,tasks,chunksize=4))
  np.savez_compressed(refcache,starts=np.array([x['start'] for x in items]),points=np.array(references,object))
 # Match reference event set/clinical durations to existing phase_pd reference.
 existing=shared.OUT/f'{ds}_fold_{fold:02d}_pairs.csv'
 expected={}
 for row in csv.DictReader(existing.open()):
  if row['record']==rid and row['metric'] in ('qrs','qt'):expected[int(row['start']),int(row['ref_r']),row['metric']]=float(row['ref'])
 generated={}
 for item,ref in zip(items,references):
  for row in shared.make_rows(ds,fold,item['subject'],rid,item['start'],ref,ref,('qrs','qt')):generated[row['start'],row['ref_r'],row['metric']]=row['ref']
 assert generated==expected,'Raw clinical scoring reference differs from existing REM-Net'
 shape_cache=directory/(rid+'_shape_cache.npz')
 if shape_cache.exists():
  with np.load(shape_cache,allow_pickle=True) as z:
   rs=list(z['anchors']);old_starts=list(map(int,z['context_starts']));values=z['shapes']
   if 'checkpoint_sha256' in z:assert str(z['checkpoint_sha256'])==model._clinical_checkpoint_sha
   else:
    original=json.loads((OUT/'archive_initial_central_support/checkpoints.json').read_text())
    assert original[ds+':'+str(fold)+':'+model.name]['sha256']==model._clinical_checkpoint_sha
  lookup=dict(zip(old_starts,values))
 else:rs=coarse_anchors(ds,model,items,radar,native);lookup={}
 starts=set();missing=0
 for item,r in zip(items,rs):
  for a,b in zip(r[:-1],r[1:]):
   if not 60<=b-a<=400:continue
   origin=int(round(item['start']+(a+b)/2-800))
   if origin<0 or origin+1600>length:missing+=1;continue
   starts.add(origin)
 starts=sorted(starts);needed=[s for s in starts if s not in lookup]
 print('Full-window predicted-beat contexts',ds,fold,model.name,rid,len(starts),'new',len(needed),flush=True)
 lookup.update(shapes_for(ds,model,radar,needed,pool))
 np.savez_compressed(shape_cache,anchors=np.array(rs,object),context_starts=np.array(starts),shapes=np.stack([lookup[s] for s in starts]) if starts else np.empty((0,200),np.float32),missing_contexts=missing,checkpoint_sha256=model._clinical_checkpoint_sha)
 tasks=[];fallback=[];invalid_total=0
 for item,ref,r in zip(items,references,rs):
  y,covered,invalid=assemble_native_pairs(r,item['start'],lookup)
  invalid_total+=invalid;fallback.append(float(1-covered[288:1312].mean()))
  tasks.append((ds,fold,item['subject'],rid,item['start'],ref,y,r))
 rows=[]
 for a in pool.map(clinical_worker,tasks,chunksize=4):rows.extend(a)
 rows=shared.dedup(rows)
 # Every eligible reference pair is kept, including predictions with NaN endpoints.
 assert {(r['start'],r['ref_r'],r['metric']) for r in rows}<=set(expected)
 meta=dict(dataset=ds,fold=fold,model=model.name,record=rid,subject=items[0]['subject'],n_windows=len(items),n_independent_contexts=len(starts),missing_contexts=missing,fallback_window_mean=float(np.mean(fallback)),fallback_fractions=fallback,invalid_pairs=invalid_total,reference_matches_remnet=True,full_window_synthesis=True)
 save(path,rows);write_json(meta_path,meta);return rows,meta

def collect():
 rows=[];metas=[];sources=[]
 for path in sorted((OUT/'records').glob('*/*/*/*_pairs.csv')):
  model=path.parent.name
  rows.extend(dict(**r,model=model) for r in shared.numeric(list(csv.DictReader(path.open()))));sources.append(path)
  metas.append(json.loads(path.with_name(path.name.replace('_pairs.csv','_meta.json')).read_text()))
 subjects=[];summary=[]
 for ds,n in (('mmecg',11),('crrvs',30)):
  for model in METHODS:
   ids=sorted(set(m['subject'] for m in metas if m['dataset']==ds and m['model']==model));assert len(ids)==n
   for sid in ids:
    for metric in ('qrs','qt'):
     a=[r for r in rows if r['dataset']==ds and r['model']==model and int(r['subject'])==sid and r['metric']==metric]
     valid=[r for r in a if np.isfinite(r['pred'])]
     subjects.append(dict(dataset=ds,model=model,subject=sid,metric=metric,eligible=len(a),matched=len(valid),coverage=len(valid)/len(a) if a else 0.,mae_ms=float(np.mean([abs(r['pred']-r['ref']) for r in valid])) if valid else None))
   for metric in ('qrs','qt'):
    a=[r for r in subjects if r['dataset']==ds and r['model']==model and r['metric']==metric];v=[r['mae_ms'] for r in a if r['mae_ms'] is not None]
    summary.append(dict(dataset=ds,model=model,metric=metric,mean_mae_ms=float(np.mean(v)) if v else None,std_population_ms=float(np.std(v)) if v else None,n_assessable_subjects=len(v),n_total_subjects=n,coverage_subject_mean=float(np.mean([r['coverage'] for r in a])),eligible=sum(r['eligible'] for r in a),matched=sum(r['matched'] for r in a)))
 save(OUT/'paired_values.csv',rows);save(OUT/'subject_metrics.csv',subjects);save(OUT/'summary.csv',summary);save(OUT/'fallback_summary.csv',[{k:v for k,v in m.items() if k!='fallback_fractions'} for m in metas])
 write_json(OUT/'manifest.json',dict(training=False,checkpoint_selection=False,clinical_pipeline_source=str(Path(shared.__file__)),clinical_pipeline_sha256=sha(shared.__file__),roundtrip_gate=json.loads((shared.OUT/'gate.json').read_text()),script_sha256=sha(__file__),protocol_spec_sha256=sha(OUT/'spec.json'),checkpoints=json.loads((OUT/'checkpoints.json').read_text()),record_pair_sources_sha256={str(p):sha(p) for p in sources},complete=True))
 lines=['# Five adapted baseline clinical intervals, frozen inference only','',__doc__,'Same raw reference, NeuroKit2 prominence, predicted R anchors with native soft-argmax refinement,RR60–400samples,fixed-zero fallback,central1024,endpoint±30samples,R-pair±30samples,centre-distance dedup as phase_pd BA. Native baselines use their own full8s beat-centred shape head contexts; REM instead has its learned phase-grid decoder. Neither path uses GT to generate waveforms. This unavoidable native decoder distinction is disclosed,not presented as identical architecture. Same scoring reference asserted exact against existing REM-Net per-window clinical rows. No copied representative beat, new filter, threshold search,offset subtraction or continuous-model weights.','All means/SD in summary are subject-macro(population SD),not pooled BA MAE; unmatched endpoints remain eligible,not assigned zero error. Entire process uses frozen seed42 fold-specific weights. CPU vectorized WSST is equivalent to original independent50bin transforms;preflight bitwise float16 audit saved.','|Dataset|Method|Metric|MAE±SD(ms)|Assessable subjects|Coverage|','|---|---|---|---|---|---|']
 for r in summary:lines.append(f"|{r['dataset']}|{r['model']}|{r['metric']}|{r['mean_mae_ms']} ± {r['std_population_ms']}|{r['n_assessable_subjects']}/{r['n_total_subjects']}|{r['coverage_subject_mean']:.3%}|")
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');state(stage='COMPLETE',summary=str(OUT/'summary.csv'))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');ap.add_argument('--workers',type=int,default=12);args=ap.parse_args()
 OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
 spec=dict(task='author approved filling clinical baseline cells',training=False,datasets={'mmecg':11,'crrvs':5},models=list(METHODS),seed_base=42,window=1600,synthesis_support='all valid adjacent predicted RR pairs in full1600',central=[288,1312],rr_samples=[60,400],matching_tolerance_samples=30,soft_argmax_radius_samples=10,fallback_value=0.,shape_assignment='own exact predicted-RR-centred8s input, no replication, noGT',detector='frozen NeuroKit2 prominence via phase_pd BA',aggregation='per-subject matched MAE then subject mean/populationSD',workers=args.workers)
 specpath=OUT/'spec.json'
 if specpath.exists():assert json.loads(specpath.read_text())==spec
 else:write_json(specpath,spec)
 assert json.loads((shared.OUT/'manifest.json').read_text())['gate_passed']
 audit_path=OUT/'engineering_check.json'
 if not audit_path.exists():
  checks=[]
  for rid in ('MMECG_0001','MMECG_0052'):
   raw=helper.rawio.record_data('MMECG',rid,with_radar=True);x=helper.rawio.radar_window('MMECG',raw['radar'],1600)
   a=helper.transform_task(('MMECG',x));b=tf_worker(('mmecg',x));assert np.array_equal(a,b),'Vectorization changes input'
   checks.append(dict(record=rid,context_start=1600,original_per_bin_vs_vectorized_bitwise_equal=True,shape=list(b.shape)))
  write_json(audit_path,dict(wsst_checks=checks,smoke_record='MMECG_0001/Chen fold1',training=False))
 items=shared.inventory();checkpoints={};ctx=multiprocessing.get_context('spawn')
 with ProcessPoolExecutor(max_workers=args.workers,mp_context=ctx) as pool:
  for ds,nfold in (('mmecg',11),('crrvs',5)):
   for fold in range(1,nfold+1):
    folditems=[r for r in items if r['dataset']==ds and r['fold']==fold]
    native=NativeTF(ds,fold,folditems)
    try:
     for name in METHODS:
      cp=helper.frozen_inference.checkpoint_path('MMECG' if ds=='mmecg' else 'CR_RVS',fold,name)
      saved=torch.load(cp,map_location='cpu',weights_only=False);model=DualEndpointBaseline(name,in_bins=50 if ds=='mmecg' else 1).to('cuda').eval();model.load_state_dict(saved['model'],strict=True);model._clinical_checkpoint_sha=sha(cp)
      checkpoints[ds+':'+str(fold)+':'+name]=dict(path=str(cp),sha256=sha(cp),epoch=saved.get('epoch'));write_json(OUT/'checkpoints.json',checkpoints)
      records=sorted(set(r['record'] for r in folditems))
      for rid in records:
       group=[r for r in folditems if r['record']==rid];state(stage='INFERENCE',dataset=ds,fold=fold,model=name,record=rid)
       rows,meta=process_record(ds,fold,model,group,native,pool)
       print('Record complete',ds,fold,name,rid,'contexts',meta['n_independent_contexts'],flush=True)
       if args.check:state(stage='CHECK_COMPLETE',dataset=ds,fold=fold,model=name,record=rid);return
      del model,saved;gc.collect();torch.cuda.empty_cache()
    finally:native.close()
 collect()
if __name__=='__main__':main()
