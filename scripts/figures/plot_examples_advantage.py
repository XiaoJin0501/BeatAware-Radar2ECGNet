"""Author-requested advantage illustrations, NOT representative test examples.

Original dual-head seed42 frozen inference. No training/model/benchmark selection.
Same test window for radar, GT and all methods. Ranking before centred inference:
representative REM PCC >=.85 and >both baselines; descending minimum paired gap,
then descending REM PCC, fold/index. Apnea keeps previous lower-quartile reference
HF display filter. Verify actual independent-per-beat PCC exceeds BOTH baselines;
try next ranked candidate if not, keeping all attempted outputs and audit rows.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('SSQ_PARALLEL','0')
import sys,csv,json,warnings,shutil
from pathlib import Path
import numpy as np
from scipy.signal import welch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.figures import plot_examples_v4 as v4
from scripts.figures import plot_examples_v4_standalone as drawing
from experiments.phase_resampling_20261027.frozen_bridge import shape_target,pcc_per_row
FIG=v4.FIG;OUT=FIG/'source_data/examples_advantage_20261031';STEM='fig_examples_advantage'

def rank_candidates():
 rows=v4.previous.csvread(v4.previous.CACHE/'selection_candidates.csv');sources=[]
 for ds,folds in (('mmecg',11),('crrvs',5)):
  for fold in range(1,folds+1):
   rp=v4.previous.old.SOURCE/f'references/{ds}/fold_{fold:02d}.npz';ref=v4.previous.old.load_npz(rp);sources.append(rp)
   targets=np.stack([shape_target(e,v4.previous.old.refs_at(ref,'r',i)) for i,e in enumerate(ref['ecg'])])
   scores={};predictions={}
   for method in v4.METHODS:
    p=v4.previous.old.SOURCE/f'predictions/{ds}/{method}/seed_42/fold_{fold:02d}/test_predictions.npz';sources.append(p)
    with np.load(p) as z:predictions[method]=z['shape'].reshape(-1,200).astype(float)
    scores[method]=pcc_per_row(predictions[method],targets)
   for r in rows:
    if r['dataset']!=ds or int(r['fold'])!=fold:continue
    for k in ('fold','index','subject','start','stop'):r[k]=int(r[k])
    r['window_pcc']=float(r['window_pcc']);i=r['index']
    if abs(float(scores['rem_net'][i])-r['window_pcc'])>2e-6:
     # Only if stored reference R differs, reuse the SAME frozen delineator.
     from experiments.phase_resampling_20261027.frozen_bridge import extract_points
     t=shape_target(ref['ecg'][i],extract_points(ref['ecg'][i])['r'])
     for method in v4.METHODS:
      pp=predictions[method][i:i+1]
      scores[method][i]=pcc_per_row(pp.astype(float),t[None])[0]
     assert abs(float(scores['rem_net'][i])-r['window_pcc'])<2e-6
    for method in v4.METHODS:r[method+'_representative_pcc']=float(scores[method][i])
    r['min_representative_gap']=min(r['rem_net_representative_pcc']-r[m+'_representative_pcc'] for m in v4.METHODS[1:])
    y=ref['ecg'][i,288:1312];f,p=welch(y,200,nperseg=512)
    r['hf_fraction']=float(p[(f>=10)&(f<=40)].sum()/max(p[(f>=.5)&(f<=40)].sum(),1e-12))
   print('Candidate scoring',ds,fold,flush=True)
 apnea=[r for r in rows if r['dataset']=='crrvs' and r['scene']=='SP'];threshold=float(np.quantile([r['hf_fraction'] for r in apnea],.25))
 for r in rows:r['eligible']=r['rem_net_representative_pcc']>=.85 and r['min_representative_gap']>0 and (not(r['dataset']=='crrvs' and r['scene']=='SP') or r['hf_fraction']<=threshold)
 v4.save(OUT/'candidate_ranking.csv',rows)
 return rows,threshold,sources

def write_csv(cases):
 fields=['dataset','scene','scene_label','subject','fold','window_index','record_id','input_start_sample','central_start_sample','sample_index','time_s',*drawing.ROWS,*(k+'_pcc' for k in v4.METHODS),'reference_r_samples','reference_annotation_samples']
 with (FIG/(STEM+'_source.csv')).open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
  for meta,a in cases:
   events=v4.centre_events(a) if meta['scene']=='NB' else {}
   for i in range(1024):
    writer.writerow(dict(dataset=meta['dataset'],scene=meta['scene'],scene_label=meta['scene_label'],subject=meta['subject'],fold=meta['fold'],window_index=meta['index'],record_id=meta['record_id'],input_start_sample=meta['start'],central_start_sample=meta['start']+288,sample_index=i,time_s=repr(i/200),**{k:repr(float(a[k][i])) for k in drawing.ROWS},**{k+'_pcc':meta['panel_pcc'][k] for k in v4.METHODS},reference_r_samples=json.dumps(a['gt_r'].tolist()) if i==0 else '',reference_annotation_samples=json.dumps(events) if i==0 else ''))

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 preserved={str(FIG/f'{stem}.{ext}'):v4.sha(FIG/f'{stem}.{ext}') for stem in ('fig_examples_v4','fig_examples_v3') for ext in ('pdf','svg','png')}
 rankpath=OUT/'candidate_ranking.csv'
 if rankpath.exists():
  rows=v4.previous.csvread(rankpath)
  for r in rows:
   for k in ('fold','index','subject','start','stop'):r[k]=int(r[k])
   for k in ('window_pcc','hf_fraction','min_representative_gap',*(m+'_representative_pcc' for m in v4.METHODS)):r[k]=float(r[k])
   r['eligible']=r['eligible']=='True'
  threshold=float(np.quantile([r['hf_fraction'] for r in rows if r['dataset']=='crrvs' and r['scene']=='SP'],.25));sources=[rankpath]
 else:rows,threshold,sources=rank_candidates()
 from scripts.figures import plot_per_beat_morphology as inference
 cases=[];attempts=[];beatrows=[]
 clinical,_=v4.previous.clinical()
 for ds,scenes in v4.previous.old.SCENES.items():
  for scene,label in scenes:
   candidates=sorted([r for r in rows if r['dataset']==ds and r['scene']==scene and r['eligible']],key=lambda r:(-r['min_representative_gap'],-r['rem_net_representative_pcc'],r['fold'],r['index']))
   found=False
   for rank,row in enumerate(candidates,1):
    if rank>30:break
    rawpath=ROOT/'experiments/development/structured_decoder_v0_20261017_ab/raw_reference'/('MMECG' if ds=='mmecg' else 'CR_RVS')/(row['record_id']+'.npz')
    with np.load(rawpath) as z:length=len(z['ecg'])
    if row['start']<800 or row['stop']+800>length:
     attempts.append(dict(dataset=ds,scene=scene,rank=rank,subject=row['subject'],fold=row['fold'],index=row['index'],status='insufficient full beat-centred context',scores=''));continue
    inference.ASSETS=OUT/'attempts'/f'{ds}_{scene}_{rank:02d}';inference.ASSETS.mkdir(parents=True,exist_ok=True)
    native={k:row[k] for k in ('dataset','fold','index','subject','scene','record_id','start','stop','window_pcc')};native['scene_label']=label
    print('Infer',ds,scene,'rank',rank,'record',row['record_id'],'row',row['index'],flush=True)
    meta,a=inference.per_case(native);scores=meta['window_pcc_by_method']
    passed=all(scores['rem_net']>scores[m] for m in v4.METHODS[1:])
    attempts.append(dict(dataset=ds,scene=scene,rank=rank,subject=row['subject'],fold=row['fold'],index=row['index'],status='accepted' if passed else 'actual per-beat advantage not confirmed',scores=json.dumps(scores)))
    v4.save(OUT/'attempt_log.csv',attempts)
    if not passed:continue
    refpath=v4.previous.old.SOURCE/f'references/{ds}/fold_{row["fold"]:02d}.npz';ref=v4.previous.old.load_npz(refpath);ecg=ref['ecg'][row['index']]
    with warnings.catch_warnings():warnings.simplefilter('ignore');points=clinical.delineate_signal(ecg,fs=200,method='prominence')
    assert np.array_equal(points['r']-288,a['gt_r'])
    for e in 'qrst':a['ref_'+e]=np.asarray(points[e],float)-288
    assert np.allclose(v4.previous.old.reference_display(ecg,points['r'])[288:1312],a['gt'],rtol=0,atol=1e-10)
    with np.load(rawpath) as z:raw=z['ecg'][row['start']:row['stop']]
    assert v4.previous.old.pcc(raw,ecg)>1-1e-10
    for method in v4.METHODS:
     values=v4.previous.score_shapes(a[method+'_shapes'],a['reference_shapes']);assert abs(float(values.mean())-scores[method])<2e-6
     for bi,((left,right),value) in enumerate(zip(a['beat_bounds'],values)):
      beatrows.append(dict(dataset=ds,scene=scene,method=method,subject=row['subject'],record_id=row['record_id'],window_index=row['index'],beat_index=bi,reference_left=row['start']+288+int(left),reference_right=row['start']+288+int(right),context_start=meta['reference_context_starts'][bi],pcc=float(value)))
    meta.update(panel_pcc=scores,ranking_position=rank,hf_fraction=row['hf_fraction'],cached_reference_ecg_path=str(refpath),filtered_raw_record_path=str(rawpath));cases.append((meta,a))
    np.savez_compressed(OUT/f'{ds}_{scene}_plotted.npz',**a);found=True;break
   if not found:raise RuntimeError(f'No verified shared-window advantage within30 candidates: {ds}/{scene}. No substituted method rows allowed.')
 write_csv(cases);drawing.plot(drawing.load_csv(FIG/(STEM+'_source.csv')),FIG/STEM)
 # Portable renderer: exact same layout, only default input/output stem renamed.
 portable=(ROOT/'scripts/figures/plot_examples_v4_standalone.py').read_text().replace('fig_examples_v4',STEM).replace('Standalone Fig. examples v4 (Apnea revision)','Standalone advantage examples (selection disclosed in REPORT)')
 (FIG/('plot_'+STEM+'.py')).write_text(portable)
 v4.save(OUT/'per_window_results.csv',[dict(dataset=m['dataset'],scene=m['scene'],subject=m['subject'],fold=m['fold'],window_index=m['index'],record_id=m['record_id'],start=m['start'],central_start=m['start']+288,central_stop=m['start']+1312,selection_representative_pcc=m['window_pcc'],**{k+'_pcc':m['panel_pcc'][k] for k in v4.METHODS}) for m,a in cases])
 v4.save(OUT/'per_beat_results.csv',beatrows)
 report=['# REM-Net advantage examples (deliberately selected, not representative)','',__doc__,
  f'Apnea HF lower-quartile threshold={threshold:.8f}. Only frozen original dual-head seed42 outputs, NOT phase_pd. Test results are used for illustration selection only; no training/model/checkpoint selection or quantitative table changes. All seven examples are newly selected; no REM/GT/baseline row is spliced from another window. No smoothing/filter changes. Annotation absent from figure per author request; this disclosure must appear in manuscript caption or prose.',
  'Layout identical to v4:7.16x6.3in,4 MMECG+3 CR_RVS columns,5 rows; reference-only labels in first GT columns; no REM markers. Same morphology scoring and reference-RR placement; timing evaluated separately.',
  'Candidate ranking uses existing representative-window outputs. Accept the first ranked candidate with actual independently decoded per-beat REM PCC greater than BOTH baseline PCCs; every attempted case retained. This is a post-hoc advantage-case figure and must not be called median/typical/unbiased examples or used to infer population superiority. Original v4 remains unchanged.',
  'Source: ../../fig_examples_advantage_source.csv (all waveforms, exact cached PCCs, references, metadata), standalone ../../plot_fig_examples_advantage.py. CSV is the only input needed for redraw. Per-window/per-beat tables, ranked candidates, attempted cases, checkpoint hashes and manifest accompany this report.',
  '|Dataset|Scene|Subject|Record|Window|REM PCC|Chen PCC|radarODE PCC|','|---|---|---|---|---:|---:|---:|---:|']
 for m,a in cases:report.append(f"|{m['dataset']}|{m['scene_label']}|{m['subject']}|{m['record_id']}|{m['index']}|{m['panel_pcc']['rem_net']:.6f}|{m['panel_pcc']['mmecg_tcn']:.6f}|{m['panel_pcc']['radarode_mtl']:.6f}|")
 (OUT/'REPORT.md').write_text('\n'.join(report)+'\n')
 (OUT/'caption.tex').write_text(r'\caption{Selected cases in which REM-Net has higher mean per-beat morphology PCC than both adapted baselines on the same test window. Beats are independently decoded and placed on reference R--R intervals; timing is evaluated separately. Case selection is detailed in the accompanying report. Gray dashed lines indicate reference R peaks.}'+'\n')
 assert all(v4.sha(p)==h for p,h in preserved.items())
 (OUT/'manifest.json').write_text(json.dumps(dict(training=False,model_inference=True,selection='post-hoc verified shared-window advantage, not representative',cases=[m for m,a in cases],old_figures_sha256=preserved,script_sha256=v4.sha(__file__),source_sha256={str(p):v4.sha(p) for p in sources},outputs={str(p):v4.sha(p) for p in [FIG/(STEM+'_source.csv'),*(FIG/f'{STEM}.{e}' for e in ('pdf','svg','png'))]}),indent=2))
 print('Complete:',OUT,flush=True)

if __name__=='__main__':main()
