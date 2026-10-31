"""Fill real seed42 test results; no training/inference/metric modification.

Use current phase_pd A4; five adapted dual-head baseline outputs. Common formal
Shape-200 targets, within-subject Fisher PCC, subject-macro mean/population SD.
QRS/QT only where actual endpoint-based continuous clinical evaluation exists.
"""
import sys,csv,json,hashlib,re,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.phase_resampling_20261027 import frozen_bridge as frozen
FORMAL=ROOT/'experiments/phase_resampling_20261027/formal_ablation_20261027'
SNAP=ROOT/'docs/paper/source_data/2026-10-16_existing_remnet_v1'
OUT=ROOT/'docs/paper/sections/results_verified_20261101'
TARGET=ROOT/'docs/paper/sections/Table_baseline_compare.tex'
MODELS=('mmecg_tcn','radarode_mtl','cross_modal_conformer','wavegru_net','attention_dcgan','rem_net')
NAMES=dict(zip(MODELS,('Chen-TCN','radarODE-MTL','Conformer','WaveGRU-Net','Attention-DCGAN','Ours')))
FIELDS=('pcc','rmse','q','r','s','t','rr','qrs','qt')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,rows):
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 original=OUT/'Table_baseline_compare_before.tex'
 if not original.exists():shutil.copy2(TARGET,original)
 subject={};sources=[]
 baseline_events=SNAP/'original_reports/q1_formal_anchor_intervals/subject_seed_metrics.csv';sources.append(baseline_events)
 eventrows=list(csv.DictReader(baseline_events.open()))
 baseline_clinical=ROOT/'docs/paper/sections/figures/source_data/baseline_clinical_20261101'
 clinical_complete=(baseline_clinical/'manifest.json').exists() and json.loads((baseline_clinical/'manifest.json').read_text()).get('complete',False)
 clinical_rows=list(csv.DictReader((baseline_clinical/'subject_metrics.csv').open())) if clinical_complete else []
 if clinical_complete:sources.extend((baseline_clinical/'subject_metrics.csv',baseline_clinical/'manifest.json'))
 oldformal=list(csv.DictReader((FORMAL/'per_subject.csv').open()));sources.append(FORMAL/'per_subject.csv')
 for ds,nfold in (('mmecg',11),('crrvs',5)):
  for fold in range(1,nfold+1):
   lab=FORMAL/f'label_cache/{ds}/fold_{fold:02d}/test.npz';phase=FORMAL/f'runs/{ds}/fold_{fold:02d}/A4/test_predictions.npz';sources.extend((lab,phase))
   with np.load(lab,allow_pickle=True) as z:targets=z['shapes'];points=list(z['points'])
   with np.load(phase) as z:sids=z['subject_id'];remshape=z['shape'];prob=z['anchor_probability'];rids=z['record_id'];starts=z['start_idx']
   # One same fold/index/subject mapping for ALL methods.
   refpath=SNAP/f'references/{ds}/fold_{fold:02d}.npz';sources.append(refpath)
   with np.load(refpath) as z:
    assert np.array_equal(z['subject_id'],sids)
    assert np.array_equal(z['record_id'].astype(str),rids.astype(str))
    assert np.array_equal(z['start_idx'],starts)
   for model in MODELS:
    if model=='rem_net':shape=remshape.reshape(-1,200)
    else:
     path=SNAP/f'predictions/{ds}/{model}/seed_42/fold_{fold:02d}/test_predictions.npz';sources.append(path)
     with np.load(path) as z:shape=z['shape'].reshape(-1,200)
    assert shape.shape==targets.shape
    pc=frozen.pcc_per_row(shape,targets);rm=np.sqrt(np.mean((shape-targets)**2,axis=1))
    for sid in np.unique(sids):
     mask=sids==sid;sid=int(sid);key=(ds,model,sid)
     assert key not in subject,'Test subject duplicated across folds'
     row=dict(dataset=ds,model=model,fold=fold,subject=sid,n_windows=int(mask.sum()),pcc=float(np.tanh(np.arctanh(np.clip(pc[mask],-1+1e-7,1-1e-7)).mean())),rmse=float(rm[mask].mean()),pcc_window_arithmetic=float(pc[mask].mean()))
     if model=='rem_net':
      bucket=frozen.empty_bucket()
      for i in np.flatnonzero(mask):frozen.accumulate_window(bucket,points[i],frozen.structured_decode(prob[i]))
      results=frozen.finalise(bucket)
      check=next(r for r in oldformal if r['dataset']==ds and r['arm']=='A4' and int(r['subject'])==sid)
      assert abs(row['pcc']-float(check['pcc']))<1e-7 and abs(row['rmse']-float(check['rmse']))<1e-7
      for event in ('q','r','s','t','rr'):
       v=results[event];row[event]=v['mae_ms'];row[event+'_coverage']=v['coverage'];row[event+'_eligible']=v['eligible'];row[event+'_matched']=v['evaluable']
       if event!='rr':assert abs(row[event]-float(check[event+'_mae_ms']))<1e-7
     else:
      for event in ('q','r','s','t','rr'):
       v=next(r for r in eventrows if r['dataset']==ds and r['model']==model and int(r['seed'])==42 and int(r['subject'])==sid and r['metric']==event)
       assert int(v['fold'])==fold
       row[event]=float(v['mae_ms']) if v['mae_ms'] else None;row[event+'_coverage']=float(v['coverage']);row[event+'_eligible']=int(v['eligible']);row[event+'_matched']=int(v['evaluable'])
     for event in ('qrs','qt'):row[event]=row[event+'_coverage']=row[event+'_eligible']=row[event+'_matched']=None
     subject[key]=row
   print('Read test predictions',ds,fold,flush=True)
 # Add newly completed native-baseline clinical inference when available; no zero-error imputation.
 for key,row in subject.items():
  ds,model,sid=key
  if model=='rem_net' or not clinical_complete:continue
  for metric in ('qrs','qt'):
   c=next(x for x in clinical_rows if x['dataset']==ds and x['model']==model and int(x['subject'])==sid and x['metric']==metric)
   row[metric]=float(c['mae_ms']) if c['mae_ms'] else None
   row[metric+'_coverage']=float(c['coverage']);row[metric+'_eligible']=int(c['eligible']);row[metric+'_matched']=int(c['matched'])
 # True clinical onset/offset intervals: existing phase_pd assembled inference.

 clinical_dir=ROOT/'docs/paper/sections/figures/source_data/phase_pd_bland_altman_20261029'
 pairs=clinical_dir/'paired_values.csv';manifest=clinical_dir/'manifest.json';sources.extend((pairs,manifest))
 assert json.loads(manifest.read_text())['gate_passed'],'Do not report failed roundtrip clinical outputs'
 raw=list(csv.DictReader(pairs.open()))
 for key,row in subject.items():
  ds,model,sid=key
  if model!='rem_net':continue
  for event in ('qrs','qt'):
   a=[x for x in raw if x['dataset']==ds and int(x['subject'])==sid and x['metric']==event]
   assert a,'Clinical subject missing'
   valid=[x for x in a if np.isfinite(float(x['pred'])) and np.isfinite(float(x['ref']))]
   assert valid,'No assessable clinical pairs; do not silently drop subject'
   row[event]=float(np.mean([abs(float(x['pred'])-float(x['ref'])) for x in valid]))
   row[event+'_coverage']=len(valid)/len(a);row[event+'_eligible']=len(a);row[event+'_matched']=len(valid)
 subjectrows=list(subject.values());save(OUT/'per_subject_verified.csv',subjectrows)
 summary=[]
 for ds,count in (('mmecg',11),('crrvs',30)):
  for model in MODELS:
   rows=[v for k,v in subject.items() if k[:2]==(ds,model)];assert len(rows)==count
   for metric in FIELDS:
    values=[x[metric] for x in rows if x[metric] is not None]
    if values and len(values)!=count:
     if metric in ('qrs','qt') and clinical_complete:
      assert all(x[metric+'_matched']==0 for x in rows if x[metric] is None),'Missing clinical MAE despite matched pairs'
     else:
      assert (ds,model,metric,len(values)) in {('mmecg','cross_modal_conformer',e,10) for e in ('q','r','s','t','rr')},'Unexpected missing subject'
      assert next(r for r in rows if r[metric] is None)['subject']==9,'Only documented zero-match subject9 allowed'
    values=np.asarray(values,float);assert np.isfinite(values).all()
    coverage=[x[metric+'_coverage'] for x in rows if metric+'_coverage' in x and x[metric+'_coverage'] is not None]
    summary.append(dict(dataset=ds,model=model,metric=metric,mean=float(values.mean()) if len(values) else None,std_population=float(values.std(ddof=0)) if len(values) else None,n_subjects=len(values),n_subjects_total=count,coverage_subject_mean=float(np.mean(coverage)) if coverage else None,source=('common frozen Shape200 targets / formal test predictions' if metric in ('pcc','rmse') else 'phase_pd continuous onset-offset pairs; roundtrip PASS' if metric in ('qrs','qt') and model=='rem_net' else ('native baseline independently decoded per-predicted-beat clinical endpoints' if clinical_complete else 'unavailable: no verified clinical assembly evaluation') if metric in ('qrs','qt') else 'frozen structured event-head matching, seed42')))
 save(OUT/'baseline_compare_verified.csv',summary)
 idx={(r['dataset'],r['model'],r['metric']):r for r in summary}
 def cell(ds,model,metric):
  row=idx[ds,model,metric]
  if row['mean'] is None:return r'---'
  mean=row['mean'];std=row['std_population']
  value=f'{mean*100:.2f}' if metric=='pcc' else f'{mean:.2f}' if metric=='rmse' else f'{mean:.1f} \\pm {std:.1f}'
  available=[idx[ds,m,metric]['mean'] for m in MODELS if idx[ds,m,metric]['mean'] is not None]
  # Rank unrounded values; single-method clinical values are NOT comparative winners.
  if len(available)<2:return '$'+value+'$'
  ranks=sorted(set(available),reverse=metric=='pcc')
  if mean==ranks[0]:return r'\best{$'+value+'$}'
  if len(ranks)>1 and mean==ranks[1]:return r'$\underline{'+value+'}$'
  return '$'+value+'$'
 text=original.read_text()
 # Preserve original table layout and citation keys; replace only row cells.
 split=text.index('% Dataset 2: CR_RVS')
 first,second=text[:split],text[split:]
 for ds,part in [('mmecg',first),('crrvs',second)]:
  for model in MODELS:
   name=NAMES[model];prefix=r'\textbf{Ours}' if model=='rem_net' else name
   pattern=re.escape(prefix)+(r'~\\cite\{[^}]+\}' if model!='rem_net' else '')+r'\s*\n&.*?\\\\'
   match=re.search(pattern,part,re.S);assert match,(ds,model)
   label=match.group().split('\n&',1)[0]
   replacement=label+'\n& '+'\n& '.join(cell(ds,model,metric) for metric in FIELDS)+'\n'+r'\\'
   part=part[:match.start()]+replacement+part[match.end():]
  if ds=='mmecg':first=part
  else:second=part
 text=first+second
 for old,new in [('Event-Based Timing Errors','Fiducial Timing MAE (ms)'),('Clinical Interval Errors','Clinical Interval MAE (ms)'),('Q-Q','Q'),('R-R','R'),('S-S','S'),('T-T','T')]:text=text.replace(r'\textbf{'+old+'}',r'\textbf{'+new+'}')
 text=text.replace('Best and second-best results are shown in bold and underlined, respectively.','Best results are shown with gray shading and boldface; second-best results are underlined.')
 start=text.index(r'\textit{Note:}');end=text.index(r'\end{minipage}',start)
 note=r'''\textit{Note:}
Results use the complete seed-42 test sets (11 MMECG and 30 CR\_RVS subjects).
PCC is averaged with Fisher-$z$ within each subject and reported as a percentage;
RMSE is dimensionless for beat-normalized Shape-200 targets.
Timing/interval entries are held-out-subject macro mean $\pm$ population standard deviation, in ms.
Q/R/S/T are event-head localization MAEs after the common ordered decoder;
RR uses consecutive one-to-one R matches within $\pm150$ ms. Q/S/T inherit those beat matches.
QRS duration is QRS offset minus onset; QT is T offset minus QRS onset,
measured on assembled phase\_pd REM-Net waveforms using the frozen delineator after a passing roundtrip check.
``---'' denotes no verified clinical onset/offset evaluation for that baseline, not zero error.
Q--S and Q--T peak intervals and legacy continuous-model results are not substituted.
All assessable matched-event errors are reported conditionally; coverage and sample counts are exported in the source CSV.
$^{\dagger}$MMECG Conformer timing and RR use 10/11 assessable subjects: subject 9 has zero matched events. Its zero coverage is retained; morphology uses all 11 subjects.
Ranks use unrounded means; clinical columns available for only REM-Net are not ranked.
'''
 if clinical_complete:
  note=note.replace(r'measured on assembled phase\_pd REM-Net waveforms using the frozen delineator after a passing roundtrip check.',
   'measured using the frozen delineator after a passing roundtrip check. Baselines independently decode each predicted beat from its own centered 8-s radar context; REM-Net uses its native phase-conditioned decoder. Thus scoring is shared, while decoder input support differs.')
  note=note.replace("``---'' denotes no verified clinical onset/offset evaluation for that baseline, not zero error.",
   r"``---'' denotes no assessable matched clinical endpoints, not zero error. Clinical coverage and assessable-subject counts are given in Table~\ref{tab:baseline_clinical_coverage}.")
  note=note.replace('Ranks use unrounded means; clinical columns available for only REM-Net are not ranked.',
   'Ranks use unrounded conditional means; coverage must be considered when interpreting clinical errors.')
 text=text[:start]+note+text[end:]
 text=text.replace('Conformer~\\cite{cross-modalautoencoder}',r'Conformer~\cite{cross-modalautoencoder}$^{\dagger}$',1)
 TARGET.write_text(text);(OUT/'Table_baseline_compare_verified.tex').write_text(text)
 report=['# Verified full-test baseline comparison, 2026-11-01','',
 'This summarizer performs no training/model inference/delineation and uses no selected advantage-case data. Baseline clinical inputs, when available, come from separately authorized frozen inference. Screenshot numbers ignored. REM-Net=full phase_pd/A4 seed-base42; baseline=original five adapted dual-head seed42 checkpoints. All formal test subjects retained for morphology/coverage. Conditional Conformer MMECG timing/RR has10/11 assessable subjects; subject9 has zero matches and is explicitly disclosed, never assigned zero error.',
 '## Common aggregation (important change from older paper table)',
 'ALL six methods scored using exactly the same stored formal Shape-200 targets. Within-subject PCC uses frozen Fisher-z averaging, matching A4 formal ablation; then arithmetic mean over subjects. per_subject_verified.csv also preserves the alternative within-subject window-arithmetic PCC for audit; it is NOT the displayed table PCC. PCC and RMSE shown without SD,2decimals; event/interval MAE±populationSD over subjects1decimal. No fold-macro versus subject-macro mixing on CR_RVS. Thus baseline PCC differs from old window-arithmetic/fold-macro summary (e.g. Chen MM84.57% vs current86.54%); this is documented common aggregation, NOT improvement of weights or prediction changes. Current A4 PCC91.397%/90.445% matches formal ablation exactly. Supplement per_subject_verified.csv includes both actual values and all event sample counts/coverage.',
 'Events: original raw subject_seed_metrics.csv seed42 rows for five baselines; same frozen accumulator used for A4 with existing formal cached reference points and ordered-decoded saved probabilities. A4 Q/R/S/T rechecked against formal per_subject.csv to1e-7. RR is the direct event-head interval; no waveform/continuous-model RR substitution. All conditional errors are shown, including low-coverage Conformer, rather than NA gating. Coverage exported; no previously selected advantage-case subsets used. MMECG Conformer has zero matches for subject9: conditional MAE summaries use10/11 subjects, disclosed by dagger in table; morphology/coverage retain11.',
 'Clinical: existing phase_pd assembled paired_values.csv, NeuroKit prominence, QRS onset/offset and T offset, passing roundtrip gate. Reaggregate per subject FIRST, then mean±populationSD. Consequently these MAEs differ from pooled-beat Bland-Altman report MAE7.54/18.98 etc; no pooled numbers are mixed with subject-macro table cells. Five baselines have no equivalent verified clinical assembled evaluation in this source, so QRS/QT cells are unavailable. Never fill them with QS/QT peak spacings or legacy CONTINUOUS_ROOT results. Clinical comparison ranking disabled with only one method.',
 '## Outputs',
 'Table_baseline_compare.tex updated in place; exact previous file preserved in Table_baseline_compare_before.tex. Table_baseline_compare_verified.tex is a standalone copy. baseline_compare_verified.csv has108 cells (2datasets×6methods×9metrics), per_subject_verified.csv has246 method-subject rows. Null clinical baseline values intentional. Original layout/citations retained; Q-Q/R-R/S-S/T-T header corrected to Q/R/S/T because those columns contain localization, not successive intervals. Ranking recalculated from full precision; gray/bold best, underline second; no superiority marks for REM-only clinical columns.',
 '|Dataset|Method|PCC(%)|RMSE|Q|R|S|T|RR|QRS|QT|','|---|---|---:|---:|---|---|---|---|---|---|---|']
 for ds in ('mmecg','crrvs'):
  for model in MODELS:
   values=[]
   for metric in FIELDS:
    x=idx[ds,model,metric];v=x['mean'];sd=x['std_population']
    values.append('—' if v is None else f'{v*100:.2f}' if metric=='pcc' else f'{v:.2f}' if metric=='rmse' else f'{v:.1f} ± {sd:.1f}')
   report.append('|'+ds+'|'+NAMES[model]+'|'+'|'.join(values)+'|')
 if clinical_complete:
  report=[line.replace('Five baselines have no equivalent verified clinical assembled evaluation in this source, so QRS/QT cells are unavailable.',
   'Five baselines now use completed frozen native per-beat centered8s clinical inference in baseline_clinical_20261101. Their scoring reference/detector/matching/dedup are identical to REM-Net; their native shape heads require separately centered contexts and therefore radar input support is NOT identical to the REM single-parent-window decoder. All clinical means are conditional; incomplete subject assessment and coverage are supplied explicitly.')
   .replace('Clinical comparison ranking disabled with only one method.','Clinical ranks use unrounded conditional means; low coverage can produce deceptively small errors, so do not infer clinical superiority from MAE alone.')
   .replace('Null clinical baseline values intentional.','Any null clinical values mean no valid matched endpoints, never zero-imputed.')
   .replace('no superiority marks for REM-only clinical columns.','all available clinical means ranked, with coverage disclosure.') for line in report]
  report+=['','## Completed baseline clinical inference','Source: '+str(baseline_clinical/'REPORT.md'),
   'Original baseline clinical cells were unavailable before this newly authorized inference. No weights or event/morphology numbers changed. Clinical baseline inputs use their own per-beat radar contexts rather than repeated representative templates. This uses potentially different radar context from REM-Net; do not call the full synthesis paths identical or compare total assembly latency with per-pass complexity table latency.',
   '|Dataset|Method|Metric|Assessable/total subjects|Subject-mean coverage|','|---|---|---|---:|---:|']
  for x in summary:
   if x['metric'] in ('qrs','qt'):report.append(f"|{x['dataset']}|{NAMES[x['model']]}|{x['metric']}|{x['n_subjects']}/{x['n_subjects_total']}|{x['coverage_subject_mean']:.3%}|")
  coverage=[dict(dataset=x['dataset'],model=x['model'],metric=x['metric'],n_assessable=x['n_subjects'],n_total=x['n_subjects_total'],coverage=x['coverage_subject_mean']) for x in summary if x['metric'] in ('qrs','qt')]
  save(OUT/'clinical_coverage.csv',coverage)
 (OUT/'REPORT.md').write_text('\n'.join(report)+'\n')
 (OUT/'manifest.json').write_text(json.dumps(dict(training=False,inference=False,new_delineation=False,selected_advantage_cases=False,main_model='phase_pd A4 seed-base42',aggregation='subject macro; within-subject Fisher PCC; population SD',script_sha256=sha(__file__),previous_table_sha256=sha(original),source_sha256={str(p):sha(p) for p in set(sources)},output_sha256={str(p):sha(p) for p in (TARGET,OUT/'baseline_compare_verified.csv',OUT/'per_subject_verified.csv')}),indent=2))
 print('Complete:',OUT)
if __name__=='__main__':main()
