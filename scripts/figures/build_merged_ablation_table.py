"""Unified component/task ablation table from existing full-test subject rows.

No training/inference. Preserve planned architecture family (four comparisons
per dataset). S1-vs-A4 contrasts are NEW post-hoc descriptive training-task
comparisons; Holm across their TWO dataset p-values, separately disclosed.
Never combine existing corrected p-values as if from one larger test family.
"""
import sys,csv,json,hashlib,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.phase_resampling_20261027 import ablation_statistics as stats
SOURCE=ROOT/'experiments/phase_resampling_20261027/formal_ablation_20261027'
OUT=ROOT/'docs/paper/sections/ablation_merged_20261101';TARGET=ROOT/'docs/paper/sections/Table_ablation.tex'
ARMS=('A4','A0','A1','A2','A3','S1','S2')
NAMES={'A4':'REM-Net (full)','A0':'Parallel heads','A1':'w/o phase alignment','A2':'w/o event priors','A3':'w/o window context','S1':'Shape head only','S2':'Event head only'}
COMP={'A4':(1,1,1),'A0':(0,0,1),'A1':(0,1,1),'A2':(1,0,1),'A3':(1,1,0),'S1':(0,0,1),'S2':(None,None,None)}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return list(csv.DictReader(Path(p).open()))
def save(p,rows):
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def number(r,k):return float(r[k]) if r[k] else None

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 spec=dict(training=False,inference=False,source=str(SOURCE),architecture_family='preserve original four A0-A3 vs A4 comparisons per dataset',new_shape_only_family='post-hoc S1 vs A4 across two datasets,Holm2',bootstrap=dict(n=10000,seed=42,unit='paired subject',estimator='mean PCC difference in percentage points'),coverage='within subject: total matched Q/R/S/T endpoints / total eligible endpoints; then equal-subject mean')
 (OUT/'spec.json').write_text(json.dumps(spec,indent=2))
 old=OUT/'Table_ablation_before.tex'
 if not old.exists():shutil.copy2(TARGET,old)
 subject=read(SOURCE/'per_subject.csv');planned=read(SOURCE/'table1_modules.csv');multi=read(SOURCE/'table3_multitask.csv');supp=read(SOURCE/'supplement_S1.csv')
 grouped={}
 for ds,n in (('mmecg',11),('crrvs',30)):
  for arm in ARMS:
   a=[r for r in subject if r['dataset']==ds and r['arm']==arm];assert len(a)==n
   ids=[int(r['subject']) for r in a];assert len(set(ids))==n
   grouped[ds,arm]={int(r['subject']):r for r in a}
 new=[]
 for ds in ('mmecg','crrvs'):
  ids=sorted(grouped[ds,'A4']);differences=np.array([(float(grouped[ds,'S1'][s]['pcc'])-float(grouped[ds,'A4'][s]['pcc']))*100 for s in ids])
  delta,lo,hi=stats.bootstrap_mean(differences,n_boot=10000,seed=42)
  new.append(dict(dataset=ds,arm='S1',delta_pp=delta,ci_low_pp=lo,ci_high_pp=hi,p_raw=stats.signed_rank(differences)))
 for row,p in zip(new,stats.holm([r['p_raw'] for r in new])):row['p_holm']=float(p);row['star']='*' if p<.05 else '';row['family']='post-hoc S1 across2datasets'
 save(OUT/'shape_only_posthoc_statistics.csv',new)
 rows=[]
 for ds,n in (('mmecg',11),('crrvs',30)):
  for arm in ARMS:
   a=list(grouped[ds,arm].values());mean=lambda k:float(np.mean([float(r[k]) for r in a if r[k]])) if any(r[k] for r in a) else None
   row=dict(dataset=ds,arm=arm,variant=NAMES[arm],phase=COMP[arm][0],event_prior=COMP[arm][1],context=COMP[arm][2],n_subjects=n,pcc_percent=mean('pcc')*100 if mean('pcc') is not None else None,rmse=mean('rmse'),r_mae_ms=mean('r_mae_ms'),coverage_percent=None,delta_pp=None,ci_low_pp=None,ci_high_pp=None,p_raw=None,p_holm=None,star='',family='none:reference/not applicable')
   if arm!='S1':
    row['coverage_percent']=100*float(np.mean([sum(int(r[e+'_evaluable']) for e in 'qrst')/sum(int(r[e+'_eligible']) for e in 'qrst') for r in a]))
   if arm in ('A0','A1','A2','A3'):
    original=next(r for r in planned if r['dataset']==ds and r['arm']==arm)
    for k in ('delta_pp','ci_low_pp','ci_high_pp','p_raw','p_holm'):row[k]=float(original[k])
    row['star']=original['star'];row['family']='planned4architecture comparisons per dataset'
    assert abs(row['delta_pp']-(row['pcc_percent']-float(next(r for r in planned if r['dataset']==ds and r['arm']=='A4')['pcc_percent'])))<1e-9
   elif arm=='S1':
    extra=next(r for r in new if r['dataset']==ds)
    for k in ('delta_pp','ci_low_pp','ci_high_pp','p_raw','p_holm','star','family'):row[k]=extra[k]
   # Exact agreement with existing metrics/supplement, not a new coverage denominator.
   sr=next(r for r in supp if r['dataset']==ds and r['arm']==arm)
   for event in 'qrst':row[event+'_mae_ms']=mean(event+'_mae_ms');row[event+'_coverage_percent']=float(sr[event+'_coverage'])*100 if sr[event+'_coverage'] else None
   row['fallback_percent']=float(sr['fallback_rate'])*100 if sr['fallback_rate'] else None
   row['parameters_allocated']=int(sr['parameters_allocated']);row['parameters_active']=int(sr['parameters_active'])
   if arm in ('A4','A0','S1','S2'):
    oldrow=next(r for r in multi if r['dataset']==ds and r['arm']==arm)
    for key in ('pcc_percent','rmse','r_mae_ms'):
     if oldrow[key]:assert abs(row[key]-float(oldrow[key]))<1e-9
    if oldrow['overall_coverage_pct']:assert abs(row['coverage_percent']-float(oldrow['overall_coverage_pct']))<1e-9
   rows.append(row)
 save(OUT/'merged_ablation.csv',rows);index={(r['dataset'],r['arm']):r for r in rows}
 def metric(ds,arm,key,digits):
  value=index[ds,arm][key]
  if value is None:return '--'
  text=f'{value:.{digits}f}';available=[index[ds,a][key] for a in ARMS if index[ds,a][key] is not None]
  ranked=sorted(set(available),reverse=key in ('pcc_percent','coverage_percent'))
  if value==ranked[0]:return '$\\mathbf{'+text+'}$'
  if len(ranked)>1 and value==ranked[1]:return '$\\underline{'+text+'}$'
  return '$'+text+'$'
 def delta(ds,arm):
  r=index[ds,arm]
  if r['delta_pp'] is None:return '--'
  markers=r['star']+(r'\dagger' if arm=='S1' else '')
  return '$'+f"{r['delta_pp']:+.2f}"+('^{'+markers+'}' if markers else '')+'$'
 def cells(arm):
  c=['--' if v is None else r'$\checkmark$' if v else r'$\times$' for v in COMP[arm]]
  for ds in ('mmecg','crrvs'):c.extend((metric(ds,arm,'pcc_percent',2),delta(ds,arm),metric(ds,arm,'rmse',3),metric(ds,arm,'r_mae_ms',2),metric(ds,arm,'coverage_percent',2)))
  return c
 lines=[r'\begin{table*}[!t]',r'\centering',r'\caption{Ablation of REM-Net: shape-branch components and training tasks (seed 42, subject-independent test folds).}',r'\label{tab:mechanism_ablation}',r'\scriptsize',r'\setlength{\tabcolsep}{2.7pt}',r'\renewcommand{\arraystretch}{1.12}',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{@{}lccc*{10}{r}@{}}',r'\toprule',r'\multirow{2}{*}{Variant} & \multicolumn{3}{c}{Components} & \multicolumn{5}{c}{MMECG (77-GHz FMCW)} & \multicolumn{5}{c}{CR\_RVS (24-GHz CW)} \\',r'\cmidrule(lr){2-4}\cmidrule(lr){5-9}\cmidrule(lr){10-14}',r' & Phase & $P$ & Ctx & PCC$\uparrow$ & $\Delta$PCC & RMSE$\downarrow$ & R-MAE$\downarrow$ & Cov.$\uparrow$ & PCC$\uparrow$ & $\Delta$PCC & RMSE$\downarrow$ & R-MAE$\downarrow$ & Cov.$\uparrow$ \\',r'\midrule',r'\rowcolor{gray!12}',r'\textbf{REM-Net (full)} & '+' & '.join(cells('A4'))+r' \\',r'\addlinespace[2pt]',r'\multicolumn{14}{l}{\textsc{Shape-branch components}} \\']
 for arm in ('A0','A1','A2','A3'):lines.append(NAMES[arm]+' & '+' & '.join(cells(arm))+r' \\')
 lines.extend([r'\addlinespace[3pt]',r'\multicolumn{14}{l}{\textsc{Training tasks}} \\'])
 for arm in ('S1','S2'):lines.append(NAMES[arm]+' & '+' & '.join(cells(arm))+r' \\')
 lines.extend([r'\bottomrule',r'\end{tabular}%',r'}',r'\vspace{2pt}',r'\begin{minipage}{\textwidth}',r'\scriptsize',r'Phase: beat-synchronous resampling on predicted R anchors; $P$: event-probability priors; Ctx: window-level context features. Components refer to the shape branch, so they are inapplicable to event-only. Parallel heads decode shape in window time without event conditioning. Shape-only uses $\lambda=0$; event-only has no shape output. PCC and Cov. are percentages; R-MAE is in ms. Cov. is the mean across subjects of matched Q/R/S/T endpoints divided by eligible endpoints within each subject. $\Delta$PCC is variant minus full model in percentage points. $^{*}$Two-sided paired Wilcoxon, Holm-adjusted $p<0.05$: the original four architecture comparisons remain one planned family per dataset. $^{\dagger}$Shape-only contrasts are additional post-hoc comparisons; their two dataset $p$-values form a separate Holm family. Bootstrap 95\% CIs, Q/S/T errors, endpoint coverage, fallback rates and parameter counts are provided in Table~\ref{tab:ablation_supplement_merged}. Shape-head input widths and parameter counts differ across variants. Best in bold, second underlined, ranked on unrounded means across all applicable rows.',r'\end{minipage}',r'\end{table*}'])
 text='\n'.join(lines)+'\n';TARGET.write_text(text);(OUT/'Table_ablation_merged.tex').write_text(text)
 supptex=[r'\begin{table*}[!t]\centering\scriptsize',r'\caption{Supplementary ablation statistics, event errors and availability.}',r'\label{tab:ablation_supplement_merged}',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{llrrrrrrrrrrr}\toprule',r'Dataset & Variant & $\Delta$PCC [95\% CI] & $p_{\rm Holm}$ & Q-MAE & S-MAE & T-MAE & Q-Cov. & R-Cov. & S-Cov. & T-Cov. & Fallback & Params (M) \\ \midrule']
 for r in rows:
  fmt=lambda x,d=2:'--' if x is None else f'{x:.{d}f}'
  ci='--' if r['delta_pp'] is None else f"{r['delta_pp']:+.2f} [{r['ci_low_pp']:+.2f}, {r['ci_high_pp']:+.2f}]"
  vals=[r['dataset'].replace('crrvs',r'CR\_RVS').replace('mmecg','MMECG'),NAMES[r['arm']],ci,fmt(r['p_holm'],4),*[fmt(r[e+'_mae_ms']) for e in ('q','s','t')],*[fmt(r[e+'_coverage_percent']) for e in 'qrst'],fmt(r['fallback_percent']),fmt(r['parameters_allocated']/1e6,3)]
  supptex.append(' & '.join(vals)+r' \\')
 supptex.extend([r'\bottomrule\end{tabular}}',r'\parbox{\textwidth}{\scriptsize CI: paired-subject percentile bootstrap, 10,000 resamples, seed 42. Planned architecture and additional post-hoc shape-only tests use separate Holm families as described in the main table. MAE in ms; coverage/fallback in percent. Parameters include all allocated trainable layers; active counts are in merged\_ablation.csv.}',r'\end{table*}'])
 (ROOT/'docs/paper/sections/Table_ablation_S1.tex').write_text('\n'.join(supptex)+'\n');(OUT/'Table_ablation_S1.tex').write_text('\n'.join(supptex)+'\n')
 report=['# Unified component/training-task ablation','',__doc__,'All numbers from existing full-test seed42 experiments; no extra model runs. Full reference shown ONCE; A0 is shared between component/task interpretation and not duplicated. S1/S2 remain native ablations, not missing-data models. No event scores/coverage for S1 and no morphology scores for S2 can be invented.','Coverage missing in the draft is directly recoverable as the mean of per-subject total-matched/total-eligible Q/R/S/T endpoint coverage, matching original Table_multitask overall_coverage_pct exactly. No pooling-window denominator substitution. Shape-only deltas/paired bootstrap and tests now computed from existing paired subject PCCs. Original4-per-dataset Holm results are preserved rather than silently expanded to5 comparisons; S1 tests explicitly post-hoc,Holm across2datasets,marked dagger. If the author instead wants a single5-per-dataset family,stars must be recomputed and may differ; not silently done here.','Main table14columns,same supplied structure. All seven rows ranked jointly for applicable metrics using unrounded means; CR A2/A4 PCC round to90.45 but A2 is slightly higher,so A4 is underlined. Full row shading only identifies reference,not metric superiority. Existing experiments Table_modules/Table_multitask/Table_S1 remain unchanged. Original manuscript Table_ablation.tex saved as Table_ablation_before.tex.', '|Dataset|Variant|PCC|Delta [CI]|RMSE|R-MAE|Coverage|Holm p|','|---|---|---:|---|---:|---:|---:|---:|']
 for r in rows:
  fmt=lambda x,d=2:'—' if x is None else f'{x:.{d}f}'
  ci='—' if r['delta_pp'] is None else f"{r['delta_pp']:+.2f} [{r['ci_low_pp']:+.2f},{r['ci_high_pp']:+.2f}]{r['star']}"
  report.append('|'+r['dataset']+'|'+NAMES[r['arm']]+'|'+fmt(r['pcc_percent'])+'|'+ci+'|'+fmt(r['rmse'],3)+'|'+fmt(r['r_mae_ms'])+'|'+fmt(r['coverage_percent'])+'|'+fmt(r['p_holm'],6)+'|')
 (OUT/'REPORT.md').write_text('\n'.join(report)+'\n')
 sourcepaths=[SOURCE/f for f in ('per_subject.csv','table1_modules.csv','table3_multitask.csv','supplement_S1.csv')]
 (OUT/'manifest.json').write_text(json.dumps(dict(training=False,inference=False,source_sha256={str(p):sha(p) for p in sourcepaths},original_table_sha256=sha(old),script_sha256=sha(__file__),spec=spec,output_sha256={str(p):sha(p) for p in (TARGET,OUT/'merged_ablation.csv',OUT/'shape_only_posthoc_statistics.csv')}),indent=2))
 print('Complete:',OUT)
if __name__=='__main__':main()
