"""Finish authorized clinical table refill after the frozen inference queue.

No new training/inference. Waits for its explicit PID or complete manifest;
protects concurrent manuscript edits, checks old nonclinical metrics unchanged,
and writes clinical coverage/subject-count supplementary table.
"""
import os,sys,time,json,csv,hashlib,argparse,subprocess,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'docs/paper/sections/figures/source_data/baseline_clinical_20261101'
RESULTS=ROOT/'docs/paper/sections/results_verified_20261101'
TARGET=ROOT/'docs/paper/sections/Table_baseline_compare.tex'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return list(csv.DictReader(Path(p).open()))
def alive(pid):
 try:
  os.kill(pid,0)
  stat=Path(f'/proc/{pid}/stat')
  return not(stat.exists() and stat.read_text().split()[2]=='Z')
 except ProcessLookupError:return False

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--wait-pid',type=int);args=ap.parse_args()
 request=DATA/'finalization_request.json'
 if not request.exists():request.write_text(json.dumps(dict(manuscript_sha256=sha(TARGET),target=str(TARGET),wait_pid=args.wait_pid),indent=2))
 original=json.loads(request.read_text())
 while True:
  complete=DATA/'manifest.json'
  if complete.exists() and json.loads(complete.read_text()).get('complete'):break
  if args.wait_pid and not alive(args.wait_pid):
   (DATA/'FINALIZATION_FAILED.json').write_text(json.dumps(dict(reason='Inference stopped before complete manifest; no table cells filled',pid=args.wait_pid),indent=2));raise RuntimeError('Incomplete inference; preserve existing table')
  time.sleep(10)
 before=read(RESULTS/'baseline_compare_verified.csv')
 spec=importlib.util.spec_from_file_location('fill_verified_clinical',ROOT/'scripts/figures/fill_verified_baseline_table.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 concurrent_edit=sha(TARGET)!=original['manuscript_sha256']
 if concurrent_edit:module.TARGET=RESULTS/'Table_baseline_compare_auto_generated.tex'
 module.main()
 after=read(RESULTS/'baseline_compare_verified.csv');old={(r['dataset'],r['model'],r['metric']):r for r in before}
 for row in after:
  if row['metric'] in ('qrs','qt'):continue
  prior=old[row['dataset'],row['model'],row['metric']]
  for k in ('mean','std_population','n_subjects'):assert row[k]==prior[k],('Unexpected nonclinical change',row,k)
 assert len(after)==108
 counts=read(RESULTS/'clinical_coverage.csv')
 metas=[]
 for file in (DATA/'records').glob('*/*/*/*_meta.json'):metas.append(json.loads(file.read_text()))
 fallback={}
 for m in metas:
  key=(m['dataset'],m['model'],int(m['subject']));v=fallback.setdefault(key,[0.,0]);v[0]+=m['fallback_window_mean']*m['n_windows'];v[1]+=m['n_windows']
 # Add unchanged REM-Net fallback from formal per-subject test source.
 for m in read(ROOT/'experiments/phase_resampling_20261027/formal_ablation_20261027/per_subject.csv'):
  if m['arm']=='A4':fallback[m['dataset'],'rem_net',int(m['subject'])]=[float(m['fallback_rate']),1]
 models=('mmecg_tcn','radarode_mtl','cross_modal_conformer','wavegru_net','attention_dcgan','rem_net')
 labels=dict(zip(models,('Chen-TCN','radarODE-MTL','Conformer','WaveGRU-Net','Attention-DCGAN','REM-Net')))
 lines=[r'\begin{table*}[t]\centering\scriptsize',r'\caption{Clinical interval availability and synthesis fallback on all test subjects.}',r'\label{tab:baseline_clinical_coverage}',r'\begin{tabular}{llrrrrr}\toprule',r'Dataset & Method & QRS subjects & QRS cov. (\%) & QT subjects & QT cov. (\%) & Fallback (\%) \\ \midrule']
 for ds in ('mmecg','crrvs'):
  for model in models:
   a={r['metric']:r for r in counts if r['dataset']==ds and r['model']==model}
   fb=[v[0]/v[1] for k,v in fallback.items() if k[:2]==(ds,model)];meanfb=sum(fb)/len(fb)*100
   row=[('MMECG' if ds=='mmecg' else r'CR\_RVS'),labels[model],a['qrs']['n_assessable']+'/'+a['qrs']['n_total'],f"{float(a['qrs']['coverage'])*100:.2f}",a['qt']['n_assessable']+'/'+a['qt']['n_total'],f"{float(a['qt']['coverage'])*100:.2f}",f'{meanfb:.2f}']
   lines.append(' & '.join(row)+r' \\')
 lines.extend([r'\bottomrule\end{tabular}',r'\parbox{\textwidth}{\scriptsize Coverage: subject-macro matched/eligible clinical endpoint-pair fraction; missing detections remain eligible. Fallback: subject-macro uncovered central-window fraction, filled with the fixed zero baseline. Baselines use independent predicted-beat-centered 8-s contexts; REM-Net reuses its parent-window phase-conditioned features. Clinical scoring is shared, but waveform decoder input support and total inference cost differ.}',r'\end{table*}'])
 cover=ROOT/'docs/paper/sections/Table_baseline_clinical_coverage.tex';cover.write_text('\n'.join(lines)+'\n')
 report=DATA/'REPORT.md'
 with report.open('a') as f:f.write('\n## Decoder support and automated table refill\nBaselines use separately centered8s radar contexts per predicted RR; REM-Net reuses its parent8s-window features. Thus scoring/detection/matching are shared but radar input support and total assembly inference cost differ. Do not call these identical fixed-input-budget synthesis paths or substitute per-pass complexity latency for full clinical assembly latency. Native-baseline model weights, head design and training are unchanged.\nThe verified comparison table and all20clinical baseline cells were reconsidered from these computed endpoint pairs; no nonclinical mean/std/count changed. Any subject without matched clinical endpoints has undefined conditional MAE, explicitly counted as unassessable and retained in coverage. No zero-error imputation. Coverage and assessable counts: '+str(cover)+'\n')
 result=dict(complete=True,table_path=str(module.TARGET),table_sha256=sha(module.TARGET),protected_concurrent_manuscript_edits=concurrent_edit,source_summary=str(DATA/'summary.csv'),coverage_table=str(cover),all_nonclinical_values_unchanged=True,training=False)
 (DATA/'FINALIZATION_COMPLETE.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
