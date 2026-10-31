"""Selected SAME-subject localization advantage CDFs; existing errors only.

For each dataset, select the common held-out subject maximizing the equal-weight
mean Q/R/S/T MAE gap (baseline minus REM-Net). One subject for ALL four panels,
SAME subject for both methods. Not overall best REM against another subject's
worst baseline; not a representative/whole-dataset performance comparison.
No model inference, new matching, error correction or discarded large errors.
"""
import os
os.environ.setdefault('MPLBACKEND','Agg')
import sys,csv,json,hashlib
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.figures.paper_method_styles import METHOD_STYLES
FIG=ROOT/'docs/paper/sections/figures';OUT=FIG/'source_data/localization_cdf_advantage_20261031';STEM='fig_localization_cdf_advantage'
BASELINES={'mmecg':'mmecg_tcn','crrvs':'cross_modal_conformer'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,rows):
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def quantile(x,f,p):return float(x[np.flatnonzero(f>=p-1e-12)[0]])

def render(groups,medians):
 plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':7,
  'axes.labelsize':8,'axes.titlesize':8,'xtick.labelsize':7,'ytick.labelsize':7,
  'axes.linewidth':.6,'axes.unicode_minus':True,'pdf.fonttype':42,'svg.fonttype':'none',
  'xtick.direction':'out','ytick.direction':'out','xtick.major.size':3,'ytick.major.size':3})
 fig,axes=plt.subplots(2,4,figsize=(7.16,3.65),sharey=True)
 fig.subplots_adjust(left=.075,right=.985,bottom=.13,top=.88,wspace=.35,hspace=1.0)
 for row,ds in enumerate(('mmecg','crrvs')):
  pos=axes[row,0].get_position();title='MMECG (77-GHz FMCW)' if row==0 else 'CR_RVS (24-GHz CW)'
  fig.text(.075,pos.y1+.075,title+' — vs. '+METHOD_STYLES[BASELINES[ds]]['label'],fontsize=8,fontweight='bold',va='bottom')
  for col,event in enumerate('qrst'):
   ax=axes[row,col]
   for model in (BASELINES[ds],'rem_net'):
    x,f=groups[ds,model,event];style={k:v for k,v in METHOD_STYLES[model].items() if k!='label'}
    grid=np.unique(np.r_[np.linspace(max(0,x[0]),min(120,x[-1]),1000),x[x<=120]])
    ax.plot(grid,PchipInterpolator(x,f,extrapolate=False)(grid),**style)
    if x[-1]<120:ax.plot([x[-1],120],[f[-1],f[-1]],**style)
    if x[0]==0 and f[0]>0:ax.vlines(0,0,f[0],color=style['color'],linewidth=style['linewidth'],linestyle=style['linestyle'],zorder=style['zorder'])
   ax.set(xlim=(0,120),ylim=(0,1),xlabel='Error (ms)',title=event.upper())
   if col==0:ax.set_ylabel('CDF')
   ax.set_xticks([0,40,80,120]);ax.set_yticks([0,.5,1]);ax.tick_params(labelleft=True)
   ax.spines[['top','right']].set_visible(True);ax.grid(False)
   ax.text(-.20,1.055,f'({chr(97+row*4+col)})',transform=ax.transAxes,fontsize=8,fontweight='bold',va='bottom')
   a=medians[ds,'rem_net',event];b=medians[ds,BASELINES[ds],event]
   ax.text(.98,.035,f'Median {a:g} vs {b:g} ms',transform=ax.transAxes,ha='right',va='bottom',fontsize=6.3)
   if col==0:
    handles=[Line2D([0],[0],**{k:v for k,v in METHOD_STYLES[m].items() if k!='zorder'}) for m in ('rem_net',BASELINES[ds])]
    ax.legend(handles=handles,loc='lower right',bbox_to_anchor=(1.,.16),frameon=False,fontsize=6.3,handlelength=1.8,labelspacing=.3,borderaxespad=.25,handletextpad=.4)
 for ext in ('pdf','svg','png'):fig.savefig(FIG/f'{STEM}.{ext}',dpi=600)
 plt.close(fig)

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 preserved={str(FIG/f'{stem}.{ext}'):sha(FIG/f'{stem}.{ext}') for stem in ('fig_localization_cdf_main','fig_localization_cdf_all') for ext in ('pdf','svg','png')}
 paths={'rem_net':FIG/'source_data/remnet_two_dataset_cdf_20261026/deduplicated_errors.json',**{m:FIG/f'source_data/baseline_event_cdf_20261027/{m}/deduplicated_errors.json' for m in BASELINES.values()}}
 data={k:json.loads(p.read_text()) for k,p in paths.items()};ranks=[];selected={};metrics=[];raw=[];nodes=[];groups={};medians={}
 for ds,baseline in BASELINES.items():
  eligible=[]
  for subject in sorted(set(data['rem_net']['errors'][ds])&set(data[baseline]['errors'][ds]),key=int):
   errors={m:data[m]['errors'][ds][subject] for m in ('rem_net',baseline)}
   if not all(len(errors[m][e]) for m in errors for e in 'qrst'):continue
   gap=[float(np.mean(errors[baseline][e])-np.mean(errors['rem_net'][e])) for e in 'qrst']
   eligible.append(dict(dataset=ds,subject=int(subject),baseline=baseline,mean_qrst_mae_gap_ms=float(np.mean(gap)),**{e+'_mae_gap_ms':v for e,v in zip('qrst',gap)}))
  ranked=sorted(eligible,key=lambda r:(-r['mean_qrst_mae_gap_ms'],r['subject']))
  winner=ranked[0];selected[ds]=winner;subject=str(winner['subject'])
  ranks.extend(dict(**r,rank=i+1,selected=i==0) for i,r in enumerate(ranked))
  for model in ('rem_net',baseline):
   for event in 'qrst':
    values=np.asarray(data[model]['errors'][ds][subject][event],float)
    assert np.isfinite(values).all() and (values>=0).all()
    # Same source ECDF construction, including zero node and all tails.
    x=np.unique(np.r_[0.,values]);f=np.searchsorted(np.sort(values),x,side='right')/len(values)
    groups[ds,model,event]=(x,f);medians[ds,model,event]=quantile(x,f,.5)
    counts=data[model]['window_event_counts'][ds][subject][event]
    metrics.append(dict(dataset=ds,subject=int(subject),model=model,metric=event,n_matched_deduplicated=len(values),mae_ms=float(values.mean()),p50_ms=quantile(x,f,.5),p90_ms=quantile(x,f,.9),eligible_window_observations=counts['eligible'],matched_window_observations=counts['matched'],window_observation_coverage=counts['matched']/counts['eligible'],fraction_above_120_ms=float(np.mean(values>120))))
    raw.extend(dict(dataset=ds,subject=int(subject),model=model,metric=event,error_index=i,error_ms=float(v)) for i,v in enumerate(values))
    nodes.extend(dict(dataset=ds,subject=int(subject),model=model,metric=event,error_ms=float(xx),cdf=float(ff)) for xx,ff in zip(x,f))
 save(OUT/'subject_selection.csv',ranks);save(OUT/'metrics_and_coverage.csv',metrics);save(FIG/(STEM+'_errors.csv'),raw);save(FIG/(STEM+'_nodes.csv'),nodes)
 render(groups,medians)
 caption=('Illustrative conditional localization-error CDFs on selected shared test subjects: MMECG S'+str(selected['mmecg']['subject'])+' (REM-Net versus Chen-TCN) and CR\\_RVS S'+str(selected['crrvs']['subject'])+' (REM-Net versus Conformer). Within each dataset the subject maximizes the equal-weight mean Q/R/S/T MAE gap in favor of REM-Net; the same subject is used for both methods and all four panels. These post-hoc selected examples do not represent overall dataset performance. Errors and overlap deduplication are unchanged from the original seed-42 dual-head benchmark. R pairing is one-to-one within $\\pm150$ ms; Q/S/T inherit the R match and may exceed150 ms. CDFs are conditional on each method\'s own matches; coverage and counts are reported separately. PCHIP passes through unchanged ECDF knots for display; medians are empirical inverse-CDF nodes and all tails remain in the denominator.')
 (OUT/'caption.tex').write_text('\\caption{'+caption+'}\n')
 lines=['# Shared-subject localization advantage CDFs','',__doc__,
 'No model inference/training, new delineation, matching, offset subtraction or error manipulation. All scalar errors are exact copies from existing deduplicated_errors.json. Original dual-head seed42, NOT phase_pd. No per-panel subject selection. Raw event identifiers were not retained in these error arrays, so methods use their OWN matched observations on the SAME subject, not claimed identical matched endpoint sets. Coverage and matched counts exported.',
 'Display unchanged from main:7.16x3.65in,2x4,STIX,full frames,Error(ms) on all8 axes,0–120ms,0/40/80/120 ticks,CDF0–1,row-specific legends,canonical colors,PCHIP exact knots. The figure itself has no selection banner; manuscript caption/prose MUST identify these selected subjects and not call them whole-dataset or representative CDFs. Use supplied caption.tex.',
 'MMECG best shared-subject mean advantage is only '+f"{selected['mmecg']['mean_qrst_mae_gap_ms']:.3f} ms"+'; cannot manufacture a large global gap. Positive mean MAE advantage does NOT imply first-order CDF dominance; curves may intersect and medians can tie or reverse. CR_RVS best shared-subject mean advantage is '+f"{selected['crrvs']['mean_qrst_mae_gap_ms']:.3f} ms.",
 'Files: subject_selection.csv(all ranked subjects);metrics_and_coverage.csv;../../'+STEM+'_errors.csv(exact selected scalar errors);../../'+STEM+'_nodes.csv(plot nodes);caption.tex;manifest.json. Source errors cannot support independently choosing the best REM subject and worst baseline subject as if paired.',
 '|Dataset|Subject|Method|Endpoint|N matched|MAE(ms)|P50(ms)|Window-observation coverage|','|---|---:|---|---|---:|---:|---:|---:|']
 for m in metrics:lines.append(f"|{m['dataset']}|{m['subject']}|{m['model']}|{m['metric'].upper()}|{m['n_matched_deduplicated']}|{m['mae_ms']:.3f}|{m['p50_ms']:g}|{m['window_observation_coverage']:.3%}|")
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
 assert all(sha(p)==h for p,h in preserved.items())
 (OUT/'manifest.json').write_text(json.dumps(dict(training=False,inference=False,new_matching=False,new_scalar_errors=False,selected_subjects=selected,selection='post-hoc best mean shared-subject gap, not population comparison',source_sha256={str(p):sha(p) for p in paths.values()},original_figures_sha256=preserved,script_sha256=sha(__file__),outputs={str(p):sha(p) for p in [*(FIG/f'{STEM}.{e}' for e in ('pdf','svg','png')),FIG/(STEM+'_errors.csv'),FIG/(STEM+'_nodes.csv')]}),indent=2))
 print(json.dumps(selected,indent=2));print('Complete:',OUT)

if __name__=='__main__':main()
