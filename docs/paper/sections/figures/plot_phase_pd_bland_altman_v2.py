"""Presentation-only BA revision; original paired statistics are never refitted.

HR attribution reads existing probability outputs, never a model checkpoint.
"""
import os
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('OMP_NUM_THREADS','2')
import sys,csv,json,hashlib
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter,Formatter,MaxNLocator
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
FIG=Path(__file__).resolve().parent
SOURCE=FIG/'source_data/phase_pd_bland_altman_20261029'
OUT=FIG/'source_data/phase_pd_bland_altman_v2_20261029'
FORMAL=ROOT/'experiments/phase_resampling_20261027/formal_ablation_20261027/runs'
METRICS=('rr','qrs','qt')
COLOR='#1F4E79'

def read(path):return list(csv.DictReader(path.open()))
def save(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def minus(x):return str(x).replace('-', '\N{MINUS SIGN}')

def counts(ref,pred,match):
    matched=len(match(ref,pred,30))
    missing=len(ref)-matched;extra=len(pred)-matched
    category=('both' if missing and extra else 'missed_only' if missing else 'extra_only' if extra else 'neither')
    return missing,extra,category

def refined_r(prob,decode):
    r=decode(prob)['r'];out=[]
    for p in r:
        a=max(0,int(p)-10);b=min(1600,int(p)+11)
        pr=np.clip(prob[1,a:b].astype(float),1e-12,1-1e-12)
        logits=np.log(pr)-np.log1p(-pr);w=np.exp(logits-logits.max())
        out.append(np.sum(w*np.arange(a,b))/w.sum())
    return np.asarray(out)

def hr_attribution(pairs):
    # Original frozen definitions only. No forward, data loader, or training.
    from experiments.phase_resampling_20261027 import frozen_bridge as frozen
    audit=json.loads((SOURCE/'roundtrip_window_audit.json').read_text())
    refs={(r['dataset'],int(r['fold']),r['record'],int(r['start'])):np.asarray(r['reference_r']) for r in audit}
    hr={(r['dataset'],int(r['fold']),r['record'],int(r['start'])):r for r in pairs if r['metric']=='hr'}
    details=[];sources=[]
    for ds,n in (('mmecg',11),('crrvs',5)):
        for fold in range(1,n+1):
            path=FORMAL/ds/f'fold_{fold:02d}/A4/test_predictions.npz';sources.append(dict(path=str(path),sha256=sha(path)))
            with np.load(path,allow_pickle=False) as z:
                for i,prob in enumerate(z['anchor_probability']):
                    rid=z['record_id'][i];rid=rid.decode() if isinstance(rid,bytes) else str(rid)
                    start=int(z['start_idx'][i]);key=(ds,fold,rid,start)
                    if key not in hr:continue
                    stored=hr[key];gt=refs[key];gt=gt[(gt>=288)&(gt<1312)]
                    pred=refined_r(prob,frozen.structured_decode);pred=pred[(pred>=288)&(pred<1312)]
                    estimate=12000/np.diff(pred).mean() if len(pred)>1 else np.nan
                    reference=12000/np.diff(gt).mean() if len(gt)>1 else np.nan
                    actual=float(stored['ref']);old_est=float(stored['pred'])
                    if not np.isclose(reference,actual,atol=1e-8):raise RuntimeError('Reference HR mismatch')
                    same=(np.isnan(estimate) and np.isnan(old_est)) or np.isclose(estimate,old_est,atol=1e-3,rtol=0)
                    if not same:raise RuntimeError(f'Cached probability HR differs from saved result: {key} {estimate} {old_est}')
                    missing,extra,category=counts(gt,pred,frozen.clinical.match_beats)
                    details.append(dict(dataset=ds,fold=fold,subject=str(z['subject_id'][i]),record=rid,start=start,
                        ref_hr=actual,pred_hr=old_est,abs_hr_error=abs(old_est-actual),n_ref=len(gt),n_pred=len(pred),
                        missed=missing,extra=extra,category=category,
                        excessive_error=bool(np.isfinite(old_est) and abs(old_est-actual)>5),
                        unavailable=bool(not np.isfinite(old_est)),hr_replay_abs_difference=abs(old_est-estimate)))
    summary=[]
    for ds in ('mmecg','crrvs'):
        a=[r for r in details if r['dataset']==ds];bad=[r for r in a if r['excessive_error']];N=len(bad)
        count={k:sum(r['category']==k for r in bad) for k in ('missed_only','extra_only','both','neither')}
        summary.append(dict(dataset=ds,total_windows=len(a),finite_hr_windows=sum(not r['unavailable'] for r in a),
            unavailable_hr_windows=sum(r['unavailable'] for r in a),bad_windows=N,
            bad_fraction_finite=N/sum(not r['unavailable'] for r in a),
            any_missed_n=count['missed_only']+count['both'],any_missed_fraction=(count['missed_only']+count['both'])/N,
            any_extra_n=count['extra_only']+count['both'],any_extra_fraction=(count['extra_only']+count['both'])/N,
            neither_n=count['neither'],neither_fraction=count['neither']/N,
            **{k+'_n':v for k,v in count.items() if k!='neither'},
            **{k+'_fraction':v/N for k,v in count.items() if k!='neither'}))
    save(OUT/'hr_attribution_windows.csv',details);save(OUT/'hr_attribution_summary.csv',summary)
    return summary,sources

def render(pairs,stats):
    plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':7,
        'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':.6,
        'axes.unicode_minus':True,'pdf.fonttype':42,'svg.fonttype':'none',
        'xtick.direction':'out','ytick.direction':'out'})
    fig=plt.figure(figsize=(7.16,3.6));outer=fig.add_gridspec(2,3,left=.085,right=.985,bottom=.115,top=.875,wspace=.33,hspace=.83)
    points=[];overflow=[]
    titles=('RR (ms)','QRS duration (ms)','QT interval (ms)')
    for row,ds in enumerate(('mmecg','crrvs')):
        fig.text(.085,.985 if row==0 else .48,
            'MMECG (77-GHz FMCW)' if row==0 else 'CR_RVS (24-GHz CW)',
            fontsize=8,fontweight='bold',ha='left',va='top')
        for col,(metric,title) in enumerate(zip(METRICS,titles)):
            inner=outer[row,col].subgridspec(1,2,width_ratios=[4,1],wspace=.055)
            ax=fig.add_subplot(inner[0]);side=fig.add_subplot(inner[1],sharey=ax)
            s=next(v for v in stats if v['dataset']==ds and v['metric']==metric)
            subset=[r for r in pairs if r['dataset']==ds and r['metric']==metric and np.isfinite(float(r['pred']))]
            ref=np.array([float(r['ref']) for r in subset]);pred=np.array([float(r['pred']) for r in subset]);d=pred-ref;x=(pred+ref)/2
            subjects=np.array([r['subject'] for r in subset]);sx=[];sy=[]
            for subject in sorted(set(subjects),key=int):
                mask=subjects==subject;sx.append(x[mask].mean());sy.append(d[mask].mean())
                points.append(dict(dataset=ds,metric=metric,subject=subject,n_pairs=int(mask.sum()),mean_pair_average=float(sx[-1]),mean_difference=float(sy[-1])))
            assert len(sx)==(11 if ds=='mmecg' else 30)
            loa_limit=1.15*max(abs(float(s['loa_low'])),abs(float(s['loa_high'])))
            # Every subject must remain visible, including a mean outside beat LoA.
            limit=max(loa_limit,1.15*np.max(np.abs(sy)))
            overflow.append(dict(dataset=ds,metric=metric,n_pairs=len(d),display_limit_ms=limit,
                minimum_loa_display_limit_ms=loa_limit,expanded_to_show_subject_means=bool(limit>loa_limit),
                beats_below=int((d<-limit).sum()),beats_above=int((d>limit).sum()),
                beat_fraction_outside=float((abs(d)>limit).mean()),subjects_outside=int((abs(np.asarray(sy))>limit).sum())))
            ax.scatter(sx,sy,s=14,c=COLOR,edgecolors='white',linewidths=.5,zorder=4)
            ax.axhline(0,color='#BDBDBD',lw=.5)
            for key,ls in (('bias','-'),('loa_low','--'),('loa_high','--')):ax.axhline(float(s[key]),color=COLOR,ls=ls,lw=.8)
            significant=float(s['slope_ci_low'])>0 or float(s['slope_ci_high'])<0
            if significant:
                xx=np.array([min(sx),max(sx)]);span=max(np.ptp(xx)*.12,1);xx+=np.array([-span,span])
                ax.plot(xx,float(s['bias'])+float(s['slope'])*(xx-x.mean()),':',color='#D55E00',lw=.8)
            text=minus(f"Bias {float(s['bias']):.1f} [{float(s['bias_ci_low']):.1f}, {float(s['bias_ci_high']):.1f}] ms\nLoA [{float(s['loa_low']):.1f}, {float(s['loa_high']):.1f}] ms")
            ax.text(.025,.975,text,transform=ax.transAxes,ha='left',va='top',fontsize=6.3,
                bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=.7),zorder=5)
            lo=5*np.floor(d.min()/5);hi=5*np.ceil(d.max()/5)
            edges=np.arange(lo,hi+5.01,5)
            side.hist(d,bins=edges,orientation='horizontal',color='#9E9E9E',edgecolor='none')
            side.set_xticks([]);side.tick_params(axis='y',left=False,labelleft=False);side.set_title('beats',fontsize=6.3,pad=3)
            side.set_ylim(-limit,limit);ax.set_ylim(-limit,limit)
            for a in (ax,side):a.spines[['top','right']].set_visible(False);a.grid(False)
            side.spines[['left','bottom']].set_visible(False)
            ax.yaxis.set_major_locator(MaxNLocator(4));ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:Formatter.fix_minus(f'{v:g}')))
            ax.set_xlabel('Mean (ms)')
            if col==0:ax.set_ylabel('Predicted − reference (ms)')
            ax.set_title(title,fontsize=8,pad=7)
            ax.text(-.17,1.08,f'({chr(97+row*3+col)})',transform=ax.transAxes,fontsize=8,fontweight='bold',va='bottom')
    for ext in ('pdf','svg','png'):fig.savefig(FIG/f'fig_bland_altman_v2.{ext}',dpi=600)
    plt.close(fig);save(OUT/'subject_mean_points.csv',points);save(OUT/'display_overflow.csv',overflow)
    return overflow

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    stats=read(FIG/'bland_altman_stats.csv');pairs=read(SOURCE/'paired_values.csv')
    selected=[r for r in stats if r['metric'] in METRICS]
    save(FIG/'bland_altman_stats_v2.csv',selected)
    overflow=render(pairs,stats);hr,sources=hr_attribution(pairs)
    lines=['# REM-Net phase_pd Bland–Altman v2 (presentation-only)', '',
        'Original paired values and all statistics are unchanged. No training or model inference. Original plots and REPORT.md are retained.',
        'Main scatter: one mean paired-average and one mean paired-difference per subject (11/30 subjects); bias, repeated-observation LoA and regression remain from the original individual beat pairs.',
        'Regression is shown only where the stored subject-bootstrap slope95%CI excludes0. The sidebar includes ALL paired beat differences with5ms histogram bins; its shared display range clips the view, not statistics or histogram input.',
        'Symmetric limits cover the original LoA with15% margin. Where a subject mean exceeds this range, limits expand to include that mean with15% margin so that all11/30 subjects remain visible. This affects CR_RVS QT (subject22 mean difference −95.90ms); no subject point or beat is omitted and no triangles are used.',
        'Source: ../phase_pd_bland_altman_20261029/paired_values.csv and ../../bland_altman_stats.csv. Coverage, slope/CI and ICC are copied unchanged to bland_altman_stats_v2.csv.',
        '', '## Beats outside symmetric display limits', '|Dataset|Metric|N pairs|Below|Above|Outside %|','|---|---|---:|---:|---:|---:|']
    for r in overflow:lines.append(f"|{r['dataset']}|{r['metric']}|{r['n_pairs']}|{r['beats_below']}|{r['beats_above']}|{100*r['beat_fraction_outside']:.3f}|")
    lines+=['','## HR errors >5 bpm: R-count attribution',
        'Uses EXACTLY the original central[288,1312) windows and saved HR pairs; unavailable predicted HR windows are reported separately, never classed as zero error.',
        'Existing A4 anchor_probability is decoded with the frozen ordered decoder. Native soft-refined R is recovered by logit(probability) and the original±10sample softmax; recovered HR is checked against every saved HR pair with tolerance0.001bpm. No checkpoint is loaded.',
        'An order-preserving one-to-one R match within150ms defines missed=reference−matched and extra=predicted−matched. Any-missed and any-extra percentages overlap when BOTH occur. Exclusive categories below sum100%.',
        'These are associations with matching/count errors, not a causal decomposition: temporal mislocalization beyond150ms and central-window boundary crossings may also create unmatched events. Neither means counts/matches are intact but RR-span placement still changes HR.',
        '', '|Dataset|Finite HR windows|Unavailable|Error>5 bpm|Any missed %|Any extra %|Neither %|','|---|---:|---:|---:|---:|---:|---:|']
    for r in hr:lines.append(f"|{r['dataset']}|{r['finite_hr_windows']}|{r['unavailable_hr_windows']}|{r['bad_windows']}|{100*r['any_missed_fraction']:.2f}|{100*r['any_extra_fraction']:.2f}|{100*r['neither_fraction']:.2f}|")
    lines+=['','|Dataset|Missed only|Extra only|Both|Neither|','|---|---:|---:|---:|---:|']
    for r in hr:lines.append(f"|{r['dataset']}|{r['missed_only_n']} ({100*r['missed_only_fraction']:.2f}%)|{r['extra_only_n']} ({100*r['extra_only_fraction']:.2f}%)|{r['both_n']} ({100*r['both_fraction']:.2f}%)|{r['neither_n']} ({100*r['neither_fraction']:.2f}%)|")
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (OUT/'manifest.json').write_text(json.dumps(dict(training=False,model_inference=False,
        sources=[dict(path=str(p),sha256=sha(p)) for p in (FIG/'bland_altman_stats.csv',SOURCE/'paired_values.csv',SOURCE/'roundtrip_window_audit.json')]+sources,
        script_sha256=sha(Path(__file__)),unchanged_statistics=True),indent=2)+'\n')
    print('V2 complete:',OUT)

if __name__=='__main__':main()
