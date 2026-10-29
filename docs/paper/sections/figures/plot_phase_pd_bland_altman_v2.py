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

def subject_slope(x,y,repeats=10000,seed=42):
    """Equal-subject OLS; bootstrap whole subjects (one mean point per subject)."""
    x=np.asarray(x,float);y=np.asarray(y,float)
    den=np.sum((x-x.mean())**2)
    if den<=1e-12:raise ValueError('Subject means have no horizontal variation')
    slope=float(np.sum((x-x.mean())*(y-y.mean()))/den)
    rng=np.random.default_rng(seed);idx=rng.integers(0,len(x),(repeats,len(x)))
    bx=x[idx];by=y[idx];bx=bx-bx.mean(1,keepdims=True);by=by-by.mean(1,keepdims=True)
    bd=(bx*bx).sum(1);valid=bd>1e-12
    boots=(bx[valid]*by[valid]).sum(1)/bd[valid]
    low,high=np.percentile(boots,[2.5,97.5])
    return dict(subject_mean_slope=slope,subject_mean_intercept=float(y.mean()-slope*x.mean()),
        subject_mean_slope_ci_low=float(low),subject_mean_slope_ci_high=float(high),
        subject_mean_bootstrap_valid=int(valid.sum()),subject_mean_bootstrap_total=repeats,
        regression_line_drawn=bool(low>0 or high<0))

def add_slopes(pairs,stats):
    augmented=[]
    for row in stats:
        if row['metric'] not in METRICS:continue
        group={}
        for r in pairs:
            if r['dataset']==row['dataset'] and r['metric']==row['metric'] and np.isfinite(float(r['pred'])):
                ref,pred=float(r['ref']),float(r['pred'])
                group.setdefault(r['subject'],[]).append(((pred+ref)/2,pred-ref))
        means=np.array([np.mean(group[s],axis=0) for s in sorted(group,key=int)])
        augmented.append(dict(**row,beat_slope=row['slope'],beat_slope_ci_low=row['slope_ci_low'],
            beat_slope_ci_high=row['slope_ci_high'],**subject_slope(means[:,0],means[:,1])))
    return augmented

def place_annotation(fig,ax,side,text,x,y,s):
    """Lower-left first; if occupied, upper-left with extra symmetric headroom."""
    note=ax.text(.025,.025,text,transform=ax.transAxes,ha='left',va='bottom',fontsize=6.3,
        bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=.7),zorder=5)
    limit=ax.get_ylim()[1];gap=fig.dpi*2.5/72
    def clear():
        fig.canvas.draw();bbox=note.get_bbox_patch().get_window_extent()
        pts=ax.transData.transform(np.c_[x,y]);near=bbox.padded(fig.dpi*3/72)
        hit=any(near.contains(*p) for p in pts)
        linehit=any(bbox.y0-gap<=ax.transData.transform((0,float(s[k])))[1]<=bbox.y1+gap for k in ('loa_low','loa_high'))
        return not hit and not linehit
    # The lower LoA is often close to the frame; try just above it as well.
    for ypos in (.025,max(.025,(float(s['loa_low'])+limit)/(2*limit)+.07)):
        note.set_position((.025,ypos))
        if clear():return limit,'lower_left',ypos
    note.set_va('top');note.set_position((.025,.975))
    for _ in range(25):
        if clear():return limit,'upper_left',.975
        limit*=1.12;ax.set_ylim(-limit,limit);side.set_ylim(-limit,limit)
    raise RuntimeError('Could not place annotation without covering a point/LoA')

def worst_qt_subject(pairs):
    eligible=[r for r in pairs if r['dataset']=='crrvs' and r['metric']=='qt']
    groups={}
    for r in eligible:
        if np.isfinite(float(r['pred'])):groups.setdefault(r['subject'],[]).append(float(r['pred'])-float(r['ref']))
    subject=max(groups,key=lambda k:abs(np.mean(groups[k])))
    rows=[r for r in eligible if r['subject']==subject];d=np.asarray(groups[subject])
    meta_path=Path('/home/qhh2237/Datasets/CR_RVS/metadata/crrvs_metadata.csv')
    metadata={r['record_id']:r for r in read(meta_path)}
    windows={};scene_sources=[]
    for fold in sorted(set(int(r['fold']) for r in rows)):
        path=FORMAL/'crrvs'/f'fold_{fold:02d}/A4/test_predictions.npz'
        scene_sources.append(dict(path=str(path),sha256=sha(path)))
        with np.load(path) as z:
            for rid,sid in zip(z['record_id'],z['subject_id']):
                if str(sid)==subject:
                    rid=rid.decode() if isinstance(rid,bytes) else str(rid);windows[rid]=windows.get(rid,0)+1
    scenes=[];export=[]
    for rid in sorted(set(r['record'] for r in rows)):
        a=[r for r in rows if r['record']==rid];v=np.array([float(r['pred'])-float(r['ref']) for r in a if np.isfinite(float(r['pred']))])
        m=metadata[rid]
        scenes.append(dict(subject=subject,official_subject_id=m['subject_id'],record=rid,scene=m['scenario_name'],
            test_windows=windows[rid],eligible_qt_beats=len(a),matched_qt_beats=len(v),coverage=len(v)/len(a),
            bias_ms=float(v.mean()),mae_ms=float(abs(v).mean()),p25_ms=float(np.percentile(v,25)),
            median_ms=float(np.median(v)),p75_ms=float(np.percentile(v,75))))
        for r in a:export.append(dict(**r,scene=m['scenario_name'],official_subject_id=m['subject_id'],signed_error_ms=float(r['pred'])-float(r['ref'])))
    quantiles=np.percentile(d,[0,5,10,25,50,75,90,95,100]);absq=np.percentile(abs(d),[50,90,95])
    summary=dict(subject=subject,official_subject_id=scenes[0]['official_subject_id'],eligible=len(rows),matched=len(d),coverage=len(d)/len(rows),
        bias_ms=float(d.mean()),mae_ms=float(abs(d).mean()),std_ms=float(d.std(ddof=1)),
        signed_quantiles_ms=dict(zip(('min','p5','p10','p25','p50','p75','p90','p95','max'),map(float,quantiles))),
        absolute_quantiles_ms=dict(zip(('p50','p90','p95'),map(float,absq))),fraction_negative=float((d<0).mean()),
        fraction_zero=float((d==0).mean()),fraction_positive=float((d>0).mean()))
    edges=np.arange(5*np.floor(d.min()/5),5*np.ceil(d.max()/5)+5.01,5);counts,_=np.histogram(d,edges)
    bins=[dict(left_ms=float(a),right_ms=float(b),n=int(n),fraction=float(n/len(d))) for a,b,n in zip(edges[:-1],edges[1:],counts)]
    save(OUT/'worst_qt_subject_beats.csv',export);save(OUT/'worst_qt_subject_scenes.csv',scenes);save(OUT/'worst_qt_subject_histogram.csv',bins)
    (OUT/'worst_qt_subject.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary,scenes,[dict(path=str(meta_path),sha256=sha(meta_path))]+scene_sources

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
            ax.scatter(sx,sy,s=14,c=COLOR,edgecolors='white',linewidths=.5,zorder=4)
            ax.axhline(0,color='#BDBDBD',lw=.5)
            for key,ls in (('bias','-'),('loa_low','--'),('loa_high','--')):ax.axhline(float(s[key]),color=COLOR,ls=ls,lw=.8)
            significant=str(s['regression_line_drawn']).lower()=='true'
            if significant:
                xx=np.array([min(sx),max(sx)]);span=max(np.ptp(xx)*.12,1);xx+=np.array([-span,span])
                # Retain the previous horizontal limits without drawing a fit.
                ax.update_datalim(np.c_[xx,np.zeros(2)])
                ax.autoscale_view(scalex=True,scaley=False)
            text=minus(f"Bias {float(s['bias']):.1f} [{float(s['bias_ci_low']):.1f}, {float(s['bias_ci_high']):.1f}] ms\nLoA [{float(s['loa_low']):.1f}, {float(s['loa_high']):.1f}] ms")
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
            initial_limit=limit
            limit,position,note_y=place_annotation(fig,ax,side,text,sx,sy,s)
            overflow.append(dict(dataset=ds,metric=metric,n_pairs=len(d),display_limit_ms=limit,
                minimum_loa_display_limit_ms=loa_limit,expanded_to_show_subject_means=bool(initial_limit>loa_limit),
                expanded_for_annotation=bool(limit>initial_limit),annotation_location=position,annotation_axes_y=note_y,
                beats_below=int((d<-limit).sum()),beats_above=int((d>limit).sum()),
                beat_fraction_outside=float((abs(d)>limit).mean()),subjects_outside=int((abs(np.asarray(sy))>limit).sum())))
    for ext in ('pdf','svg','png'):fig.savefig(FIG/f'fig_bland_altman_v2.{ext}',dpi=600)
    plt.close(fig);save(OUT/'subject_mean_points.csv',points);save(OUT/'display_overflow.csv',overflow)
    return overflow

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    stats=read(FIG/'bland_altman_stats.csv');pairs=read(SOURCE/'paired_values.csv')
    selected=add_slopes(pairs,stats)
    save(FIG/'bland_altman_stats_v2.csv',selected)
    overflow=render(pairs,selected);hr,sources=hr_attribution(pairs)
    worst,scenes,worst_sources=worst_qt_subject(pairs);sources+=worst_sources
    lines=['# REM-Net phase_pd Bland–Altman v2 (presentation-only)', '',
        'Original paired values, bias, LoA, coverage, ICC and beat-level slope/CI are unchanged. Subject-mean slope/CI newly added. No training or model inference. Original plots and REPORT.md are retained.',
        'Main scatter: one mean paired-average and one mean paired-difference per subject (11/30 subjects); bias and repeated-observation LoA remain from original individual beat pairs.',
        'All proportional-bias regression lines are omitted from the current figure at author request2026-10-30. Both beat-level and subject-mean slopes/CI remain in the statistics. The legacy regression_line_drawn column records the PREVIOUS CI eligibility rule, not the current display. The sidebar includes ALL paired beat differences with5ms histogram bins; its shared display range clips the view, not statistics or histogram input.',
        'Subject-mean OLS uses10000 subject resamples, local RNGseed42; degenerate x-variance bootstrap draws are excluded and their counts recorded. No within-subject resampling. Legacy slope/slope_ci columns remain aliases of beat-level values; explicit beat_slope and subject_mean_slope columns remove ambiguity.',
        'Beat-level slope and its95%CI retain the original pooled paired-beat OLS and10000 whole-subject cluster bootstrap (seed42), with every selected subject contributing all its paired beats on each draw. It is not a regression on subject means.',
        'Symmetric limits cover the original LoA with15% margin. Where a subject mean exceeds this range, limits expand to include that mean with15% margin so that all11/30 subjects remain visible. This affects CR_RVS QT (subject22 mean difference −95.90ms); no subject point or beat is omitted and no triangles are used.',
        'Source: ../phase_pd_bland_altman_20261029/paired_values.csv and ../../bland_altman_stats.csv. Coverage, original beat slope/CI and ICC are copied unchanged to bland_altman_stats_v2.csv.',
        'Annotations first try lower-left, including a position just above the lower LoA. Point-marker extent and LoA-line clearance are checked in rendered coordinates. If occupied, use upper-left and expand the symmetric y-range as needed. Display-overflow fractions below use FINAL display limits. No data points are moved.',
        '', '## Beats outside symmetric display limits', '|Dataset|Metric|N pairs|Below|Above|Outside %|','|---|---|---:|---:|---:|---:|']
    for r in overflow:lines.append(f"|{r['dataset']}|{r['metric']}|{r['n_pairs']}|{r['beats_below']}|{r['beats_above']}|{100*r['beat_fraction_outside']:.3f}|")
    lines+=['','## Proportional bias: two levels', '|Dataset|Metric|Beat slope [95% CI]|Subject-mean slope [95% CI]|Historical CI eligibility|','|---|---|---|---|---|']
    for r in selected:lines.append(f"|{r['dataset']}|{r['metric']}|{float(r['beat_slope']):.4f} [{float(r['beat_slope_ci_low']):.4f}, {float(r['beat_slope_ci_high']):.4f}]|{r['subject_mean_slope']:.4f} [{r['subject_mean_slope_ci_low']:.4f}, {r['subject_mean_slope_ci_high']:.4f}]|{r['regression_line_drawn']}|")
    lines+=['','## CR_RVS worst absolute subject-mean QT bias',
        f"- Pipeline subject ID **{worst['subject']}**, official ID **{worst['official_subject_id']}**; mean signed error **{worst['bias_ms']:.2f}ms**, MAE **{worst['mae_ms']:.2f}ms**.",
        f"- Coverage **{worst['matched']}/{worst['eligible']} = {100*worst['coverage']:.2f}%** of eligible unique reference QT beats. This distribution is conditional on both endpoints matched within150ms; unmatched eligible beats remain in the denominator and per-beat export.",
        '- Test prediction scene labels are unknown; scene names are resolved from the existing official metadata record_id mapping, not inferred from signal appearance. Window counts describe the test windows actually scored, not full raw-record duration.',
        '', '|Record|Scene|Test windows|Eligible QT beats|Matched|Coverage %|Bias ms|MAE ms|','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in scenes:lines.append(f"|{r['record']}|{r['scene']}|{r['test_windows']}|{r['eligible_qt_beats']}|{r['matched_qt_beats']}|{100*r['coverage']:.2f}|{r['bias_ms']:.2f}|{r['mae_ms']:.2f}|")
    lines+=['','### Matched beat QT error distribution (prediction minus reference)',
        '- Signed quantiles (ms): '+', '.join(f'{k}={v:.2f}' for k,v in worst['signed_quantiles_ms'].items()),
        '- Absolute-error quantiles (ms): '+', '.join(f'{k}={v:.2f}' for k,v in worst['absolute_quantiles_ms'].items()),
        f"- SD={worst['std_ms']:.2f}ms; negative/zero/positive={100*worst['fraction_negative']:.2f}%/{100*worst['fraction_zero']:.2f}%/{100*worst['fraction_positive']:.2f}%.",
        '- Exact5ms histogram counts: worst_qt_subject_histogram.csv; all eligible individual pairs with scene/official subject mapping: worst_qt_subject_beats.csv; scene summary: worst_qt_subject_scenes.csv. No outlier exclusion or offset subtraction.',
        '- Low coverage and a large conditional negative bias are both retained. This descriptive result does not identify the cause (model morphology, delineation, or both) and is not a diagnosis of the participant.']
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
        script_sha256=sha(Path(__file__)),original_statistics_unchanged=True,added_subject_mean_slopes=True),indent=2)+'\n')
    print('V2 complete:',OUT)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--render-only',action='store_true',
        help='Redraw saved paired values/statistics without recomputing any statistics')
    args=parser.parse_args()
    if args.render_only:render(read(SOURCE/'paired_values.csv'),read(FIG/'bland_altman_stats_v2.csv'))
    else:main()
