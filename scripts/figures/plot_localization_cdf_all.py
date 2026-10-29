"""Read ONLY existing CDF nodes: no error recomputation, inference or training."""
import os
os.environ.setdefault('MPLBACKEND','Agg')
import sys,csv,json,hashlib
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.figures.paper_method_styles import METHODS,METHOD_STYLES
FIG=ROOT/'docs/paper/sections/figures'
SOURCE=FIG/'source_data/baseline_event_cdf_20261027/exact_cdf_nodes.csv'
OUT=FIG/'source_data/localization_cdf_all_20261029'
MAIN_BASELINES={'mmecg':'mmecg_tcn','crrvs':'cross_modal_conformer'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def visible_step_nodes(x,f,xmax=150.):
    """Keep the full-distribution denominator; extend last level, never normalize."""
    x=np.asarray(x,float);f=np.asarray(f,float)
    if len(x)!=len(f) or not len(x) or not np.isfinite(x).all() or not np.isfinite(f).all():raise ValueError('Invalid nodes')
    if np.any(np.diff(x)<=0) or np.any(np.diff(f)<-1e-12) or np.any(f<0) or np.any(f>1+1e-12):raise ValueError('Invalid empirical CDF')
    mask=x<=xmax;xx=x[mask];ff=f[mask]
    if not len(xx):return np.array([0.,xmax]),np.zeros(2)
    if xx[0]>0:xx=np.r_[0.,xx];ff=np.r_[0.,ff]
    if xx[-1]<xmax:xx=np.r_[xx,xmax];ff=np.r_[ff,ff[-1]]
    return xx,ff

def main(main_figure=False):
    out=FIG/'source_data/localization_cdf_main_20261029' if main_figure else OUT
    stem='fig_localization_cdf_main' if main_figure else 'fig_localization_cdf_all'
    out.mkdir(parents=True,exist_ok=True)
    previous=[FIG/f'fig_event_interval_cdf_remnet_vs_chen.{e}' for e in ('png','pdf','svg')]
    if main_figure:previous.extend(FIG/f'fig_localization_cdf_all.{e}' for e in ('png','pdf','svg'))
    original={str(p):sha(p) for p in previous if p.exists()}
    with SOURCE.open() as f:raw=list(csv.DictReader(f))
    selected=[r for r in raw if r['metric'] in 'qrst']
    used=[r for r in selected if not main_figure or r['model'] in ('rem_net',MAIN_BASELINES[r['dataset']])]
    save(out/'source_cdf_nodes.csv',used)
    groups={}
    for r in selected:groups.setdefault((r['dataset'],r['model'],r['metric']),[]).append(r)
    if len(groups)!=48:raise RuntimeError('Expected 2 datasets × 6 methods × 4 fiducials')
    plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':7,
        'axes.labelsize':8,'axes.titlesize':8,'xtick.labelsize':7,'ytick.labelsize':7,
        'axes.linewidth':.6,'axes.unicode_minus':True,'pdf.fonttype':42,'svg.fonttype':'none',
        'xtick.direction':'out','ytick.direction':'out','xtick.major.size':3,'ytick.major.size':3})
    fig,axes=plt.subplots(2,4,figsize=(7.16,3.85),sharey=True)
    fig.subplots_adjust(left=.075,right=.985,bottom=.115,top=.82,wspace=.35,hspace=.8)
    plotted=[];tails=[]
    for row,ds in enumerate(('mmecg','crrvs')):
        pos=axes[row,0].get_position()
        fig.text(.075,pos.y1+.055,'MMECG (77-GHz FMCW)' if row==0 else 'CR_RVS (24-GHz CW)',
            fontsize=8,fontweight='bold',va='bottom')
        for col,metric in enumerate('qrst'):
            ax=axes[row,col]
            active=(MAIN_BASELINES[ds],'rem_net') if main_figure else METHODS[1:]+METHODS[:1]
            for model in active:
                nodes=groups[ds,model,metric];x=np.array([float(r['error_ms']) for r in nodes]);f=np.array([float(r['cdf']) for r in nodes])
                xx,ff=visible_step_nodes(x,f);style=METHOD_STYLES[model]
                ax.step(xx,ff,where='post',**{k:v for k,v in style.items() if k!='label'})
                if xx[0]==0 and ff[0]>0:
                    ax.vlines(0,0,ff[0],color=style['color'],linewidth=style['linewidth'],linestyle=style['linestyle'],zorder=style['zorder'])
                plotted.extend(dict(dataset=ds,model=model,metric=metric,error_ms=float(a),cdf=float(b)) for a,b in zip(xx,ff))
                tails.append(dict(dataset=ds,model=model,metric=metric,cdf_at_150_ms=float(ff[-1]),
                    fraction_above_150_ms=float(1-ff[-1]),maximum_existing_error_ms=float(x[-1])))
            ax.set(xlim=(0,150),ylim=(0,1),xlabel='Absolute error (ms)',title=metric.upper())
            if col==0:ax.set_ylabel('CDF')
            ax.set_xticks([0,50,100,150]);ax.set_yticks([0,.25,.5,.75,1]);ax.tick_params(labelleft=True)
            ax.spines[['top','right']].set_visible(False);ax.grid(False)
            ax.text(-.20,1.055,f'({chr(97+row*4+col)})',transform=ax.transAxes,fontsize=8,fontweight='bold',va='bottom')
    legend_methods=('rem_net','mmecg_tcn','cross_modal_conformer') if main_figure else METHODS
    handles=[]
    for model in legend_methods:
        style={k:v for k,v in METHOD_STYLES[model].items() if k!='zorder'}
        if main_figure and model!='rem_net':style['label']+=' (MMECG)' if model=='mmecg_tcn' else ' (CR_RVS)'
        handles.append(Line2D([0],[0],**style))
    legend=fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.995),ncol=len(handles),
        fontsize=7,frameon=False,handlelength=2.1,columnspacing=1.0,handletextpad=.4)
    fig.canvas.draw()
    if legend.get_window_extent().width>fig.bbox.width*.96:
        legend.remove();fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.995),ncol=3,
            fontsize=7,frameon=False,handlelength=2.1,columnspacing=1.3,handletextpad=.4)
    for ext in ('pdf','svg','png'):fig.savefig(FIG/f'{stem}.{ext}',dpi=600)
    plt.close(fig);save(out/'plotted_step_nodes.csv',plotted);save(out/'visible_tail_fractions.csv',tails)
    caption=('Conditional localization-error CDFs for REM-Net and five adapted baselines on MMECG (top) and CR\\_RVS (bottom). '
        'Columns show Q, R, S and T absolute localization errors. All methods use a common four-fiducial event-head design '
        '(independently trained weights and backbone-specific input channels) and the same ordered decoder. '
        'Errors are computed for endpoints on heartbeats paired one-to-one using an R-peak tolerance of $\\pm150$ ms; '
        'Q/S/T retain the corresponding R-matched beat pairing and may have errors above150 ms. '
        'Only0--150 ms is displayed; tails beyond150 ms remain in the CDF denominator. '
        'Overlapping-window absolute errors are deduplicated as in the original source; each subject receives equal CDF weight. '
        'Curves are exact steps, without interpolation or smoothing. R-MDR is reported in the main comparison table.')
    rule=('For the main-text figure, one baseline is selected per dataset by the lowest equal-weight mean of '
        'the existing subject-macro Q/R/S/T localization MAEs across the five adapted baselines: '
        'Chen-TCN for MMECG and Conformer for CR\\_RVS. The same selected baseline is used in all four fiducial panels '
        'of that dataset; no panel-wise baseline selection is performed. All six methods are shown in the supplementary figure.')
    if main_figure:
        supplement=caption+' '+rule
        caption=caption.replace('REM-Net and five adapted baselines','REM-Net and one dataset-specific adapted baseline')+' '+rule
        for filename in ('caption.tex','caption_supplement.tex'):
            (OUT/filename).write_text('\\caption{Supplementary figure. '+supplement+'}\n')
        selection=[]
        with (SOURCE.parent/'all_methods_metrics.csv').open() as f:metrics=list(csv.DictReader(f))
        for ds in ('mmecg','crrvs'):
            candidates=[]
            for model in METHODS[1:]:
                values=[float(r['subject_macro_mae_ms']) for r in metrics if r['dataset']==ds and r['model']==model and r['metric'] in ('q','r','s','t')]
                if len(values)!=4:raise RuntimeError('Missing precomputed event MAE for selection')
                candidates.append((sum(values)/4,model))
            if min(candidates)[1]!=MAIN_BASELINES[ds]:raise RuntimeError('Predeclared baseline disagrees with existing four-event average')
            for rank,(value,model) in enumerate(sorted(candidates),1):
                selection.append(dict(dataset=ds,model=model,rank=rank,mean_existing_qrst_mae_ms=value,selected=model==MAIN_BASELINES[ds]))
        save(out/'baseline_selection.csv',selection)
    else:caption+=' '+rule
    (out/'caption.tex').write_text('\\caption{'+caption+'}\n')
    report=['# Six-method localization CDF: existing data only','',
        'No new error calculation, delineation, matching, model inference or training. Every displayed measured node is copied from the existing exact_cdf_nodes.csv; only a constant final step to150ms is added when needed.',
        'Source is the original dual-head seed42 benchmark underlying the requested REM-Net-versus-Chen figure, exported2026-10-16 and cached2026-10-27. REM-Net here is the ORIGINAL dual-head model, NOT the newer phase_pd model. No mixed model lineage or replacement of REM-Net errors.',
        'CDF weighting remains source-defined: window overlap deduplication by median absolute error per reference event, then equal-subject empirical CDF average. This is not pooled-beat weighting.',
        '## Matching-definition correction',
        'The existing error source uses ±150ms one-to-one R matching. Q/S/T inherit that beat match; individual Q/S/T errors can exceed150ms. Filtering these errors or forcing F(150)=1 would change the source data/denominator, which the user prohibited. The caption therefore specifies R-anchored matching rather than falsely claiming every individual endpoint is within150ms.',
        'Common-head means the same Conv5→BN→GELU→Conv5 four-channel readout design, with each method trained independently and the first convolution accepting its own backbone channel count. It does NOT mean shared parameter values or identical head parameter counts.',
        'R-MDR main-table reference is included as requested. The current local Table_baseline_compare.tex has no literal R-MDR heading; verify that the final manuscript main table includes the corresponding version before publication. This task does not edit that table.',
        '## Style and preservation',
        'Canonical method encodings are in scripts/figures/paper_method_styles.py for reuse by future paper plots. Existing figures are not silently recolored; the requested original two-method version is preserved byte-for-byte.',
        '7.16in wide;2x4 panels Q/R/S/T, STIX,8pt axis labels/7pt ticks, one shared legend,0–150ms and0–1 axes. REM-Net is drawn last at highest zorder. No smoothing, PCHIP or endpoint-error recomputation.',
        '', 'See visible_tail_fractions.csv for conditional mass above150ms, and source_cdf_nodes.csv for the unchanged input nodes.']
    if main_figure:
        report[0]='# Main-text localization CDF: dataset-specific strongest baseline'
        report+=['','## Main/supplement split',rule.replace('\\_','_'),
            'Selection reads existing all_methods_metrics.csv only; no per-beat errors or MAEs are recomputed. Mean of the four stored MAEs only determines this descriptive display selection, not model/checkpoint selection or new statistical evidence.',
            'Main: fig_localization_cdf_main.pdf/svg/png. Supplement: existing fig_localization_cdf_all.pdf/svg/png, unchanged. Supplement caption: ../localization_cdf_all_20261029/caption_supplement.tex.',
            'Selected source distributions:16; all measured nodes match the corresponding distributions in the six-method figure exactly. Method colors, line styles and CDF denominators are unchanged.']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    assert all(sha(Path(p))==h for p,h in original.items()),'Original comparison figure changed'
    head=Path('/tmp/rem_dual_head_formal/src/models/baselines/dual_endpoint.py')
    rem=ROOT/'docs/paper/source_data/2026-10-16_existing_remnet_v1/code_snapshot/src/models/continuous/dual_head_rem.py'
    decoder=ROOT/'docs/paper/source_data/2026-10-16_existing_remnet_v1/code_snapshot/scripts/run_dual_head_interaction.py'
    (out/'manifest.json').write_text(json.dumps(dict(training=False,inference=False,error_recalculation=False,main_figure=main_figure,
        selected_baselines=MAIN_BASELINES if main_figure else None,
        source_path=str(SOURCE),source_sha256=sha(SOURCE),original_comparison_sha256=original,
        script_sha256=sha(Path(__file__)),styles_sha256=sha(Path(__file__).with_name('paper_method_styles.py')),
        head_architecture_sources={str(p):sha(p) for p in (head,rem,decoder)},
        output_sha256={str(p):sha(p) for p in [*(FIG/f'{stem}.{e}' for e in ('pdf','svg','png')),out/'plotted_step_nodes.csv']}),indent=2)+'\n')
    print('Complete:',stem,'; existing errors unchanged; original comparison preserved.')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--main',action='store_true');args=parser.parse_args()
    main(main_figure=args.main)
