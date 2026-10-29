"""Layout-only revision of v3 caches: no inference, no new examples or scores."""
import os
os.environ.setdefault('MPLBACKEND','Agg')
import sys,csv,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.figures import plot_examples_v3 as previous

FIG=ROOT/'docs/paper/sections/figures'
SOURCE=FIG/'source_data/examples_v3_20261030'
OUT=FIG/'source_data/examples_v4_20261030'
WIDTH,HEIGHT=7.16,6.3
LEFT,RIGHT=.91,.07
METHODS=previous.METHODS
COLORS=previous.COLORS
ROWS=('radar','gt','rem_net','mmecg_tcn','radarode_mtl')
HEIGHTS=(.42,.48,.48,.48,.48)
VGAP=.04

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def centre_events(a):
    """One complete ordered group nearest centre, from unchanged predicted times."""
    points={e:np.asarray(a['pred_'+e],float) for e in 'qrst'};groups=[]
    for ri,r in enumerate(points['r']):
        if not 0<=r<1024:continue
        q=points['q'][(points['q']<r)&(points['q']>=r-20)&(points['q']>=0)]
        s=points['s'][(points['s']>r)&(points['s']<=r+21)&(points['s']<1024)]
        if not len(q) or not len(s):continue
        q=float(q.max());s=float(s.min())
        upper=min(1024,s+91,float(points['r'][ri+1]) if ri+1<len(points['r']) else 1024)
        t=points['t'][(points['t']>=s+8)&(points['t']<upper)]
        if len(t):groups.append(dict(q=q,r=float(r),s=s,t=float(t.min())))
    if not groups:raise ValueError('No complete visible predicted Q/R/S/T group; never replace the selected example')
    return min(groups,key=lambda p:(abs(p['r']-512),p['r']))

def make_panel(fig,x,y,w,h,radar=False):
    ax=fig.add_axes([x/WIDTH,y/HEIGHT,w/WIDTH,h/HEIGHT]);ax.set_xlim(0,5.12)
    ax.set_facecolor('#EEF5EE' if radar else '#FDF0F0')
    ax.tick_params(which='both',bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft=False)
    for s in ax.spines.values():s.set_color('#888888');s.set_linewidth(.35)
    if not radar:
        ax.set_ylim(-.08,1.30)
        ax.xaxis.set_major_locator(MultipleLocator(.2));ax.xaxis.set_minor_locator(MultipleLocator(.04))
        ax.yaxis.set_major_locator(MultipleLocator(.5));ax.yaxis.set_minor_locator(MultipleLocator(.1))
        ax.grid(which='major',color='#F3D3D3',lw=.30)
        ax.grid(which='minor',color='#FAEAEA',lw=.18);ax.set_axisbelow(True)
    return ax

def load_cases():
    m=json.loads((SOURCE/'manifest.json').read_text());cases=[];sources=[SOURCE/'manifest.json',SOURCE/'per_window_results.csv']
    for meta in m['cases']:
        path=SOURCE/f"{meta['dataset']}_{meta['scene']}_plotted.npz"
        arrays=previous.old.load_npz(path);sources.append(path)
        for method in METHODS:
            value=float(previous.score_shapes(arrays[method+'_shapes'],arrays['reference_shapes']).mean())
            assert abs(value-meta['panel_pcc'][method])<2e-6
        # Recheck SAME native ECG and200-point reference targets, without delineating.
        ref=previous.old.load_npz(Path(meta['cached_reference_ecg_path']));ecg=ref['ecg'][meta['index']]
        display=previous.old.reference_display(ecg,arrays['gt_r']+288)[288:1312]
        assert np.allclose(display,arrays['gt'],atol=1e-10,rtol=0)
        targets=[]
        for left,right in arrays['beat_bounds']+288:
            cycle=ecg[int(left):int(right)]
            v=np.interp(np.linspace(0,len(cycle)-1,200),np.arange(len(cycle)),cycle).astype(np.float32)
            targets.append(previous.old.minmax(v))
        assert np.allclose(targets,arrays['reference_shapes'],atol=2e-6,rtol=0)
        with np.load(meta['filtered_raw_record_path']) as z:
            raw=z['ecg'][meta['start']:meta['start']+1600]
        assert previous.old.pcc(raw,ecg)>1-1e-10
        sources.extend([Path(meta['cached_reference_ecg_path']),Path(meta['filtered_raw_record_path'])])
        cases.append((meta,arrays))
    return cases,sources

def plot(cases):
    plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':7.5,
        'axes.unicode_minus':True,'pdf.fonttype':42,'svg.fonttype':'none','path.simplify':False})
    fig=plt.figure(figsize=(WIDTH,HEIGHT));geometry=[];annotations=[];scales=[];pcc_texts=[]
    top=HEIGHT-.06
    for block,ds in enumerate(('mmecg','crrvs')):
        data=[c for c in cases if c[0]['dataset']==ds];n=len(data)
        mm_width=(WIDTH-LEFT-RIGHT-3*.06)/4
        w=mm_width if n==4 else mm_width*4/3
        gap=(WIDTH-LEFT-RIGHT-n*w)/(n-1)
        scene_top=top-8.5/72-.08
        content_top=scene_top-7.5/72-.04
        fig.text(LEFT/WIDTH,top/HEIGHT,
            '(a) MMECG (77-GHz FMCW)' if block==0 else '(b) CR_RVS (24-GHz CW)',
            fontsize=8.5,fontweight='bold',ha='left',va='top')
        for col,(meta,a) in enumerate(data):
            x=LEFT+col*(w+gap);y=content_top
            fig.text((x+w/2)/WIDTH,scene_top/HEIGHT,meta['scene_label'],ha='center',va='top',fontsize=7.5,fontweight='bold')
            for key,h in zip(ROWS,HEIGHTS):
                bottom=y-h;ax=make_panel(fig,x,bottom,w,h,radar=key=='radar');wave=a[key]
                if key=='radar':
                    limit=max(float(abs(wave).max())*1.1,1e-6);ax.set_ylim(-limit,limit)
                ax.plot(np.arange(1024)/200,wave,color='black',lw=.6,zorder=3)
                if key!='radar':
                    for r in a['gt_r']:
                        if 0<=r<1024:ax.axvline(r/200,color='#BDBDBD',ls=(0,(2,2)),lw=.4,zorder=2)
                if key in METHODS:
                    label=ax.text(.985,.985,f"PCC {meta['panel_pcc'][key]:.2f}",transform=ax.transAxes,
                        ha='right',va='top',fontsize=6.3,zorder=7)
                    pcc_texts.append((ax,label))
                if key=='rem_net':
                    event=centre_events(a)
                    for e,sample in event.items():
                        value=float(np.interp(sample,np.arange(1024),wave));dx={'q':-7,'r':0,'s':7,'t':2}[e]
                        dy=min(3 if e=='r' else 5,max(0,(1.16-value)/1.38*h*72))
                        ax.plot(sample/200,value,'o',ms=1.8,color=COLORS[e],mew=0,zorder=6)
                        ax.annotate(e.upper(),(sample/200,value),xytext=(dx,dy),textcoords='offset points',
                            ha='center',va='center',fontsize=6.3,fontweight='bold',color=COLORS[e],
                            arrowprops=dict(arrowstyle='-',color=COLORS[e],lw=.35,shrinkA=1,shrinkB=1),zorder=6)
                        annotations.append(dict(dataset=ds,scene=meta['scene'],event=e,predicted_sample=sample,
                            predicted_time_seconds=sample/200,waveform_value=value))
                if col==0:
                    name={'radar':'Radar input','gt':'GT ECG',**previous.NAMES}[key]
                    ax.text(-.035,.5,name,transform=ax.transAxes,ha='right',va='center',fontsize=7.5,
                        fontweight='bold' if key=='rem_net' else 'normal')
                geometry.append(dict(dataset=ds,scene=meta['scene'],row=key,left_in=x,bottom_in=bottom,width_in=w,height_in=h))
                y=bottom-VGAP
        content_bottom=content_top-sum(HEIGHTS)-VGAP*4
        bar_y=content_bottom-.055;bar_width=w/5.12;right=WIDTH-RIGHT;left=right-bar_width
        fig.add_artist(Line2D([left/WIDTH,right/WIDTH],[bar_y/HEIGHT]*2,transform=fig.transFigure,color='black',lw=.8))
        for edge in (left,right):
            fig.add_artist(Line2D([edge/WIDTH]*2,[(bar_y-.01)/HEIGHT,(bar_y+.01)/HEIGHT],transform=fig.transFigure,color='black',lw=.6))
        fig.text((left+right)/2/WIDTH,(bar_y-.025)/HEIGHT,'1 s',ha='center',va='top',fontsize=6.5)
        scales.append(dict(dataset=ds,panel_width_in=w,horizontal_gap_in=gap,scale_bar_width_in=bar_width,
            seconds_represented=bar_width/w*5.12,content_bottom_in=content_bottom))
        top=bar_y-.025-6.5/72-.15
    fig.canvas.draw()
    for ax,label in pcc_texts:
        box=label.get_window_extent()
        bottom=ax.transData.inverted().transform((box.x0,box.y0))[1]
        if bottom<=1.0:raise RuntimeError('PCC text overlaps normalized waveform range')
    for ext in ('pdf','svg','png'):fig.savefig(FIG/f'fig_examples_v4.{ext}',dpi=600,facecolor='white')
    plt.close(fig);save(OUT/'panel_geometry.csv',geometry);save(OUT/'event_annotations.csv',annotations);save(OUT/'scale_bars.csv',scales)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    oldfiles=[FIG/f'{stem}.{e}' for stem in ('fig_mmecg_waveform_examples','fig_examples_v3') for e in ('pdf','svg','png')]
    preserved={str(p):sha(p) for p in oldfiles};cases,sources=load_cases();plot(cases)
    # Scores and examples are byte-for-byte the previous audit, never recomputed exports.
    (OUT/'per_window_results.csv').write_bytes((SOURCE/'per_window_results.csv').read_bytes())
    old_report=(SOURCE/'REPORT.md').read_text()
    # Carry over scientific provenance, not the superseded v3 strip/legend layout.
    old_report=old_report.split('## Layout and integrity')[0].split('\n',1)[1]
    old_report=old_report.replace(
        'Predicted Q/R/S/T strips retain cached event times from independently centred REM-Net event outputs, associated to the reference cycle for this reference-assisted morphology view. Reference endpoints use frozen prominence on the same reference ECG. No markers on waveform curves.',
        'Cached predicted event times originate from independently centred REM-Net event outputs and are retained unchanged in v4. The earlier strip presentation is replaced by direct annotations described above.')
    for name in ('baseline_selection.csv','per_beat_results.csv'):
        (OUT/name).write_bytes((SOURCE/name).read_bytes())
    report=['# Examples v4: layout-only revision','',
        'Same7 test examples, model arrays and PCC values as examples_v3. No model inference, training, example replacement, new filtering or score changes. Original dual-head seed42 lineage, not phase_pd.',
        'Waveform/event arrays are read directly from ../examples_v3_20261030/*_plotted.npz; no altered or replacement arrays are generated. per_window_results.csv, baseline_selection.csv and per_beat_results.csv are copied byte-for-byte from that source.',
        '## Revised display',
        '7.16x6.3in IEEE double-column. Both datasets fill the same panel-area extent (row-label margin retained): MMECG4x1.5in columns with0.06in gaps; CR_RVS3x2.0in columns with0.09in gaps, exactly4/3 width. No unused column, event strip or in-figure legend. Each block has its own right-aligned true1s scale below the final row.',
        'Radar0.42in and ECG0.48in row heights;0.04in gaps; title/scenario gap0.08in; between blocks approximately0.15in after the first block scale bar. Full geometry and scale calibration exported.',
        'All normalized ECG axes use[−0.08,1.30], black0.6pt waves, pale grids#F3D3D3/#FAEAEA. Radar retains native signed amplitude and its existing limits, not the ECG0–1 range, to avoid hiding negative phase data. Radar waveform line is also0.6pt. AllPCC labels6.3pt, upper-right, no white patch; rendered text bottoms checked above maximum normalized signal1.0.',
        'Q/R/S/T now appear directly on REM-Net, one complete ordered predicted-event group nearest the5.12s centre (deterministic, no quality selection). Colors unchanged. Marker x coordinates are the original predicted times, never moved to nearby wave maxima; marker y is sampled from the displayed morphology. This reference-RR morphology view can therefore have event times different from visible extrema. Direct labels do NOT assert waveform/anchor identity. Exact selected times saved in event_annotations.csv.',
        '## Apnea reference recheck',
        'CR_RVS Apnea: subject20, CRRVS_0055, fold2 test row953, input start41600; central[41888,42912). Same cached filtered ECG reference used in allPCC target construction; native display and200point targets reverified, as well as positive-affine identity with the0.5–40Hz200Hz raw-record reference cache (PCC>1−1e−10). No smoothing or noise removal added.',
        'This original reference contains substantial10–40Hz content (43.1085% of0.5–40Hz Welch power for this window), retained in both reference target and plotted trace. Beat min–max can make high-frequency variations prominent. It is not caused by a new renderer or a different ECG version. Physiological detail versus acquisition/artifact origin cannot be established from this audit; no unsupported cause is claimed.',
        '', '## Inherited source, selection and PCC audit (unchanged)',old_report]
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n')
    caption=('Reference-timed morphology examples of REM-Net, Chen-TCN and radarODE-MTL, '
        'the two adapted baselines with the highest Table-I morphology PCCs. Each scenario uses the same median-rule test '
        'subject/window as the previous examples; all methods share that window and independently decode every beat. '
        'Beats are placed on reference R--R intervals; timing is evaluated separately. Colored Q/R/S/T labels mark '
        'unchanged predicted event times for a centre-nearest complete group, not a new delineation of the reference-placed waveform. '
        'Gray dashed lines denote reference R peaks. PCC values are the original mean per-beat morphology scores. '
        'Every panel spans5.12s; each block has its own correctly calibrated1s scale. The Apnea reference retains '
        'its original high-frequency content and uses the same preprocessing as its scoring reference.')
    (OUT/'caption.tex').write_text('\\caption{'+caption+'}\n')
    (OUT/'figure_ieee.tex').write_text('\\begin{figure*}[!t]\n\\centering\n\\includegraphics[width=\\textwidth]{figures/fig_examples_v4.pdf}\n\\input{figures/source_data/examples_v4_20261030/caption.tex}\n\\label{fig:examples-v4}\n\\end{figure*}\n')
    assert all(sha(p)==h for p,h in preserved.items()),'Earlier figures changed'
    manifest=dict(training=False,model_inference=False,selection_unchanged=True,source_sha256={str(p):sha(p) for p in sources},
        previous_figures_sha256=preserved,script_sha256=sha(Path(__file__)),
        output_sha256={str(p):sha(p) for p in [*(FIG/f'fig_examples_v4.{e}' for e in ('pdf','svg','png')),OUT/'per_window_results.csv']})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Examples v4 complete:',OUT)

if __name__=='__main__':main()
