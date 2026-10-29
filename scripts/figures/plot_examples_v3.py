"""IEEE morphology examples from audited independent per-beat inference caches.

No training or new model inference. Existing waveform figures are read-only.
"""
import os
os.environ.setdefault('MPLBACKEND','Agg')
import sys,csv,json,hashlib,importlib.util
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.signal import welch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.figures import plot_dual_head_waveform_grid as old

ROOT=old.ROOT;FIG=ROOT/'docs/paper/sections/figures'
CACHE=FIG/'source_data/fig4_per_beat_20261026'
OUT=FIG/'source_data/examples_v3_20261030'
METHODS=old.METHODS
NAMES={'rem_net':'REM-Net','mmecg_tcn':'Chen-TCN','radarode_mtl':'radarODE-MTL'}
COLORS={'q':'#D9822B','r':'#2A9D8F','s':'#D17DB8','t':'#E0B03A'}
HEIGHTS=(.42,.48,.48,.12,.48,.48)
GAPS=(.04,.04,0.,.04,.04)
WIDTH,HEIGHT=7.16,7.4
LEFT,RIGHT,HGAP=.91,.07,.06
PANEL_WIDTH=(WIDTH-LEFT-RIGHT-3*HGAP)/4
BLOCK_CONTENT=sum(HEIGHTS)+sum(GAPS)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csvread(p):return list(csv.DictReader(p.open()))
def save(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def select():
    """Same frozen arithmetic morphology-PCC median rule, before any baseline."""
    candidates=csvread(CACHE/'selection_candidates.csv');selected=[];subject_rows=[]
    for ds,scenes in old.SCENES.items():
        for scene,label in scenes:
            group=[r for r in candidates if r['dataset']==ds and r['scene']==scene]
            subjects=[]
            for sid in sorted(set(int(r['subject']) for r in group)):
                a=[r for r in group if int(r['subject'])==sid]
                subjects.append(dict(dataset=ds,scene=scene,subject=sid,pcc=float(np.mean([float(r['window_pcc']) for r in a])),n_windows=len(a)))
            median=float(np.median([r['pcc'] for r in subjects]));subject=min(subjects,key=lambda r:(abs(r['pcc']-median),r['subject']))
            windows=[r for r in group if int(r['subject'])==subject['subject']];wm=float(np.median([float(r['window_pcc']) for r in windows]))
            row=min(windows,key=lambda r:(abs(float(r['window_pcc'])-wm),r['record_id'],int(r['start'])))
            selected.append(dict(dataset=ds,scene=scene,scene_label=label,subject=subject['subject'],fold=int(row['fold']),
                index=int(row['index']),record_id=row['record_id'],start=int(row['start']),stop=int(row['stop']),
                window_pcc=float(row['window_pcc']),subject_pcc=subject['pcc'],scenario_median_pcc=median,subject_window_median=wm))
            subject_rows+=subjects
    return selected,subject_rows

def score_shapes(pred,ref):
    """Reproduce frozen per-row PCC float32 formula used by cache, not new scores."""
    a=pred-pred.mean(1,keepdims=True);b=ref-ref.mean(1,keepdims=True)
    return (a*b).sum(1)/np.sqrt((a*a).sum(1)*(b*b).sum(1)+1e-8)

def clinical():
    p=Path('/tmp/rem_dual_head_formal/src/utils/clinical_endpoints.py')
    spec=importlib.util.spec_from_file_location('examples_v3_frozen_clinical',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m,p

def load_cases():
    import warnings
    chosen,subjects=select();previous=json.loads((CACHE/'manifest.json').read_text());actual,clinical_path=clinical()
    cases=[];logs=[];sources=[CACHE/'selection_candidates.csv',CACHE/'manifest.json',clinical_path];beat_rows=[]
    for row in chosen:
        assert row['stop']-row['start']==1600
        meta=next(m for m in previous['cases'] if m['dataset']==row['dataset'] and m['scene']==row['scene'])
        for key in ('subject','fold','index','record_id','start'):assert row[key]==meta[key],(row,key)
        prefix=f"{row['dataset']}_{row['scene']}"
        plotted=CACHE/(prefix+'_plotted.npz');infer=CACHE/(prefix+'_inference.npz')
        arrays=old.load_npz(plotted);cache=old.load_npz(infer);sources.extend([plotted,infer,infer.with_suffix('.json')])
        refpath=old.SOURCE/'references'/row['dataset']/f"fold_{row['fold']:02d}.npz";ref=old.load_npz(refpath);sources.append(refpath)
        i=row['index'];ecg=np.asarray(ref['ecg'][i]);assert str(ref['record_id'][i])==row['record_id'] and int(ref['start_idx'][i])==row['start']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');points=actual.delineate_signal(ecg,fs=200,method='prominence')
        assert np.array_equal(points['r']-288,arrays['gt_r']),'Reference peak/delineator version changed'
        assert np.allclose(old.reference_display(ecg,points['r'])[288:1312],arrays['gt'],rtol=0,atol=1e-10),'GT display changed'
        bounds=arrays['beat_bounds']+288;targets=[]
        covered=np.zeros(1600,bool)
        for a,b in bounds:
            covered[int(a):int(b)]=True
            cycle=ecg[int(a):int(b)];v=np.interp(np.linspace(0,len(cycle)-1,200),np.arange(len(cycle)),cycle).astype(np.float32)
            targets.append(old.minmax(v))
        assert covered[288:1312].all(),'Reference beat assembly has an uncovered displayed segment'
        assert np.allclose(targets,arrays['reference_shapes'],atol=2e-6,rtol=0),'Reference targets differ from displayed preprocessing'
        origins=cache['reference_origins'];assert len(origins)==len(bounds)==len(set(origins)), 'Non-independent beat contexts'
        lookup={int(s):j for j,s in enumerate(cache['starts'])};pccs={}
        for method in METHODS:
            shapes=np.stack([cache[method+'_shape'][lookup[int(s)]] for s in origins])
            assert np.array_equal(shapes,arrays[method+'_shapes']),'Beat shape is not its own context output'
            check=float(score_shapes(shapes,arrays['reference_shapes']).mean());stored=float(meta['window_pcc_by_method'][method])
            if abs(check-stored)>2e-6:raise RuntimeError(f'Per-window PCC mismatch {prefix} {method}: {check} {stored}')
            pccs[method]=stored
            wave=np.zeros(1600)
            for bi,((a,b),s) in enumerate(zip(bounds,origins)):
                a,b=int(a),int(b);wave[a:b]=np.interp(np.linspace(0,199,b-a),np.arange(200),old.minmax(shapes[bi]))
                beat_rows.append(dict(dataset=row['dataset'],scene=row['scene'],method=method,subject=row['subject'],
                    record_id=row['record_id'],window_index=i,beat_index=bi,reference_left=row['start']+a,
                    reference_right=row['start']+b,context_start=int(s),pcc=float(score_shapes(shapes[bi:bi+1],arrays['reference_shapes'][bi:bi+1])[0])))
            assert np.allclose(wave[288:1312],arrays[method],rtol=0,atol=1e-10),'Assembly differs from independent per-beat shapes'
        radar,radar_meta=old.radar_input(row)
        assert np.allclose(radar,arrays['radar'],rtol=0,atol=1e-10),'Radar preprocessing changed'
        # radar_input additionally verifies record/start against the real test.h5.
        with h5py.File(radar_meta['path'],'r') as h:
            assert int(h['subject_id'][i])==row['subject']
            stored_ecg=np.asarray(h['ecg'][i]).reshape(-1)
            alignment=old.pcc(stored_ecg,ecg)
            assert alignment>1-1e-10,'Snapshot reference differs from current test preprocessing'
        rawpath=ROOT/'experiments/development/structured_decoder_v0_20261017_ab/raw_reference'/('MMECG' if row['dataset']=='mmecg' else 'CR_RVS')/(row['record_id']+'.npz')
        with np.load(rawpath) as z:raw_ecg=z['ecg'][row['start']:row['start']+1600]
        raw_alignment=old.pcc(raw_ecg,ecg)
        assert raw_alignment>1-1e-10,'Raw-record filtered ECG is not the same reference version'
        sources.append(rawpath)
        for event in 'qrst':arrays['ref_'+event]=np.asarray(points[event],float)-288
        used_keys=('gt','radar','gt_r','beat_bounds','reference_shapes',*METHODS,
            *(m+'_shapes' for m in METHODS),*(p+e for p in ('pred_','ref_') for e in 'qrst'))
        np.savez_compressed(OUT/(prefix+'_plotted.npz'),**{k:arrays[k] for k in used_keys})
        freq,power=welch(ecg,fs=200,nperseg=800)
        hf=float(power[(freq>=10)&(freq<=40)].sum()/max(power[(freq>=.5)&(freq<=40)].sum(),1e-12))
        log=dict(**row,panel_pcc=pccs,reference_beats=len(bounds),reference_pcc_against_test_cache=alignment,
            reference_pcc_against_raw_record=raw_alignment,filtered_raw_record_path=str(rawpath),
            radar=radar_meta,reference_high_frequency_fraction_10_40hz=hf,all_windows_test=True,
            independent_beat_contexts=list(map(int,origins)),cached_reference_ecg_path=str(refpath),
            cached_case_manifest_matches=True,checkpoints=meta['checkpoints'])
        cases.append((log,arrays));logs.append(log)
    save(OUT/'per_beat_results.csv',beat_rows);save(OUT/'subject_scenario_selection_scores.csv',subjects)
    save(OUT/'per_window_results.csv',[dict(dataset=r['dataset'],scene=r['scene'],subject=r['subject'],fold=r['fold'],window_index=r['index'],
        record_id=r['record_id'],start=r['start'],central_start=r['start']+288,central_stop=r['start']+1312,
        selection_representative_pcc=r['window_pcc'],**{m+'_pcc':r['panel_pcc'][m] for m in METHODS}) for r in logs])
    return cases,logs,sources

def panel(fig,left,bottom,width,height,radar=False,strip=False):
    ax=fig.add_axes([left/WIDTH,bottom/HEIGHT,width/WIDTH,height/HEIGHT])
    ax.set_xlim(0,5.12);ax.set_facecolor('#EEF5EE' if radar else '#FDF0F0')
    ax.tick_params(which='both',bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft=False)
    for spine in ax.spines.values():spine.set_linewidth(.35);spine.set_color('#888888')
    if radar or strip:ax.grid(False)
    else:
        ax.set_ylim(-.1,1.16)
        ax.xaxis.set_major_locator(MultipleLocator(.2));ax.xaxis.set_minor_locator(MultipleLocator(.04))
        ax.yaxis.set_major_locator(MultipleLocator(.5));ax.yaxis.set_minor_locator(MultipleLocator(.1))
        ax.grid(which='major',color='#F2C9C9',lw=.30,zorder=0)
        ax.grid(which='minor',color='#F8E1E1',lw=.18,zorder=0);ax.set_axisbelow(True)
    return ax

def plot(cases):
    plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':7.5,
        'axes.unicode_minus':True,'pdf.fonttype':42,'svg.fonttype':'none','path.simplify':False})
    fig=plt.figure(figsize=(WIDTH,HEIGHT));axes_log=[]
    # Fixed requested panel heights. Remaining page space is reserved for headings/margins.
    header=.64;top=HEIGHT-.25
    for block,ds in enumerate(('mmecg','crrvs')):
        title_y=top-.03;content_top=top-header
        fig.text(LEFT/WIDTH,title_y/HEIGHT,
            '(a) MMECG (77-GHz FMCW)' if block==0 else '(b) CR_RVS (24-GHz CW)',
            fontsize=8.5,fontweight='bold',ha='left',va='bottom')
        data=[c for c in cases if c[0]['dataset']==ds]
        for col,(meta,arrays) in enumerate(data):
            x=LEFT+col*(PANEL_WIDTH+HGAP);y=content_top
            fig.text((x+PANEL_WIDTH/2)/WIDTH,(content_top+.08)/HEIGHT,meta['scene_label'],
                ha='center',va='bottom',fontsize=7.5,fontweight='bold')
            for ri,(key,height) in enumerate(zip(('radar','gt','rem_net','fiducials','mmecg_tcn','radarode_mtl'),HEIGHTS)):
                bottom=y-height;ax=panel(fig,x,bottom,PANEL_WIDTH,height,radar=key=='radar',strip=key=='fiducials')
                if key=='fiducials':
                    ax.set_ylim(0,1)
                    for event,color in COLORS.items():
                        p=arrays['pred_'+event];p=p[np.isfinite(p)&(p>=0)&(p<1024)]
                        r=arrays['ref_'+event];r=r[np.isfinite(r)&(r>=0)&(r<1024)]
                        ax.vlines(p/200,.56,.94,color=color,lw=1.0,zorder=5)
                        ax.vlines(r/200,.06,.44,color=color,lw=.5,alpha=.6,zorder=4)
                else:
                    wave=arrays[key]
                    if key=='radar':
                        lim=max(float(abs(wave).max())*1.1,1e-6);ax.set_ylim(-lim,lim)
                    ax.plot(np.arange(1024)/200,wave,color='black',lw=.6,zorder=3)
                    if key in METHODS:
                        ax.text(.985,.98,f"PCC {meta['panel_pcc'][key]:.2f}",transform=ax.transAxes,fontsize=6.3,
                            ha='right',va='top',bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=.5),zorder=7)
                    if col==0:
                        label={'radar':'Radar input','gt':'GT ECG',**NAMES}[key]
                        ax.text(-.035,.5,label,transform=ax.transAxes,ha='right',va='center',fontsize=7.5,
                            fontweight='bold' if key=='rem_net' else 'normal')
                if key!='radar':
                    for r in arrays['gt_r']:
                        if 0<=r<1024:ax.axvline(r/200,color='#BDBDBD',ls=(0,(2,2)),lw=.4,zorder=2)
                axes_log.append(dict(dataset=ds,scene=meta['scene'],row=key,left_in=x,bottom_in=bottom,width_in=PANEL_WIDTH,height_in=height))
                y=bottom-(GAPS[ri] if ri<len(GAPS) else 0)
        content_bottom=content_top-BLOCK_CONTENT
        if ds=='crrvs':
            legend=fig.add_axes([(LEFT+3*(PANEL_WIDTH+HGAP))/WIDTH,content_bottom/HEIGHT,PANEL_WIDTH/WIDTH,BLOCK_CONTENT/HEIGHT]);legend.axis('off');legend.set(xlim=(0,1),ylim=(0,1))
            from matplotlib.patches import Rectangle
            for j,(event,color) in enumerate(COLORS.items()):
                x=.08+j*.235;legend.add_patch(Rectangle((x,.69),.07,.027,facecolor=color,edgecolor='none'))
                legend.text(x+.09,.703,event.upper(),fontsize=6.5,va='center')
            legend.plot([.12,.12],[.58,.62],color='black',lw=1.)
            legend.text(.22,.60,'Predicted',fontsize=6.5,va='center')
            legend.plot([.12,.12],[.49,.53],color='black',lw=.5,alpha=.6)
            legend.text(.22,.51,'Reference',fontsize=6.5,va='center')
            legend.plot([.12,.12],[.40,.44],color='#BDBDBD',lw=.4,ls=(0,(2,2)))
            legend.text(.22,.42,'Reference R peak',fontsize=6.5,va='center')
            start=.5-.5/5.12;stop=.5+.5/5.12
            legend.plot([start,stop],[.28,.28],color='black',lw=.8)
            legend.plot([start,start],[.27,.29],color='black',lw=.6);legend.plot([stop,stop],[.27,.29],color='black',lw=.6)
            legend.text(.5,.25,'1 s',fontsize=6.5,ha='center',va='top')
        top=content_bottom-.18
    for ext in ('pdf','svg','png'):fig.savefig(FIG/f'fig_examples_v3.{ext}',dpi=600,facecolor='white')
    plt.close(fig);save(OUT/'panel_geometry.csv',axes_log)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    oldfiles=[FIG/f'fig_mmecg_waveform_examples.{e}' for e in ('png','pdf','svg')]
    preserved={str(p):sha(p) for p in oldfiles if p.exists()}
    cases,logs,sources=load_cases();plot(cases)
    table=old.SOURCE/'tables/all_models_official_shape_macro_metrics.csv'
    allrows=csvread(table);rank=[]
    for ds in ('mmecg','crrvs'):
        a=sorted([r for r in allrows if r['dataset']==ds and r['metric']=='pcc' and r['model']!='rem_net'],key=lambda r:-float(r['mean']))
        if set(r['model'] for r in a[:2])!=set(METHODS[1:]):raise RuntimeError('Requested baseline selection does not match morphology table')
        rank.extend(dict(dataset=ds,rank=i+1,method=r['model'],table_pcc=float(r['mean']),selected=i<2) for i,r in enumerate(a))
    save(OUT/'baseline_selection.csv',rank);sources.append(table)
    lines=['# Fig. examples v3: reference-timed morphology examples','',
        'No training or new inference. Existing independent per-beat centred inference caches reused for all methods; seven median-selected test windows. OLD original dual-head REM-Net seed42 lineage, not phase_pd. No representative-beat copying.',
        '## Selection and PCC protocol',
        'Per scenario: arithmetic subject mean of existing Table-I morphology-window PCC; subject closest to scenario median; within that subject choose window closest to its median (ties by subject ID, record ID, start). All methods use the same chosen window. Selection reproduced from frozen candidate rows and exactly matches the existing pre-inference selection manifest.',
        'Baselines: highest two Table-I morphology PCCs among five adapted baselines, independently checked for each dataset: radarODE-MTL and Chen-TCN. Selected once, never by example appearance. See baseline_selection.csv.',
        'Each reference R–R beat has its own8s radar context centred on the beat midpoint and its own shape-head output for ALL methods. Cached context-start/shape associations audited. Reference R timing is used for morphology placement; this is NOT deployable continuous ECG timing. Beats placed on reference R–R intervals; timing evaluated separately.',
        'Displayed PCC is the arithmetic mean of per-beat200point Pearson scores for reference beats intersecting the central5.12s. Same target construction/formula as Table I but additional beat-centred sampling. Values exactly match the cached per-window per-beat inference manifest (tolerance2e−6), and are exported in per_window_results.csv.',
        'The original selection-window score was from one representative beat per8s window. It is separately exported as selection_representative_pcc and MUST NOT be substituted for the independent per-beat panel PCC. The latter need not equal that original representative-beat test score.',
        'Fold numbers are1-based; test-window indices and channels are0-based. Start/stop coordinates are200Hz samples, half-open. Source origins and window identifiers are exported for every independently decoded beat.',
        '', '|Dataset|Scene|Subject|Fold|Test index|Record|Window start|REM-Net PCC|Chen PCC|radarODE PCC|','|---|---|---:|---:|---:|---|---:|---:|---:|---:|']
    for r in logs:lines.append(f"|{r['dataset']}|{r['scene_label']}|{r['subject']}|{r['fold']}|{r['index']}|{r['record_id']}|{r['start']}|{r['panel_pcc']['rem_net']:.6f}|{r['panel_pcc']['mmecg_tcn']:.6f}|{r['panel_pcc']['radarode_mtl']:.6f}|")
    lines+=['','## Radar input and reference checks',
        'Radar: same test.h5 RCG source as time-domain inputs; additional display-only fourth-order Butterworth0.8–3Hz cardiac-band filtering with zero-phase sosfiltfilt. Select channel with highest cardiac-band/total AC energy over the FULL8s window, without ECG or output information. Amplitude unchanged; only y limits adjust. Channels are zero-based.',
        'Native GT ECG is the SAME cached preprocessed reference used to build every200point PCC target. Every displayed native reference beat and each cached200point target was independently checked against this same reference window; all test HDF5 record/subject/start identities and positive-affine preprocessing equivalence verified. The raw-record reference cache (0.5–40Hz filtered,200Hz) also matches each selected window with PCC>1−1e−10; paths/hashes retained. No new ECG smoothing/filtering for display.',
        'Per-beat min–max is applied independently to each method and reference for display. Reference ECG stays at its native sample count; each predicted200point beat is inverse-resampled to its reference RR length. Complete coverage of retained centre; no copied shape, arbitrary alignment or GT amplitude fitting.',
        'Predicted Q/R/S/T strips retain cached event times from independently centred REM-Net event outputs, associated to the reference cycle for this reference-assisted morphology view. Reference endpoints use frozen prominence on the same reference ECG. No markers on waveform curves.',
        '', '|Dataset|Scene|Channel index|Filter Hz|Reference10–40Hz energy fraction|','|---|---|---:|---|---:|']
    for r in logs:lines.append(f"|{r['dataset']}|{r['scene_label']}|{r['radar']['channel_zero_based']}|0.8–3|{r['reference_high_frequency_fraction_10_40hz']:.4%}|")
    lines+=['', '### Apnea reference high-frequency content',
        'Any fine/high-frequency variation in Apnea GT is already present in the cached paired ECG and its corresponding test HDF5 reference; it is not added by rendering. Beat min–max can make small-amplitude variations visually prominent. The same variation is retained in PCC target construction. Spectral10–40Hz fraction is descriptive, not an artifact diagnosis; this audit cannot distinguish physiological high-frequency detail from acquisition noise.',
        '', '## Layout and integrity',
        'Canvas7.16x7.4in, fixed shared4column grid with0.06in gaps. Radar0.42in; ECG rows0.48in; REM event strip0.12in, zero REM-to-strip gap; other row gaps0.04in. Block gap0.18in; remaining page area accommodates block/scenario headings and outer margins. Panel geometry exported. CR_RVS column4 holds legend and one1s bar, whose physical length equals1s in every5.12s panel.',
        'STIX, embedded TrueType fonts, vector PDF/SVG,600dpi PNG. No GT PCC label, no single-beat zoom, no separate per-block scale bars. PCC labels shown only for model rows, no Shape-200 wording inside the figure. Original figures preserved.',
        'ECG grid time spacing0.2s/0.04s. Horizontal/amplitude grid spacing is decorative on independently normalized beats, not calibrated millivolts. Colors and time scale are identical across methods. Standalone plotted data including predicted/reference event times are saved as seven *_plotted.npz files in this output directory.',
        'All displayed windows and all centred radar contexts belong to the held-out test RECORD/SUBJECT; the extra centred contexts are not necessarily original cached-window grid starts. No training examples, model selection or hand-picked waveforms.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    caption=('Morphology examples of REM-Net and the two adapted baselines with the highest Table-I morphology PCCs '
        '(radarODE-MTL and Chen-TCN). Rows show radar input, reference ECG, REM-Net, Chen-TCN and radarODE-MTL; '
        'each REM-Net row is followed by a predicted/reference Q/R/S/T timing strip. '
        'Each scenario uses the subject nearest its subject-level median REM-Net PCC and the window nearest that subject\'s median; '
        'all methods share the same test window. Every beat is independently decoded from a beat-centred radar context. '
        'Beats are placed on reference R--R intervals; timing is evaluated separately. '
        'PCC labels are mean per-beat normalized morphology correlations. Native reference ECG and the scoring targets use the same preprocessing. '
        'Gray dashed lines mark reference R peaks. All panels span5.12s; the single1s scale applies to both blocks.')
    (OUT/'caption.tex').write_text('\\caption{'+caption+'}\n')
    (OUT/'figure_ieee.tex').write_text('\\begin{figure*}[!t]\n\\centering\n\\includegraphics[width=\\textwidth]{figures/fig_examples_v3.pdf}\n\\input{figures/source_data/examples_v3_20261030/caption.tex}\n\\label{fig:examples-v3}\n\\end{figure*}\n')
    assert all(sha(p)==h for p,h in preserved.items()),'Old waveform figure changed'
    manifest=dict(training=False,new_model_inference=False,model_lineage='original dual-head seed42; cached per-beat inference',
        cases=logs,source_sha256={str(p):sha(p) for p in sources},original_figure_sha256=preserved,script_sha256=sha(Path(__file__)),
        output_sha256={str(p):sha(p) for p in [*(FIG/f'fig_examples_v3.{e}' for e in ('pdf','svg','png')),OUT/'per_window_results.csv']})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Examples v3 complete:',OUT)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--render-only',action='store_true',
        help='Redraw standalone plotted NPZ data and stored per-window PCCs; skip source audits/delineation')
    args=parser.parse_args()
    if args.render_only:
        metadata=json.loads((OUT/'manifest.json').read_text())['cases']
        plot([(r,old.load_npz(OUT/f"{r['dataset']}_{r['scene']}_plotted.npz")) for r in metadata])
    else:main()
