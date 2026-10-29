"""Standalone Fig. examples v4 (Apnea revision): ONE CSV, no project imports.

Requires Python, numpy, matplotlib only. No model inference, NPZ, NeuroKit2,
raw data, resampling, filtering, normalization or metric recomputation.
CSV waveform values are the final plotted values. *_pcc is the original mean
per-beat Shape-200 PCC, NOT correlation of the assembled display trace.
Reference annotation samples are fixed; only letter layout is automatic.

Usage:
  python plot_examples_v4_standalone.py fig_examples_v4_source.csv
  python plot_examples_v4_standalone.py fig_examples_v4_source.csv --output out/fig_examples_v4
Outputs: PDF, SVG, PNG (600 dpi), 7.16 x 6.3 in.
"""
import os
os.environ.setdefault('MPLBACKEND','Agg')
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
WIDTH,HEIGHT=7.16,6.3
LEFT,RIGHT=.91,.07
METHODS=('rem_net','mmecg_tcn','radarode_mtl')
NAMES={'rem_net':'REM-Net','mmecg_tcn':'Chen-TCN','radarode_mtl':'radarODE-MTL'}
COLORS={'q':'#D9822B','r':'#2A9D8F','s':'#D17DB8','t':'#E0B03A'}
ROWS=('radar','gt','rem_net','mmecg_tcn','radarode_mtl')
HEIGHTS=(.42,.48,.48,.48,.48)
VGAP=.04


def load_csv(path):
    groups={}
    with Path(path).open(newline='') as f:
        for row in csv.DictReader(f):
            groups.setdefault((row['dataset'],row['scene']),[]).append(row)
    expected=(('mmecg','NB'),('mmecg','IB'),('mmecg','SP'),('mmecg','PE'),
              ('crrvs','NB'),('crrvs','IB'),('crrvs','SP'))
    if set(groups)!=set(expected):raise ValueError('Expected exactly seven dataset/scenario groups')
    cases=[]
    for ds,scene in expected:
        rows=sorted(groups[ds,scene],key=lambda r:int(r['sample_index']))
        if [int(r['sample_index']) for r in rows]!=list(range(1024)):
            raise ValueError(f'{ds}/{scene}: need exactly samples 0..1023')
        if not np.allclose([float(r['time_s']) for r in rows],np.arange(1024)/200,rtol=0,atol=1e-12):
            raise ValueError('Time grid must be 200 Hz')
        first=rows[0]
        for key in ('subject','fold','window_index','record_id','input_start_sample','central_start_sample','scene_label',*(m+'_pcc' for m in METHODS)):
            if any(r[key]!=first[key] for r in rows):raise ValueError(f'Inconsistent metadata: {key}')
        arrays={key:np.array([float(r[key]) for r in rows]) for key in ROWS}
        if not all(np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite waveform')
        arrays['gt_r']=np.asarray(json.loads(first['reference_r_samples']),float)
        arrays['reference_annotation_samples']=json.loads(first['reference_annotation_samples'])
        if scene=='NB' and set(arrays['reference_annotation_samples'])!=set('qrst'):
            raise ValueError('First-column reference annotations missing')
        meta=dict(dataset=ds,scene=scene,scene_label=first['scene_label'],
                  panel_pcc={m:float(first[m+'_pcc']) for m in METHODS})
        cases.append((meta,arrays))
    return cases

def reference_labels(fig,ax,events,wave):
    """Automated text layout; endpoint samples and waveform values remain fixed."""
    fig.canvas.draw();renderer=fig.canvas.get_renderer();occupied=[];logs=[]
    wave_path=ax.lines[0].get_path().transformed(ax.transData)
    margin=fig.dpi*.8/72
    # One generic search for all panels/classes, not hand-tuned by example.
    candidates=sorted([(dx,dy) for dy in (3,5,8,11,15,19) for dx in (0,-7,7,-12,12,-18,18,-24,24)],
        key=lambda v:(v[0]**2+v[1]**2,abs(v[0]),v[1],v[0]))
    for event,sample in events.items():
        value=float(np.interp(sample,np.arange(1024),wave));anchor=np.array([sample/200,value])
        pixel=ax.transData.transform(anchor)
        text=ax.text(*anchor,event.upper(),ha='center',va='center',fontsize=6.3,fontweight='bold',color=COLORS[event],zorder=7)
        for dx,dy in candidates:
            label_xy=ax.transData.inverted().transform(pixel+np.array([dx,dy])*fig.dpi/72)
            text.set_position(label_xy);box=text.get_window_extent(renderer).padded(margin)
            inside=box.x0>=ax.bbox.x0 and box.x1<=ax.bbox.x1 and box.y0>=ax.bbox.y0 and box.y1<=ax.bbox.y1
            if inside and not wave_path.intersects_bbox(box,filled=False) and not any(box.overlaps(b) for b in occupied):
                occupied.append(box);break
        else:raise RuntimeError('No nonoverlapping automatic reference label placement')
        ax.plot(*anchor,'o',ms=1.8,color=COLORS[event],mew=0,zorder=6)
        ax.annotate('',xy=anchor,xytext=label_xy,arrowprops=dict(arrowstyle='-',color=COLORS[event],lw=.35,shrinkA=3,shrinkB=1),zorder=6)
        logs.append(dict(event=event,reference_sample=sample,reference_time_seconds=sample/200,waveform_value=value,
            label_offset_x_pt=dx,label_offset_y_pt=dy,source='stored_NeuroKit2_reference',row='gt'))
    return logs

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

def plot(cases, output):
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
                if key=='gt' and col==0:
                    event=a['reference_annotation_samples']
                    annotations.extend(dict(dataset=ds,scene=meta['scene'],**r) for r in reference_labels(fig,ax,event,wave))
                if col==0:
                    name={'radar':'Radar input','gt':'GT ECG',**NAMES}[key]
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
    for ext in ('pdf','svg','png'):fig.savefig(output.with_suffix('.'+ext),dpi=600,facecolor='white')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv',nargs='?',type=Path,default=Path(__file__).with_name('fig_examples_v4_source.csv'))
    parser.add_argument('--output',type=Path,help='Output prefix, default: CSV folder/fig_examples_v4')
    args=parser.parse_args()
    output=args.output or args.csv.parent/'fig_examples_v4'
    output.parent.mkdir(parents=True,exist_ok=True)
    plot(load_csv(args.csv),output)
    print('Saved:',*[str(output.with_suffix('.'+e)) for e in ('pdf','svg','png')])

if __name__=='__main__':main()
