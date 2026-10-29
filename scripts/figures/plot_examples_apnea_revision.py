"""Apnea-only quality-screened illustration; frozen-weight inference, no training.

Selection fixed before inference: lowest quartile of central-reference 10–40 Hz /
0.5–40 Hz Welch energy, then the existing subject/window PCC median rule within
eligible Apnea windows. This is an illustration quality proxy, not clinical SQI.
"""
import os
os.environ.setdefault('MPLBACKEND','Agg')
import sys,json,csv,shutil,warnings
from pathlib import Path
import numpy as np
from scipy.signal import welch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.figures import plot_examples_v4 as v4
OUT=v4.FIG/'source_data/examples_v4_apnea_revision_20261030'

def select_apnea():
    candidates=v4.previous.csvread(v4.previous.CACHE/'selection_candidates.csv')
    rows=[r.copy() for r in candidates if r['dataset']=='crrvs' and r['scene']=='SP']
    for fold in sorted(set(int(r['fold']) for r in rows)):
        ref=v4.previous.old.load_npz(v4.previous.old.SOURCE/'references/crrvs'/f'fold_{fold:02d}.npz')
        for r in rows:
            if int(r['fold'])!=fold:continue
            for k in ('fold','index','subject','start','stop'):r[k]=int(r[k])
            r['window_pcc']=float(r['window_pcc'])
            y=ref['ecg'][r['index'],288:1312]
            f,p=welch(y,fs=200,nperseg=512)
            r['hf_fraction']=float(p[(f>=10)&(f<=40)].sum()/max(p[(f>=.5)&(f<=40)].sum(),1e-12))
    threshold=float(np.quantile([r['hf_fraction'] for r in rows],.25))
    eligible=[r for r in rows if r['hf_fraction']<=threshold]
    subjects=[]
    for sid in sorted(set(r['subject'] for r in eligible)):
        windows=[r for r in eligible if r['subject']==sid]
        subjects.append((sid,float(np.mean([r['window_pcc'] for r in windows]))))
    middle=float(np.median([p for _,p in subjects]));sid,subject_pcc=min(subjects,key=lambda t:(abs(t[1]-middle),t[0]))
    windows=[r for r in eligible if r['subject']==sid];wm=float(np.median([r['window_pcc'] for r in windows]))
    chosen=min(windows,key=lambda r:(abs(r['window_pcc']-wm),r['record_id'],r['start']))
    for r in rows:r['eligible']=r['hf_fraction']<=threshold;r['selected']=r['index']==chosen['index'] and r['fold']==chosen['fold']
    v4.save(OUT/'apnea_quality_candidates.csv',rows)
    chosen={**chosen,'scene_label':'Apnea','subject_pcc':subject_pcc,'scenario_median_pcc':middle,'subject_window_median':wm}
    return chosen,threshold,len(rows),len(eligible)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    # Preserve pre-revision figure/data before replacing only this column.
    backup=OUT/'previous_v4';backup.mkdir(exist_ok=True)
    for e in ('pdf','svg','png'):
        p=v4.FIG/f'fig_examples_v4.{e}'
        if not (backup/p.name).exists():shutil.copy2(p,backup/p.name)
    oldout=v4.OUT
    for name in ('REPORT.md','caption.tex','per_window_results.csv','per_beat_results.csv','manifest.json'):
        if not (backup/name).exists():shutil.copy2(oldout/name,backup/name)
    cases,sources=v4.load_cases();row,threshold,total,eligible=select_apnea()
    print('Selected Apnea before model inference:',row,'quality threshold',threshold,flush=True)
    from scripts.figures import plot_per_beat_morphology as inference
    inference.ASSETS=OUT
    meta,arrays=inference.per_case({k:v for k,v in row.items() if k!='hf_fraction'})
    refpath=v4.previous.old.SOURCE/'references/crrvs'/f"fold_{row['fold']:02d}.npz"
    ref=v4.previous.old.load_npz(refpath);ecg=ref['ecg'][row['index']]
    clinical,_=v4.previous.clinical()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');points=clinical.delineate_signal(ecg,fs=200,method='prominence')
    assert np.array_equal(points['r']-288,arrays['gt_r'])
    for e in 'qrst':arrays['ref_'+e]=np.asarray(points[e],float)-288
    rawpath=ROOT/'experiments/development/structured_decoder_v0_20261017_ab/raw_reference/CR_RVS'/(row['record_id']+'.npz')
    with np.load(rawpath) as z:raw=z['ecg'][row['start']:row['start']+1600]
    assert v4.previous.old.pcc(raw,ecg)>1-1e-10
    assert np.allclose(v4.previous.old.reference_display(ecg,points['r'])[288:1312],arrays['gt'],atol=1e-10,rtol=0)
    np.savez_compressed(OUT/'crrvs_SP_plotted.npz',**arrays)
    replacement=dict(**row,panel_pcc=meta['window_pcc_by_method'],checkpoints=meta['checkpoints'],cached_reference_ecg_path=str(refpath),filtered_raw_record_path=str(rawpath),radar=meta['radar'])
    cases=[(replacement,arrays) if m['dataset']=='crrvs' and m['scene']=='SP' else (m,a) for m,a in cases]
    v4.OUT=OUT;v4.plot(cases)
    logs=[dict(dataset=m['dataset'],scene=m['scene'],subject=m['subject'],fold=m['fold'],window_index=m['index'],record_id=m['record_id'],start=m['start'],central_start=m['start']+288,central_stop=m['start']+1312,selection_representative_pcc=m['window_pcc'],**{k+'_pcc':m['panel_pcc'][k] for k in v4.METHODS}) for m,a in cases]
    v4.save(OUT/'per_window_results.csv',logs)
    beats=v4.previous.csvread(oldout/'per_beat_results.csv');beats=[r for r in beats if not(r['dataset']=='crrvs' and r['scene']=='SP')]
    for method in v4.METHODS:
        for i,((left,right),shape,target) in enumerate(zip(arrays['beat_bounds'],arrays[method+'_shapes'],arrays['reference_shapes'])):
            beats.append(dict(dataset='crrvs',scene='SP',method=method,subject=row['subject'],record_id=row['record_id'],window_index=row['index'],beat_index=i,reference_left=row['start']+288+int(left),reference_right=row['start']+288+int(right),context_start=meta['reference_context_starts'][i],pcc=float(v4.previous.score_shapes(shape[None],target[None])[0])))
    v4.save(OUT/'per_beat_results.csv',beats)
    oldrows=v4.previous.csvread(backup/'per_window_results.csv')
    for a,b in zip(oldrows,logs):
        if b['dataset']=='crrvs' and b['scene']=='SP':continue
        assert all(str(b[k])==a[k] for k in a),('Other selection/score changed',b)
    report=f'''# Apnea-only example revision

Only CR_RVS Apnea is replaced. Other six windows, arrays, PCC scores and all layout settings remain unchanged. This is a quality-screened illustration, not an unqualified median example of all Apnea recordings. No training; original dual-head seed42 frozen checkpoints, NOT phase_pd.

## Selection fixed before new inference
All {total} existing Apnea test-window candidates were scored using central 5.12s reference Welch power (nperseg512): 10–40Hz / 0.5–40Hz. Keep the lowest quartile (threshold {threshold:.8f}; {eligible} candidates). This is a high-frequency-content proxy for display clarity, not validated clinical signal quality; high-frequency power may include genuine morphology. Among eligible windows use subject-mean PCC nearest the eligible subject median, then that subject's median-PCC window, ties by subject/record/start. No baseline outputs or newly inferred panel scores enter selection. Full candidate audit: apnea_quality_candidates.csv. This selection revision was requested after inspecting the earlier illustration; not a prospective benchmark protocol.

Selected subject {row['subject']}, fold {row['fold']}, record {row['record_id']}, test row {row['index']}; input [{row['start']},{row['stop']}), displayed [{row['start']+288},{row['start']+1312}). Reference HF fraction {row['hf_fraction']:.6f}. Selection representative PCC {row['window_pcc']:.8f}; independent per-beat panel scores: {json.dumps(meta['window_pcc_by_method'])}.

## Reference and inference integrity
Native displayed ECG and every Shape-200 target use the SAME cached reference; checked reference_display equality and raw-record filtered ECG positive-affine agreement PCC>1−1e−10. No additional ECG filter, smoothing, timing shift or noise suppression. Radar preprocessing remains 0.8–3Hz, channel chosen without ECG. All methods use the same test window, their own frozen shape heads and independent beat-centred 8s contexts (no replication), reference RR placement. Panel scores use unchanged mean per-beat Shape-200 PCC. Native ECG appearance and these scores have different sampling/normalization, as documented in the prior report.

No REM-Net markers. Optional first-column GT reference annotations unchanged. Timing evaluated separately. Baselines remain Chen-TCN and radarODE-MTL, the highest two adapted Table-I morphology baselines. Prior figures and source audits retained in previous_v4/. Full original report and caption there. Other six per-window rows checked exactly unchanged.
'''
    (OUT/'REPORT.md').write_text(report)
    cap=(backup/'caption.tex').read_text().replace('Each scenario uses the same median-rule test subject/window as the previous examples;', 'Six scenarios retain the original median-rule test examples; Apnea uses the same median rule after a reference high-frequency-content lower-quartile screen (see REPORT);')
    cap=cap.replace('PCC values are the original mean per-beat morphology scores.','PCC values are the mean per-beat morphology scores; Apnea scores are updated for the replacement window.')
    (OUT/'caption.tex').write_text(cap)
    (OUT/'manifest.json').write_text(json.dumps(dict(training=False,model_inference=True,changed_scenarios=['CR_RVS Apnea'],selection_rule=__doc__,selected=row,cases=[m for m,a in cases],source_sha256={str(p):v4.sha(p) for p in sources+[refpath,rawpath]},script_sha256=v4.sha(__file__),outputs={e:v4.sha(v4.FIG/f'fig_examples_v4.{e}') for e in ('pdf','svg','png')}),indent=2))
    # Discoverability at the original report location; full prior report archived.
    (oldout/'REPORT.md').write_text('Apnea-only revision: see ../examples_v4_apnea_revision_20261030/REPORT.md\n\n'+report)
    (oldout/'caption.tex').write_text(cap)
    print('Completed:',OUT,flush=True)

if __name__=='__main__':main()
