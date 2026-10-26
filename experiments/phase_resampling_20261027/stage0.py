"""Validate ALL val windows in two folds/dataset. No test/training API."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys
import csv
import json
import hashlib
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
import h5py
from tqdm.auto import tqdm

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.phase_resampling_20261027.architecture import AnchorConditionedREM,phase_alignment,phase_resample
from experiments.phase_resampling_20261027.frozen_bridge import extract_points,shape_target,center_bounds,SOURCES

OUT=Path(__file__).resolve().parent/'stage0'
DATA={'mmecg':Path('/home/qhh2237/Datasets/MMECG/processed/loso_v2'),
      'crrvs':Path('/home/qhh2237/Datasets/CR_RVS/processed/q1_continuous_kfold')}

def reference(ecg):
    points=extract_points(ecg);bounds=center_bounds(points['r'],len(ecg))
    return bounds,shape_target(ecg,points['r'])

def main():
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    config=json.loads((OUT.parent/'spec.json').read_text())
    assert config['stage']==0 and not config['test_access_authorized']
    rows=[];summaries=[];sourcefiles=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for ds,root in DATA.items():
            for fold in (1,2):
                path=root/f'fold_{fold:02d}/val.h5'
                assert path.name=='val.h5'
                with h5py.File(path,'r') as f:ecg=f['ecg'][:,0].astype(np.float32)
                sourcefiles.append(dict(dataset=ds,fold=fold,path=str(path),n=len(ecg),split='val',
                    ecg_array_sha256=hashlib.sha256(ecg.tobytes()).hexdigest()))
                labels=list(tqdm(pool.map(reference,ecg,chunksize=16),total=len(ecg),
                    desc=f'{ds} fold{fold:02d} identity references',disable=not sys.stderr.isatty(),leave=False))
                bounds=np.array([x[0] for x in labels],np.float32)
                targets=np.array([x[1] for x in labels],np.float32)
                with torch.no_grad():
                    ti=phase_alignment(torch.from_numpy(bounds))
                    prediction=phase_resample(torch.from_numpy(ecg[:,None]),ti)[:,0]
                    lo=prediction.min(-1,keepdim=True).values;hi=prediction.max(-1,keepdim=True).values
                    prediction=torch.where(hi-lo<1e-8,torch.zeros_like(prediction),(prediction-lo)/(hi-lo).clamp_min(1e-8)).numpy()
                diff=np.max(abs(prediction-targets),axis=1)
                a=prediction-prediction.mean(1,keepdims=True);b=targets-targets.mean(1,keepdims=True)
                den=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
                pcc=np.divide((a*b).sum(1),den,out=np.full(len(a),np.nan),where=den>1e-12)
                good=(pcc>=.999)&(diff<1e-3)
                for i in range(len(ecg)):
                    rows.append(dict(dataset=ds,fold=fold,split='val',window=i,reference_start=int(bounds[i,0]),
                        reference_end_exclusive=int(bounds[i,1]),pcc=float(pcc[i]),max_abs_diff=float(diff[i]),passed=bool(good[i])))
                summaries.append(dict(dataset=ds,fold=fold,windows=len(ecg),failures=int((~good).sum()),
                    min_pcc=float(np.nanmin(pcc)),max_abs_diff=float(diff.max())))
                print(summaries[-1],flush=True)
    with (OUT/'identity_windows.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tests=subprocess.run([sys.executable,'-B','-m','pytest','-q','tests/test_anchor_conditioned_phase.py'],
                         cwd=ROOT,text=True,capture_output=True)
    (OUT/'unit_tests.txt').write_text(tests.stdout+tests.stderr)
    parameters={}
    for variant in ('interp','phase','phase_p','phase_pd'):
        torch.manual_seed(42);m=AnchorConditionedREM(variant)
        parameters[variant]=dict(total=sum(p.numel() for p in m.parameters()),
            shape_head=sum(p.numel() for p in m.shape_head.parameters()),input_channels=m.shape_head[0].in_channels)
    passed=tests.returncode==0 and all(x['failures']==0 for x in summaries)
    result=dict(stage=0,passed=passed,T1=summaries,T2_T3_pytest_passed=tests.returncode==0,parameters=parameters,
        frozen_sources=SOURCES,validation_sources=sourcefiles,training_started=False,test_accessed=False,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),config=config)
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(tests.stdout,flush=True);print('Stage0 PASS' if passed else 'Stage0 FAIL',flush=True)
    if not passed:raise SystemExit(1)

if __name__=='__main__':main()
