"""Compile only selected, unchanged definitions from the original dual-head code.

No training runner is imported; no dataset/checkpoint/test results are opened.
Source hashes are exported so this reuse is auditable.
"""
import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import warnings
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.signal import find_peaks

FROZEN=Path(os.environ.get('REM_PHASE_FROZEN_ROOT','/tmp/rem_dual_head_formal'))
SOURCES={}

def definitions(relative,names,namespace):
    path=FROZEN/relative
    content=path.read_text();SOURCES[str(path)]=hashlib.sha256(content.encode()).hexdigest()
    nodes=[n for n in ast.parse(content).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes}!=set(names):raise RuntimeError(f'Missing frozen definitions: {path}')
    code=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]))
    exec(compile(code,str(path),'exec'),namespace)

namespace=dict(torch=torch,nn=nn,F=F,np=np,warnings=warnings,find_peaks=find_peaks,
               WIN=1600,FS=200,LATENT_T=400,FIDUCIALS=('q','r','s','t'))
definitions('src/models/continuous/ce_net.py',{'WSSTStem','_TCN'},namespace)
definitions('src/models/continuous/dual_head_rem.py',{'DualHeadREM'},namespace)
definitions('src/data/mmecg_dataset.py',{'_center_cycle_bounds_from_rpeaks'},namespace)
clinical_path=FROZEN/'src/utils/clinical_endpoints.py'
spec=importlib.util.spec_from_file_location('phase_frozen_clinical',clinical_path)
clinical=importlib.util.module_from_spec(spec);spec.loader.exec_module(clinical)
SOURCES[str(clinical_path)]=hashlib.sha256(clinical_path.read_bytes()).hexdigest()
namespace['delineate_signal']=clinical.delineate_signal
definitions('scripts/run_dual_head_interaction.py',{'shape_target','structured_decode','extract_points','losses'},namespace)
FrozenDualHeadREM=namespace['DualHeadREM']
shape_target=namespace['shape_target']
structured_decode=namespace['structured_decode']
extract_points=namespace['extract_points']
center_bounds=namespace['_center_cycle_bounds_from_rpeaks']
losses=namespace['losses']
