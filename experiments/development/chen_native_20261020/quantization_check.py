"""Validation-only codec round trip; no model, optimizer, GPU, or test access."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import json
import subprocess
import numpy as np
import torch
import bridge
from data import reference, statistics
from model import mulaw_encode, mulaw_decode
from unified_metrics import continuous_pair


def main():
    torch.set_num_threads(1)
    destination = bridge.OUT / 'validation_quantization_check'
    if destination.exists():
        raise FileExistsError(destination)
    rows = []
    folds = []
    for fold in (1, 2, 3):
        stats = statistics(fold)
        subjects = {}
        for row in bridge.split_rows('MMECG', fold, 'val'):
            subjects[row['record_id']] = row['subject_id']
        levels = set()
        samples = clipped = 0
        squared_z = squared_raw = 0.
        local = []
        for rid, subject in sorted(subjects.items()):
            raw = reference(rid)['ecg']
            z = ((raw - stats['median']) / stats['iqr']).astype(np.float32)
            u = z / stats['mu_scale']
            code = mulaw_encode(torch.from_numpy(u))
            recovered_z = mulaw_decode(code).numpy() * stats['mu_scale']
            recovered_raw = recovered_z * stats['iqr'] + stats['median']
            levels.update(code.unique().tolist())
            count = len(raw)
            nclip = int(np.count_nonzero(np.abs(u) > 1))
            zmetrics = continuous_pair(recovered_z, z)
            rawmetrics = continuous_pair(recovered_raw, raw)
            entry = dict(fold=fold, record_id=rid, subject_id=subject,
                         samples=count, clipped=nclip, used_levels=int(code.unique().numel()),
                         normalized=zmetrics, original=rawmetrics)
            rows.append(entry)
            local.append(entry)
            samples += count
            clipped += nclip
            squared_z += float(np.square(recovered_z.astype(float)-z).sum())
            squared_raw += float(np.square(recovered_raw.astype(float)-raw).sum())
        # Equal-record Fisher-z PCC; RMSE separately pooled over native samples.
        pcc = [r['normalized']['pcc'] for r in local]
        folds.append(dict(fold=fold, records=len(local), subjects=sorted(set(subjects.values())),
                          train_statistics=stats, samples=samples, clipped=clipped,
                          clipping_fraction=clipped/samples, used_levels=len(levels),
                          level_indices=sorted(levels),
                          record_fisher_pcc=float(np.tanh(np.mean(np.arctanh(np.clip(pcc,-1+1e-12,1-1e-12))))),
                          sample_pooled_rmse_z=float(np.sqrt(squared_z/samples)),
                          sample_pooled_rmse_original=float(np.sqrt(squared_raw/samples))))
    result = dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=bridge.HERE,text=True).strip(),
                  frozen_metrics=bridge.verify_frozen(), training=False, cuda_initialized=torch.cuda.is_initialized(),
                  support='complete validation reference records, no window cache; no time alignment',
                  pcc_aggregation='equal-record Fisher-z mean; per-record values retained',
                  rmse_aggregation='sample pooled within fold; original units are dataset reference units, not asserted mV',
                  folds=folds, records=rows)
    destination.mkdir(parents=True)
    bridge.atomic_json(destination/'results.json', result)
    print(json.dumps([{k:v for k,v in f.items() if k not in ('train_statistics','level_indices')} for f in folds],indent=2))


if __name__ == '__main__':
    main()
