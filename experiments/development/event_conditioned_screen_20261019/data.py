"""Lazy exact-key WSST training input; raw record targets, train-only scaling."""
import json
import environment as env
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from src.data.wsst8s_subject_pool import SubjectWSSTPool, window_key
from src.data.mmecg_dataset import _gaussian_mask
from rawio import split_rows, record_data, radar_window, MM
from layout import approved_layout


def reference(rid):
    # Reading already-audited raw-reference artifacts must never write to old A.
    path = env.A / 'raw_reference/MMECG' / f'{rid}.npz'
    if not path.exists():
        raise FileNotFoundError(f'Raw reference unavailable; no historical writes allowed: {path}')
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def statistics(fold):
    path = env.A / 'training_statistics/MMECG' / f'fold_{fold:02d}.json'
    s = json.loads(path.read_text())
    tr, va = split_rows('MMECG', fold, 'train'), split_rows('MMECG', fold, 'val')
    assert s['train_subjects'] == sorted({r['subject_id'] for r in tr})
    assert not set(s['train_subjects']) & {r['subject_id'] for r in va}
    return s


def local_reference(data, start):
    return {k: data[f'event_{k}'][(data['event_r'] >= start) &
                               (data['event_r'] < start+1600)]-start
            for k in ('q', 'r', 's', 't')}


def event_targets(points):
    return np.stack([_gaussian_mask(np.asarray(points[k])[np.isfinite(points[k])].astype(int),
                                    1600, 4.) for k in ('q', 'r', 's', 't')]).astype(np.float32)


class TrainWindows(Dataset):
    def __init__(self, fold, limit=None):
        self.stats = statistics(fold)
        self.rows = split_rows('MMECG', fold, 'train')
        if limit is not None:
            # Deterministic coverage across source ordering; no RNG consumed.
            ix = np.linspace(0, len(self.rows)-1, limit, dtype=int)
            self.rows = [self.rows[i] for i in ix]
        self.pool = SubjectWSSTPool(MM / 'processed/wsst8s_subject_pool')
        self.references = {r['record_id']: reference(r['record_id']) for r in self.rows}
        self.features = None

    def preload(self):
        """Read each exact-key float16 representation once; values unchanged."""
        grouped = {}
        for i, row in enumerate(self.rows):
            loc = self.pool.index[window_key(row['subject_id'], row['record_id'], row['start'], row['stop'])]
            grouped.setdefault(loc.path, []).append((i, loc.row))
        self.features = np.empty((len(self.rows), 50, 71, 240), np.float16)
        for path, pairs in grouped.items():
            with h5py.File(path, 'r') as h:
                for i, j in pairs:
                    self.features[i] = h['wsst'][j]
        self.targets = []
        self.events = []
        for row in self.rows:
            ref = self.references[row['record_id']]
            a = row['start']
            self.targets.append(((ref['ecg'][a:a+1600]-self.stats['median'])/self.stats['iqr']).astype(np.float32)[None])
            self.events.append(event_targets(local_reference(ref, a)))
        return self

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        if self.features is not None:
            x = self.features[i].astype(np.float32)
        else:
            loc = self.pool.index[window_key(row['subject_id'], row['record_id'], row['start'], row['stop'])]
            with h5py.File(loc.path, 'r') as h:
                x = np.asarray(h['wsst'][loc.row], np.float32)
        if self.features is not None:
            return {'wsst': torch.from_numpy(x), 'target': torch.from_numpy(self.targets[i]),
                    'events': torch.from_numpy(self.events[i]), 'index': i}
        ref = self.references[row['record_id']]
        a = row['start']
        y = ((ref['ecg'][a:a+1600]-self.stats['median'])/self.stats['iqr']).astype(np.float32)
        if len(y) != 1600 or x.shape != (50, 71, 240):
            raise ValueError('Misaligned support')
        return {'wsst': torch.from_numpy(x), 'target': torch.from_numpy(y[None]),
                'events': torch.from_numpy(event_targets(local_reference(ref, a))),
                'index': i}


def compute_feature(task):
    """Exact A radar preprocessing and WSST, no ECG-dependent input selection."""
    rid, start = task
    from inference import one_wsst
    data = record_data('MMECG', rid, with_radar=True)
    return one_wsst(radar_window('MMECG', data['radar'], start))


def prepare_validation(fold, out, record_limit=None, workers=8):
    """All A1 positions, tail exception included; optional record subset is smoke-only."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    from src.utils.atomic_state import atomic_npz, atomic_json
    rows = {r['record_id']: r for r in split_rows('MMECG', fold, 'val')}
    rows = [rows[k] for k in sorted(rows)]
    if record_limit is not None:
        rows = rows[:record_limit]
    prepared = []
    for row in rows:
        data = reference(row['record_id'])
        segments = approved_layout(len(data['ecg']))
        starts = sorted({s['input_start'] for s in segments})
        dest = out / 'validation_features' / f"{row['record_id']}.npz"
        if not dest.exists():
            context = multiprocessing.get_context('spawn')
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                features = list(pool.map(compute_feature, [(row['record_id'], s) for s in starts]))
            atomic_npz(dest, dict(starts=np.asarray(starts), features=np.stack(features)))
        with np.load(dest) as z:
            assert np.array_equal(z['starts'], starts)
        prepared.append(dict(**row, samples=len(data['ecg']), segments=segments,
                             starts=starts, features=str(dest)))
        print(f"VALIDATION FEATURES {row['record_id']} {len(starts)} windows ready", flush=True)
    atomic_json(out / 'validation_inventory.json', prepared)
    return prepared


