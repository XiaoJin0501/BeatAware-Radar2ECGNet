"""Reuse exact WSST keys/targets and baseline native RCG preprocessing."""
import bridge
import numpy as np
import torch


def raw_radar(dataset, rid):
    # Existing A caches MUST already exist; record_data cannot create old outputs.
    if not (bridge.old_environment.A/'raw_reference'/dataset/f'{rid}.npz').exists():
        raise FileNotFoundError('Historical references must not be generated here')
    data = bridge.record_data(dataset, rid, with_radar=True)
    if not np.isfinite(data['radar']).all():
        raise ValueError('No unapproved missing-radar imputation')
    return data['radar']


def aligned_td(dataset, radar, start, stop):
    if stop-start != 1600:
        raise ValueError('Exact 8s WSST support required')
    return bridge.radar_window(dataset, radar, start)


class TrainWindows(bridge.old_data.TrainWindows):
    def __init__(self, fold, dual, limit=None):
        super().__init__(fold, limit)
        self.dual = dual
        self.radars = {rid:raw_radar('MMECG', rid) for rid in self.references} if dual else {}

    def __getitem__(self, i):
        item = super().__getitem__(i)
        if self.dual:
            row = self.rows[i]
            item['td'] = torch.from_numpy(aligned_td('MMECG', self.radars[row['record_id']], row['start'], row['stop']))
        return item


def inventory(fold):
    import json
    path = bridge.old_environment.OUT/'data'/f'fold_{fold:02d}'/'validation_inventory.json'
    rows = json.loads(path.read_text())
    # Read only; do not write derived features into the previous experiment.
    bridge.old_run.preload_validation(rows, bridge.old_data.statistics(fold))
    for row in rows:
        radar = raw_radar('MMECG', row['record_id'])
        row['_td'] = np.stack([aligned_td('MMECG', radar, start, start+1600) for start in row['starts']])
        assert row['_td'].shape == (len(row['starts']), 50, 1600)
    return rows
