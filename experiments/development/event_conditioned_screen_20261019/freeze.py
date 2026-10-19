"""Version and hash the metric contract BEFORE A supplement or screening."""
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import environment as env
from src.utils.atomic_state import atomic_json

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def dependencies():
    paths = [env.A/k for k in ('unified_metrics.py', 'bootstrap.py', 'rawio.py', 'layout.py', 'inference.py', 'structured.py')]
    paths += [env.A.parent/'structured_decoder_v0_20261016_a3/metrics.py']
    paths += [env.FROZEN/k for k in (
        'scripts/run_dual_head_interaction.py', 'scripts/train_continuous_ecg_pilot.py',
        'scripts/evaluate_formal_anchor_intervals.py', 'scripts/build_crrvs_wsst_shape_h5.py',
        'src/models/continuous/ce_net.py', 'src/data/mmecg_dataset.py',
        'src/data/wsst8s_subject_pool.py', 'src/utils/clinical_endpoints.py', 'src/utils/atomic_state.py')]
    paths += [env.RESOURCE_ROOT/'archive/legacy_scripts/misc/precompute_wsst8s_mmecg.py',
              Path('/home/qhh2237/Datasets/MMECG/04B_preprocess_records.py')]
    return paths

def freeze():
    target = env.OUT/'METRIC_FREEZE.json'
    if target.exists():
        return verify()
    local = {p.name: sha(p) for p in sorted(env.HERE.iterdir()) if p.suffix in ('.py', '.json')}
    lock = dict(version='continuous_screen_v2_record150_harmonic', source=local,
                packages={name: version(name) for name in ('torch', 'numpy', 'scipy', 'neurokit2', 'h5py', 'ssqueezepy')},
                dependencies={str(p): sha(p) for p in dependencies()},
                metric_contract=json.loads((env.HERE/'metric_contract.json').read_text()),
                created_before_A_supplement=False, A_supplement_is_frozen_v1=True,created_before_screen_training=True,
                changes_after_start_forbidden=True)
    atomic_json(target, lock)
    return lock

def verify():
    lock = json.loads((env.OUT/'METRIC_FREEZE.json').read_text())
    assert all(version(name) == recorded for name, recorded in lock['packages'].items()), 'Frozen package version changed'
    for name, digest in lock['source'].items():
        assert sha(env.HERE/name) == digest, f'Frozen source changed: {name}'
    for name, digest in lock['dependencies'].items():
        assert sha(name) == digest, f'Frozen dependency changed: {name}'
    return lock

if __name__ == '__main__':
    freeze()
    print('PASS: metric/architecture/training sources and dependencies frozen')
