"""Recoverable v1 run archival and v2 immutable preflight, no model inference."""
import json
import os
from pathlib import Path
import shutil
import environment as env
from freeze import sha,freeze,verify
from run import code_state
from src.utils.atomic_state import atomic_json

ARCHIVE=env.RESOURCE_ROOT/'archive/experiments/event_conditioned_screen_v1_20261020'

def main():
    commit=code_state()
    assert not (env.OUT/'runs').exists(),'Fresh v2 output required'
    assert (env.V1_OUT/'AUTHOR_PAUSED.json').exists(),'v1 must remain paused'
    assert (env.OUT/'integration_check/COMPLETE.json').exists(),'Integration check required'
    assert json.loads((env.OUT/'integration_check/COMPLETE.json').read_text())['passed']
    v1runs=env.V1_OUT/'runs'
    archived=ARCHIVE/'runs'
    if v1runs.exists():
        if archived.exists():raise RuntimeError('Archive destination already exists')
        # No active legacy training can retain open handles to moved outputs.
        for p in Path('/proc').iterdir():
            if not p.name.isdigit() or int(p.name)==os.getpid():continue
            try:cmd=(p/'cmdline').read_bytes().decode(errors='replace')
            except (FileNotFoundError,PermissionError,ProcessLookupError):continue
            if 'python' in cmd and ('event_conditioned_screen_20261019/run.py' in cmd or 'run_continuous_screen_parallel.py' in cmd):
                raise RuntimeError('Legacy screening process remains active')
        before={str(p.relative_to(v1runs)):sha(p) for p in sorted(v1runs.rglob('*')) if p.is_file()}
        ARCHIVE.mkdir(parents=True,exist_ok=True)
        atomic_json(ARCHIVE/'ARCHIVE_MANIFEST.json',dict(source=str(v1runs),destination=str(archived),
            excluded_from_v2=True,files=before,v1_implementation_commit='4300687',v2_commit=commit,
            legacy_identity_rule='anonymous SHA identity uses original source directory path; path translation is source->destination'))
        v1runs.rename(archived)
        assert all(sha(archived/k)==digest for k,digest in before.items())
        atomic_json(ARCHIVE/'VERIFIED.json',dict(passed=True,files=len(before),all_file_hashes_identical=True,recoverable=True))
    else:
        assert (ARCHIVE/'VERIFIED.json').exists() and archived.exists()
    # Copy cached radar features only; NO weights, history, metrics or best-epoch reuse.
    for fold in (1,2,3):
        source=env.V1_OUT/'data'/f'fold_{fold:02d}'/'validation_features'
        target=env.OUT/'data'/f'fold_{fold:02d}'/'validation_features'
        if not target.exists():shutil.copytree(source,target)
        for p in source.glob('*.npz'):assert sha(p)==sha(target/p.name)
    model_source=Path('/home/qhh2237/Projects/BeatAware-Radar2ECGNet-continuous-screen/experiments/development/event_conditioned_screen_20261019')
    unchanged={name:sha(env.HERE/name) for name in ('model.py','data.py','metrics.py')}
    assert all(digest==sha(model_source/name) for name,digest in unchanged.items())
    from rawio import split_path
    input_hashes={}
    for fold in (1,2,3):
        for split in ('train','val'):
            p=split_path('MMECG',fold,split);input_hashes[str(p)]=sha(p)
        p=env.A/'training_statistics/MMECG'/f'fold_{fold:02d}.json'
        input_hashes[str(p)]=sha(p)
        for p in (env.OUT/'data'/f'fold_{fold:02d}'/'validation_features').glob('*.npz'):
            input_hashes[str(p)]=sha(p)
    atomic_json(env.OUT/'INPUT_FINGERPRINTS.json',input_hashes)
    freeze();verify()
    atomic_json(env.OUT/'AUTHOR_APPROVED.json',dict(protocol_version='v2',author_date='2026-10-20',
        approved_runs=36,folds=[1,2,3],seeds=[42,43,44],arms=[1,2,3,4],commit=commit,
        source_worktree_clean=True,confirmation_authorized=False,test_access=False,
        frozen_sha256=sha(env.OUT/'METRIC_FREEZE.json'),unchanged_v1_numeric_sources=unchanged,
        old_runs_archive=str(archived),preflight=True))
    print('PASS: v1 archived and verified, v2 cache-only reuse, rules frozen; no training yet',flush=True)

if __name__=='__main__':main()
