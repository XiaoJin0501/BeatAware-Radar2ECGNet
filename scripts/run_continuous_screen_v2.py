"""Disjoint fold workers; automatic full36-run report, confirmation blocked."""
import argparse
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/development/event_conditioned_screen_20261019'))
import environment as env
from freeze import verify,sha
from src.utils.atomic_state import atomic_json

def worker(fold):
    import torch
    from data import TrainWindows,prepare_validation
    from run import train_one,preload_validation,code_state
    code_state();verify()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    log=env.OUT/f'fold_{fold:02d}.log'
    with log.open('a',buffering=1) as stream:
        sys.stdout=sys.stderr=stream
        try:
            data=TrainWindows(fold).preload()
            inventory=prepare_validation(fold,env.OUT/'data'/f'fold_{fold:02d}')
            preload_validation(inventory,data.stats)
            for seed in (42,43,44):
                for arm in (1,2,3,4):
                    train_one(fold,seed,arm,data,inventory,'cuda:0')
        except BaseException:
            traceback.print_exc();raise

def main():
    from run import code_state
    commit=code_state();verify()
    if not (env.OUT/'AUTHOR_APPROVED.json').exists():raise RuntimeError('Run prepare_v2 first')
    context=multiprocessing.get_context('spawn')
    processes=[context.Process(target=worker,args=(f,),name=f'v2-fold{f}') for f in (1,2,3)]
    atomic_json(env.OUT/'LAUNCH.json',dict(commit=commit,protocol_version='v2',pid=os.getpid(),
        metric_freeze_sha256=sha(env.OUT/'METRIC_FREEZE.json'),folds=[1,2,3],
        parallelism='one isolated process per fold, exactly12 unique runs each',
        expected_runs=36,confirmation_authorized=False,test_access=False))
    try:
        for p in processes:p.start()
        while any(p.is_alive() for p in processes):
            complete=list((env.OUT/'runs').glob('fold_*/seed_*/arm_*/COMPLETE.json'))
            atomic_json(env.OUT/'progress.json',dict(protocol_version='v2',completed=len(complete),total=36,
                worker_pids=[p.pid for p in processes],worker_alive=[p.is_alive() for p in processes],
                confirmation_authorized=False))
            time.sleep(5)
        for p in processes:p.join()
        if any(p.exitcode for p in processes):
            raise RuntimeError('Fold worker failed; stopped, no partial comparison: '+str([p.exitcode for p in processes]))
        from v2_report import generate
        generate()
        atomic_json(env.OUT/'FINISHED.json',dict(runs=36,complete=True,confirmation_authorized=False))
        print('COMPLETE v2:36 fresh runs, complete report; confirmation BLOCKED',flush=True)
    except BaseException as e:
        atomic_json(env.OUT/'QUEUE_FAILED.json',dict(error=str(e),confirmation_authorized=False))
        for p in processes:
            if p.is_alive():p.terminate();p.join()
        raise

if __name__=='__main__':main()
