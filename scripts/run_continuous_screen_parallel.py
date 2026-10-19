"""Scheduling only: disjoint approved folds2/3 alongside existing fold1 queue.

All numerical code comes from the already frozen committed run.train_one.
The original queue is suspended after11 fold1 completions if necessary, then
resumed after workers finish, so it never duplicates their fold2/3 jobs.
"""
import argparse
import json
import multiprocessing
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/development/event_conditioned_screen_20261019'))
import environment as env
from freeze import verify, sha
from src.utils.atomic_state import atomic_json

def worker(fold):
    import torch
    from data import TrainWindows, prepare_validation
    from run import train_one, preload_validation
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = TrainWindows(fold).preload()
    inventory = prepare_validation(fold, env.OUT/'data'/f'fold_{fold:02d}')
    preload_validation(inventory, data.stats)
    for seed in (42, 43, 44):
        for arm in (1, 2, 3, 4):
            train_one(fold, seed, arm, data, inventory, 'cuda:0')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--main-pid', required=True, type=int)
    args = parser.parse_args()
    command = Path(f'/proc/{args.main_pid}/cmdline').read_bytes().decode().replace('\0', ' ')
    if 'event_conditioned_screen_20261019/run.py --stage screen' not in command:
        raise RuntimeError('Refusing to suspend unrelated process')
    if len(list((env.OUT/'runs/fold_01').glob('seed_*/arm_*/COMPLETE.json'))) >= 11:
        raise RuntimeError('Launch coordinator earlier; do not risk duplicate ownership')
    from run import code_state
    commit = code_state()
    verify()
    context = multiprocessing.get_context('spawn')
    jobs = [context.Process(target=worker, args=(fold,), name=f'approved-fold{fold}') for fold in (2, 3)]
    paused = False
    atomic_json(env.OUT/'SCHEDULING.json', dict(numerical_code_changed=False,
        metric_freeze_sha256=sha(env.OUT/'METRIC_FREEZE.json'), orchestration_commit=commit,
        main_pid=args.main_pid, worker_folds=[2, 3],
        seed_values=[42, 43, 44], arms=[1, 2, 3, 4],
        exclusion='main suspended at11 fold1 completions until workers exit; completed markers make main skip fold2/3',
        confirmation_authorized=False))
    try:
        for process in jobs:
            process.start()
        while any(process.is_alive() for process in jobs):
            fold1 = len(list((env.OUT/'runs/fold_01').glob('seed_*/arm_*/COMPLETE.json')))
            if fold1 >= 11 and not paused:
                os.kill(args.main_pid, signal.SIGSTOP)
                paused = True
                print('SCHEDULER: original queue paused before fold2 ownership', flush=True)
            done = len(list((env.OUT/'runs').glob('fold_*/seed_*/arm_*/COMPLETE.json')))
            atomic_json(env.OUT/'parallel_progress.json', dict(completed=done, total=36,
                original_queue_paused=paused, worker_pids=[p.pid for p in jobs],
                workers_alive=[p.is_alive() for p in jobs], confirmation_authorized=False))
            time.sleep(2)
        for process in jobs:
            process.join()
        atomic_json(env.OUT/'SCHEDULING_COMPLETE.json', dict(worker_exitcodes=[p.exitcode for p in jobs],
            original_queue_resumed=True, numerical_sources_unchanged=True,
            recover_failures='Original epoch-resilient queue resumes incomplete jobs if any worker failed'))
    finally:
        # No worker may overlap original queue after it resumes, even on Ctrl-C.
        for process in jobs:
            if process.is_alive():
                process.terminate()
                process.join()
        if paused:
            os.kill(args.main_pid, signal.SIGCONT)
        print('SCHEDULER: workers exited; original queue owns completion/report', flush=True)

if __name__ == '__main__':
    main()
