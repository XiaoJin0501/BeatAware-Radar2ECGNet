"""Three fold workers, then deterministic complete report. Never starts Chen."""
import os
import subprocess
import sys
import time
import bridge
import lock


def main():
    lock.verify()
    # Guard against stale launch while earlier screening or another GPU job runs.
    if not (bridge.old_environment.OUT/'FINISHED.json').exists():raise RuntimeError('v2 not finished')
    workers=[];logs=[]
    for fold in (1,2,3):
        log=(bridge.OUT/f'fold_{fold:02d}.log').open('a',buffering=1);logs.append(log)
        workers.append(subprocess.Popen([sys.executable,'-B',str(bridge.HERE/'train.py'),'--fold',str(fold)],
            cwd=bridge.ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}))
    try:
        while any(p.poll() is None for p in workers):
            completed=len(list((bridge.OUT/'runs').glob('fold_*/seed_*/*/COMPLETE.json')))
            bridge.atomic_json(bridge.OUT/'progress.json',dict(completed=completed,total=27,
                worker_pids=[p.pid for p in workers],worker_alive=[p.poll() is None for p in workers]))
            if any(p.poll() not in (None,0) for p in workers):raise RuntimeError('A fold worker failed; no comparison/report generated')
            time.sleep(5)
        if any(p.returncode!=0 for p in workers):raise RuntimeError('Worker failure')
        bridge.atomic_json(bridge.OUT/'progress.json',dict(completed=27,total=27,worker_alive=[False]*3))
        from report_final import generate
        generate()
    finally:
        for p in workers:
            if p.poll() is None:p.terminate()
        for p in workers:
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for log in logs:log.close()


if __name__=='__main__':main()
