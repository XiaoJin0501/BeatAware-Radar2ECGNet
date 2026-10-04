#!/usr/bin/env python3
"""Run the registered folds 3/7 checkpoint-balance mechanism pilot."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.run_dual_head_interaction as core  # noqa: E402
from src.utils.atomic_state import atomic_json  # noqa: E402

TAG = "q1_dual_head_checkpoint_balance"
SPEC = ROOT / "experiments/specs/q1_dual_head_checkpoint_balance.json"
OUT = Path(os.environ.get("REM_CHECKPOINT_BALANCE_OUT", ROOT/"experiments_mmecg"/TAG))
CONTROL = Path(os.environ.get(
    "REM_DUAL_HEAD_FULL_OUT", ROOT/"experiments_mmecg/q1_dual_head_full_loso_seed42"))
FOLDS = (3, 7)


def preflight():
    spec=json.loads(SPEC.read_text())
    if not spec["status"].startswith("PREFLIGHT_PASSED"):
        raise SystemExit(f"spec not launchable: {spec['status']}")
    commit=spec.get("implementation_commit")
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    if not commit or subprocess.run(["git","merge-base","--is-ancestor",commit,head],cwd=ROOT).returncode:
        raise SystemExit("implementation commit missing or not an ancestor")
    for f in FOLDS:
        if not (CONTROL/"A1"/f"fold_{f:02d}/test_metrics.json").exists():
            raise SystemExit(f"missing control fold {f}")


def aggregate():
    folds={}; clauses=[]
    for f in FOLDS:
        cand=json.loads((OUT/"A1"/f"fold_{f:02d}/test_metrics.json").read_text())
        ctrl=json.loads((CONTROL/"A1"/f"fold_{f:02d}/test_metrics.json").read_text())
        c_f1=cand["structured"]["macro_f1_100ms"]
        b_f1=ctrl["structured"]["macro_f1_100ms"]
        c_cov=sum(cand["structured"][p]["coverage"] for p in core.FIDUCIALS)/4
        b_cov=sum(ctrl["structured"][p]["coverage"] for p in core.FIDUCIALS)/4
        shape_delta=cand["shape"]["pcc"]-ctrl["shape"]["pcc"]
        row={"control":ctrl,"candidate":cand,"delta_macro_f1_100ms":c_f1-b_f1,
             "delta_mean_coverage":c_cov-b_cov,"delta_shape_pcc":shape_delta}
        row["pass"]={"event_recovery":c_f1-b_f1>=.30 and c_cov-b_cov>=.30,
                     "shape_safety":shape_delta>=-.05,
                     "ordering":cand["structured"]["order_violation_rate"] in (0,0.0)}
        clauses.extend(row["pass"].values()); folds[str(f)]=row
    result={"experiment_id":TAG,"folds":folds,"overall_pass":all(clauses)}
    atomic_json(OUT/"summary.json",result); return result


def main():
    p=argparse.ArgumentParser(); p.add_argument("--device",default="cuda:0")
    p.add_argument("--preflight_only",action="store_true"); p.add_argument("--aggregate_only",action="store_true")
    a=p.parse_args(); preflight()
    if a.preflight_only: print("PASS",TAG); return
    if a.aggregate_only: print(json.dumps(aggregate(),indent=2)); return
    core.OUT=OUT; core.SEED=42; core.CHECKPOINT_RULE="shape_event_hmean"
    for f in FOLDS: core.train(f,"A1",core.torch.device(a.device))
    print(json.dumps(aggregate(),indent=2))


if __name__=="__main__": main()
