#!/usr/bin/env python3
"""Registered A0/A1/A2 interaction gate for the dual-head REM-Net.

The test split is constructed only after a validation-selected checkpoint is
frozen.  A2 is a deterministic, reference-free decoder of A1 probabilities.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_continuous_ecg_pilot import Fold, fit_scaler  # noqa: E402
from src.data.mmecg_dataset import _center_cycle_bounds_from_rpeaks, _gaussian_mask  # noqa: E402
from src.models.continuous.dual_head_rem import DualHeadREM  # noqa: E402
from src.utils.atomic_state import (atomic_json, atomic_npz, atomic_torch_save,
                                    capture_rng_state, restore_rng_state,
                                    transition_state)  # noqa: E402
from src.utils.clinical_endpoints import delineate_signal, match_beats  # noqa: E402

TAG = "q1_dual_head_interaction_gate"
OUT = ROOT / "experiments_mmecg" / TAG
SPEC = ROOT / "experiments/specs/q1_dual_head_interaction_gate.json"
BASELINE_DIR = ROOT / "docs/audits/q1_timing_screen_v2"
FOLDS, ARMS, FIDUCIALS = (2, 6, 10), ("A0", "A1"), ("q", "r", "s", "t")
FS, WIN, BATCH, SEED = 200, 1600, 16, 42
MAX_EPOCHS, PATIENCE = 100, 15


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_points(ecg: np.ndarray) -> dict[str, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return delineate_signal(ecg, fs=FS, method="prominence")


def shape_target(ecg: np.ndarray, r: np.ndarray) -> np.ndarray:
    a, b = _center_cycle_bounds_from_rpeaks(r, len(ecg))
    cycle = np.asarray(ecg[a:b], np.float32)
    if cycle.size < 2: cycle = np.asarray(ecg[700:900], np.float32)
    target = np.interp(np.linspace(0, cycle.size - 1, 200),
                       np.arange(cycle.size), cycle).astype(np.float32)
    lo, hi = float(target.min()), float(target.max())
    return np.zeros(200, np.float32) if hi - lo < 1e-8 else (target - lo) / (hi - lo)


class DualFold(Dataset):
    constructions: list[str] = []

    def __init__(self, fold: int, split: str, arm: str, allow_test=False, limit=None):
        if arm not in ARMS: raise ValueError(arm)
        DualFold.constructions.append(split)
        self.arm = arm
        self.base = Fold(fold, split, "B1", fit_scaler(fold), limit=limit,
                         allow_test=allow_test, wsst_source="subject_pool")
        self.points = [extract_points(x) for x in self.base.ecg]
        self.shapes = np.stack([shape_target(x, p["r"]) for x, p in
                                zip(self.base.ecg, self.points)]).astype(np.float32)

    def __len__(self): return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        if self.arm == "A0":
            r = np.asarray(self.points[i]["r"], float)
            r = r[np.isfinite(r)].astype(int)
            anchor = np.stack([_gaussian_mask(r, WIN, s) for s in (4., 16.)])
        else:
            anchor = np.stack([_gaussian_mask(
                np.asarray(self.points[i][p], float)[np.isfinite(self.points[i][p])].astype(int),
                WIN, 4.) for p in FIDUCIALS])
        return {"wsst": item["wsst"],
                "shape": torch.from_numpy(self.shapes[i][None]),
                "anchor": torch.from_numpy(anchor.astype(np.float32))}


def pcc_per_row(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    return (a*b).sum(1) / np.maximum(np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1), 1e-8)


def shape_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    pcc = pcc_per_row(pred, target)
    rmse = np.sqrt(np.mean((pred-target)**2, axis=1))
    den = np.sum((target-target.mean(1, keepdims=True))**2, axis=1)
    r2 = 1 - np.sum((pred-target)**2, axis=1) / np.maximum(den, 1e-8)
    return {"pcc": float(np.mean(pcc)), "rmse": float(np.mean(rmse)),
            "r2": float(np.mean(r2)), "n": int(len(pred))}


def independent_decode(prob: np.ndarray) -> dict[str, np.ndarray]:
    return {p: find_peaks(prob[k], height=.5, distance=40)[0].astype(float)
            for k, p in enumerate(FIDUCIALS)}


def structured_decode(prob: np.ndarray) -> dict[str, np.ndarray]:
    """R-centred Q/R/S/T decoding using only predicted probabilities."""
    r = find_peaks(prob[1], height=.5, distance=40)[0]
    out = {p: [] for p in FIDUCIALS}
    for k, anchor in enumerate(r):
        q0, q1 = max(0, anchor-20), anchor
        s0, s1 = anchor+1, min(WIN, anchor+21)
        if q1 <= q0 or s1 <= s0: continue
        q = q0 + int(np.argmax(prob[0, q0:q1]))
        s = s0 + int(np.argmax(prob[2, s0:s1]))
        t0 = s + 8
        t1 = min(WIN, s+91, int(r[k+1]) if k+1 < len(r) else WIN)
        if t1 <= t0: continue
        t = t0 + int(np.argmax(prob[3, t0:t1]))
        out["q"].append(q); out["r"].append(anchor)
        out["s"].append(s); out["t"].append(t)
    return {p: np.asarray(v, float) for p, v in out.items()}


def score_events(probabilities: np.ndarray, refs: list[dict], structured: bool) -> dict:
    rows = {p: {"gt": 0, "pred": 0, "match": 0, "match100": 0, "err": []}
            for p in FIDUCIALS}
    complete = violations = 0
    for prob, ref in zip(probabilities, refs):
        pred = structured_decode(prob) if structured else independent_decode(prob)
        for p in FIDUCIALS:
            gt = np.asarray(ref[p], float); gt = gt[np.isfinite(gt)]
            pairs = match_beats(gt, pred[p], 30); pairs100 = match_beats(gt, pred[p], 20)
            x = rows[p]; x["gt"] += len(gt); x["pred"] += len(pred[p])
            x["match"] += len(pairs); x["match100"] += len(pairs100)
            x["err"].extend(abs(gt[i]-pred[p][j])*5 for i, j in pairs.items())
        if structured:
            complete += len(pred["r"])
            violations += sum(not (q < r < s < t) for q, r, s, t in
                              zip(pred["q"], pred["r"], pred["s"], pred["t"]))
        else:
            for r in pred["r"]:
                near = {p: pred[p][np.argmin(abs(pred[p]-r))] for p in ("q","s","t")
                        if len(pred[p])}
                if len(near) == 3:
                    complete += 1; violations += int(not (near["q"] < r < near["s"] < near["t"]))
    out = {}
    for p, x in rows.items():
        err = np.asarray(x.pop("err"), float)
        out[p] = {**x, "medae_ms": float(np.median(err)) if len(err) else None,
                  "coverage": x["match"]/x["gt"] if x["gt"] else None,
                  "mdr_150ms": 1-x["match"]/x["gt"] if x["gt"] else None,
                  "f1_100ms": 2*x["match100"]/(x["gt"]+x["pred"])
                  if x["gt"]+x["pred"] else None}
    out["macro_medae_ms"] = float(np.mean([out[p]["medae_ms"] for p in FIDUCIALS]))
    out["macro_f1_100ms"] = float(np.mean([out[p]["f1_100ms"] for p in FIDUCIALS]))
    out["order_complete"] = complete; out["order_violations"] = violations
    out["order_violation_rate"] = violations/complete if complete else None
    return out


def losses(output, batch):
    pred, target = output["shape"], batch["shape"]
    l1 = F.l1_loss(pred, target)
    pc = ((pred-pred.mean(-1, keepdim=True))*(target-target.mean(-1, keepdim=True))).sum(-1)
    pc = pc / torch.clamp(torch.linalg.vector_norm(pred-pred.mean(-1, keepdim=True), dim=-1) *
                          torch.linalg.vector_norm(target-target.mean(-1, keepdim=True), dim=-1), 1e-8)
    shape = l1 + .5*(1-pc.mean())
    anchor = F.binary_cross_entropy_with_logits(output["anchor_logits"], batch["anchor"])
    return shape + .3*anchor, shape, anchor


@torch.no_grad()
def evaluate(model, data: DualFold, device, arm: str):
    model.eval(); shapes=[]; probs=[]
    for b in DataLoader(data, batch_size=BATCH, shuffle=False, num_workers=0):
        o = model(b["wsst"].to(device)); shapes.append(o["shape"].cpu().numpy()[:,0])
        probs.append(torch.sigmoid(o["anchor_logits"]).cpu().numpy())
    shapes=np.concatenate(shapes); probs=np.concatenate(probs)
    result={"shape": shape_metrics(shapes, data.shapes)}
    if arm == "A0":
        # Control selection score is the narrow R read-out only.
        rprob=np.zeros((len(probs),4,WIN), np.float32); rprob[:,1]=probs[:,0]
        refs=[{"q":np.array([]), "r":p["r"], "s":np.array([]), "t":np.array([])} for p in data.points]
        # Avoid undefined empty-fiducial aggregate: score R directly.
        gt=pr=ma=0
        for x,p in zip(rprob,refs):
            pred=find_peaks(x[1],height=.5,distance=40)[0]; ref=np.asarray(p["r"],float)
            pairs=match_beats(ref,pred,20); gt+=len(ref); pr+=len(pred); ma+=len(pairs)
        result["anchor_selection"]={"r_f1_100ms":2*ma/(gt+pr) if gt+pr else 0.}
    else:
        result["independent"]=score_events(probs,data.points,False)
        result["structured"]=score_events(probs,data.points,True)
    return result, shapes, probs


def rank(metrics: dict, arm: str) -> tuple:
    s=metrics["shape"]
    anchor=(metrics["anchor_selection"]["r_f1_100ms"] if arm=="A0"
            else metrics["independent"]["macro_f1_100ms"])
    return (s["pcc"], -s["rmse"], anchor)


def train(fold: int, arm: str, device, max_epochs=MAX_EPOCHS, limit=None,
          interrupt_after=None):
    d=OUT/arm/f"fold_{fold:02d}"; ck=d/"checkpoints"; ck.mkdir(parents=True,exist_ok=True)
    state,last,best=d/"run_state.json",ck/"last_state.pt",ck/"best.pt"
    if state.exists():
        stage = json.loads(state.read_text())["stage"]
        if stage == "COMPLETE": return
        if stage in ("CHECKPOINT_FROZEN", "TEST_EVALUATING"):
            run_test(fold, arm, device, limit); return
    set_seed(SEED+fold); mode="r_multiscale" if arm=="A0" else "qrst"
    model=DualHeadREM(mode).to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    start=1; wait=0; best_rank=(-np.inf,-np.inf,-np.inf); history=[]
    if last.exists():
        x=torch.load(last,map_location=device,weights_only=False); model.load_state_dict(x["model"])
        opt.load_state_dict(x["optimizer"]); restore_rng_state(x["rng"])
        start=x["epoch"]+1; wait=x["wait"]; best_rank=tuple(x["best_rank"]); history=x["history"]
    if not state.exists(): transition_state(state,"TRAINING",fold=fold,arm=arm,test_accessed=False)
    tr=DualFold(fold,"train",arm,limit=limit); va=DualFold(fold,"val",arm,limit=limit)
    loader=DataLoader(tr,batch_size=BATCH,shuffle=True,num_workers=0,
                      generator=torch.Generator().manual_seed(SEED+fold))
    for epoch in range(start,max_epochs+1):
        model.train(); total=0.; n=0
        for b in loader:
            b={k:(v.to(device) if torch.is_tensor(v) else v) for k,v in b.items()}
            loss,_,_=losses(model(b["wsst"]),b); opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step(); total+=float(loss.detach()); n+=1
        m,_,_=evaluate(model,va,device,arm); current=rank(m,arm); history.append({"epoch":epoch,"loss":total/max(n,1),"validation":m})
        if current>best_rank:
            best_rank=current; wait=0; atomic_torch_save(best,{"model":model.state_dict(),"epoch":epoch,"validation":m})
        else: wait+=1
        atomic_torch_save(last,{"model":model.state_dict(),"optimizer":opt.state_dict(),"epoch":epoch,
                                "wait":wait,"best_rank":best_rank,"history":history,"rng":capture_rng_state()})
        atomic_json(d/"history.json",history)
        print(f"{arm} fold{fold:02d} ep{epoch:03d} loss={total/max(n,1):.4f} shapePCC={m['shape']['pcc']:.3f}",flush=True)
        if interrupt_after==epoch: raise RuntimeError("SIMULATED_INTERRUPTION")
        if wait>=PATIENCE: break
    transition_state(state,"CHECKPOINT_FROZEN",best_rank=list(best_rank),test_accessed=False)
    run_test(fold,arm,device,limit)


def run_test(fold:int,arm:str,device,limit=None):
    d=OUT/arm/f"fold_{fold:02d}"; state=d/"run_state.json"; s=json.loads(state.read_text())
    if s["stage"]=="CHECKPOINT_FROZEN": transition_state(state,"TEST_EVALUATING",test_accessed=True)
    elif s["stage"]!="TEST_EVALUATING": raise RuntimeError(f"test access from {s['stage']}")
    mode="r_multiscale" if arm=="A0" else "qrst"; model=DualHeadREM(mode).to(device)
    model.load_state_dict(torch.load(d/"checkpoints/best.pt",map_location=device,weights_only=False)["model"])
    test=DualFold(fold,"test",arm,allow_test=True,limit=limit); m,shape,prob=evaluate(model,test,device,arm)
    atomic_json(d/"test_metrics.json",m); atomic_npz(d/"test_predictions.npz",{"shape":shape.astype(np.float16),"anchor_probability":prob.astype(np.float16)})
    transition_state(state,"COMPLETE",test_accessed=True)


def chen(fold:int)->dict:
    rows=json.loads((BASELINE_DIR/f"mmecg_tcn_42_fold_{fold:02d}.json").read_text())["rows"]
    return {r["metric"]:r for r in rows if r["metric"] in FIDUCIALS}


def aggregate():
    a0={f:json.loads((OUT/"A0"/f"fold_{f:02d}/test_metrics.json").read_text()) for f in FOLDS}
    a1={f:json.loads((OUT/"A1"/f"fold_{f:02d}/test_metrics.json").read_text()) for f in FOLDS}
    base={f:chen(f) for f in FOLDS}
    shape_pcc=all(a1[f]["shape"]["pcc"]>=a0[f]["shape"]["pcc"]-.02 for f in FOLDS)
    rmse=np.mean([a1[f]["shape"]["rmse"] for f in FOLDS])<=np.mean([a0[f]["shape"]["rmse"] for f in FOLDS])+.01
    ours={p:float(np.median([a1[f]["structured"][p]["medae_ms"] for f in FOLDS])) for p in FIDUCIALS}
    theirs={p:float(np.median([base[f][p]["medae_ms"] for f in FOLDS])) for p in FIDUCIALS}
    wins=[p for p in FIDUCIALS if ours[p]<theirs[p]]
    reduction=1-np.mean([a1[f]["structured"]["macro_medae_ms"] for f in FOLDS])/np.mean([np.mean([base[f][p]["medae_ms"] for p in FIDUCIALS]) for f in FOLDS])
    coverage=all(np.median([a1[f]["structured"][p]["coverage"] for f in FOLDS])>=np.median([base[f][p]["coverage"] for f in FOLDS])-.03 for p in wins)
    order=all(a1[f]["structured"]["order_violation_rate"]<=.01 for f in FOLDS)
    gate={"shape_noninferiority":bool(shape_pcc and rmse),"anchor_accuracy":bool(len(wins)>=3 and reduction>=.10),"coverage":bool(coverage),"ordering":bool(order)}
    gate["overall_pass"]=all(gate.values())
    result={"A0":a0,"A1_A2":a1,"chen_tcn":base,"event_medians_ms":{"A2":ours,"Chen-TCN":theirs},"wins":wins,"macro_relative_reduction":float(reduction),"gate":gate}
    atomic_json(OUT/"summary.json",result); return result


def preflight() -> None:
    spec=json.loads(SPEC.read_text())
    if not spec["status"].startswith("PREFLIGHT_PASSED"):
        raise SystemExit(f"spec not launchable: {spec['status']}")
    commit=spec.get("implementation_commit")
    if not commit:
        raise SystemExit("spec lacks implementation_commit")
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    if subprocess.run(["git","merge-base","--is-ancestor",commit,head],cwd=ROOT).returncode:
        raise SystemExit(f"implementation commit {commit} is not an ancestor of {head}")
    for f in FOLDS:
        for split in ("train","val","test"):
            path=Path(f"/home/qhh2237/Datasets/MMECG/processed/loso_v2/fold_{f:02d}/{split}.h5")
            if not path.exists(): raise SystemExit(f"missing {path}")
        if not (BASELINE_DIR/f"mmecg_tcn_42_fold_{f:02d}.json").exists():
            raise SystemExit(f"missing frozen comparator fold {f}")


def main():
    p=argparse.ArgumentParser(); p.add_argument("--device",default="cuda:0"); p.add_argument("--fold",type=int,choices=FOLDS); p.add_argument("--arm",choices=ARMS); p.add_argument("--aggregate_only",action="store_true"); p.add_argument("--preflight_only",action="store_true"); p.add_argument("--max_epochs",type=int,default=MAX_EPOCHS); p.add_argument("--limit",type=int); p.add_argument("--interrupt_after",type=int)
    a=p.parse_args()
    if a.aggregate_only: print(json.dumps(aggregate()["gate"],indent=2)); return
    preflight()
    if a.preflight_only: print("PASS",TAG); return
    dev=torch.device(a.device); folds=(a.fold,) if a.fold else FOLDS; arms=(a.arm,) if a.arm else ARMS
    for arm in arms:
        for fold in folds: train(fold,arm,dev,a.max_epochs,a.limit,a.interrupt_after)
    if all((OUT/x/f"fold_{f:02d}/test_metrics.json").exists() for x in ARMS for f in FOLDS): print(json.dumps(aggregate()["gate"],indent=2))


if __name__=="__main__": main()
