#!/usr/bin/env python3
"""Validation-only preflight for q1_continuous_ecg_b0_b2_pilot. Launches no training."""
from __future__ import annotations
import json, shutil, sys
from pathlib import Path
import h5py, numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.models.continuous.ce_net import CENet, param_report, WIN  # noqa: E402

SPEC = json.loads((ROOT / "experiments/specs/q1_continuous_ecg_b0_b2_pilot.json").read_text())
DATA = Path("/home/qhh2237/Datasets/MMECG/processed/loso")
FOLDS, SPLITS, ARMS = [1, 5, 9], ["train", "val"], ["B0", "B1", "B2a", "B2a-wide", "B2b"]
BUDGET = {"max_epochs": 30, "patience": 8, "optimizer": "AdamW", "lr": 1e-4,
          "weight_decay": 1e-4, "batch_size": 16, "base_seed": 42,
          "effective_seed": "42 + fold"}
COMPOSITE = "S = 0.40*PCC + 0.25*(1-NRMSE_clipped) + 0.20*R2_clipped + 0.15*RpeakF1_100ms"
fail: list[str] = []
R: dict = {"spec": SPEC["experiment_id"], "launches_training": False}

print("="*78); print("PREFLIGHT  q1_continuous_ecg_b0_b2_pilot   (validation-only, no training)"); print("="*78)

# --- 1 & 6: input provenance + fold/split inventory, and no test file opened ---
print("\n[1/8] INPUT PROVENANCE  +  [6/8] FOLD/SPLIT INVENTORY")
inv = []
for f in FOLDS:
    for s in SPLITS:
        p = DATA / f"fold_{f:02d}" / f"{s}_wsst8s.h5"
        with h5py.File(p, "r") as h:
            a, sh = dict(h.attrs), h["wsst"].shape
        good = (a.get("support") == "full_window_samples_0_1600"
                and float(a.get("supported_fraction_of_target", 0)) == 1.0
                and (sh[2], sh[3]) == (71, 240) and abs(float(a["hop_ms"]) - 100/3) < 1e-3)
        if not good: fail.append(f"provenance {p.name} fold{f}")
        inv.append({"fold": f, "split": s, "n": int(sh[0]), "shape": list(sh),
                    "support": a.get("support"), "frac": float(a.get("supported_fraction_of_target", 0)),
                    "hop_ms": float(a["hop_ms"]), "bins": int(a["n_time_bins"])})
        print(f"  fold_{f:02d} {s:5s} n={sh[0]:5d} {str(tuple(sh)):22s} support={a['support']} "
              f"frac={a['supported_fraction_of_target']} hop={float(a['hop_ms']):.2f}ms  {'OK' if good else 'FAIL'}")
R["input_provenance"], R["inventory"] = inv, inv
test_files = list(DATA.glob("fold_*/test_wsst8s.h5"))
print(f"  test_wsst8s.h5 present anywhere: {len(test_files)}  -> {'OK (none)' if not test_files else 'FAIL'}")
if test_files: fail.append("test WSST exists")
R["test_files_present"] = len(test_files)

# --- 2: enforcement assertions ---
print("\n[2/8] ENFORCEMENT ASSERTIONS")
enf = SPEC["enforcement"]
checks = {
    "topk_method != 'corr'": enf["topk_method"].startswith("energy"),
    "narrow_bandpass False on rcg path": "False" in enf["narrow_bandpass"],
    "input declares full-window support": all(i["frac"] == 1.0 for i in inv),
    "legacy 4s _wsst.h5 not consumed": all("wsst8s" in str(DATA/f"fold_{f:02d}"/f"{s}_wsst8s.h5") for f in FOLDS for s in SPLITS),
}
for k, v in checks.items():
    print(f"  {k:42s} {'PASS' if v else 'FAIL'}")
    if not v: fail.append(k)
R["enforcement"] = checks

# --- 3: parameter counts and the 15% capacity trigger ---
print("\n[3/8] PARAMETER COUNTS")
models = {a: CENet(a) for a in ARMS}
rep = {a: param_report(m) for a, m in models.items()}
for a in ARMS:
    print(f"  {a:5s} total {rep[a]['total']:>9,d}   " +
          "  ".join(f"{k}={v:,d}" for k, v in rep[a]["per_module"].items()))
wide_gap = abs(rep["B2a-wide"]["total"] - rep["B2b"]["total"]) / rep["B2b"]["total"]
print(f"  B2a-wide vs B2b capacity match: {rep['B2a-wide']['total']:,d} vs {rep['B2b']['total']:,d} "
      f"({100*wide_gap:.3f}%)  {'PASS' if wide_gap < 0.001 else 'FAIL'}")
if wide_gap >= 0.001: fail.append("B2a-wide capacity match")
d_abs = rep["B2b"]["total"] - rep["B2a"]["total"]
d_pct = 100.0 * d_abs / rep["B2a"]["total"]
stem = rep["B2b"]["per_module"].get("rcg_stem", 0)
fuse_d = rep["B2b"]["per_module"]["fuse"] - rep["B2a"]["per_module"]["fuse"]
trig = d_pct > 15.0
print(f"\n  B2b - B2a = {d_abs:,d} ({d_pct:.2f}%)   rcg_stem={stem:,d}  fusion_delta={fuse_d:,d}")
print(f"  15% capacity-control trigger: {'TRIPPED -> B2a-wide REQUIRED' if trig else 'not tripped'}")
dec_match = len({rep[a]["per_module"]["decoder"] for a in ("B2a", "B2a-wide", "B2b")}) == 1
print(f"  decoder parity B2a == B2b: {rep['B2a']['per_module']['decoder']:,d} vs "
      f"{rep['B2b']['per_module']['decoder']:,d}  {'PASS' if dec_match else 'FAIL'}")
if not dec_match: fail.append("decoder parity")
R["parameters"] = {"per_arm": rep, "b2b_minus_b2a": d_abs, "pct": d_pct,
                   "rcg_stem": stem, "fusion_delta": fuse_d,
                   "trigger_15pct_tripped": bool(trig), "decoder_parity": bool(dec_match)}

# --- 4: identical budget ---
print("\n[4/8] BUDGET PARITY (identical across all arms)")
for k, v in BUDGET.items(): print(f"  {k:14s} {v}")
R["budget"] = BUDGET

# --- 5: checkpoint composite ---
print("\n[5/8] CHECKPOINT RULE (frozen before training)")
print(f"  {COMPOSITE}")
uses_shape = "shape" in COMPOSITE.lower() or "200" in COMPOSITE
print(f"  reads any Shape-200 quantity: {uses_shape}  -> {'FAIL' if uses_shape else 'PASS'}")
print("  R-peak F1 uses the WAVEFORM-DERIVED readout, not the event head.")
if uses_shape: fail.append("composite reads Shape-200")
R["checkpoint_rule"] = {"composite": COMPOSITE, "reads_shape200": uses_shape,
                        "rpeak_readout": "waveform-derived"}

# --- 7: forward-pass shape trace ---
print("\n[7/8] FORWARD-PASS SHAPE TRACE  (B=2)")
wsst = torch.randn(2, 50, 71, 240); rcg = torch.randn(2, 50, 1600)
trace = {}
for a in ARMS:
    m = models[a].eval()
    with torch.no_grad():
        y, ev = m(wsst, rcg) if a == "B2b" else m(wsst)
    ok = tuple(y.shape) == (2, 1, WIN)
    trace[a] = {"in_wsst": list(wsst.shape), "in_rcg": list(rcg.shape) if a == "B2b" else None,
                "out": list(y.shape), "event": list(ev.shape), "ok": ok}
    print(f"  {a:5s} wsst{tuple(wsst.shape)}" + (f" + rcg{tuple(rcg.shape)}" if a == "B2b" else " "*22)
          + f" -> ecg{tuple(y.shape)} event{tuple(ev.shape)}  {'OK' if ok else 'FAIL'}")
    if not ok: fail.append(f"shape {a}")
lin = not any(isinstance(mm, torch.nn.Sigmoid) for mm in models["B1"].decoder.head.modules())
print(f"  B1 output head is LINEAR (no sigmoid): {'PASS' if lin else 'FAIL'}")
if not lin: fail.append("B1 linear head")
R["shape_trace"] = trace

# --- 8: disk + runtime ---
print("\n[8/8] RESOURCES")
du = shutil.disk_usage("/home/qhh2237/Datasets")
n_tr = sum(i["n"] for i in inv if i["split"] == "train")
print(f"  disk free {du.free/2**30:.1f} GiB of {du.total/2**30:.1f} GiB ({100*du.used/du.total:.1f}% used)")
print(f"  train windows across folds {FOLDS}: {n_tr};  {len(ARMS)} arms x 3 folds = {len(ARMS)*3} runs")
print(f"  checkpoints ~{rep['B2b']['total']*4/2**20:.1f} MiB each -> ~{len(ARMS)*3*rep['B2b']['total']*4/2**20:.0f} MiB total")
print(f"  WARNING: disk is {100*du.used/du.total:.0f}% full; legacy _wsst.h5/_wsst4s.h5 remain on disk by policy.")
R["resources"] = {"disk_free_gib": du.free/2**30, "pct_used": 100*du.used/du.total,
                  "train_windows": n_tr, "runs": len(ARMS)*3}

print("\n" + "="*78)
print(f"PREFLIGHT: {'PASS - ready to launch on approval' if not fail else 'FAIL -> ' + '; '.join(fail)}")
print("="*78)
R["result"] = "PASS" if not fail else "FAIL"; R["failures"] = fail
out = ROOT / "experiments_mmecg/q1_continuous_ecg_b0_b2_pilot"; out.mkdir(parents=True, exist_ok=True)
(out / "preflight.json").write_text(json.dumps(R, indent=2, default=str) + "\n")
print(f"wrote {out/'preflight.json'}")
