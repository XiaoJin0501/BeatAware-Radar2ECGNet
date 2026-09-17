#!/usr/bin/env python3
"""CHARACTERIZATION ONLY -- not part of the frozen audit protocol.

Answers one question about arm C3/C4 of q1_continuous_gap_decomposition_audit:
do the predicted cycle bounds change POSITION only, or position AND duration?
Reports no reconstruction metric and alters nothing in the frozen run.
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.evaluate_continuous_gap_decomposition import (  # noqa: E402
    _load_model, _infer, _global_cycles, _centre_cycle, WIN)
from scripts.train_anchor_head import peaks_from_heatmap  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp_tag", default="q1_final_remnet_incumbent_mmecg_seed42")
    ap.add_argument("--folds", default="1-11")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    lo, hi = a.folds.split("-"); folds = list(range(int(lo), int(hi) + 1))
    dev = torch.device(a.device)
    start_off, mid_off, len_ratio, len_gt, len_pr = [], [], [], [], []

    for fold in folds:
        run = ROOT / "experiments_mmecg" / a.exp_tag / f"fold_{fold:02d}"
        model, cfg, _ = _load_model(run, dev, "best.pt")
        ws = _infer(model, cfg, Path(cfg.loso_h5_dir) / f"fold_{fold:02d}", "test", dev)
        by = defaultdict(list)
        for w in ws: by[w["record"]].append(w)
        for rec, wl in by.items():
            _, cyc = _global_cycles(wl)
            pk = np.unique(np.concatenate(
                [w["start"] + peaks_from_heatmap(w["rmask"]).astype(np.int64) for w in wl]))
            pc = np.stack([pk[:-1], pk[1:]], axis=1) if len(pk) >= 2 else np.empty((0, 2), np.int64)
            for w in wl:
                if len(w["rpeaks_local"]) < 2: continue
                centre = w["start"] + WIN // 2
                j = _centre_cycle(cyc, centre)
                if j < 0: continue
                A, B = int(cyc[j, 0]), int(cyc[j, 1])
                if B <= A + 10 or A < w["start"] or B > w["start"] + WIN: continue
                jp = _centre_cycle(pc, centre)
                if jp < 0: continue
                A2, B2 = int(pc[jp, 0]), int(pc[jp, 1])
                if B2 <= A2 + 10: continue
                start_off.append(A2 - A); mid_off.append((A2 + B2) / 2 - (A + B) / 2)
                len_gt.append(B - A); len_pr.append(B2 - A2); len_ratio.append((B2 - A2) / (B - A))

    MS = 1000.0 / 200.0  # 200 Hz -> ms
    so, mo, lr = map(np.asarray, (start_off, mid_off, len_ratio))
    lg, lp = np.asarray(len_gt), np.asarray(len_pr)
    def q(x, s=1.0): return {k: float(np.percentile(x, v) * s) for k, v in
                             (("p05",5),("p25",25),("median",50),("p75",75),("p95",95))}
    out = {
        "note": "CHARACTERIZATION ONLY. Not part of the frozen protocol; no metric altered.",
        "n_matched_cycles": int(len(so)),
        "start_offset_ms": {"mean": float(so.mean()*MS), "sd": float(so.std()*MS),
                            "mean_abs": float(np.abs(so).mean()*MS), **q(so, MS)},
        "midpoint_offset_ms": {"mean": float(mo.mean()*MS), "sd": float(mo.std()*MS),
                               "mean_abs": float(np.abs(mo).mean()*MS), **q(mo, MS)},
        "duration_gt_ms": {"mean": float(lg.mean()*MS), **q(lg, MS)},
        "duration_pred_ms": {"mean": float(lp.mean()*MS), **q(lp, MS)},
        "duration_ratio_pred_over_gt": {"mean": float(lr.mean()), "sd": float(lr.std()),
                                        "mean_abs_log2": float(np.abs(np.log2(lr)).mean()), **q(lr)},
        "frac_cycles_duration_within_5pct": float(np.mean(np.abs(lr - 1) <= 0.05)),
        "frac_cycles_duration_within_10pct": float(np.mean(np.abs(lr - 1) <= 0.10)),
        "frac_cycles_start_within_25ms": float(np.mean(np.abs(so) * MS <= 25)),
        "frac_cycles_start_within_50ms": float(np.mean(np.abs(so) * MS <= 50)),
    }
    p = ROOT / "experiments_mmecg/q1_continuous_gap_decomposition_audit/predicted_bounds_characterization.json"
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
