#!/usr/bin/env python3
"""Scenario-specific breakdown of the frozen continuous benchmark.

Registered analysis `q1_scenario_breakdown` (MMECG) and A1 of
`q1_crrvs_diagnostics_seed42` (CR_RVS). Frozen test predictions only; nothing is
trained or selected.

Scenarios are confounded with subjects in MMECG -- two test subjects contribute only
normal breathing, and sleep is dominated by one subject -- so every statistic is
computed **per (subject, scenario)** first and then aggregated over subjects. The
number of contributing subjects is reported beside each scenario.

Codes: MMECG NB normal breathing, IB irregular breathing, PE post-exercise, SP sleep.
CR_RVS reuses the same codes for NB Resting, IB Valsalva, SP Apnea.

Usage:
  python scripts/evaluate_scenario_breakdown.py --dataset mmecg
  python scripts/evaluate_scenario_breakdown.py --dataset crrvs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_continuous_ecg_pilot import timing_metrics  # noqa: E402
from src.utils.atomic_state import atomic_csv, atomic_json  # noqa: E402

RR_GATE = 0.80
EXPERIMENTS = ROOT / "experiments_mmecg"
CONFIG = {
    "mmecg": {
        "tag": "q1_scenario_breakdown_mmecg",
        "splits": Path("/home/qhh2237/Datasets/MMECG/processed/loso_v2"),
        "folds": list(range(1, 12)), "seeds": (42, 123, 2024),
        "scenarios": {"NB": "Normal breathing", "IB": "Irregular breathing",
                      "PE": "Post-exercise", "SP": "Sleep"},
        "rem_net": "q1_remnet_continuous_strict_loso_seed{seed}",
        "baselines": "q1_matched_baselines_continuous_strict_loso_seed{seed}",
    },
    "crrvs": {
        "tag": "q1_scenario_breakdown_crrvs",
        "splits": Path("/home/qhh2237/Datasets/CR_RVS/processed/q1_continuous_kfold"),
        "folds": list(range(1, 6)), "seeds": (42,),
        "scenarios": {"NB": "Resting", "IB": "Valsalva", "SP": "Apnea"},
        "rem_net": "q1_remnet_continuous_crrvs_seed{seed}",
        "baselines": "q1_matched_baselines_continuous_crrvs_seed{seed}",
    },
}
MODELS = ("rem_net", "mmecg_tcn", "wavegru_net")
LABELS = {"rem_net": "REM-Net (ours)", "mmecg_tcn": "Chen-TCN", "wavegru_net": "WaveGRU-Net"}


def fold_dir(cfg: dict, model: str, seed: int, fold: int) -> Path:
    if model == "rem_net":
        return EXPERIMENTS / cfg["rem_net"].format(seed=seed) / "L3" / f"fold_{fold:02d}"
    return EXPERIMENTS / cfg["baselines"].format(seed=seed) / model / f"fold_{fold:02d}"


def window_pcc(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pc = pred - pred.mean(1, keepdims=True)
    yc = gt - gt.mean(1, keepdims=True)
    den = np.sqrt((pc ** 2).sum(1) * (yc ** 2).sum(1))
    return np.where(den > 1e-12, (pc * yc).sum(1) / np.maximum(den, 1e-12), np.nan)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(CONFIG), default="mmecg")
    cfg = CONFIG[ap.parse_args().dataset]
    out_dir = EXPERIMENTS / cfg["tag"]

    rows = []
    for fold in cfg["folds"]:
        with h5py.File(cfg["splits"] / f"fold_{fold:02d}" / "test.h5", "r") as h:
            status = np.array([x.decode() if isinstance(x, bytes) else x
                               for x in h["physistatus"][:]])
            subjects = h["subject_id"][:]
            ecg = h["ecg"][:, 0]
        for model in MODELS:
            for seed in cfg["seeds"]:
                d = np.load(fold_dir(cfg, model, seed, fold) / "test_predictions.npz",
                            allow_pickle=True)
                pred, gt, rp = d["prediction"], d["ground_truth"], d["rpeaks"]
                if not np.allclose(gt, ecg):
                    raise SystemExit(f"{model} seed{seed} fold{fold}: rows are not aligned")
                pcc = window_pcc(pred, gt)
                for code in cfg["scenarios"]:
                    for subject in np.unique(subjects):
                        idx = np.flatnonzero((status == code) & (subjects == subject))
                        if len(idx) == 0:
                            continue
                        t = timing_metrics(pred[idx], np.zeros_like(pred[idx]),
                                           [np.asarray(r, float) for r in rp[idx]])
                        rows.append({"model": model, "seed": seed, "fold": fold,
                                     "subject": int(subject), "scenario": cfg["scenarios"][code],
                                     "n_windows": int(len(idx)),
                                     "pcc_median": float(np.nanmedian(pcc[idx])),
                                     "pcc_mean": float(np.nanmean(pcc[idx])),
                                     "rr_error_ms": float(t["rr_error_ms"]),
                                     "rr_coverage": float(t["rr_coverage"]),
                                     "f1_100ms": float(t["f1_100ms"])})
        print(f"fold {fold:2d} done", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(out_dir / "subject_scenario_metrics.csv", list(rows[0]), rows)

    summary = {}
    for model in MODELS:
        summary[model] = {"label": LABELS[model], "scenarios": {}}
        for name in cfg["scenarios"].values():
            sel = [r for r in rows if r["model"] == model and r["scenario"] == name]
            if not sel:
                continue
            # average the seeds within each (subject, scenario), then aggregate subjects
            per_subject = {}
            for r in sel:
                per_subject.setdefault(r["subject"], []).append(r)
            pcc = [float(np.mean([x["pcc_median"] for x in v])) for v in per_subject.values()]
            gated = [x for v in per_subject.values() for x in v if x["rr_coverage"] >= RR_GATE]
            rr_by_subject = {}
            for x in gated:
                rr_by_subject.setdefault(x["subject"], []).append(x["rr_error_ms"])
            rr = [float(np.mean(v)) for v in rr_by_subject.values()]
            summary[model]["scenarios"][name] = {
                "n_subjects": len(per_subject),
                "n_windows": int(sum(r["n_windows"] for r in sel) / len(cfg["seeds"])),
                "pcc_median_over_subjects": float(np.median(pcc)),
                "pcc_iqr": [float(np.percentile(pcc, 25)), float(np.percentile(pcc, 75))],
                "pcc_min": float(np.min(pcc)), "pcc_max": float(np.max(pcc)),
                "rr_error_ms_median_over_subjects": float(np.median(rr)) if rr else None,
                "rr_subjects_gated_out": len(per_subject) - len(rr_by_subject)}
    atomic_json(out_dir / "scenario_summary.json", {
        "experiment_id": cfg["tag"], "seeds": list(cfg["seeds"]),
        "aggregation": "per (subject, scenario) median over windows, seeds averaged, then "
                       "median over subjects; RR gated at rr_coverage >= 0.80",
        "confound_note": "scenarios are unevenly distributed over subjects; the subject count "
                         "per scenario is reported and no scenario is compared across models "
                         "without it",
        "models": summary})

    for name in cfg["scenarios"].values():
        print(f"\n[{name}]")
        for model in MODELS:
            s = summary[model]["scenarios"].get(name)
            if not s:
                continue
            rr = ("n/a" if s["rr_error_ms_median_over_subjects"] is None
                  else f"{s['rr_error_ms_median_over_subjects']:6.1f} ms")
            print(f"  {LABELS[model]:16s} PCC median {s['pcc_median_over_subjects']:.3f} "
                  f"[{s['pcc_iqr'][0]:.3f}, {s['pcc_iqr'][1]:.3f}]  range "
                  f"{s['pcc_min']:.3f}-{s['pcc_max']:.3f}  RR {rr}  "
                  f"({s['n_subjects']} subjects, {s['n_windows']} windows"
                  + (f", {s['rr_subjects_gated_out']} gated out)" if s["rr_subjects_gated_out"]
                     else ")"))


if __name__ == "__main__":
    main()
