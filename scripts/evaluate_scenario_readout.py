#!/usr/bin/env python3
"""Per-scenario readout of the frozen CR_RVS continuous benchmark (analysis A1).

Registered analysis `q1_crrvs_diagnostics_seed42`. The frozen test predictions of
REM-Net, Chen-TCN and WaveGRU-Net are scored per physiological scenario with the
frozen evaluator functions. CR_RVS stores its scenarios with MMECG's codes:
NB = Resting, IB = Valsalva, SP = Apnea.

Resting is the scenario every cited CR_RVS paper uses, so it is the like-for-like
comparison with the literature. The per-subject median PCC over Resting windows is
the quantity Wu et al. 2024 report under leave-one-subject-out (0.39-0.90).

Usage:
  python scripts/evaluate_scenario_readout.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from statistics import mean, median, pstdev

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_template_timing import score  # noqa: E402
from src.utils.atomic_state import atomic_csv, atomic_json  # noqa: E402

TAG = "q1_scenario_readout_crrvs_seed42"
OUT = ROOT / "experiments_mmecg" / TAG
SPLITS = Path("/home/qhh2237/Datasets/CR_RVS/processed/q1_continuous_kfold")
FOLDS = list(range(1, 6))
SCENARIOS = {"NB": "Resting", "IB": "Valsalva", "SP": "Apnea"}
ARMS = {
    "rem_net": ROOT / "experiments_mmecg" / "q1_remnet_continuous_crrvs_seed42" / "L3",
    "mmecg_tcn": ROOT / "experiments_mmecg" / "q1_matched_baselines_continuous_crrvs_seed42" / "mmecg_tcn",
    "wavegru_net": ROOT / "experiments_mmecg" / "q1_matched_baselines_continuous_crrvs_seed42" / "wavegru_net",
}
KEYS = ("pcc", "r2", "nrmse", "rmse", "mae", "f1_100ms", "amplitude_variance_ratio")


def window_pcc(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pc = pred - pred.mean(1, keepdims=True)
    yc = gt - gt.mean(1, keepdims=True)
    den = np.sqrt((pc ** 2).sum(1) * (yc ** 2).sum(1))
    return np.where(den > 1e-12, (pc * yc).sum(1) / np.maximum(den, 1e-12), np.nan)


def main() -> None:
    rows, per = [], {a: {s: [] for s in SCENARIOS.values()} for a in ARMS}
    resting_pcc = {a: {} for a in ARMS}
    for fold in FOLDS:
        with h5py.File(SPLITS / f"fold_{fold:02d}" / "test.h5", "r") as h:
            status = np.array([x.decode() if isinstance(x, bytes) else x for x in h["physistatus"][:]])
            subjects = h["subject_id"][:]
            ecg = h["ecg"][:, 0]
        for arm, root in ARMS.items():
            d = np.load(root / f"fold_{fold:02d}" / "test_predictions.npz", allow_pickle=True)
            pred, gt, rp = d["prediction"], d["ground_truth"], d["rpeaks"]
            if not np.allclose(gt, ecg):
                raise SystemExit(f"{arm} fold {fold}: predictions are not row-aligned with test.h5")
            for code, name in SCENARIOS.items():
                idx = np.flatnonzero(status == code)
                if len(idx) == 0:
                    continue
                m = score(pred[idx], gt[idx], rp[idx])
                per[arm][name].append(m)
                rows.append({"arm": arm, "fold": fold, "scenario": name,
                             "n_windows": int(len(idx)), **{k: m[k] for k in KEYS}})
            rest = np.flatnonzero(status == "NB")
            pcc = window_pcc(pred[rest], gt[rest])
            for s, v in zip(subjects[rest], pcc):
                resting_pcc[arm].setdefault(int(s), []).append(float(v))

    OUT.mkdir(parents=True, exist_ok=True)
    atomic_csv(OUT / "scenario_fold_metrics.csv", list(rows[0]), rows)
    summary = {arm: {name: {k: {"mean": mean(r[k] for r in rs), "std_population": pstdev(r[k] for r in rs)}
                            for k in KEYS} for name, rs in scen.items() if rs}
               for arm, scen in per.items()}
    wins = {}
    for arm in ("mmecg_tcn", "wavegru_net"):
        wins[arm] = {name: {"rem_net_minus_baseline_pcc": float(np.mean(
            [a["pcc"] - b["pcc"] for a, b in zip(per["rem_net"][name], per[arm][name])])),
                            "rem_net_wins": int(sum(a["pcc"] > b["pcc"] for a, b in
                                                    zip(per["rem_net"][name], per[arm][name]))),
                            "folds": len(per[arm][name])} for name in SCENARIOS.values()}
    subject_level = {}
    for arm, by_subject in resting_pcc.items():
        med = {s: float(np.nanmedian(v)) for s, v in by_subject.items()}
        vals = np.array(list(med.values()))
        subject_level[arm] = {"n_subjects": int(len(vals)), "min": float(vals.min()),
                              "max": float(vals.max()), "median": float(np.median(vals)),
                              "subjects_above_0_7": int((vals > 0.7).sum()),
                              "per_subject_median_pcc": med}
    atomic_json(OUT / "scenario_summary.json", {
        "experiment_id": "q1_crrvs_diagnostics_seed42 / A1", "scenario_codes": SCENARIOS,
        "aggregation": "window-macro within fold and scenario, fold-macro over 5 folds, population std",
        "scenarios": summary, "rem_net_vs_baseline": wins,
        "resting_subject_level_pcc": subject_level,
        "literature_reference": "Wu 2024 RSSRnet, leave-one-subject-out, Resting: per-subject "
                                "median PCC 0.39-0.90, more than half above 0.7"})

    for name in SCENARIOS.values():
        print(f"\n[{name}]")
        for arm in ARMS:
            s = summary[arm][name]
            print(f"  {arm:12s} PCC {s['pcc']['mean']:.3f}±{s['pcc']['std_population']:.3f}  "
                  f"R2 {s['r2']['mean']:+.3f}  NRMSE {s['nrmse']['mean']:.3f}  "
                  f"F1@100 {s['f1_100ms']['mean']:.3f}  var {s['amplitude_variance_ratio']['mean']:.3f}")
    print("\nResting, per-subject median PCC (Wu 2024 LOSO: 0.39-0.90, >half above 0.7):")
    for arm, d in subject_level.items():
        print(f"  {arm:12s} {d['n_subjects']} subjects  range {d['min']:.2f}-{d['max']:.2f}  "
              f"median {d['median']:.2f}  above 0.7: {d['subjects_above_0_7']}/{d['n_subjects']}")


if __name__ == "__main__":
    main()
