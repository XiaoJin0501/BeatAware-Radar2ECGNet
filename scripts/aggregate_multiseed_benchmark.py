#!/usr/bin/env python3
"""Three-seed matched continuous benchmark on MMECG (Table 1).

Registered analysis `q1_multiseed_benchmark_aggregation`. Aggregates the 198 sealed
folds -- six models x three seeds (42 / 123 / 2024) x eleven `loso_v2` folds -- under
the binding rules of `docs/METRIC_CONTRACT.md`:

  * per seed: fold-macro over the 11 test subjects, population std across folds;
  * headline: mean +/- std ACROSS SEEDS of those per-seed values;
  * paired tests: metrics averaged over seeds per fold, then a paired two-sided
    Wilcoxon on the 11 seed-averaged fold values against REM-Net;
  * RR/HR only over entries with `rr_coverage >= 0.80`, excluded entries listed;
  * `above_10hz_energy_ratio` as a median, never a bare mean;
  * RMSE/MAE/region MAE are per-record z-score units.

Nothing is trained or selected.

Usage:
  python scripts/aggregate_multiseed_benchmark.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, median, pstdev

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.atomic_state import atomic_csv, atomic_json  # noqa: E402

TAG = "q1_multiseed_benchmark_summary"
OUT = ROOT / "experiments_mmecg" / TAG
SEEDS = (42, 123, 2024)
FOLDS = list(range(1, 12))
MODELS = ("rem_net", "mmecg_tcn", "radarode_mtl", "wavegru_net", "attention_dcgan",
          "cross_modal_conformer")
LABELS = {"rem_net": "REM-Net (ours)", "mmecg_tcn": "Chen-TCN",
          "radarode_mtl": "radarODE-MTL", "wavegru_net": "WaveGRU-Net",
          "attention_dcgan": "Attention-DCGAN", "cross_modal_conformer": "Cross-Modal Conformer"}
METRICS = ("pcc", "r2", "nrmse", "rmse", "mae", "f1_50ms", "f1_100ms", "f1_150ms",
           "qrs_region_mae", "stt_region_mae", "amplitude_variance_ratio")
RATE = ("rr_error_ms", "hr_error_bpm")
RR_GATE = 0.80


def fold_dir(model: str, seed: int, fold: int) -> Path:
    if model == "rem_net":
        return (ROOT / "experiments_mmecg" / f"q1_remnet_continuous_strict_loso_seed{seed}"
                / "L3" / f"fold_{fold:02d}")
    return (ROOT / "experiments_mmecg"
            / f"q1_matched_baselines_continuous_strict_loso_seed{seed}" / model
            / f"fold_{fold:02d}")


def load_all() -> dict:
    """metrics[model][seed][fold]; every fold must be sealed."""
    out = {}
    for model in MODELS:
        out[model] = {}
        for seed in SEEDS:
            out[model][seed] = {}
            for fold in FOLDS:
                d = fold_dir(model, seed, fold)
                state = json.loads((d / "run_state.json").read_text())
                if state["stage"] != "COMPLETE":
                    raise SystemExit(f"{model} seed{seed} fold{fold}: stage {state['stage']}")
                out[model][seed][fold] = json.loads((d / "test_metrics.json").read_text())["metrics"]
    return out


def main() -> None:
    m = load_all()
    n = sum(len(f) for s in m.values() for f in s.values())
    print(f"sealed folds: {n} (6 models x 3 seeds x 11 folds)\n")

    rows, summary = [], {}
    for model in MODELS:
        per_seed = {}
        for seed in SEEDS:
            vals = {k: [m[model][seed][f][k] for f in FOLDS] for k in METRICS}
            per_seed[seed] = {k: mean(v) for k, v in vals.items()}
            per_seed[seed]["fold_std"] = {k: pstdev(v) for k, v in vals.items()}
            rows.append({"model": model, "seed": seed,
                         **{k: round(per_seed[seed][k], 6) for k in METRICS}})
        summary[model] = {
            "label": LABELS[model],
            "across_seeds": {k: {"mean": mean(per_seed[s][k] for s in SEEDS),
                                 "std_across_seeds": pstdev(per_seed[s][k] for s in SEEDS),
                                 "fold_std_seed42": per_seed[42]["fold_std"][k]}
                             for k in METRICS},
            "per_seed_fold_macro": {s: {k: per_seed[s][k] for k in METRICS} for s in SEEDS},
        }
        hf = [m[model][s][f]["above_10hz_energy_ratio"] for s in SEEDS for f in FOLDS]
        summary[model]["above_10hz_energy_ratio"] = {
            "median": median(hf), "min": min(hf), "max": max(hf)}
        gated = [(s, f) for s in SEEDS for f in FOLDS
                 if m[model][s][f]["rr_coverage"] >= RR_GATE]
        excluded = [{"seed": s, "fold": f, "rr_coverage": m[model][s][f]["rr_coverage"]}
                    for s in SEEDS for f in FOLDS if m[model][s][f]["rr_coverage"] < RR_GATE]
        summary[model]["rate_metrics_gated"] = {
            "rule": f"rr_coverage >= {RR_GATE}", "n_entries": len(gated),
            "of_total": len(SEEDS) * len(FOLDS), "excluded": excluded,
            **{k: {"mean": mean(m[model][s][f][k] for s, f in gated),
                   "std": pstdev(m[model][s][f][k] for s, f in gated)} for k in RATE}}

    # Paired tests on seed-averaged fold values.
    seed_avg = {model: {k: np.array([mean(m[model][s][f][k] for s in SEEDS) for f in FOLDS])
                        for k in METRICS} for model in MODELS}
    comparisons = {}
    for model in MODELS:
        if model == "rem_net":
            continue
        comparisons[model] = {}
        for k, higher_is_better in (("pcc", True), ("r2", True), ("f1_100ms", True),
                                    ("nrmse", False), ("qrs_region_mae", False)):
            a, b = seed_avg["rem_net"][k], seed_avg[model][k]
            d = a - b
            comparisons[model][k] = {
                "rem_minus_baseline": float(d.mean()),
                "wilcoxon_p": float(wilcoxon(a, b).pvalue),
                "rem_net_better_folds": int(((d > 0) if higher_is_better else (d < 0)).sum()),
                "folds": len(FOLDS)}

    OUT.mkdir(parents=True, exist_ok=True)
    atomic_csv(OUT / "per_seed_fold_macro.csv", list(rows[0]), rows)
    atomic_json(OUT / "table1_multiseed_summary.json", {
        "experiment_id": TAG, "seeds": list(SEEDS), "folds": len(FOLDS),
        "sealed_folds": n,
        "headline_convention": "mean +/- std across the three seeds of per-seed fold-macro values",
        "units": "RMSE/MAE/region MAE in per-record z-score units",
        "paired_statistics": "paired two-sided Wilcoxon on 11 seed-averaged fold values; "
                             "five comparisons, uncorrected, descriptive",
        "models": summary, "rem_net_vs_baseline": comparisons})

    print(f"{'model':24s} {'PCC':>16s} {'R2':>16s} {'NRMSE':>14s} {'F1@100':>14s} "
          f"{'QRS MAE':>14s} {'var ratio':>12s}")
    order = sorted(MODELS, key=lambda x: -summary[x]["across_seeds"]["pcc"]["mean"])
    for model in order:
        a = summary[model]["across_seeds"]
        print(f"{LABELS[model]:24s}"
              + "".join(f"  {a[k]['mean']:+.3f}±{a[k]['std_across_seeds']:.3f}"
                        for k in ("pcc", "r2", "nrmse", "f1_100ms", "qrs_region_mae"))
              + f"  {a['amplitude_variance_ratio']['mean']:.3f}")
    print("\nrate metrics (coverage-gated) and >10 Hz ratio:")
    for model in order:
        g = summary[model]["rate_metrics_gated"]
        hf = summary[model]["above_10hz_energy_ratio"]
        print(f"  {LABELS[model]:24s} RR {g['rr_error_ms']['mean']:6.1f} ms  "
              f"HR {g['hr_error_bpm']['mean']:5.2f} bpm  ({g['n_entries']}/{g['of_total']} entries)"
              f"   >10Hz median {hf['median']:.3f} [{hf['min']:.3f}, {hf['max']:.3f}]")
    print("\nREM-Net vs baseline (paired Wilcoxon on 11 seed-averaged folds):")
    for model, c in comparisons.items():
        print(f"  {LABELS[model]:24s} " + " | ".join(
            f"{k}: Δ{v['rem_minus_baseline']:+.4f} p={v['wilcoxon_p']:.3f} "
            f"win {v['rem_net_better_folds']}/11" for k, v in c.items()))


if __name__ == "__main__":
    main()
