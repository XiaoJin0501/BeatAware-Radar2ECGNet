#!/usr/bin/env python3
"""W3: REM-Net and the five matched baselines on CR_RVS under the continuous contract.

CR_RVS is the external 24 GHz CW dataset: 5 subject-independent folds, one radar
phase channel, a log-STFT time-frequency map, and a per-record z-scored ECG target
(`processed/q1_continuous_kfold`). Every arm runs through the same training scripts,
evaluator, budget, checkpoint rule and test state machine as on MMECG; only the
dataset switch differs. Each fold opens its held-out subjects exactly once.

Usage:
  python scripts/run_crrvs_continuous_benchmark.py --preflight_only
  python scripts/run_crrvs_continuous_benchmark.py --device cuda:0
  python scripts/run_crrvs_continuous_benchmark.py --arms rem_net wavegru_net
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, median, pstdev

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.atomic_state import atomic_csv, atomic_json, atomic_text  # noqa: E402

DATA = Path("/home/qhh2237/Datasets/CR_RVS/processed/q1_continuous_kfold")
REGISTRY = ROOT / "experiments_mmecg" / "EXPERIMENT_REGISTRY.csv"
FOLDS = list(range(1, 6))
# CR_RVS scope (2026-09-19): the native time-domain and MODWT-wavelet families only.
# radarODE-MTL, Attention-DCGAN and the Conformer are MMECG-only; the other CR_RVS
# methods enter as reported literature (docs/audits/Q1_CRRVS_LITERATURE_PROTOCOL_AUDIT.md).
BASELINES = ("mmecg_tcn", "wavegru_net")
ARMS = ("rem_net",) + BASELINES
RR_COVERAGE_GATE = 0.80
METRICS = ["pcc", "rmse", "nrmse", "mae", "r2", "amplitude_variance_ratio",
           "above_10hz_energy_ratio", "f1_50ms", "f1_100ms", "f1_150ms",
           "rr_error_ms", "hr_error_bpm", "rr_coverage", "qrs_region_mae", "stt_region_mae"]


def tags(seed_base: int) -> dict[str, str]:
    return {"rem_net": f"q1_remnet_continuous_crrvs_seed{seed_base}",
            "baselines": f"q1_matched_baselines_continuous_crrvs_seed{seed_base}"}


def fold_dir(arm: str, fold: int, seed_base: int) -> Path:
    t = tags(seed_base)
    if arm == "rem_net":
        return ROOT / "experiments_mmecg" / t["rem_net"] / "L3" / f"fold_{fold:02d}"
    return ROOT / "experiments_mmecg" / t["baselines"] / arm / f"fold_{fold:02d}"


def require_preflight(seed_base: int) -> tuple[str, str]:
    for tag in tags(seed_base).values():
        spec = json.loads((ROOT / "experiments" / "specs" / f"{tag}.json").read_text())
        if spec.get("status") not in {"DATA_READY", "PREFLIGHT_PASSED"}:
            raise SystemExit(f"{tag}: spec status {spec.get('status')!r} does not authorise training")
    dirty = [line for line in subprocess.check_output(
        ["git", "status", "--short", "--untracked-files=no"], text=True).splitlines()
        if line.strip() and not line.endswith("experiments_mmecg/EXPERIMENT_REGISTRY.csv")]
    if dirty:
        raise SystemExit("Refusing to train from a dirty worktree:\n" + "\n".join(dirty))
    for fold in FOLDS:
        subjects = {}
        for split in ("train", "val", "test"):
            path = DATA / f"fold_{fold:02d}" / f"{split}.h5"
            tf = DATA / f"fold_{fold:02d}" / f"{split}_wsst8s.h5"
            if not path.exists() or not tf.exists():
                raise SystemExit(f"Missing CR_RVS input: {path} / {tf}")
            with h5py.File(tf, "r") as h:
                if h.attrs.get("support") != "full_window_samples_0_1600":
                    raise SystemExit(f"Time-frequency map is not full-window: {tf}")
            with h5py.File(path, "r") as h:
                subjects[split] = set(h["subject_id"][:].tolist())
                ecg = h["ecg"][: min(64, h["ecg"].shape[0])]
                if float(ecg.min()) >= 0.0 and float(ecg.max()) <= 1.0:
                    raise SystemExit(f"{path}: ECG looks min-max normalised, not z-scored")
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            if subjects[a] & subjects[b]:
                raise SystemExit(f"fold {fold}: subjects shared between {a} and {b}")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                     text=True).strip()
    return commit, branch


def update_registry(seed_base: int, commit: str, branch: str, status: str) -> None:
    lines = REGISTRY.read_text().splitlines()
    rows = {
        tags(seed_base)["rem_net"]:
            (f'{tags(seed_base)["rem_net"]},CR_RVS / Nature,5-fold subject-independent continuous; '
             f'validation selection then one-time test,final_candidate,false,{commit},{branch},'
             f'REM-Net B1 + L3 on CR_RVS (external CW validation),{status},'
             f'"W3. Per-record z-scored ECG target (q1_continuous_kfold); 1-channel phase, '
             f'log-STFT input; same evaluator and rules as MMECG."'),
        tags(seed_base)["baselines"]:
            (f'{tags(seed_base)["baselines"]},CR_RVS / Nature,5-fold subject-independent '
             f'continuous; validation selection then one-time test,baseline,false,{commit},'
             f'{branch},Matched-protocol continuous baselines on CR_RVS (adapted),{status},'
             f'"W3. Chen-TCN and WaveGRU-Net with their MMECG adaptations; in_bins = 1. '
             f'Other CR_RVS methods are reported literature only."'),
    }
    for tag, row in rows.items():
        for i, line in enumerate(lines):
            if line.startswith(tag + ","):
                lines[i] = row
                break
        else:
            lines.append(row)
    atomic_text(REGISTRY, "\n".join(lines) + "\n")


def run_arm(arm: str, seed_base: int, device: str) -> None:
    t = tags(seed_base)
    for fold in FOLDS:
        out = fold_dir(arm, fold, seed_base)
        state_path = out / "run_state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else None
        if state and state.get("stage") == "COMPLETE":
            if not (out / "test_metrics.json").exists():
                raise SystemExit(f"Corrupt COMPLETE fold without metrics: {out}")
            print(f"[skip] {arm} fold {fold:02d} already sealed", flush=True)
            continue
        if arm == "rem_net":
            cmd = [sys.executable, "scripts/train_continuous_ecg_pilot.py", "--arm", "B1",
                   "--loss_arm", "L3", "--dataset", "crrvs", "--fold", str(fold),
                   "--exp_tag", t["rem_net"], "--final_test", "--wsst_source", "per_split",
                   "--seed_base", str(seed_base), "--device", device]
        else:
            cmd = [sys.executable, "scripts/train_continuous_baseline.py", "--baseline", arm,
                   "--dataset", "crrvs", "--fold", str(fold), "--seed_base", str(seed_base),
                   "--final_test", "--device", device]
        print("+ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)


def aggregate(seed_base: int) -> dict:
    per_arm = {}
    for arm in ARMS:
        rows = []
        for fold in FOLDS:
            path = fold_dir(arm, fold, seed_base) / "test_metrics.json"
            if not path.exists():
                break
            rows.append({"fold": fold, **{k: json.loads(path.read_text())["metrics"].get(k)
                                          for k in METRICS}})
        if len(rows) == len(FOLDS):
            per_arm[arm] = rows
    out = ROOT / "experiments_mmecg" / tags(seed_base)["baselines"]
    out.mkdir(parents=True, exist_ok=True)
    table = [{"arm": arm, **r} for arm, rows in per_arm.items() for r in rows]
    if table:
        atomic_csv(out / "crrvs_fold_metrics_all_arms.csv", list(table[0]), table)

    def col(rows, k):
        return [float(r[k]) for r in rows]

    summary = {}
    for arm, rows in per_arm.items():
        gated = [r for r in rows if float(r["rr_coverage"]) >= RR_COVERAGE_GATE]
        summary[arm] = {
            "metrics": {k: {"mean": mean(col(rows, k)), "std_population": pstdev(col(rows, k))}
                        for k in METRICS},
            "above_10hz_energy_ratio_median": median(col(rows, "above_10hz_energy_ratio")),
            "rate_metrics_gated": {
                "n_folds": len(gated),
                "excluded_folds": [r["fold"] for r in rows if r not in gated],
                "rr_error_ms": mean(float(r["rr_error_ms"]) for r in gated) if gated else None,
                "hr_error_bpm": mean(float(r["hr_error_bpm"]) for r in gated) if gated else None}}
    comparisons = {}
    if "rem_net" in per_arm:
        ref = per_arm["rem_net"]
        for arm, rows in per_arm.items():
            if arm == "rem_net":
                continue
            comparisons[arm] = {}
            for k in ("pcc", "r2", "f1_100ms"):
                d = np.array(col(ref, k)) - np.array(col(rows, k))
                comparisons[arm][k] = {"rem_minus_baseline": float(d.mean()),
                                       "rem_net_wins": int((d > 0).sum()),
                                       "folds": len(d)}
    result = {"seed_base": seed_base, "dataset": "CR_RVS q1_continuous_kfold",
              "aggregation": "fold-macro over 5 folds, population std",
              "statistics_note": ("With 5 folds a two-sided Wilcoxon cannot reach p < 0.05 "
                                  "(minimum 0.0625); fold wins and effect sizes are reported "
                                  "instead and the three-seed audit carries inference."),
              "arms": summary, "rem_net_vs_baseline": comparisons}
    atomic_json(out / "crrvs_benchmark_summary.json", result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed_base", type=int, default=42, choices=(42, 123, 2024))
    ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--preflight_only", action="store_true")
    ap.add_argument("--aggregate_only", action="store_true")
    a = ap.parse_args()
    if a.aggregate_only:
        print(json.dumps(aggregate(a.seed_base), indent=1)[:3000])
        return
    commit, branch = require_preflight(a.seed_base)
    update_registry(a.seed_base, commit, branch,
                    "preflight-passed" if a.preflight_only else "running")
    if a.preflight_only:
        print(f"[preflight] CR_RVS seed {a.seed_base} authorised at {commit[:8]} on {branch}")
        return
    for arm in a.arms:
        print(f"\n=== {arm} ===", flush=True)
        run_arm(arm, a.seed_base, a.device)
    result = aggregate(a.seed_base)
    update_registry(a.seed_base, commit, branch, "completed; awaiting metric audit")
    for arm, s in result["arms"].items():
        print(f"{arm:24s} PCC {s['metrics']['pcc']['mean']:+.4f} +/- "
              f"{s['metrics']['pcc']['std_population']:.4f}")


if __name__ == "__main__":
    main()
