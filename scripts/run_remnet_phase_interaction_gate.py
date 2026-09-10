#!/usr/bin/env python3
"""Run the registered validation-only REM-Net phase interaction gate."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/home/qhh2237/Datasets/MMECG/processed/loso_v2")
OUT = ROOT / "experiments_mmecg" / "q1_remnet_phase_interaction_gate"
SPEC = ROOT / "experiments/specs/q1_remnet_phase_interaction_gate.json"


@dataclass(frozen=True)
class Arm:
    code: str
    arch: str
    beta_peak: float
    canonicalize: bool = False
    condition: bool = False
    use_alignment: bool = False


ARMS = (
    Arm("I0", "rem_align_mtl", 0.3),
    Arm("F00", "rem_phase_mtl", 0.0, False, False),
    Arm("F10", "rem_phase_mtl", 0.3, False, True),
    Arm("F01", "rem_phase_mtl", 0.0, True, False),
    Arm("F11", "rem_phase_mtl", 0.3, True, True),
)


def tag(arm: Arm) -> str:
    return f"q1_remnet_phase_gate_{arm.code.lower()}"


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def preflight(folds: list[int], selected: list[Arm]) -> None:
    spec = json.loads(SPEC.read_text())
    if spec.get("status") != "approved; implemented; unit-tested; awaiting clean commit and pilot launch":
        raise RuntimeError("Unexpected spec status; inspect before launch")
    for fold in folds:
        for split in ("train", "val"):
            for suffix in (".h5", "_wsst8s.h5"):
                path = DATA / f"fold_{fold:02d}" / f"{split}{suffix}"
                if not path.exists():
                    raise FileNotFoundError(path)
        # Development training must not require or inspect test WSST.
        if (DATA / f"fold_{fold:02d}" / "test_wsst8s.h5").exists():
            raise RuntimeError(f"Unexpected test WSST exists for fold {fold}")
    for arm in selected:
        for fold in folds:
            fold_dir = ROOT / "experiments_mmecg" / tag(arm) / f"fold_{fold:02d}"
            if fold_dir.exists():
                raise FileExistsError(f"Refusing to overwrite {fold_dir}")


def train(arm: Arm, fold: int, epochs: int, patience: int) -> None:
    cmd = [
        sys.executable, "scripts/train_mmecg.py",
        "--exp_tag", tag(arm), "--fold_idx", str(fold),
        "--loso_h5_dir", str(DATA), "--n_range_bins", "50",
        "--model_arch", arm.arch, "--epochs", str(epochs),
        "--early_stop_patience", str(patience), "--lr", "0.0001",
        "--weight_decay", "0.0001", "--target_norm", "minmax",
        "--wsst_suffix", "_wsst8s.h5", "--beta_peak", str(arm.beta_peak),
        "--use_shape200_loss", "true", "--lambda_shape200", "0.8",
        "--shape200_pcc_weight", "0.5", "--baseline_hidden", "128",
        "--use_alignment", str(arm.use_alignment).lower(),
        "--use_phase_canonicalizer", str(arm.canonicalize).lower(),
        "--use_event_conditioning", str(arm.condition).lower(),
        "--phase_temperature", "0.10", "--seed", "42",
    ]
    run(cmd)
    run([
        sys.executable, "scripts/evaluate_remnet_validation.py",
        "--exp_tag", tag(arm), "--fold", str(fold),
        "--checkpoint", "best_shape200.pt",
    ])


def aggregate(folds: list[int], selected: list[Arm]) -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, object]] = []
    for arm in selected:
        for fold in folds:
            path = ROOT / "experiments_mmecg" / tag(arm) / f"fold_{fold:02d}/results_validation/metrics.json"
            result = json.loads(path.read_text())
            row: dict[str, object] = {
                "arm": arm.code, "fold": fold,
                "checkpoint_epoch": result["checkpoint_epoch"],
                "num_windows": result["num_windows"],
            }
            row.update({f"shape_{k}": v for k, v in result["shape200"].items()})
            row.update({f"event_{k}": v for k, v in result["event_head"].items()})
            rows.append(row)
    fields = list(rows[0])
    with (OUT / "fold_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)

    summaries = []
    for arm in selected:
        subset = [row for row in rows if row["arm"] == arm.code]
        summary: dict[str, object] = {"arm": arm.code, "n_folds": len(subset)}
        for metric in ("shape_pcc", "shape_rmse", "shape_mae", "shape_r2", "event_f1_50ms"):
            vals = [float(row[metric]) for row in subset]
            summary[f"{metric}_mean"] = mean(vals)
            summary[f"{metric}_std"] = pstdev(vals)
        summaries.append(summary)
    with (OUT / "summary.json").open("w") as f:
        json.dump(summaries, f, indent=2)

    by_arm_fold = {(str(r["arm"]), int(r["fold"])): r for r in rows}
    interaction = []
    if all(code in {arm.code for arm in selected} for code in ("F00", "F10", "F01", "F11")):
        for fold in folds:
            p = {code: float(by_arm_fold[(code, fold)]["shape_pcc"]) for code in ("F00", "F10", "F01", "F11")}
            interaction.append({
                "fold": fold,
                "pcc_difference_in_differences": (p["F11"] - p["F10"]) - (p["F01"] - p["F00"]),
            })
        with (OUT / "interaction.json").open("w") as f:
            json.dump({
                "per_fold": interaction,
                "mean": mean(float(x["pcc_difference_in_differences"]) for x in interaction),
            }, f, indent=2)
    print(f"Wrote aggregate results to {OUT}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", default="1,5,9")
    parser.add_argument("--arms", default="I0,F00,F10,F01,F11")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()
    folds = [int(x) for x in args.folds.split(",")]
    wanted = {x.strip().upper() for x in args.arms.split(",")}
    selected = [arm for arm in ARMS if arm.code in wanted]
    preflight(folds, selected)
    # Fold-major ordering preserves a complete paired comparison if interrupted.
    for fold in folds:
        for arm in selected:
            train(arm, fold, args.epochs, args.patience)
    aggregate(folds, selected)


if __name__ == "__main__":
    main()
