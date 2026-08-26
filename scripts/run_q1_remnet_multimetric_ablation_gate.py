#!/usr/bin/env python3
"""Run the RQ1 multi-metric REM-Net ablation gate.

This is a pilot gate, not a publishable final run. It trains controlled ablation
arms on representative MMECG folds and evaluates morphology in Shape-200 space.
Component claims must wait for the aggregate table and later metric audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "experiments_mmecg" / "EXPERIMENT_REGISTRY.csv"
OUT_DIR = ROOT / "experiments_mmecg" / "q1_remnet_multimetric_ablation_gate"


@dataclass(frozen=True)
class Arm:
    code: str
    name: str
    model_arch: str
    role: str
    use_alignment: bool = False
    beta_peak: float = 0.3
    use_shape200_loss: bool = True
    lambda_shape200: float = 0.8
    shape200_pcc_weight: float = 0.5
    eval_mode: str = "full_shape200"
    extra_args: tuple[str, ...] = field(default_factory=tuple)


ARMS = (
    Arm(
        code="A0",
        name="wsst_tcn",
        model_arch="wsst_tcn",
        role="Representation-only WSST-TCN baseline",
        beta_peak=0.0,
        eval_mode="full_shape200",
    ),
    Arm(
        code="A1",
        name="rem_event",
        model_arch="rem_event",
        role="Event-structured model without morphology MTL decoder",
        eval_mode="full_shape200",
    ),
    Arm(
        code="A2",
        name="remnet_no_event_loss",
        model_arch="rem_align_mtl",
        role="REM-Net architecture without event supervision",
        beta_peak=0.0,
        eval_mode="shape_head",
    ),
    Arm(
        code="A3",
        name="remnet_no_shape_pcc",
        model_arch="rem_align_mtl",
        role="REM-Net without the Shape-200 PCC loss term",
        shape200_pcc_weight=0.0,
        eval_mode="shape_head",
    ),
    Arm(
        code="A4",
        name="remnet_final",
        model_arch="rem_align_mtl",
        role="Final no-alignment REM-Net configuration",
        eval_mode="shape_head",
    ),
)


def run_capture(cmd: list[str]) -> str:
    print("+ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, flush=True)
    if result.stderr:
        print(result.stderr, file=sys.stderr, flush=True)
    result.check_returncode()
    return result.stdout.strip()


def run_stream(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def git_commit() -> str:
    return run_capture(["git", "rev-parse", "--short", "HEAD"])


def git_branch() -> str:
    return run_capture(["git", "branch", "--show-current"])


def tracked_code_clean() -> bool:
    # Ignore unrelated ICONIP manuscript files; this runner only depends on configs,
    # scripts, src, docs/specs, and the committed experiment code.
    out = run_capture(
        [
            "git",
            "status",
            "--short",
            "--untracked-files=no",
            "--",
            ".",
            ":(exclude)ICONIP2026",
        ]
    )
    return out == ""


def load_registry() -> list[dict[str, str]]:
    with REGISTRY.open(newline="") as f:
        return list(csv.DictReader(f))


def write_registry(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise RuntimeError("Refusing to write an empty registry.")
    with REGISTRY.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def tag_for(arm: Arm) -> str:
    return f"q1_ablation_gate_{arm.code.lower()}_{arm.name}"


def ensure_registry_row(arm: Arm, commit: str, branch: str) -> None:
    tag = tag_for(arm)
    rows = load_registry()
    if any(row.get("tag") == tag for row in rows):
        return
    rows.append(
        {
            "tag": tag,
            "dataset": "MMECG public",
            "protocol": "3-fold strict LOSO multi-metric ablation gate (folds 01/05/09)",
            "category": "pilot",
            "publishable": "false",
            "commit_id": commit,
            "branch": branch,
            "role": f"{arm.code}: {arm.role}",
            "status": "registered",
            "reason": "Controlled RQ1 ablation gate; no causal manuscript claim until audited.",
        }
    )
    write_registry(rows)


def update_registry_status(tag: str, status: str, reason: str) -> None:
    rows = load_registry()
    for row in rows:
        if row.get("tag") == tag:
            row["status"] = status
            row["reason"] = reason
            write_registry(rows)
            return
    raise KeyError(tag)


def train_cmd(tag: str, arm: Arm, fold: int, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "scripts/train_mmecg.py",
        "--exp_tag",
        tag,
        "--fold_idx",
        str(fold),
        "--loso_h5_dir",
        "/home/qhh2237/Datasets/MMECG/processed/loso",
        "--n_range_bins",
        "50",
        "--model_arch",
        arm.model_arch,
        "--epochs",
        str(args.epochs),
        "--early_stop_patience",
        str(args.patience),
        "--lr",
        "0.0001",
        "--weight_decay",
        "0.0001",
        "--narrow_bandpass",
        "true",
        "--target_norm",
        "minmax",
        "--wsst_suffix",
        "_wsst.h5",
        "--beta_peak",
        str(arm.beta_peak),
        "--use_shape200_loss",
        str(arm.use_shape200_loss).lower(),
        "--lambda_shape200",
        str(arm.lambda_shape200),
        "--shape200_pcc_weight",
        str(arm.shape200_pcc_weight),
        "--baseline_hidden",
        "128",
        "--use_alignment",
        str(arm.use_alignment).lower(),
        "--seed",
        str(args.seed),
        *arm.extra_args,
    ]


def eval_cmd(tag: str, arm: Arm, fold: int, device: str) -> list[str]:
    if arm.eval_mode == "shape_head":
        return [
            sys.executable,
            "scripts/evaluate_remalign_shape_head.py",
            "--exp_tag",
            tag,
            "--fold_idx",
            str(fold),
            "--checkpoint",
            "best.pt",
            "--device",
            device,
        ]
    return [
        sys.executable,
        "scripts/evaluate_mmecg_shape200.py",
        "--model_type",
        "continuous",
        "--exp_tag",
        tag,
        "--fold_idx",
        str(fold),
        "--device",
        device,
    ]


def summary_path(tag: str, arm: Arm, fold: int) -> Path:
    fold_dir = ROOT / "experiments_mmecg" / tag / f"fold_{fold:02d}"
    if arm.eval_mode == "shape_head":
        return fold_dir / "results_shape_head" / "global_summary_shape_head.json"
    return fold_dir / "results_shape200" / "global_summary_shape200.json"


def read_metric(path: Path, key_options: tuple[str, ...]) -> float:
    data = json.loads(path.read_text())
    for key in key_options:
        if key in data and data[key] is not None:
            return float(data[key])
    raise KeyError(f"None of {key_options} found in {path}")


def aggregate(folds: list[int]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for arm in ARMS:
        tag = tag_for(arm)
        for fold in folds:
            path = summary_path(tag, arm, fold)
            if not path.exists():
                continue
            rows.append(
                {
                    "arm": arm.code,
                    "tag": tag,
                    "fold": fold,
                    "eval_mode": arm.eval_mode,
                    "pcc": read_metric(path, ("pcc_mean", "pcc")),
                    "rmse": read_metric(path, ("rmse_mean", "rmse")),
                    "mae": read_metric(path, ("mae_mean", "mae")),
                    "r2": read_metric(path, ("r2_mean", "r2")),
                    "summary_path": str(path.relative_to(ROOT)),
                }
            )
    if not rows:
        raise RuntimeError("No ablation summaries found.")
    detail = OUT_DIR / "fold_metrics.csv"
    with detail.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    overall: list[dict[str, object]] = []
    by_arm: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_arm.setdefault(str(row["arm"]), []).append(row)
    for arm in ARMS:
        vals = by_arm.get(arm.code, [])
        if not vals:
            continue
        overall.append(
            {
                "arm": arm.code,
                "tag": tag_for(arm),
                "role": arm.role,
                "eval_mode": arm.eval_mode,
                "n_folds": len(vals),
                "pcc_mean": mean(float(v["pcc"]) for v in vals),
                "pcc_std": pstdev(float(v["pcc"]) for v in vals) if len(vals) > 1 else 0.0,
                "rmse_mean": mean(float(v["rmse"]) for v in vals),
                "mae_mean": mean(float(v["mae"]) for v in vals),
                "r2_mean": mean(float(v["r2"]) for v in vals),
            }
        )
    overall_path = OUT_DIR / "ablation_gate_summary.csv"
    with overall_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(overall[0].keys()))
        writer.writeheader()
        writer.writerows(overall)
    print(f"[aggregate] wrote {detail}")
    print(f"[aggregate] wrote {overall_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", default="1,5,9")
    ap.add_argument("--arms", default="A0,A1,A2,A3,A4")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    ap.add_argument("--skip_existing", action="store_true", default=True)
    args = ap.parse_args()

    if not tracked_code_clean():
        raise SystemExit("Tracked code/docs are dirty. Commit before training.")
    commit = git_commit()
    branch = git_branch()
    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    selected = {x.strip().upper() for x in args.arms.split(",") if x.strip()}
    arms = [arm for arm in ARMS if arm.code in selected]
    if not arms:
        raise SystemExit("No matching arms selected.")

    for arm in arms:
        ensure_registry_row(arm, commit, branch)
    for arm in arms:
        tag = tag_for(arm)
        update_registry_status(tag, "running", "Ablation gate training/evaluation running.")
        for fold in folds:
            sp = summary_path(tag, arm, fold)
            if args.skip_existing and sp.exists():
                print(f"[skip] {tag} fold_{fold:02d} already has {sp.name}", flush=True)
                continue
            run_stream(train_cmd(tag, arm, fold, args))
            run_stream(eval_cmd(tag, arm, fold, args.device))
        update_registry_status(tag, "completed", "Ablation gate folds completed; awaiting audit/promotion decision.")
    aggregate(folds)


if __name__ == "__main__":
    main()
