#!/usr/bin/env python3
"""Summarize fold-level MMECG shape-200 evaluation results.

The script reads `global_summary_shape200.json` files produced by
`scripts/evaluate_mmecg_shape200.py` and writes a fold-level CSV plus an
overall JSON summary. It is intentionally lightweight so reproduced baselines
such as radarODE-MTL can be frozen and cited without rerunning training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = (
    "pcc_raw_mean",
    "pcc_raw_median",
    "rmse_norm_mean",
    "mae_norm_mean",
    "r2_mean",
    "qualified_monitoring_rate",
    "rpeak_f1_150ms_mean",
    "r_peak_error_ms_mean",
    "q_peak_error_ms_mean",
    "s_peak_error_ms_mean",
    "t_peak_error_ms_mean",
    "qrs_duration_error_ms_mean",
    "qt_interval_error_ms_mean",
)


def _json_load(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _as_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _valid(values: list[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


def _metric_summary(values: list[float]) -> dict[str, float | int | None]:
    vals = _valid(values)
    if not vals:
        return {"mean": None, "std": None, "median": None, "min": None, "max": None, "n_valid": 0}
    return {
        "mean": mean(vals),
        "std": pstdev(vals) if len(vals) > 1 else 0.0,
        "median": median(vals),
        "min": min(vals),
        "max": max(vals),
        "n_valid": len(vals),
    }


def _discover_fold_files(exp_dir: Path) -> list[Path]:
    return sorted(exp_dir.glob("fold_*/results_shape200/global_summary_shape200.json"))


def summarize(exp_tag: str, output_dir: Path, metrics: tuple[str, ...]) -> dict[str, Any]:
    exp_dir = ROOT / "experiments_mmecg" / exp_tag
    if not exp_dir.exists():
        raise FileNotFoundError(exp_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    summary_paths = _discover_fold_files(exp_dir)
    if not summary_paths:
        raise FileNotFoundError(f"No shape-200 summaries found under {exp_dir}")

    rows: list[dict[str, Any]] = []
    for path in summary_paths:
        row = _json_load(path)
        row.setdefault("fold_idx", int(path.parents[1].name.split("_")[-1]))
        row["summary_path"] = str(path.relative_to(ROOT))
        rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_csv = output_dir / f"{exp_tag}_shape200_folds.csv"
    fieldnames = [
        "exp_tag",
        "model_type",
        "protocol",
        "fold_idx",
        "num_segments",
        *metrics,
        "summary_path",
    ]
    with fold_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    overall: dict[str, Any] = {
        "exp_tag": exp_tag,
        "num_folds": len(rows),
        "total_segments": sum(int(row.get("num_segments", 0) or 0) for row in rows),
        "fold_csv": str(fold_csv.relative_to(ROOT)),
        "metrics": {},
        "note": (
            "Fold-level summary of shape-200 morphology evaluation. Peak and interval "
            "metrics in shape-200 space are exploratory and should not be interpreted "
            "as full-window clinical monitoring metrics."
        ),
    }

    for metric in metrics:
        values = [_as_float(row.get(metric)) for row in rows]
        overall["metrics"][metric] = _metric_summary(values)

    overall_json = output_dir / f"{exp_tag}_shape200_overall.json"
    with overall_json.open("w") as f:
        json.dump(overall, f, indent=2)

    return {"fold_csv": str(fold_csv), "overall_json": str(overall_json), "overall": overall}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp_tag", required=True, help="Experiment tag under experiments_mmecg.")
    parser.add_argument(
        "--output_dir",
        default=str(ROOT / "experiments_mmecg" / "q1_supervisor_shape200_snapshot"),
        help="Directory for summary CSV/JSON outputs.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=list(DEFAULT_METRICS),
        help="Metric keys to aggregate from global_summary_shape200.json.",
    )
    args = parser.parse_args()

    result = summarize(args.exp_tag, Path(args.output_dir), tuple(args.metrics))
    print(json.dumps(result["overall"], indent=2))


if __name__ == "__main__":
    main()
