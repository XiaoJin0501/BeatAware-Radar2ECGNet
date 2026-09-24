#!/usr/bin/env python3
"""Core metric set, aligned with radarODE-MTL (TIM 2025) and mmJEPA-ECG.

Read-only evaluator over frozen test predictions; nothing is trained or selected.

  Morphology  Shape PCC, Shape R^2  single centre cycle cut at reference R peaks,
                                    resampled to 200 samples, min-max normalised
                                    (radarODE-MTL Task 1, "recovered single-cycle
                                    ECG pieces"); reported beside the radar-free
                                    template floor, as METRIC_CONTRACT requires.
              Long-term PCC, R^2    full 8-s window, stored units (radarODE-MTL
                                    Fig. 7 long-term evaluation); from the frozen
                                    evaluator's test_metrics.json.
  Anchors     MDR                   fraction of reference R peaks with no predicted
                                    anchor within +/-150 ms, greedy one-to-one
                                    matching (radarODE-MTL: deviations < 150 ms
                                    are not "missed detected").
              Timing Error          mean |offset| of the matched anchors, ms.
  Interval    Cycle Length Error    |mean predicted RR - mean reference RR| per
                                    window, ms (radarODE-MTL "PPI Error"), folds
                                    gated at rr_coverage >= 0.80.

Anchor read-outs are kept apart (docs/paper/TERMINOLOGY.md): `waveform` detects
peaks on the reconstruction and exists for every system; `event` reads REM-Net's
anchor head. Baselines carry no anchor head in the saved outputs.

Outputs (docs only, no experiment directory is created):
  docs/audits/Q1_CORE_METRICS_fold.csv       one row per system x seed x fold x read-out
  docs/audits/Q1_CORE_METRICS_scenario.csv   one row per system x seed x subject x scenario

Usage:
  python scripts/evaluate_core_metrics.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_anchor_head import peaks_from_heatmap  # noqa: E402
from scripts.train_continuous_ecg_pilot import _centre_cycle  # noqa: E402
from src.utils.metrics import detect_rpeaks  # noqa: E402

FS, SHAPE_LEN, TOL_MS, GATE = 200, 200, 150, 0.80
E = ROOT / "experiments_mmecg"
AUD = ROOT / "docs" / "audits"
DATASETS = {
    "mmecg": dict(splits=Path("/home/qhh2237/Datasets/MMECG/processed/loso_v2"),
                  folds=range(1, 12), seeds=(42, 123, 2024),
                  models=("rem_net", "mmecg_tcn", "radarode_mtl", "cross_modal_conformer",
                          "wavegru_net", "attention_dcgan"),
                  rem="q1_remnet_continuous_strict_loso_seed{s}",
                  base="q1_matched_baselines_continuous_strict_loso_seed{s}",
                  scen={"NB": "Normal breathing", "IB": "Irregular breathing",
                        "PE": "Post-exercise", "SP": "Sleep"}),
    "crrvs": dict(splits=Path("/home/qhh2237/Datasets/CR_RVS/processed/q1_continuous_kfold"),
                  folds=range(1, 6), seeds=(42,),
                  models=("rem_net", "mmecg_tcn", "wavegru_net"),
                  rem="q1_remnet_continuous_crrvs_seed{s}",
                  base="q1_matched_baselines_continuous_crrvs_seed{s}",
                  scen={"NB": "Resting", "IB": "Valsalva", "SP": "Apnea"}),
}


def fold_dir(cfg, model, seed, fold) -> Path:
    if model == "rem_net":
        return E / cfg["rem"].format(s=seed) / "L3" / f"fold_{fold:02d}"
    return E / cfg["base"].format(s=seed) / model / f"fold_{fold:02d}"


def _clean(p) -> np.ndarray:
    p = np.sort(np.unique(np.asarray(p, np.int64)))
    return p[(p >= 0) & (p < 1600)]


def _norm01(z):
    return (z - z.min()) / max(z.max() - z.min(), 1e-8)


def match(pred: np.ndarray, gt: np.ndarray, tol: int):
    """Greedy nearest one-to-one matching within tol; returns (tp, fn, |offsets| ms)."""
    if gt.size == 0:
        return 0, 0, []
    if pred.size == 0:
        return 0, int(gt.size), []
    d = np.abs(gt[:, None] - pred[None, :])
    mg, mp, offs = np.zeros(gt.size, bool), np.zeros(pred.size, bool), []
    for f in np.argsort(d.ravel()):
        gi, pi = divmod(int(f), pred.size)
        if mg[gi] or mp[pi] or d[gi, pi] > tol:
            continue
        mg[gi] = mp[pi] = True
        offs.append(float(d[gi, pi]) * 1000 / FS)
    return int(mg.sum()), int((~mg).sum()), offs


def shape_scores(p, y, r):
    """Centre-cycle PCC and R^2, or None where no complete cycle exists."""
    b = _centre_cycle(r)
    if b is None:
        return None
    a, c = b
    grid, src = np.linspace(0, 1, SHAPE_LEN), np.linspace(0, 1, c - a)
    ps, ys = _norm01(np.interp(grid, src, p[a:c])), _norm01(np.interp(grid, src, y[a:c]))
    pc, yc = ps - ps.mean(), ys - ys.mean()
    den = np.sqrt((pc ** 2).sum() * (yc ** 2).sum())
    if den <= 1e-12:
        return None
    return float((pc * yc).sum() / den), float(1 - ((ps - ys) ** 2).sum() / (yc ** 2).sum())


def window_record(pred_peaks, gt_peaks):
    tol = int(round(TOL_MS * FS / 1000))
    tp, fn, offs = match(pred_peaks, gt_peaks, tol)
    cl = np.nan
    if len(pred_peaks) >= 2 and len(gt_peaks) >= 2:
        cl = abs(np.mean(np.diff(pred_peaks)) - np.mean(np.diff(gt_peaks))) * 1000 / FS
    return tp, fn, offs, cl


def summarise(recs):
    tp = sum(r[0] for r in recs); fn = sum(r[1] for r in recs)
    offs = [o for r in recs for o in r[2]]
    cl = np.array([r[3] for r in recs], float)
    ok = np.isfinite(cl)
    return {"mdr": fn / max(tp + fn, 1),
            "timing_error_ms": float(np.mean(offs)) if offs else np.nan,
            "cycle_length_error_ms": float(np.mean(cl[ok])) if ok.any() else np.nan,
            "cl_coverage": float(ok.mean()),
            "n_ref_anchors": tp + fn}


def template_floor(splits: Path, fold: int):
    """Radar-free control on the same single-cycle protocol: training-fold mean beat."""
    grid = np.linspace(0, 1, SHAPE_LEN)

    def beats(path):
        out = []
        with h5py.File(path, "r") as h:
            ecg, peaks = h["ecg"][:, 0], h["rpeak_indices"][:]
        for e, pk in zip(ecg, peaks):
            b = _centre_cycle(_clean(pk))
            if b is None:
                continue
            a, c = b
            out.append(_norm01(np.interp(grid, np.linspace(0, 1, c - a), e[a:c])))
        return np.asarray(out)

    tmpl = _norm01(beats(splits / f"fold_{fold:02d}" / "train.h5").mean(0))
    test = beats(splits / f"fold_{fold:02d}" / "test.h5")
    tc = tmpl - tmpl.mean()
    pcc = [float((tc * (t - t.mean())).sum() / np.sqrt((tc ** 2).sum() * ((t - t.mean()) ** 2).sum()))
           for t in test]
    r2 = [float(1 - ((tmpl - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()) for t in test]
    return float(np.mean(pcc)), float(np.mean(r2))


def main() -> None:
    fold_rows, scen_rows = [], []
    for ds, cfg in DATASETS.items():
        for fold in cfg["folds"]:
            with h5py.File(cfg["splits"] / f"fold_{fold:02d}" / "test.h5", "r") as h:
                status = np.array([x.decode() if isinstance(x, bytes) else x
                                   for x in h["physistatus"][:]])
                subj = h["subject_id"][:]
            f_pcc, f_r2 = template_floor(cfg["splits"], fold)
            fold_rows.append({"dataset": ds, "model": "template_floor", "seed": "", "fold": fold,
                              "readout": "", "shape_pcc": f_pcc, "shape_r2": f_r2})
            for m in cfg["models"]:
                for s in cfg["seeds"]:
                    fd = fold_dir(cfg, m, s, fold)
                    d = np.load(fd / "test_predictions.npz", allow_pickle=True)
                    tm = json.loads((fd / "test_metrics.json").read_text())["metrics"]
                    pred, gt = d["prediction"], d["ground_truth"]
                    gtp = [_clean(r) for r in d["rpeaks"]]
                    sh = [shape_scores(p, y, r) for p, y, r in zip(pred, gt, gtp)]
                    sh = [x for x in sh if x is not None]
                    readouts = {"waveform": [_clean(detect_rpeaks(p, fs=FS)) for p in pred]}
                    if m == "rem_net":
                        readouts["event"] = [_clean(peaks_from_heatmap(e))
                                             for e in d["event_probability"]]
                    for ro, pk in readouts.items():
                        recs = [window_record(p, g) for p, g in zip(pk, gtp)]
                        fold_rows.append({
                            "dataset": ds, "model": m, "seed": s, "fold": fold, "readout": ro,
                            "shape_pcc": float(np.mean([x[0] for x in sh])),
                            "shape_r2": float(np.mean([x[1] for x in sh])),
                            "longterm_pcc": tm["pcc"], "longterm_r2": tm["r2"],
                            **summarise(recs)})
                        for code, name in cfg["scen"].items():
                            for su in np.unique(subj):
                                idx = np.flatnonzero((status == code) & (subj == su))
                                if len(idx) == 0:
                                    continue
                                scen_rows.append({
                                    "dataset": ds, "model": m, "seed": s, "fold": fold,
                                    "readout": ro, "subject": int(su), "scenario": name,
                                    "n_windows": len(idx),
                                    "longterm_r2_median": float(np.median(
                                        [1 - ((pred[i] - gt[i]) ** 2).sum()
                                         / ((gt[i] - gt[i].mean()) ** 2).sum() for i in idx])),
                                    **summarise([recs[i] for i in idx])})
            print(f"{ds} fold {fold:2d} done", flush=True)

    for name, rows in (("Q1_CORE_METRICS_fold.csv", fold_rows),
                       ("Q1_CORE_METRICS_scenario.csv", scen_rows)):
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(AUD / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {AUD / name} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
