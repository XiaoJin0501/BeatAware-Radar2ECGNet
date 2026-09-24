#!/usr/bin/env python3
"""Clinical-feature errors from frozen predictions, delineated with NeuroKit2.

Read-only over frozen test predictions; nothing is trained or selected.

Delineation method (METRIC_CONTRACT: QRS/QT errors must state it)
  NeuroKit2 0.2.12, `ecg_clean(method="neurokit")` then
  `ecg_delineate(method="peak")`, identically for reference and reconstruction.
  At 200 Hz the `dwt` method places ECG_R_Offsets on the T wave and ECG_R_Onsets near
  the P wave (median "QRS" 375 ms on reference ECG), and `cwt` returns no QRS
  boundaries, so QRS onset/offset are NOT used. Q and S peaks agree exactly across
  all three methods and are used instead:
    QRS duration   = S peak - Q peak          (Q-S interval; reference median 105-110 ms)
    QT interval    = T offset - Q peak        (reference median 335-405 ms)
    QTc            = QT / sqrt(RR), Bazett, RR = median R-R interval of the window
  Reference ECG is delineated at the reference R peaks; a reconstruction at the R
  peaks NeuroKit2 detects on it (`nk.ecg_peaks`, method "neurokit").

Per window the median over beats is taken; the window error is |reconstruction -
reference|. A window where either side yields no valid interval is MISSING and is
counted in `*_coverage`, never imputed. Fold value = mean over valid windows.

Anchor metrics use greedy one-to-one matching within +/-150 ms
(scripts/evaluate_core_metrics.py): R-F1, R-MAE (mean |offset| of matched anchors),
cycle-length error. REM-Net's two anchor read-outs are kept separate.

Outputs: docs/audits/Q1_CLINICAL_INTERVALS_fold.csv, _scenario.csv
"""
from __future__ import annotations

import csv
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_core_metrics import DATASETS, _clean, fold_dir, match  # noqa: E402

FS, TOL = 200, int(round(0.150 * 200))
AUD = ROOT / "docs" / "audits"


def intervals(x: np.ndarray, rpeaks: np.ndarray):
    """Median Q-S, QT, QTc (ms) over the beats of one window, or NaNs."""
    import neurokit2 as nk
    out = (np.nan, np.nan, np.nan)
    if len(rpeaks) < 2:
        return out
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, w = nk.ecg_delineate(nk.ecg_clean(x, sampling_rate=FS), rpeaks=rpeaks,
                                    sampling_rate=FS, method="peak")
    except Exception:
        return out
    q, s, to = (np.asarray(w.get(k, []), float) for k in
                ("ECG_Q_Peaks", "ECG_S_Peaks", "ECG_T_Offsets"))
    if not (len(q) == len(s) == len(to)) or len(q) == 0:
        return out
    qs, qt = (s - q) * 1000 / FS, (to - q) * 1000 / FS
    qs, qt = qs[np.isfinite(qs) & (qs > 0)], qt[np.isfinite(qt) & (qt > 0)]
    rr = np.median(np.diff(rpeaks)) / FS
    return (float(np.median(qs)) if len(qs) else np.nan,
            float(np.median(qt)) if len(qt) else np.nan,
            float(np.median(qt) / np.sqrt(rr)) if len(qt) and rr > 0 else np.nan)


def job(args):
    from scripts.train_anchor_head import peaks_from_heatmap
    from src.utils.metrics import detect_rpeaks
    ds, fold = args
    cfg = DATASETS[ds]
    with h5py.File(cfg["splits"] / f"fold_{fold:02d}" / "test.h5", "r") as h:
        status = np.array([x.decode() if isinstance(x, bytes) else x for x in h["physistatus"][:]])
        subj = h["subject_id"][:]
    ref = None
    fold_rows, scen_rows = [], []
    for m in cfg["models"]:
        for s in cfg["seeds"]:
            fd = fold_dir(cfg, m, s, fold)
            d = np.load(fd / "test_predictions.npz", allow_pickle=True)
            tm = json.loads((fd / "test_metrics.json").read_text())["metrics"]
            gt, pred = d["ground_truth"], d["prediction"]
            gtp = [_clean(r) for r in d["rpeaks"]]
            if ref is None:
                ref = np.array([intervals(y, g) for y, g in zip(gt, gtp)])
            wav = [_clean(detect_rpeaks(p, fs=FS)) for p in pred]
            rec = np.array([intervals(p, k) for p, k in zip(pred, wav)])
            err = np.abs(rec - ref)                               # (n, 3) QRS, QT, QTc
            readouts = {"waveform": wav}
            if m == "rem_net":
                readouts["event"] = [_clean(peaks_from_heatmap(e)) for e in d["event_probability"]]
            for ro, pk in readouts.items():
                tp = fp = fn = 0
                offs, cl = [], np.full(len(pk), np.nan)
                per_w = []
                for i, (p, g) in enumerate(zip(pk, gtp)):
                    a, b, o = match(p, g, TOL)
                    tp += a; fn += b; fp += len(p) - a; offs += o
                    if len(p) >= 2 and len(g) >= 2:
                        cl[i] = abs(np.mean(np.diff(p)) - np.mean(np.diff(g))) * 1000 / FS
                    per_w.append((a, b, len(p) - a, o))
                pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
                okcl = np.isfinite(cl)
                row = {"dataset": ds, "model": m, "seed": s, "fold": fold, "readout": ro,
                       "pcc": tm["pcc"], "rmse": tm["rmse"], "r2": tm["r2"],
                       "r_f1": 2 * pr * rc / max(pr + rc, 1e-8),
                       "r_mae_ms": float(np.mean(offs)) if offs else np.nan,
                       "cycle_length_error_ms": float(np.mean(cl[okcl])) if okcl.any() else np.nan,
                       "cl_coverage": float(okcl.mean())}
                for k, name in enumerate(("qrs", "qt", "qtc")):
                    v = err[:, k]
                    row[f"{name}_error_ms"] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
                    row[f"{name}_coverage"] = float(np.isfinite(v).mean())
                fold_rows.append(row)
                for code, name in cfg["scen"].items():
                    for su in np.unique(subj):
                        idx = np.flatnonzero((status == code) & (subj == su))
                        if len(idx) == 0:
                            continue
                        o = [x for i in idx for x in per_w[i][3]]
                        c = cl[idx]
                        scen_rows.append({
                            "dataset": ds, "model": m, "seed": s, "fold": fold, "readout": ro,
                            "subject": int(su), "scenario": name, "n_windows": len(idx),
                            "r_mae_ms": float(np.mean(o)) if o else np.nan,
                            "cycle_length_error_ms": float(np.nanmean(c)) if np.isfinite(c).any() else np.nan,
                            **{f"{n}_error_ms": (float(np.nanmean(err[idx, k]))
                                                 if np.isfinite(err[idx, k]).any() else np.nan)
                               for k, n in enumerate(("qrs", "qt", "qtc"))}})
    ref_ok = np.isfinite(ref).mean(0)
    return fold_rows, scen_rows, (ds, fold, ref_ok.tolist(), np.nanmedian(ref, 0).tolist())


def main() -> None:
    jobs = [(ds, f) for ds, cfg in DATASETS.items() for f in cfg["folds"]]
    fold_rows, scen_rows = [], []
    with ProcessPoolExecutor(max_workers=16) as ex:
        for fr, sr, (ds, fold, ok, med) in ex.map(job, jobs):
            fold_rows += fr; scen_rows += sr
            print(f"{ds} fold {fold:2d}: reference delineated QRS/QT/QTc {ok[0]:.2f}/{ok[1]:.2f}/{ok[2]:.2f}; "
                  f"medians {med[0]:.0f}/{med[1]:.0f}/{med[2]:.0f} ms", flush=True)
    for name, rows in (("Q1_CLINICAL_INTERVALS_fold.csv", fold_rows),
                       ("Q1_CLINICAL_INTERVALS_scenario.csv", scen_rows)):
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(AUD / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
        print(f"wrote {AUD / name} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
