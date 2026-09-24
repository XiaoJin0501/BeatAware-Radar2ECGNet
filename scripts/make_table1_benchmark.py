#!/usr/bin/env python3
"""Write Table I (dual-panel clinical benchmark) from the clinical-interval CSV.

Every cell is computed here from docs/audits/Q1_CLINICAL_INTERVALS_fold.csv; nothing
is typed by hand. MMECG: fold-macro per seed, mean +/- s.d. over three seeds
(METRIC_CONTRACT headline convention). CR_RVS: fold-macro, seed 42.
Cycle-length error uses only folds with anchor coverage >= 0.80.
Significance marks (MMECG only): paired two-sided Wilcoxon over the eleven
seed-averaged folds against REM-Net's anchor-head row, uncorrected.

Output: docs/paper/sections/Table_I_Benchmark.tex
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "docs" / "audits" / "Q1_CLINICAL_INTERVALS_fold.csv"
OUT = ROOT / "docs" / "paper" / "sections" / "Table_I_Benchmark.tex"
COLS = [("pcc", "PCC", True, 3, 1), ("rmse", "RMSE (z)", False, 3, 1),
        ("r_f1", "R-F1", True, 3, 1), ("r_mae_ms", "R-MAE (ms)", False, 1, 1),
        ("qrs_error_ms", "QRS err.\\ (ms)", False, 1, 1), ("qt_error_ms", "QT err.\\ (ms)", False, 1, 1),
        ("cycle_length_error_ms", "Cycle len.\\ err.\\ (ms)", False, 1, 1)]
SHARED = {"pcc", "rmse", "qrs_error_ms", "qt_error_ms"}      # waveform-only, read-out independent
PANELS = {
    "mmecg": dict(title="MMECG, 77-GHz FMCW", seeds=(42, 123, 2024), folds=range(1, 12),
                  rows=[("mmecg_tcn", "waveform", "Chen-TCN~\\cite{chenmmwave}", "0.61"),
                        ("radarode_mtl", "waveform", "radarODE-MTL~\\cite{radarodemt}", "41.6"),
                        ("cross_modal_conformer", "waveform", "Cross-Modal Conformer~\\cite{cross-modalautoencoder}", "0.42"),
                        ("wavegru_net", "waveform", "WaveGRU-Net~\\cite{xu2025wavegrunet}", "0.71"),
                        ("attention_dcgan", "waveform", "Attention-DCGAN~\\cite{ReconstructingECG-like}", "4.35")],
                  rem="1.44"),
    "crrvs": dict(title="CR\\_RVS, 24-GHz CW", seeds=(42,), folds=range(1, 6),
                  rows=[("mmecg_tcn", "waveform", "Chen-TCN~\\cite{chenmmwave}", "0.59"),
                        ("wavegru_net", "waveform", "WaveGRU-Net~\\cite{xu2025wavegrunet}", "0.70")],
                  rem="1.38"),
}


def load():
    rows = list(csv.DictReader(open(CSV)))
    f = lambda x: float(x) if x not in ("", "nan") else np.nan
    return {(r["dataset"], r["model"], r["readout"], int(r["seed"]), int(r["fold"])):
            {k: f(v) for k, v in r.items() if k not in ("dataset", "model", "readout", "seed", "fold")}
            for r in rows}


def stats(D, ds, m, ro, seeds, folds, key):
    per_seed, per_fold = [], []
    for s in seeds:
        v = [D[(ds, m, ro, s, f)][key] for f in folds
             if not (key == "cycle_length_error_ms" and D[(ds, m, ro, s, f)]["cl_coverage"] < 0.80)]
        per_seed.append(np.nanmean(v))
    for f in folds:
        v = [D[(ds, m, ro, s, f)][key] for s in seeds
             if not (key == "cycle_length_error_ms" and D[(ds, m, ro, s, f)]["cl_coverage"] < 0.80)]
        per_fold.append(np.nanmean(v) if v else np.nan)
    return np.array(per_seed), np.array(per_fold)


def main() -> None:
    D = load()
    lines = []
    cov = {}
    for ds, P in PANELS.items():
        systems = [(m, ro, lab, par) for m, ro, lab, par in P["rows"]] + \
                  [("rem_net", "event", "\\remnet, anchor head", P["rem"]),
                   ("rem_net", "waveform", "\\remnet, waveform peaks", P["rem"])]
        S = {(m, ro): {k: stats(D, ds, m, ro, P["seeds"], P["folds"], k) for k, *_ in COLS}
             for m, ro, *_ in systems}
        for k in ("qrs_coverage", "qt_coverage"):
            cov[(ds, k)] = [np.nanmean([D[(ds, m, "waveform", s, f)][k] for s in P["seeds"]
                                        for f in P["folds"]]) for m in {x[0] for x in systems}]
        best = {}
        for k, _, up, *_ in COLS:
            means = {sk: np.mean(S[sk][k][0]) for sk in S}
            best[k] = (max if up else min)(means, key=means.get)
        lines.append(f"    \\midrule\n    \\multicolumn{{9}}{{l}}{{\\textit{{{P['title']}}}}} \\\\")
        for m, ro, lab, par in systems:
            cells = []
            for k, _, up, dec, _ in COLS:
                ps, pf = S[(m, ro)][k]
                val = np.mean(ps)
                txt = f"{val:.{dec}f}"
                if best[k] == (m, ro) or (k in SHARED and m == "rem_net" and best[k][0] == "rem_net"):
                    txt = f"\\mathbf{{{txt}}}"
                if len(P["seeds"]) > 1:
                    txt += f" \\pm {np.std(ps):.{dec}f}"
                mark = ""
                if len(P["seeds"]) > 1 and m != "rem_net":
                    ref = S[("rem_net", "event")][k][1]
                    ok = np.isfinite(ref) & np.isfinite(pf)
                    if ok.sum() >= 6 and not np.allclose(ref[ok], pf[ok]):
                        p = wilcoxon(ref[ok], pf[ok]).pvalue
                        rem_better = np.mean(ref[ok]) > np.mean(pf[ok]) if up else np.mean(ref[ok]) < np.mean(pf[ok])
                        if p < 0.05:
                            mark = "^{\\ast}" if rem_better else "^{\\dagger}"
                if m == "rem_net" and ro == "waveform" and k in SHARED:
                    cells.append("\\multicolumn{1}{c}{\\textquotedbl}")
                    continue
                cells.append(f"${txt}{mark}$")
            lines.append(f"    {lab} & {par} & " + " & ".join(cells) + " \\\\")
    cov_txt = ", ".join(f"{ds.upper().replace('CRRVS','CR_RVS')} QRS {100*min(cov[(ds,'qrs_coverage')]):.0f}--{100*max(cov[(ds,'qrs_coverage')]):.0f}\\%, "
                        f"QT {100*min(cov[(ds,'qt_coverage')]):.0f}--{100*max(cov[(ds,'qt_coverage')]):.0f}\\%"
                        for ds in PANELS).replace("_", "\\_")
    head = "    Method & Params (M) & " + " & ".join(
        f"{lab} {'$\\uparrow$' if up else '$\\downarrow$'}" for _, lab, up, *_ in COLS) + " \\\\"
    tex = rf"""% Generated by scripts/make_table1_benchmark.py from
% docs/audits/Q1_CLINICAL_INTERVALS_fold.csv -- do not edit by hand.
\begin{{table*}}[t]
  \centering
  \caption{{Clinical benchmark on two radar modalities under strict subject-independent evaluation. PCC and RMSE (per-record $z$ units) on the full \SI{{8}}{{\second}} window. Anchor metrics use greedy one-to-one matching within $\pm$\SI{{150}}{{\milli\second}}: R-F1, R-peak mean absolute timing error (R-MAE), and cycle-length error ($|$predicted $-$ reference mean RR$|$ per window, folds with anchor coverage $\geq 0.80$). QRS (Q--S interval) and QT errors are absolute differences of per-window medians, both signals delineated with NeuroKit2 (\texttt{{peak}} method); windows where either side cannot be delineated are excluded, not imputed (valid windows: {cov_txt}); on MMECG no window of \remnet's reconstruction can be delineated in two seed--folds (fold~7 of seed~42 and fold~2 of seed~2024, where waveform R-peak detection fails), so \remnet's QRS and QT entries exclude its weakest folds and are favourable to it. MMECG: mean $\pm$ s.d.\ over three seeds of fold-macro values; CR\_RVS: seed 42. \remnet's two anchor read-outs are listed separately; waveform-only columns (PCC, RMSE, QRS, QT) do not depend on the read-out (\textquotedbl). Best per column in bold. MMECG baseline marks: paired Wilcoxon over eleven seed-averaged folds against \remnet\ (anchor head), uncorrected, $^{{\ast}}$\remnet\ significantly better, $^{{\dagger}}$significantly worse ($p<0.05$).}}
  \label{{tab:benchmark}}
  \setlength{{\tabcolsep}}{{3.6pt}}
  \begin{{tabular}}{{lcccccccc}}
    \toprule
{head}
{chr(10).join(lines)}
    \bottomrule
  \end{{tabular}}
\end{{table*}}
"""
    OUT.write_text(tex)
    print(f"wrote {OUT}")
    for ds, P in PANELS.items():
        print(f"\n{P['title']}")
        systems = [(m, ro) for m, ro, *_ in P["rows"]] + [("rem_net", "event"), ("rem_net", "waveform")]
        for m, ro in systems:
            print(f"  {m+'/'+ro:32s} " + "  ".join(
                f"{k.split('_')[0]}={np.mean(stats(D, ds, m, ro, P['seeds'], P['folds'], k)[0]):.3f}"
                for k, *_ in COLS))


if __name__ == "__main__":
    main()
