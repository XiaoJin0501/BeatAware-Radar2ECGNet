#!/usr/bin/env python3
"""Two-dataset, subject-balanced scenario distributions from frozen source data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DEFAULT_SOURCE = ROOT / "docs/paper/source_data/2026-10-16_existing_remnet_v1/tables/remnet_seed42_subject_state_metrics.csv"
DEFAULT_OUTPUT = ROOT / "docs/paper/sections/figures/fig_mmecg_subject_wise_results"
EVENT_COLORS = {"q": ("#D55E00", "#F2C1A6"), "r": ("#009E73", "#A6DCC9"),
                "s": ("#CC79A7", "#EBC5DA"), "t": ("#E69F00", "#F5D79A")}
SCENARIOS = {
    "mmecg": (("NB", "NB"), ("IB", "IB"), ("SP", "SP"), ("PE", "PE")),
    "crrvs": (("NB", "Resting"), ("IB", "Valsalva"), ("SP", "Apnea")),
}


def number(value):
    return float(value) if value not in (None, "", "nan", "NA") else np.nan


def reference_points(ecg):
    from src.utils.clinical_endpoints import delineate_signal
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return delineate_signal(ecg, fs=200, method="prominence")


def complete_cw_events(rows, source, data_dir):
    """Stored CW HDF5 has no Q/S/T labels; derive them with frozen prominence."""
    from scripts.run_dual_head_interaction import structured_decode
    from src.utils.clinical_endpoints import match_beats
    snapshot = source.parents[1]
    buckets = {}
    data_dir.mkdir(parents=True, exist_ok=True)
    for fold in range(1, 6):
        with np.load(snapshot / "references/crrvs" / f"fold_{fold:02d}.npz") as z:
            ref = {k:z[k] for k in z.files}
        cache = data_dir / f"crrvs_prominence_reference_fold_{fold:02d}.npz"
        if cache.exists():
            with np.load(cache) as z:
                labels = {k:z[k] for k in z.files}
        else:
            with ProcessPoolExecutor(max_workers=4) as pool:
                points = list(pool.map(reference_points, ref["ecg"], chunksize=16))
            labels = {}
            for m in "qrst":
                values = [np.asarray(p[m],float)[np.isfinite(p[m])] for p in points]
                labels[m+"_values"] = np.concatenate(values)
                labels[m+"_offsets"] = np.r_[0,np.cumsum([len(v) for v in values])]
            np.savez_compressed(cache, **labels)
        with np.load(snapshot / "predictions/crrvs/rem_net/seed_42" /
                     f"fold_{fold:02d}/test_predictions.npz") as z:
            probabilities = z["anchor_probability"]
        for i, prob in enumerate(probabilities):
            key = (str(int(ref["subject_id"][i])), str(ref["state_code"][i]))
            bucket = buckets.setdefault(key,{m:[] for m in "qrst"} |
                {m+"_count":[0,0] for m in "qrst"})
            predicted = structured_decode(prob)
            for m in "qrst":
                actual = labels[m+"_values"][labels[m+"_offsets"][i]:labels[m+"_offsets"][i+1]]
                matched = match_beats(actual, predicted[m], 30)
                bucket[m].extend(abs(actual[a]-predicted[m][b])*5 for a,b in matched.items())
                bucket[m+"_count"][0] += len(actual)
                bucket[m+"_count"][1] += len(matched)
        print(f"CR_RVS reference/event scoring complete: fold {fold}/5", flush=True)
    for r in rows:
        if r["dataset"] != "crrvs":
            continue
        bucket = buckets[(r["subject"],r["state_code"])]
        for m in "qrst":
            eligible,evaluable = bucket[m+"_count"]
            r[m+"_mae_ms"] = str(np.mean(bucket[m])) if bucket[m] else ""
            r[m+"_coverage"] = str(evaluable/eligible) if eligible else ""
            r[m+"_eligible"],r[m+"_evaluable"] = str(eligible),str(evaluable)
    return rows


def box_and_points(ax, values, position, width, edge, fill, rng):
    """Subject dots, IQR boxes for n>=5, and medians only for smaller groups."""
    values = np.asarray(values, float)
    valid = np.isfinite(values)
    values = values[valid]
    if not len(values):
        return {}
    if len(values) >= 5:
        result = ax.boxplot([values], positions=[position], widths=width,
            patch_artist=True, manage_ticks=False, showfliers=False,
            medianprops={"color": "black", "lw": 1.0},
            whiskerprops={"color": edge, "lw": .6},
            capprops={"color": edge, "lw": .6})
        result["boxes"][0].set(facecolor=fill, edgecolor=edge, linewidth=.6)
    else:
        ax.plot([position-.30*width, position+.30*width], [np.median(values)]*2,
                color="black", lw=1.0, zorder=5)
    jitter = rng.uniform(-width*.30, width*.30, len(values))
    if len(values) < 5:
        # Keep the RNG sequence for all other groups unchanged, but separate
        # small-sample observations deterministically around the group center.
        jitter = np.linspace(-width*.30,width*.30,len(values)) if len(values)>1 else np.zeros(1)
    x = position+jitter
    ax.scatter(x,values,s=9,facecolors=edge,edgecolors="white",
               linewidths=.3,zorder=4,alpha=.8)


def add_statistics(ax, dataset, metric, scenarios, statistics, pairwise):
    """Display supplied statistics only; no tests are inferred from the plot."""
    label = statistics.get((dataset, metric))
    if label:
        ax.text(.98, .98, label, transform=ax.transAxes, ha="right", va="top",
                fontsize=6.5, fontstyle="italic", color="#777777", zorder=8)
    aliases = {code.lower():i for i,(code,_) in enumerate(scenarios)}
    aliases.update({name.lower():i for i,(_,name) in enumerate(scenarios)})
    active=[]
    for ds, met, a, b, p in pairwise:
        if ds != dataset or met != metric:
            continue
        if not np.isfinite(p) or not 0 <= p <= 1:
            raise ValueError(f"invalid adjusted p-value: {p}")
        if a.lower() not in aliases or b.lower() not in aliases:
            raise ValueError(f"unknown scene pair: {a}, {b}")
        if p < .05:
            active.append((aliases[a.lower()],aliases[b.lower()],p))
    for level,(a,b,p) in enumerate(active):
        # Brackets sit above the data axes, retaining the requested fixed limits.
        y=1.025+.085*level
        ax.plot([a,a,b,b], [y,y+.025,y+.025,y], transform=ax.get_xaxis_transform(),
                color="#555555", lw=.5, clip_on=False)
        stars="***" if p < .001 else "**" if p < .01 else "*"
        ax.text((a+b)/2, y+.025, stars, transform=ax.get_xaxis_transform(),
                ha="center",va="bottom",fontsize=7,color="#555555",clip_on=False)


def plot(source, output, statistics=None, pairwise=None):
    """statistics={(dataset, metric): text}; pairwise=[(ds,metric,a,b,p_holm)]."""
    def dataset_key(name):
        return str(name).lower().replace("_", "").replace("-", "")
    statistics = {(dataset_key(d),str(m).lower()):v for (d,m),v in (statistics or {}).items()}
    pairwise = [(dataset_key(d),str(m).lower(),str(a),str(b),float(p))
                for d,m,a,b,p in (pairwise or [])]
    with source.open(newline="") as f:
        rows = list(csv.DictReader(f))
    data_dir = output.parent / "source_data"
    corrected = data_dir / (output.name+".csv")
    if any(r["dataset"]=="crrvs" and not r["q_mae_ms"] for r in rows) and corrected.exists():
        with corrected.open(newline="") as f:
            prior=list(csv.DictReader(f))
        if len(prior)==len(rows) and all(r["q_mae_ms"] for r in prior):
            expected={(r['dataset'],r['subject'],r['state_code']) for r in rows}
            if expected=={(r['dataset'],r['subject'],r['state_code']) for r in prior}:
                # Reuse the already audited CW reference scoring; morphology stays frozen.
                old={(r['dataset'],r['subject'],r['state_code']):r for r in prior}
                for r in rows:
                    if r['dataset']=='crrvs':
                        prev=old[(r['dataset'],r['subject'],r['state_code'])]
                        r.update({k:v for k,v in prev.items() if any(k.startswith(m+'_') for m in 'qrst')})
    if any(r["dataset"] == "crrvs" and not r["q_mae_ms"] for r in rows):
        rows = complete_cw_events(rows, source, data_dir)
    assert all(r["seed"] == "42" and r["model"] == "rem_net" for r in rows)
    keys = [(r["dataset"], r["subject"], r["state_code"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("multiple rows per subject/scenario: aggregation must be explicit")
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset":"stix", "svg.fonttype":"none",
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "axes.spines.top": True, "axes.spines.right": True,
        "axes.linewidth": .6, "pdf.fonttype": 42, "ps.fonttype": 42,
        "xtick.direction":"out", "ytick.direction":"out",
        "xtick.major.size":3,"ytick.major.size":3,
        "xtick.major.width":.6,"ytick.major.width":.6})
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.05), sharey="col",
                             gridspec_kw={"width_ratios":[1.8,1,1], "wspace":.34,"hspace":.48})
    rng = np.random.default_rng(42)
    for row_index, dataset in enumerate(("mmecg", "crrvs")):
        scenarios = SCENARIOS[dataset]
        subset = [r for r in rows if r["dataset"] == dataset]
        groups = [[r for r in subset if r["state_code"] == code] for code, _ in scenarios]
        positions = np.arange(len(scenarios))
        for i, group in enumerate(groups):
            for j, event in enumerate("qrst"):
                box_and_points(axes[row_index, 0], [number(r[f"{event}_mae_ms"]) for r in group],
                    i+(j-1.5)*.19, .15, *EVENT_COLORS[event], rng)
            for col,metric,scale in [(1,'rmse',1),(2,'pcc',100)]:
                box_and_points(axes[row_index,col],[scale*float(r[metric]) for r in group],
                    i,.48,"#555555","#E3E3E3",rng)
        for col_index, ylabel in enumerate(("Fiducial timing MAE (ms)", "RMSE (a.u.)", "PCC (%)")):
            ax = axes[row_index, col_index]
            ax.set_ylabel(ylabel)
            ax.set_xticks(positions, [name for _,name in scenarios])
            for label in ax.get_xticklabels():
                label.set_linespacing(1.35)
            for i,group in enumerate(groups):
                # Second line uses physical point spacing so both rows remain
                # consistent despite axes dimensions and tight-bounds export.
                ax.annotate(f"(n={len(group)})",xy=(i,0),xycoords=ax.get_xaxis_transform(),
                    xytext=(0,-(3+3.5+7*1.35)),textcoords="offset points",
                    ha="center",va="top",fontsize=6.5,color="#777777",
                    linespacing=1.35,annotation_clip=False)
            ax.set_xlim(-.57, len(scenarios)-.43)
            ax.tick_params(axis="y", labelleft=True)
            ax.grid(axis="y", color="#AEB4BC", alpha=.25, linewidth=.4)
            ax.set_axisbelow(True)
            ax.text(-.18,1.04,f"({chr(97+row_index*3+col_index)})",
                    transform=ax.transAxes,va="bottom",ha="left",fontweight="bold",fontsize=9)
            add_statistics(ax,dataset,('mae','rmse','pcc')[col_index],scenarios,statistics,pairwise)
    axes[0, 0].set_ylim(0,100)
    axes[0, 1].set_ylim(0,.3)
    axes[0, 2].set_ylim(55,100)
    axes[0,0].legend(handles=[Patch(facecolor=EVENT_COLORS[m][1], edgecolor=EVENT_COLORS[m][0], linewidth=.6,
                              label=m.upper()) for m in "qrst"],
               loc="lower center",bbox_to_anchor=(.5,1.02),ncol=4,frameon=False,
               fontsize=7,handlelength=1.2,columnspacing=1.2,
               borderaxespad=0,labelspacing=.2)
    fig.subplots_adjust(left=.085,right=.985,bottom=.115,top=.925)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".svg", ".png"):
        # Fixed physical width is essential for IEEE figure* inclusion.
        fig.savefig(output.with_suffix(suffix),dpi=600,facecolor="white")
    plt.close(fig)
    # Keep this revision's data next to the figure, independent of mutable report paths.
    data_dir.mkdir(exist_ok=True)
    data_path = data_dir / (output.name + ".csv")
    with data_path.open("w", newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    metadata = {"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "layout": "2 rows x 3 columns; sharey by column", "seed":42,
        "rows": ["MMECG", "CR_RVS"], "columns": ["Q/R/S/T localization MAE (ms)",
            "200-point morphology RMSE (dimensionless)", "200-point morphology PCC x100 (%)"],
        "sampling_unit":"one subject/scenario; boxes: median/IQR and 1.5 IQR whiskers; all subject dots shown",
        "shape_target":"central cycle using stored HDF5 R peaks, resampled to 200 and min-max normalized",
        "event_errors":"matched structured-decoder peaks; conditional MAE; overlapping windows retained",
        "style":"STIX; IEEE double-column 7.16x4.05 in (no tight-bbox resizing); width ratios 1.8:1:1; fixed column limits; small-n groups have dots and median only",
        "subject_connections":False,"subject_highlighting":False,
        "statistics":[{"dataset":d,"metric":m,"text":v} for (d,m),v in statistics.items()],
        "pairwise_holm":[{"dataset":d,"metric":m,"scene_a":a,"scene_b":b,"p_holm":p}
                         for d,m,a,b,p in pairwise],
        "crrvs_reference":"Q/R/S/T extracted with frozen NeuroKit2 prominence from exported reference ECG; per-fold labels cached beside this figure because original CW HDF5 Q/S/T arrays are empty",
        "missing_mae_subject_scenarios":[{"dataset":r["dataset"], "subject":r["subject"],
            "state_code":r["state_code"], "metric":m} for r in rows for m in "qrst"
            if not np.isfinite(number(r[f"{m}_mae_ms"]))],
        "jitter":"seed 42; horizontal only; +/-30% of box width; s=9; alpha=0.8",
        "subjects_by_scenario":{
            ds:{code:len([r for r in rows if r['dataset']==ds and r['state_code']==code])
                for code,_ in spec} for ds,spec in SCENARIOS.items()},
        "caption":"Scenario-wise REM-Net performance on MMECG (top) and CR_RVS (bottom). "
            "Columns show Q/R/S/T fiducial localization MAE, normalized beat-morphology RMSE, and PCC. "
            "Each point represents one held-out subject within a scenario; boxes show medians and IQRs "
            "for n>=5, otherwise only subject dots and a median line are shown. "
            "Q/R/S/T colors denote the four fiducial peak classes. "
            "The two datasets share the y-axis scale within each metric column. "
            "MAE is conditional on matched events; morphology is evaluated on normalized single cycles."}
    (data_dir / (output.name + ".json")).write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps({"pdf":str(output.with_suffix('.pdf')), "png":str(output.with_suffix('.png')),
                      "subjects_by_scenario":metadata['subjects_by_scenario']},indent=2))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--statistics-json",type=Path,
        help='Optional JSON: {"omnibus":[["mmecg","rmse","p = 0.31"]], "pairwise":[["mmecg","pcc","NB","IB",0.02]]}')
    args = parser.parse_args()
    supplied=json.loads(args.statistics_json.read_text()) if args.statistics_json else {}
    plot(args.source,args.output,
         statistics={(d,m):text for d,m,text in supplied.get('omnibus',[])},
         pairwise=supplied.get('pairwise',[]))
