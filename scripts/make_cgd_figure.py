#!/usr/bin/env python3
"""F11: per-fold paired slope + spread for the continuous-gap decomposition audit."""
import json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = json.load(open(ROOT/"experiments_mmecg/q1_continuous_gap_decomposition_audit/decomposition_metrics.json"))
byf = defaultdict(list)
for r in D["records"]: byf[r["fold"]].append(r)
folds = sorted(byf)
def fv(a, m="pcc", key=None):
    key = key or ("native" if a == "C5" else "common")
    return np.array([np.mean([r[f"{a}_{key}_{m}"] for r in byf[f]]) for f in folds])

C0,C1,C2,C3,C4,C5 = (fv(a) for a in ("C0","C1","C2","C3","C4","C5"))
OURS,MUT,WARN,GOOD = "#1f77b4","#7f7f7f","#d62728","#2ca02c"
fig, (a1,a2,a3) = plt.subplots(1,3, figsize=(13,4.2))

# (a) paired slopes: GT vs predicted bounds, at each affine condition
for y1,y2,c,lab in ((C1,C3,OURS,"fixed affine  (C1$\\to$C3)"),(C2,C4,GOOD,"oracle affine (C2$\\to$C4)")):
    for i in range(len(folds)):
        a1.plot([0,1],[y1[i],y2[i]],"-o",color=c,alpha=.55,ms=3.5,lw=1.1)
    a1.plot([],[],"-o",color=c,ms=4,label=lab)
a1.set_xticks([0,1]); a1.set_xticklabels(["GT cycle\nboundaries","predicted\nboundaries"])
a1.set_ylabel("continuous PCC vs original ECG\n(common support)")
a1.set_title("(a) Boundary localization/warping contrast\n11/11 folds, identical window set", fontsize=9.5)
a1.legend(fontsize=7.5, loc="center left"); a1.grid(alpha=.25); a1.set_ylim(-0.05,1.0)

# (b) all arms, fold spread
data=[C0,C1,C2,C3,C4]; labs=["C0\ndense","C1\nGT+fix","C2\nGT+orc","C3\npred+fix","C4\npred+orc"]
bp=a2.boxplot(data, labels=labs, patch_artist=True, widths=.6)
for p,c in zip(bp["boxes"],[MUT,OURS,GOOD,WARN,"#ff7f0e"]):
    p.set_facecolor(c); p.set_alpha(.45)
for i,y in enumerate(data): a2.plot(np.full(len(y),i+1)+np.random.default_rng(0).uniform(-.11,.11,len(y)),y,".",color="k",ms=3,alpha=.5)
a2.set_ylabel("continuous PCC (common support)"); a2.grid(alpha=.25); a2.set_ylim(-0.05,1.0)
a2.set_title("(b) Fold spread by condition\ncommon support 0.351$\\pm$0.039", fontsize=9.5)
a2.tick_params(axis="x", labelsize=7.5)

# (c) interaction
dd_gt, dd_pr = C2-C1, C4-C3
x=np.arange(len(folds))
a3.bar(x-.2, dd_gt, .4, label="amplitude gain @ GT bounds  (C2$-$C1)", color=GOOD, alpha=.8)
a3.bar(x+.2, dd_pr, .4, label="amplitude gain @ predicted bounds (C4$-$C3)", color=WARN, alpha=.8)
a3.set_xticks(x); a3.set_xticklabels([str(f) for f in folds], fontsize=7.5); a3.set_xlabel("MMECG LOSO fold")
a3.set_ylabel("$\\Delta$ PCC from oracle per-cycle affine")
a3.set_title("(c) Amplitude recovery is anchor-conditional\n$\\Delta\\Delta$=+0.135, 95% CI [+0.110,+0.158]", fontsize=9.5)
a3.legend(fontsize=7); a3.grid(alpha=.25, axis="y")

fig.tight_layout()
for ext in ("pdf","png"):
    fig.savefig(ROOT/f"docs/paper/figures/F11_continuous_gap_decomposition.{ext}", dpi=200, bbox_inches="tight")
print("wrote F11_continuous_gap_decomposition.{pdf,png}")
