"""Frozen subject-level ablation inference; never called during training."""
import numpy as np
from scipy.stats import wilcoxon

def bootstrap_mean(d,n_boot=10000,seed=42):
    d=np.asarray(d,float)
    if not len(d) or not np.isfinite(d).all():raise ValueError('complete finite paired subjects required')
    rng=np.random.default_rng(seed)
    sampled=d[rng.integers(0,len(d),size=(n_boot,len(d)))].mean(1)
    return float(d.mean()),float(np.percentile(sampled,2.5)),float(np.percentile(sampled,97.5))

def signed_rank(d):
    d=np.asarray(d,float)
    if not np.isfinite(d).all():raise ValueError('paired complete subjects required')
    if np.all(d==0):return 1.
    return float(wilcoxon(d,zero_method='wilcox',correction=False,alternative='two-sided',method='auto').pvalue)

def holm(p):
    p=np.asarray(p,float);order=np.argsort(p);result=np.empty(len(p));running=0.
    for rank,i in enumerate(order):
        running=max(running,min(1.,(len(p)-rank)*p[i]));result[i]=running
    return result

def contrast(subjects,weights,metric):
    ids=sorted(set(subjects[next(iter(weights))]))
    if any(set(subjects[k])!=set(ids) for k in weights):raise ValueError('subject sets differ')
    d=np.array([sum(w*subjects[k][s][metric] for k,w in weights.items()) for s in ids],float)
    if not np.isfinite(d).all():raise ValueError('unassessable subject contrast; do not silently filter')
    return d
