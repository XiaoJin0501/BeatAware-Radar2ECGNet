"""Parameter-free record assembly. Deployable APIs never accept a reference ECG.

An R-R shape contains S/T of beat k and Q of beat k+1, NOT Q_k/R_k/S_k/T_k.
The uniform sampler is the inverse resampling grid. Optional event-node warping
is an additional, explicitly audited assumption because training has no node warp.
"""
from __future__ import annotations
import numpy as np

from layout import approved_layout

FIELDS = ('q','r','s','t')

def shape_nodes(shape):
    x = np.asarray(shape)
    if x.shape != (200,) or not np.isfinite(x).all():
        raise ValueError('Expected one finite 200-point cycle')
    return np.array([0, 1+np.argmin(x[1:40]), 40+np.argmax(x[40:140]),
                     150+np.argmin(x[150:199]), 199], float)

def representative_r(points):
    r = np.asarray(points['r'], float)
    pairs = [(a,b) for a,b in zip(r[:-1],r[1:]) if b>a+10]
    containing = [p for p in pairs if p[0]<=800<=p[1]]
    if pairs:
        return float((containing or sorted(pairs,key=lambda p: abs((p[0]+p[1])/2-800)))[0][0])
    return float(r[np.argmin(abs(r-800))]) if len(r) else None

def stitch_anchors(length, starts, probabilities, decode):
    """Keep same-window Q/S/T even beyond ownership; retain outside R as context."""
    lookup = {int(s):i for i,s in enumerate(starts)}
    owned, context = [], []
    layout = approved_layout(length)
    for segment in layout:
        start = segment['input_start']; i = lookup[start]
        p = decode(probabilities[i])
        for j,r in enumerate(p['r']):
            absolute = start+float(r)
            row = {k:start+float(p[k][j]) for k in FIELDS}
            row.update(owner_start=start, owner_index=i)
            if segment['start']<=absolute<segment['stop']:
                owned.append(row)
            elif ((segment is layout[0] and absolute<288) or
                  (segment is layout[-1] and absolute>=length-288)):
                context.append(row)
    beats = sorted(owned+context,key=lambda b:b['r'])
    # Half-open ownership is the overlap deduplicator: no ad-hoc tolerance/NMS.
    if len({b['r'] for b in beats}) != len(beats):
        raise AssertionError('Duplicate R survived ownership partition')
    return beats, dict(retained_beats=len(owned), context_beats=len(context),
                       duplicate_r_count=0, segments=layout)

def allocate_shapes(beats, starts, shapes, probabilities, decode, mode, mean_shape,
                    representative_locations=None):
    if mode not in ('rep','dense'): raise ValueError(mode)
    lookup = {int(s):i for i,s in enumerate(starts)}
    candidates=[]
    if mode=='dense':
        for i,start in enumerate(starts):
            r = (representative_locations[i] if representative_locations is not None
                 else representative_r(decode(probabilities[i])))
            if r is not None and not np.isfinite(r): r=None
            if r is not None: candidates.append((float(start)+r,int(start),i))
        candidates.sort()
    out=[]; fallback=0; choices=[]
    for b in beats:
        if mode=='rep':
            i=lookup.get(int(b['owner_start']))
        elif candidates:
            i=min(candidates,key=lambda c:(abs(c[0]-b['r']),c[1]))[2]
        else: i=None
        out.append(np.asarray(mean_shape if i is None else shapes[i],np.float32))
        choices.append(-1 if i is None else int(starts[i]))
        fallback += i is None
    return out, dict(fallback_beats=fallback, chosen_window_starts=choices,
                     dense_candidate_count=len(candidates))

def synthesize(length, beats, shapes, gain, baseline, event_warp=True):
    """No GT argument. Complete finite record including all failure regions."""
    if len(beats)!=len(shapes): raise ValueError('One shape per predicted beat')
    output=np.full(length, baseline, np.float64)
    stats=dict(cycles=0,uniform_fallback_cycles=0,uncovered_samples=length)
    for i,(left,right) in enumerate(zip(beats[:-1],beats[1:])):
        a,b=int(left['r']),int(right['r'])
        if not (0<=a<b<length): raise ValueError('Bad ordered R coordinates')
        rr=b-a; source=np.array([0.,199.]); target=np.array([0.,rr-1.])
        nodes=np.array([0.,left['s']-a,left['t']-a,right['q']-a,rr-1.])
        if event_warp and np.all(np.isfinite(nodes)) and np.all(np.diff(nodes)>0):
            source=shape_nodes(shapes[i]); target=nodes
        elif event_warp: stats['uniform_fallback_cycles']+=1
        phase=np.interp(np.arange(rr),target,source)
        output[a:b]=baseline+gain*np.interp(phase,np.arange(200),shapes[i])
        # Linear interpolation between sample b-1 and next sample b is the
        # continuous join. Never independently add a piecewise baseline.
        stats['cycles']+=1; stats['uncovered_samples']-=rr
    if beats:
        r=int(beats[-1]['r'])
        if 0<=r<length: output[r]=baseline+gain*shapes[-1][0]
    assert np.isfinite(output).all()
    return output.astype(np.float32),stats
