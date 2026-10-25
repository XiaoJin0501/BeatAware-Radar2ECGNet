"""Read-only consistency checks on the visualization inference artifacts."""
import json
from pathlib import Path
import numpy as np
import pytest

ASSETS=Path(__file__).resolve().parents[1]/'docs/paper/sections/figures/source_data/fig4_per_beat_20261026'
pytestmark=pytest.mark.skipif(not (ASSETS/'manifest.json').exists(),reason='Local visualization artifacts not packaged in CI')


def test_target_recomputation_matches_table_scores():
    audit=json.loads((ASSETS/'table_target_audit.json').read_text())
    assert len(audit)==16
    assert max(r['absolute_difference'] for r in audit)<2e-5


def test_each_reference_beat_has_its_own_context_and_shape():
    cases=json.loads((ASSETS/'manifest.json').read_text())['cases']
    assert len(cases)==7
    for case in cases:
        stems=f"{case['dataset']}_{case['scene']}"
        z=np.load(ASSETS/f'{stems}_inference.npz')
        origins=case['reference_context_starts']
        assert len(origins)==case['reference_beat_count']==len(set(origins))
        lookup={int(s):i for i,s in enumerate(z['starts'])}
        plotted=np.load(ASSETS/f'{stems}_plotted.npz')
        for method in ('rem_net','mmecg_tcn','radarode_mtl'):
            assert np.array_equal(plotted[method+'_shapes'],z[method+'_shape'][[lookup[s] for s in origins]])


def test_shape_scores_and_fallback_masks_are_truthful():
    cases=json.loads((ASSETS/'manifest.json').read_text())['cases']
    for case in cases:
        z=np.load(ASSETS/f"{case['dataset']}_{case['scene']}_plotted.npz")
        y=z['reference_shapes'];b=y-y.mean(1,keepdims=True)
        for method in ('rem_net','mmecg_tcn','radarode_mtl'):
            p=z[method+'_shapes'];a=p-p.mean(1,keepdims=True)
            score=np.mean((a*b).sum(1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-8))
            assert float(score)==pytest.approx(case['window_pcc_by_method'][method],abs=2e-6)
        assert z['fallback'].dtype==bool and z['fallback'].shape==(1024,)
        assert np.all(z['assembled'][z['fallback']]==0)
