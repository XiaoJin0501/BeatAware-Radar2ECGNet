"""Read saved A O4-dense(b) waves only; do not rewrite any A artifact."""
import concurrent.futures
import json
import multiprocessing
import environment as env
import numpy as np
from freeze import freeze, verify, sha
from metrics import supplementary_record, aggregate_supplement
from src.utils.atomic_state import atomic_json, atomic_csv, atomic_text

NAMES = ('rem_net', 'mmecg_tcn', 'radarode_mtl', 'cross_modal_conformer', 'wavegru_net', 'attention_dcgan')

def compute(row):
    import torch
    torch.set_num_threads(1)
    dest = env.OUT/'A_supplement'/'records'/row['dataset']/f"{row['record_id']}.json"
    if dest.exists():
        return json.loads(dest.read_text())
    wavepath = env.A/'execution_results/record_waveforms'/row['dataset']/f"{row['record_id']}.npz"
    rawpath = env.A/'raw_reference'/row['dataset']/f"{row['record_id']}.npz"
    with np.load(rawpath) as raw:
        gt_r = raw['event_r']
    output = dict(identity=row, source_wave_sha256=sha(wavepath), source_reference_sha256=sha(rawpath), methods={})
    with np.load(wavepath) as z:
        reference = z['reference']
        for name in NAMES:
            variant = 'O4-dense' if name == 'rem_net' else name+'-dense'
            wave = z[variant]
            output['methods'][name] = supplementary_record(wave, reference, gt_r, 288, len(wave)-288, row['subject_id'])
    atomic_json(dest, output)
    return output

def main():
    freeze()
    original = env.A/'execution_results/summary.json'
    before = sha(original)
    inventory = json.loads((env.A/'execution_results/validation_inventory.json').read_text())
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=8, mp_context=multiprocessing.get_context('spawn')) as pool:
        for i, result in enumerate(pool.map(compute, inventory), 1):
            results.append(result)
            print(f"A SUPPLEMENT {i}/{len(inventory)} {result['identity']['dataset']} {result['identity']['record_id']}", flush=True)
    summaries, table = {}, []
    old = json.loads(original.read_text())
    for dataset in ('MMECG', 'CR_RVS'):
        summaries[dataset] = {}
        for name in NAMES:
            group = [r['methods'][name] for r in results if r['identity']['dataset'] == dataset]
            summary = aggregate_supplement(group)
            summaries[dataset][name] = summary
            variant = 'O4-dense' if name == 'rem_net' else name+'-dense'
            baseline_m2 = old['datasets'][dataset][variant]['window']['pcc']['mean']
            assert abs(summary['M2']['pcc']['mean']-baseline_m2) < 1e-10, 'M2 changed vs A'
            m1, m2, m3 = summary['M1'], summary['M2'], summary['M3']
            table.append(dict(dataset=dataset, model=name,
                M1_median=m1['pcc']['median'], M1_q1=m1['pcc']['q1'], M1_q3=m1['pcc']['q3'], M1_mean=m1['pcc']['mean'],
                coverage_mean=m1['coverage']['mean'], scored_coverage_mean=m1['scored_coverage']['mean'],
                M2_median=m2['pcc']['median'], M2_mean=m2['pcc']['mean'], RMSE_mean=m2['rmse']['mean'],
                M3_median=m3['pcc']['median'], M3_mean=m3['pcc']['mean'],
                lag_median_ms=m3['lag_distribution_ms']['median'],
                strict_beat_PCC_median=old['datasets'][dataset][variant]['beat']['pcc']['median']))
    assert sha(original) == before
    verify()
    output = env.OUT/'A_supplement'
    atomic_json(output/'summary.json', dict(datasets=summaries, original_A_summary_sha256=before,
        original_A_decisions_unchanged=True, no_network_inference=True, training=False))
    atomic_csv(output/'comparison.csv', list(table[0]), table)
    lines = ['# A O4-dense(b): supplementary metrics only', '',
             'M1 is reference-assisted conditional morphology; M3 is reference-selected lag diagnostic. A decisions unchanged.', '',
             '| Dataset | Method | M1 median [IQR] | M1 mean | Matched coverage | Scored coverage | M2 mean | M3 mean | Strict beat median |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in table:
        lines.append(f"| {r['dataset']} | {r['model']} | {r['M1_median']:.3f} [{r['M1_q1']:.3f},{r['M1_q3']:.3f}] | {r['M1_mean']:.3f} | {r['coverage_mean']:.1%} | {r['scored_coverage_mean']:.1%} | {r['M2_mean']:.3f} | {r['M3_mean']:.3f} | {r['strict_beat_PCC_median']:.3f} |")
    atomic_text(output/'REPORT.md', '\n'.join(lines)+'\n')
    atomic_json(output/'COMPLETE.json', dict(passed=True, records=len(results), method_dataset_rows=12,
        M2_matches_A=True, A_decisions_changed=False, training=False))
    print('COMPLETE A supplement: 12 method/dataset rows; original A unchanged', flush=True)

if __name__ == '__main__':
    main()
