"""Descriptive screening only; no p values and no automatic expansion."""
import json
import environment as env
from freeze import verify, sha
from metrics import screening_decision, distribution
from src.utils.atomic_state import atomic_json, atomic_csv, atomic_text

def generate():
    verify()
    table = []
    for fold in (1, 2, 3):
        for seed in (42, 43, 44):
            for arm in (1, 2, 3, 4):
                directory = env.OUT/'runs'/f'fold_{fold:02d}'/f'seed_{seed}'/f'arm_{arm}'
                if not (directory/'COMPLETE.json').exists():
                    raise RuntimeError('Cannot omit incomplete run: '+str(directory))
                result = json.loads((directory/'result.json').read_text())
                v = result['validation']
                assert len(v['events_by_subject']) == 1
                e = v['events_by_subject'][0]
                m1 = v['M1']
                table.append(dict(fold=fold, seed=seed, arm=arm, subject_id=e['subject_id'],
                    epochs=result['epochs'], best_epoch=result['best_epoch'],
                    M1=m1['pcc']['mean'], M1_median=m1['pcc']['median'], M1_q1=m1['pcc']['q1'], M1_q3=m1['pcc']['q3'],
                    M1_coverage=m1['coverage']['mean'], M1_scored_coverage=m1['scored_coverage']['mean'],
                    M1_matched=m1['matched'], M1_reference=m1['reference_count'], M1_scored=m1['scored'],
                    M2=v['M2']['pcc']['mean'], M3=v['M3']['pcc']['mean'],
                    M3_lag_median_ms=v['M3']['lag_distribution_ms']['median'],
                    strict_beat_PCC=v['beat']['pcc']['mean'], RMSE=v['M2']['rmse']['mean'],
                    **{f'{k}_mae_ms': e['fiducials'][k]['mae_ms'] for k in 'qrst'},
                    **{f'{k}_coverage': e['fiducials'][k]['coverage'] for k in 'qrst'},
                    head_R_F1=e['r_f1'], waveform_R_F1=m1['subjects'][0]['waveform_r_f1'],
                    predecode_order_violation=e['order_violation_rate'],
                    wave_head_consistency_mae_ms=e['consistency_mae_ms'],
                    oracle=arm == 4, result_path=str(directory/'result.json')))
    decision = screening_decision(table)
    atomic_csv(env.OUT/'screening_results.csv', list(table[0]), table)
    atomic_json(env.OUT/'screening_decision.json', decision)
    per_arm = {}
    for arm in (1, 2, 3, 4):
        subject_rows = []
        for fold in (1, 2, 3):
            g = [r for r in table if r['fold'] == fold and r['arm'] == arm]
            subject_rows.append(dict(subject_id=g[0]['subject_id'], **{
                metric: sum(r[metric] for r in g)/3 if all(r[metric] is not None for r in g) else None
                for metric in ('M1', 'M1_coverage', 'M2', 'M3', 'strict_beat_PCC', 'RMSE')}))
        per_arm[str(arm)] = dict(subject_seed_means=subject_rows,
            **{metric: distribution([r[metric] for r in subject_rows])
               for metric in ('M1', 'M1_coverage', 'M2', 'M3', 'strict_beat_PCC', 'RMSE')})
    atomic_json(env.OUT/'summary.json', dict(screening_only=True, statistical_evidence=False,
        arms=per_arm, decision=decision, confirmation_authorized=False, test_access=False))
    lines = ['# MMECG continuous-decoder screening', '',
        '**This is screening, NOT statistical evidence.** Validation only. M1 is R-aligned conditional morphology; M3 uses reference-selected lag. Arm4 is oracle. No11fold confirmation/test authorized.', '',
        'M1–M3 and strict PCC are subject Fisher-z means. Matched coverage and scored coverage differ only where full R-centered windows lack retained support. All runs use exact seeds42/43/44 (no fold offset).', '']
    for fold in (1, 2, 3):
        lines += [f'## Fold{fold}', '',
            '| Seed | Arm | M1 | Coverage/scored | M2 | M3 | Strict beat | RMSE | Q/R/S/T MAE(ms) | Head/wave R-F1 | Epoch best/final |',
            '|---:|---:|---:|---|---:|---:|---:|---:|---|---|---|']
        for r in [v for v in table if v['fold'] == fold]:
            fmt = lambda x: 'NA' if x is None else f'{x:.3f}'
            mae = '/'.join('NA' if r[f'{k}_mae_ms'] is None else f"{r[f'{k}_mae_ms']:.1f}" for k in 'qrst')
            lines.append(f"| {r['seed']} | {r['arm']}{' oracle' if r['oracle'] else ''} | {fmt(r['M1'])} | {r['M1_coverage']:.1%}/{r['M1_scored_coverage']:.1%} | {r['M2']:.3f} | {r['M3']:.3f} | {r['strict_beat_PCC']:.3f} | {r['RMSE']:.3f} | {mae} | {r['head_R_F1']:.3f}/{r['waveform_R_F1']:.3f} | {r['best_epoch']}/{r['epochs']} |")
        lines.append('')
    lines += ['## Preregistered arm3-minus-arm2 decision', '',
        f"Positive M2 pairs: {decision['positive_m2_pairs']}/9 (required >=8).", '',
        '| Validation subject/fold | M2 seed-mean delta | M1 seed-mean delta | Q/R/S/T MAE delta(ms) |',
        '|---|---:|---:|---|']
    for r in decision['subject_seed_means']:
        mae = '/'.join('NA' if r[f'{k}_mae_delta_ms'] is None else f"{r[f'{k}_mae_delta_ms']:+.2f}" for k in 'qrst')
        m1 = 'NA' if r['m1_delta'] is None else f"{r['m1_delta']:+.4f}"
        lines.append(f"| fold{r['fold']} | {r['m2_delta']:+.4f} | {m1} | {mae} |")
    lines += ['', 'Overall subject-macro deltas: `'+json.dumps(decision['overall_subject_mean_deltas'])+'`.', '',
              'Conditions: `'+json.dumps(decision['conditions'])+'`.', '',
              f"**Screening {'PASS' if decision['screening_pass'] else 'FAIL'}.** This does not authorize expansion or establish statistical superiority.", '',
              'All raw lag histograms, subject IQRs, missing endpoints and coverage are retained in each result.json. A supplement is in A_supplement/REPORT.md; original A decisions unchanged.']
    atomic_text(env.OUT/'REPORT.md', '\n'.join(lines)+'\n')
    atomic_json(env.OUT/'COMPLETE.json', dict(passed=True, runs=36,
        screening_pass=decision['screening_pass'], statistical_evidence=False,
        confirmation_11fold_authorized=False, test_access=False,
        metric_freeze_sha256=sha(env.OUT/'METRIC_FREEZE.json')))

if __name__ == '__main__':
    from v2_report import generate
    generate()
