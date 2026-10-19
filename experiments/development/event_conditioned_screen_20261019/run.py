"""Author-approved36-run screening; confirmation/test modes do not exist."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import environment as env
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from src.utils.atomic_state import (atomic_json, atomic_text, atomic_torch_save,
    atomic_npz, capture_rng_state, restore_rng_state)
from model import initialize_arm, objective, parameter_counts
from data import TrainWindows, prepare_validation, reference, local_reference, event_targets
from metrics import native_record, aggregate
from freeze import verify, sha

@torch.no_grad()
def validate(model, inventory, stats, device, save_predictions=None):
    state, was_training = capture_rng_state(), model.training
    model.eval()
    records, loss_sums, count = [], {}, 0
    try:
        for row in inventory:
            raw = row.get('_raw')
            if raw is None:
                raw = reference(row['record_id'])
            if '_features' in row:
                features = row['_features']
            else:
                with np.load(row['features']) as z:
                    features = z['features']
            waves, probs = [], []
            for a in tqdm(range(0, len(features), 16), desc='val', dynamic_ncols=True,
                          leave=False, mininterval=1., disable=not sys.stderr.isatty()):
                starts = row['starts'][a:a+16]
                if '_targets' in row:
                    target, events = row['_targets'][a:a+16], row['_events'][a:a+16]
                else:
                    target = np.stack([((raw['ecg'][s:s+1600]-stats['median'])/stats['iqr'])[None] for s in starts])
                    events = np.stack([event_targets(local_reference(raw, s)) for s in starts])
                y = torch.as_tensor(target, dtype=torch.float32, device=device)
                ev = torch.as_tensor(events, device=device)
                x = torch.as_tensor(features[a:a+16].astype(np.float32), device=device)
                output = model(x, oracle_events=ev if model.arm == 4 else None)
                _, losses = objective(output, y, ev)
                for k, v in losses.items():
                    loss_sums[k] = loss_sums.get(k, 0.)+v.item()*len(starts)
                count += len(starts)
                waves.extend(output['wave'][:, 0].cpu().numpy())
                probs.extend(torch.sigmoid(output['anchor_logits']).cpu().numpy())
            wave_array, prob_array = np.asarray(waves), np.asarray(probs)
            records.append(native_record(row, wave_array, prob_array, raw, stats))
            if save_predictions is not None:
                atomic_npz(save_predictions/f"{row['record_id']}.npz", dict(starts=np.asarray(row['starts']),
                    wave=wave_array, anchor_probability=prob_array, oracle=np.asarray(model.arm == 4)))
        result = aggregate(records)
        result['losses'] = {k: v/count for k, v in loss_sums.items()}
        return result
    finally:
        model.train(was_training)
        restore_rng_state(state)

def preload_validation(inventory, stats):
    for row in inventory:
        with np.load(row['features']) as z:
            row['_features'] = z['features']
        raw = reference(row['record_id'])
        row['_raw'] = raw
        row['_targets'] = np.stack([((raw['ecg'][s:s+1600]-stats['median'])/stats['iqr'])[None]
                                     for s in row['starts']]).astype(np.float32)
        row['_events'] = np.stack([event_targets(local_reference(raw, s)) for s in row['starts']])

def code_state():
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=env.ROOT, text=True).strip()
    status = subprocess.check_output(['git', 'status', '--porcelain=v1'], cwd=env.ROOT, text=True).strip()
    if status:
        raise RuntimeError('Screening requires clean committed source worktree: '+status[:400])
    return commit

def train_one(fold, seed, arm, data, inventory, device):
    verify()
    directory = env.OUT/'runs'/f'fold_{fold:02d}'/f'seed_{seed}'/f'arm_{arm}'
    if (directory/'COMPLETE.json').exists():
        return json.loads((directory/'result.json').read_text())
    directory.mkdir(parents=True, exist_ok=True)
    commit = code_state()
    spec = json.loads((env.HERE/'spec.json').read_text())
    env.core.set_seed(seed)
    model = initialize_arm(arm, seed).to(device)
    env.core.set_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(data, batch_size=16, shuffle=True, drop_last=True,
                        num_workers=0, generator=generator, pin_memory=device.startswith('cuda'))
    signature = sha(env.OUT/'METRIC_FREEZE.json')
    meta = dict(fold=fold, seed=seed, actual_seed=seed, arm=arm, oracle=arm == 4,
        commit=commit, spec=spec, metric_freeze_sha256=signature, parameters=parameter_counts(model),
        normalization=data.stats, training_windows=len(data), validation_records=len(inventory),
        training_subjects=sorted({r['subject_id'] for r in data.rows}),
        validation_subjects=sorted({r['subject_id'] for r in inventory}),
        test_access=False, screening_only=True, statistical_evidence=False)
    assert not set(meta['training_subjects']) & set(meta['validation_subjects'])
    if (directory/'meta.json').exists():
        assert json.loads((directory/'meta.json').read_text()) == meta
    else:
        atomic_json(directory/'meta.json', meta)
    history, best, wait, start = [], -float('inf'), 0, 1
    last = directory/'last_state.pt'
    if last.exists():
        saved = torch.load(last, map_location='cpu', weights_only=False)
        assert saved['metric_freeze_sha256'] == signature
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        history, best, wait = saved['history'], saved['best_val'], saved['patience']
        restore_rng_state(saved['rng'], generator)
        start = saved['epoch']+1
    for epoch in range(start, 101):
        if wait >= 15:
            break
        model.train()
        started, seen, gnorm_sum, sums = time.perf_counter(), 0, 0., {}
        index_order = []
        progress = tqdm(loader, desc=f'f{fold}s{seed}a{arm} ep{epoch}', dynamic_ncols=True,
                        leave=False, mininterval=1., disable=not sys.stderr.isatty())
        for step, batch in enumerate(progress, 1):
            x, y, ev = [batch[k].to(device, non_blocking=True) for k in ('wsst', 'target', 'events')]
            optimizer.zero_grad(set_to_none=True)
            output = model(x, oracle_events=ev if arm == 4 else None)
            loss, pieces = objective(output, y, ev)
            loss.backward()
            gnorm = torch.stack([p.grad.detach().square().sum() for p in model.parameters() if p.grad is not None]).sum().sqrt()
            if not torch.isfinite(loss) or not torch.isfinite(gnorm):
                raise FloatingPointError('Nonfinite loss/gradient: failed run cannot be omitted')
            optimizer.step()
            for k, v in pieces.items():
                sums[k] = sums.get(k, 0.)+v.detach().item()*len(x)
            seen += len(x)
            gnorm_sum += gnorm.detach().item()
            index_order.extend(batch['index'].tolist())
            if step % 20 == 0:
                progress.set_postfix(loss=sums['loss']/seen, anchor=sums['anchor']/seen,
                    gnorm=gnorm_sum/step, lr=optimizer.param_groups[0]['lr'])
        metrics = validate(model, inventory, data.stats, device)
        score = metrics['selection_window_pcc']
        if score is None or not np.isfinite(score):
            raise FloatingPointError('Missing/nonfinite M2 selection metric')
        improved = score > best
        if improved:
            best, wait = score, 0
            atomic_torch_save(directory/'best.pt', dict(model=model.state_dict(), epoch=epoch,
                validation=metrics, metric_freeze_sha256=signature, oracle=arm == 4))
        else:
            wait += 1
        row = dict(epoch=epoch, fold=fold, seed=seed, arm=arm,
            train={k: v/seen for k, v in sums.items()}, val=metrics,
            lr=optimizer.param_groups[0]['lr'], average_gnorm=gnorm_sum/len(loader),
            seconds=time.perf_counter()-started, best=improved, patience_used=wait,
            training_order_sha256=__import__('hashlib').sha256(np.asarray(index_order, np.int32).tobytes()).hexdigest(),
            oracle=arm == 4, screening_only=True)
        history.append(row)
        atomic_text(directory/'epoch_log.jsonl', ''.join(json.dumps(h, allow_nan=False)+'\n' for h in history))
        atomic_torch_save(last, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=None, epoch=epoch, patience=wait, best_val=best, history=history,
            rng=capture_rng_state(generator), metric_freeze_sha256=signature))
        m1 = metrics['M1']['pcc']['mean']
        tqdm.write(f"fold={fold} seed={seed} arm={arm} epoch={epoch} loss={row['train']['loss']:.5f} "
            f"M1={m1 if m1 is not None else 'NA'} cov={metrics['M1']['coverage']['mean']:.3f} "
            f"M2={score:.5f} M3={metrics['M3']['pcc']['mean']:.5f} "
            f"wait={wait}/15 seconds={row['seconds']:.1f} {'★' if improved else ''}")
    frozen = torch.load(directory/'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(frozen['model'])
    metrics = validate(model, inventory, data.stats, device, directory/'validation_predictions')
    assert abs(metrics['selection_window_pcc']-frozen['validation']['selection_window_pcc']) < 1e-8
    verify()
    result = dict(**meta, epochs=len(history), best_epoch=frozen['epoch'], validation=metrics,
                  completed=True, maximum_cuda_memory_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else None)
    atomic_json(directory/'result.json', result)
    atomic_json(directory/'COMPLETE.json', dict(passed=True, fold=fold, seed=seed, arm=arm,
        epochs=len(history), best_epoch=frozen['epoch'], test_access=False,
        screening_only=True, confirmation_authorized=False))
    print(f'RUN COMPLETE fold={fold} seed={seed} arm={arm} epochs={len(history)} best={frozen["epoch"]}', flush=True)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['screen'], required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    verify()
    if not (env.OUT/'A_supplement/COMPLETE.json').exists():
        raise RuntimeError('Step1 supplement must complete before screening')
    code_state()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    results = []
    for fold in (1, 2, 3):
        print(f'PREPARE fold={fold}', flush=True)
        data = TrainWindows(fold).preload()
        inventory = prepare_validation(fold, env.OUT/'data'/f'fold_{fold:02d}')
        preload_validation(inventory, data.stats)
        for seed in (42, 43, 44):
            for arm in (1, 2, 3, 4):
                result = train_one(fold, seed, arm, data, inventory, args.device)
                results.append(dict(fold=fold, seed=seed, arm=arm, best_epoch=result['best_epoch']))
                atomic_json(env.OUT/'progress.json', dict(completed=len(results), total=36,
                    last=results[-1], confirmation_11fold_authorized=False))
        del data, inventory
    from report import generate
    generate()
    print('COMPLETE:36 screening runs; confirmation remains BLOCKED', flush=True)

if __name__ == '__main__':
    main()
