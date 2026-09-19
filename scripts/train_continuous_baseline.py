#!/usr/bin/env python3
"""Matched-protocol continuous-ECG training for the published baselines.

Everything that defines the comparison is imported from
`scripts/train_continuous_ecg_pilot.py` and therefore byte-identical to the
frozen REM-Net run: the fold loader, the train-fold robust scaler, the metric
functions, the validation composite, the budget and the
TRAINING -> CHECKPOINT_FROZEN -> TEST_EVALUATING -> COMPLETE state machine.

Only three things differ per baseline, and each is recorded in its spec:

  * the architecture (`src/models/baselines/continuous/`);
  * the input tensor its paper consumes (WSST or native-rate rcg);
  * its own published training objective.

Checkpoints are selected on validation only. `--final_test` opens the held-out
subject exactly once, after the checkpoint is frozen, and is refused unless the
experiment tag is the registered baseline benchmark.

Usage:
  python scripts/train_continuous_baseline.py --baseline wavegru_net --fold 1
  python scripts/train_continuous_baseline.py --baseline wavegru_net --fold 1 --final_test
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_continuous_ecg_pilot import (  # noqa: E402
    BUDGET, COMPOSITE_W, DATA, FS, REGION_DEF, WIN, Fold, _set_split_root, _split_sha,
    effective_seed, evaluate, fit_scaler, set_seed,
)
from src.models.baselines.continuous import (  # noqa: E402
    ADAPTATIONS, BASELINES, DCGANDiscriminator, build_baseline, input_key,
)
from src.models.baselines.continuous.cross_modal_conformer import (  # noqa: E402
    ECGAutoEncoder, stft_magnitude,
)
from src.models.baselines.continuous.radarode_mtl import (  # noqa: E402
    SHAPE_LEN, TASKS, anchor_loss, ppi_loss, shape_loss,
)
from src.utils.atomic_state import (  # noqa: E402
    atomic_json, atomic_npz, atomic_torch_save, capture_rng_state, restore_rng_state,
    transition_state,
)

REGISTERED_TAG = "q1_matched_baselines_continuous_strict_loso_seed42"
REGISTERED_SEED_BASES = (42, 123, 2024)


def registered_tag(seed_base: int) -> str:
    return f"q1_matched_baselines_continuous_strict_loso_seed{seed_base}"
GAN_L1_WEIGHT = 10.0          # see ADAPTATIONS["attention_dcgan"] D4
PRETRAIN_EPOCHS = 10          # cross-modal conformer ECG autoencoder, train fold only


class BaselineAdapter(nn.Module):
    """Give every baseline the frozen evaluator's calling convention.

    `forward(wsst, rcg=None) -> (waveform, event_logits)`. Baselines without an
    event head emit a constant -10 logit, i.e. an all-zero event probability, so
    `event_f1_*` is structurally undefined for them and must be reported as not
    applicable rather than as a score of zero.
    """

    def __init__(self, core: nn.Module, key: str) -> None:
        super().__init__()
        self.core, self.key = core, key
        self.has_event_head = False

    def forward(self, wsst, rcg=None):
        y, ev = (self.core(wsst=wsst) if self.key == "wsst" else self.core(rcg=rcg))
        if ev is None:
            ev = torch.full((y.shape[0], 2, y.shape[-1]), -10.0,
                            device=y.device, dtype=y.dtype)
        return y, ev


def radarode_targets(batch: dict, dev: torch.device) -> dict:
    """Targets for radarODE-MTL's three published tasks, from the fold's own labels.

    ECG_shape: the centre cardiac cycle resampled to 200 samples, as in the authors'
    dataset. PPI: that cycle's length in samples, which is the class their 260-way
    head predicts. Anchor: the R-peak position signal, taken from the fold's Gaussian
    R-peak mask. Windows with fewer than two R peaks carry `valid = False` and are
    dropped from the beat-level losses only.
    """
    raw = batch["raw"].to(dev)                                   # (B, 1, 1600) raw ECG
    peaks = batch["rpeaks"]
    shapes, lengths, valid = [], [], []
    for i in range(raw.shape[0]):
        rp = peaks[i].numpy()
        rp = rp[rp >= 0]
        if len(rp) < 2:
            shapes.append(torch.zeros(SHAPE_LEN, device=dev))
            lengths.append(1); valid.append(False)
            continue
        pairs = np.stack([rp[:-1], rp[1:]], 1)
        j = int(np.argmin(np.abs(pairs.mean(1) - WIN // 2)))      # the centre cycle
        a, b = int(pairs[j, 0]), int(pairs[j, 1])
        cycle = raw[i, 0, a:b].unsqueeze(0).unsqueeze(0)
        shapes.append(F.interpolate(cycle, size=SHAPE_LEN, mode="linear",
                                    align_corners=False)[0, 0])
        lengths.append(b - a); valid.append(True)
    return {"ECG_shape": torch.stack(shapes).unsqueeze(1),
            "PPI": torch.tensor(lengths, device=dev),
            "Anchor": batch["event"][:, 0:1].to(dev),
            "valid": torch.tensor(valid, device=dev)}


def baseline_loss(name: str, y: torch.Tensor, target: torch.Tensor,
                  extras: dict | None = None) -> tuple[torch.Tensor, dict]:
    """Each baseline's own published objective, computed in the scaled target space."""
    extras = extras or {}
    if name in ("mmecg_tcn", "radarode_mtl", "wavegru_net"):
        loss = F.mse_loss(y, target)
        return loss, {"mse": float(loss.detach())}
    if name == "cross_modal_conformer":
        wave = F.mse_loss(y, target)
        # Scale-normalise the auxiliary terms. The paper sums waveform, spectrogram and
        # deep-feature MSE over ECG normalised to a small range; in the contract's
        # robust-scaled space the raw spectrogram MSE is orders of magnitude larger and
        # dominates, and because spectrogram magnitude is sign-blind the waveform then
        # trains to an inverted solution (observed: F1@100 0.92 with PCC -0.16).
        # Dividing each auxiliary term by its own target energy keeps the published
        # composition while making the sum scale-free. Recorded as adaptation D4.
        spec_pred, spec_target = stft_magnitude(y), stft_magnitude(target)
        spec = F.mse_loss(spec_pred, spec_target) / spec_target.pow(2).mean().detach().clamp_min(1e-8)
        parts = {"waveform_mse": float(wave.detach()), "spectrogram_mse_normalised": float(spec.detach())}
        loss = wave + spec
        if "deep_feature" in extras:
            df = extras["deep_feature"]
            parts["deep_feature_mse_normalised"] = float(df.detach())
            loss = loss + df
        return loss, parts
    if name == "attention_dcgan":
        adv = extras["adversarial"]
        l1 = F.l1_loss(y, target)
        loss = adv + GAN_L1_WEIGHT * l1
        return loss, {"adversarial": float(adv.detach()), "l1": float(l1.detach())}
    raise ValueError(name)


def pretrain_ecg_autoencoder(loader: DataLoader, dev: torch.device, epochs: int,
                             seed: int) -> ECGAutoEncoder | None:
    """Liu et al.'s stage (a): ECG-to-ECG autoencoder on TRAINING-fold ECG only."""
    if epochs <= 0:
        return None
    torch.manual_seed(seed + 5_000)
    ae = ECGAutoEncoder().to(dev)
    opt = torch.optim.AdamW(ae.parameters(), lr=1e-3, weight_decay=BUDGET["weight_decay"])
    ae.train()
    for ep in range(1, epochs + 1):
        tot = nb = 0
        for b in loader:
            target = b["y"].to(dev)
            rec, _ = ae(target)
            loss = F.mse_loss(rec, target) + F.mse_loss(stft_magnitude(rec),
                                                        stft_magnitude(target))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        print(f"  [pretrain ECG-AE] ep{ep:2d} loss {tot / max(nb, 1):.4f}", flush=True)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True, choices=sorted(BASELINES))
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--split_root", default="loso_v2")
    ap.add_argument("--exp_tag", default=None,
                    help="defaults to the registered benchmark tag for --seed_base")
    ap.add_argument("--seed_base", type=int, default=42,
                    help="per-fold seed is seed_base + fold; registered bases 42/123/2024")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--final_test", action="store_true",
                    help="registered mode: select on validation, then open test once")
    ap.add_argument("--smoke", action="store_true",
                    help="reduced run into a SEPARATE tree; never reportable")
    ap.add_argument("--smoke_epochs", type=int, default=1)
    ap.add_argument("--smoke_limit", type=int, default=48)
    ap.add_argument("--pretrain_epochs", type=int, default=PRETRAIN_EPOCHS,
                    help="cross-modal conformer only; 0 removes the deep-feature loss")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    if a.final_test and a.smoke:
        raise SystemExit("--final_test cannot be combined with --smoke")
    if a.final_test and a.overwrite:
        raise SystemExit("--overwrite cannot be combined with --final_test: a frozen or "
                         "COMPLETE registered fold is never restarted in place")
    a.exp_tag = a.exp_tag or registered_tag(a.seed_base)
    if a.final_test and (a.seed_base not in REGISTERED_SEED_BASES
                         or a.exp_tag != registered_tag(a.seed_base)):
        raise SystemExit(f"--final_test is restricted to registered tag "
                         f"{registered_tag(a.seed_base)} with seed_base in {REGISTERED_SEED_BASES}")
    tag = a.exp_tag + ("_smoke" if a.smoke else "")
    epochs = a.smoke_epochs if a.smoke else BUDGET["epochs"]
    limit = a.smoke_limit if a.smoke else None

    out = ROOT / "experiments_mmecg" / tag / a.baseline / f"fold_{a.fold:02d}"
    state_path = out / "run_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    if state and state["stage"] == "COMPLETE":
        print(f"[sealed] COMPLETE fold at {out}; test cannot be reopened")
        return
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    if a.overwrite and state is not None:
        # The state machine forbids stage regression, so a restart clears the fold's
        # artifacts instead of rewinding them. --final_test is refused above, so this can
        # never reopen a registered frozen or COMPLETE fold.
        for stale in (state_path, out / "best.json", out / "history.json",
                      out / "initial_loss_components.json",
                      out / "checkpoints" / "last_state.pt",
                      out / "checkpoints" / "best_composite.pt"):
            stale.unlink(missing_ok=True)
        state = None
    if state is None:
        state = transition_state(state_path, "TRAINING", fold=a.fold, arm=a.baseline,
                                 loss_arm="native", test_accessed=False)

    _set_split_root(a.split_root)
    seed = effective_seed(a.fold, a.seed_base)   # identical per-fold seed as REM-Net
    set_seed(seed)
    dev = torch.device(a.device)
    key = input_key(a.baseline)
    scaler = fit_scaler(a.fold)            # train-fold median/IQR, never val or test

    fold_kw = dict(wsst_source="subject_pool", needs_rcg=(key == "rcg"),
                   load_wsst=(key == "wsst"))
    tr = Fold(a.fold, "train", "B1", scaler, limit, **fold_kw)
    va = Fold(a.fold, "val", "B1", scaler, limit, **fold_kw)
    g = torch.Generator(); g.manual_seed(seed)
    ltr = DataLoader(tr, batch_size=BUDGET["batch_size"], shuffle=True, num_workers=0,
                     drop_last=True, generator=g)
    lva = DataLoader(va, batch_size=BUDGET["batch_size"], shuffle=False, num_workers=0)

    kwargs = {}
    if a.baseline == "attention_dcgan":
        # Fixed Tanh gain from the TRAINING fold only (ADAPTATIONS D2).
        import h5py
        with h5py.File(DATA / f"fold_{a.fold:02d}" / "train.h5", "r") as h:
            raw = np.asarray(h["ecg"][:, 0], np.float64).reshape(-1)
        med, iqr = scaler
        kwargs["out_gain"] = float(np.percentile(np.abs((raw - med) / iqr), 99.9))
    is_mtl = a.baseline == "radarode_mtl"
    if is_mtl:
        kwargs["max_epochs"] = epochs
    model = BaselineAdapter(build_baseline(a.baseline, **kwargs), key).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=BUDGET["lr"],
                            weight_decay=BUDGET["weight_decay"])

    disc = disc_opt = None
    if a.baseline == "attention_dcgan":
        disc = DCGANDiscriminator().to(dev)
        disc_opt = torch.optim.AdamW(disc.parameters(), lr=BUDGET["lr"],
                                     weight_decay=BUDGET["weight_decay"])
    ecg_ae = None
    if a.baseline == "cross_modal_conformer":
        ecg_ae = pretrain_ecg_autoencoder(ltr, dev, a.pretrain_epochs, seed)

    CODE_PATHS = ["src/", "scripts/", "tests/", "configs/"]
    try:
        commit = subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--"] + CODE_PATHS, text=True).strip()
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = head = "unknown"

    config = {
        "baseline": a.baseline, "paper": BASELINES[a.baseline][2],
        "reproduction_lane": "common-protocol (adapted)",
        "adaptations": ADAPTATIONS[a.baseline],
        "fold": a.fold, "mode": "smoke" if a.smoke else "formal",
        "epochs": epochs, "limit": limit, "effective_seed": seed,
        "seed_rule": "BUDGET.seed + fold; identical to the REM-Net run",
        "input_tensor": key, "wsst_source": "subject_pool" if key == "wsst" else None,
        "split_used": ["train", "val"], "test_accessed": False,
        "final_candidate_mode": bool(a.final_test),
        "scaler_median_iqr": scaler, "scaler_source": "ENTIRE training fold",
        "budget": BUDGET, "objective": "paper-native (see loss_components)",
        "gan_l1_weight": GAN_L1_WEIGHT if a.baseline == "attention_dcgan" else None,
        "pretrain_epochs": a.pretrain_epochs if a.baseline == "cross_modal_conformer" else None,
        "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "composite_weights": COMPOSITE_W,
        "event_head": False,
        "event_metrics_valid": False,
        "region_definitions": REGION_DEF, "rpeak_readout": "waveform-derived",
        "metric_space": "raw ECG units (train-fold robust inverse), identical to REM-Net",
        "split_root": a.split_root, "split_dir": str(DATA),
        "val_file_sha256": _split_sha(a.fold, "val"),
        "code_commit": commit, "runtime_head": head,
        "evaluator": "scripts/train_continuous_ecg_pilot.evaluate (imported, not copied)",
    }
    config_path = out / "config.json"
    if config_path.exists() and not a.overwrite:
        prior = json.loads(config_path.read_text())
        identity = ("baseline", "fold", "budget", "effective_seed", "split_root", "input_tensor")
        mismatch = [k for k in identity if prior.get(k) != config.get(k)]
        if mismatch:
            raise SystemExit(f"Resume contract mismatch in {mismatch}: {out}")
    else:
        atomic_json(config_path, config)

    best, best_ep, bad, hist, initial_components = -1e9, -1, 0, [], None
    start_epoch = 1
    last_state_path = out / "checkpoints" / "last_state.pt"
    if state["stage"] == "TRAINING" and last_state_path.exists() and not a.overwrite:
        saved = torch.load(last_state_path, map_location=dev, weights_only=False)
        expected = {"fold": a.fold, "baseline": a.baseline, "max_epochs": epochs,
                    "patience": BUDGET["patience"]}
        mismatch = [k for k, v in expected.items() if saved.get(k) != v]
        if mismatch:
            raise SystemExit(f"Checkpoint resume contract mismatch in {mismatch}")
        model.load_state_dict(saved["model_state_dict"])
        opt.load_state_dict(saved["optimizer_state_dict"])
        if disc is not None and saved.get("discriminator_state_dict") is not None:
            disc.load_state_dict(saved["discriminator_state_dict"])
            disc_opt.load_state_dict(saved["discriminator_optimizer_state_dict"])
        best, best_ep = float(saved["best_val"]), int(saved["best_epoch"])
        bad, hist = int(saved["patience_counter"]), saved["history"]
        initial_components = saved["initial_loss_components"]
        if is_mtl and saved.get("train_loss_buffer") is not None:
            model.core.train_loss_buffer = np.asarray(saved["train_loss_buffer"])
            model.core.average_loss = saved["ega_average_loss"]
            model.core.average_loss_count = saved["ega_average_loss_count"]
            model.core.nadir_vector = (None if saved["ega_nadir_vector"] is None
                                       else saved["ega_nadir_vector"].to(dev))
        restore_rng_state(saved["rng_state"], g)
        start_epoch = int(saved["epoch"]) + 1
        print(f"[resume] {a.baseline} fold{a.fold:02d} from epoch {start_epoch} "
              f"(best={best:.4f}@{best_ep}, patience={bad})")

    if state["stage"] == "TRAINING":
        for ep in range(start_epoch, epochs + 1):
            model.train(); tot = nb = 0; t0 = time.time()
            epoch_task_losses = np.zeros(len(TASKS))
            if is_mtl:
                model.core.epoch = ep - 1
            for b in ltr:
                target = b["y"].to(dev)
                wsst = b["wsst"].to(dev) if key == "wsst" else None
                rcg = b["rcg"].to(dev) if key == "rcg" else None
                extras = {}
                if disc is not None:
                    with torch.no_grad():
                        fake_detached, _ = model(wsst, rcg)
                    d_real = disc(target)
                    d_fake = disc(fake_detached)
                    d_loss = (F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
                              + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake)))
                    disc_opt.zero_grad(); d_loss.backward(); disc_opt.step()
                if is_mtl:
                    outs = model.core.all_tasks(wsst)
                    tgt = radarode_targets(b, dev)
                    keep = tgt["valid"]
                    beat = (lambda fn, key: fn(outs[key][keep], tgt[key][keep])
                            if bool(keep.any()) else outs[key].sum() * 0.0)
                    task_losses = torch.stack([
                        F.mse_loss(outs["continuous"], target),
                        beat(shape_loss, "ECG_shape"),
                        (ppi_loss(outs["PPI"][keep], tgt["PPI"][keep])
                         if bool(keep.any()) else outs["PPI"].sum() * 0.0),
                        anchor_loss(outs["Anchor"], tgt["Anchor"]),
                    ])
                    parts = {t: float(l.detach()) for t, l in zip(TASKS, task_losses)}
                    # EGA replaces loss.backward(): it aligns the four task gradients on
                    # the shared backbone before the optimiser step.
                    model.core._compute_grad_dim()
                    opt.zero_grad()
                    model.core.backward(task_losses, EGA_temp=model.core.ega_temp)
                    opt.step()
                    loss = task_losses.sum()
                    if initial_components is None:
                        initial_components = {**parts, "total": float(loss.detach())}
                    epoch_task_losses += np.array([parts[t] for t in TASKS])
                    tot += float(loss); nb += 1
                    continue
                y, _ = model(wsst, rcg)
                if disc is not None:
                    d_out = disc(y)
                    extras["adversarial"] = F.binary_cross_entropy_with_logits(
                        d_out, torch.ones_like(d_out))
                if ecg_ae is not None:
                    z_ecg = ecg_ae.embed(target)
                    z_radar = model.core.embed(rcg)
                    extras["deep_feature"] = (F.mse_loss(z_radar, z_ecg)
                                              / z_ecg.pow(2).mean().detach().clamp_min(1e-8))
                loss, parts = baseline_loss(a.baseline, y, target, extras)
                if initial_components is None:
                    initial_components = {**parts, "total": float(loss.detach())}
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss); nb += 1
            if is_mtl:
                model.core.train_loss_buffer[:, ep - 1] = epoch_task_losses / max(nb, 1)
            m = evaluate(model, lva, dev, compute_shape200=False)
            hist.append({"epoch": ep, "train_loss": tot / max(nb, 1), **m})
            print(f"  ep{ep:3d} loss {tot/max(nb,1):.4f} | PCC {m['pcc']:.4f} "
                  f"NRMSE {m['nrmse']:.4f} R2 {m['r2']:+.3f} F1@100 {m['f1_100ms']:.3f} "
                  f"| S {m['composite']:.4f} ({time.time()-t0:.0f}s)", flush=True)
            if m["composite"] > best:
                best, best_ep, bad = m["composite"], ep, 0
                atomic_torch_save(out / "checkpoints" / "best_composite.pt", {
                    "model_state_dict": model.state_dict(), "epoch": ep,
                    "composite": best, "baseline": a.baseline, "fold": a.fold})
            else:
                bad += 1
            atomic_torch_save(last_state_path, {
                "epoch": ep, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "discriminator_state_dict": disc.state_dict() if disc is not None else None,
                "discriminator_optimizer_state_dict": (disc_opt.state_dict()
                                                       if disc_opt is not None else None),
                "patience_counter": bad, "best_val": best, "best_epoch": best_ep,
                "history": hist, "initial_loss_components": initial_components,
                "rng_state": capture_rng_state(g), "fold": a.fold, "baseline": a.baseline,
                "max_epochs": epochs, "patience": BUDGET["patience"],
                "train_loss_buffer": model.core.train_loss_buffer if is_mtl else None,
                "ega_average_loss": model.core.average_loss if is_mtl else None,
                "ega_average_loss_count": model.core.average_loss_count if is_mtl else None,
                "ega_nadir_vector": model.core.nadir_vector if is_mtl else None,
            })
            if bad >= BUDGET["patience"]:
                print(f"  early stop at ep{ep}"); break
        atomic_json(out / "history.json", hist)
        atomic_json(out / "initial_loss_components.json", initial_components)
        atomic_json(out / "best.json", {
            "baseline": a.baseline, "fold": a.fold,
            "mode": "smoke" if a.smoke else "formal",
            "best_epoch": best_ep, "best_composite": best, "code_commit": commit,
            "runtime_head": head, "max_epochs": epochs, "patience": BUDGET["patience"],
            "hit_epoch_cap": best_ep == epochs, "stopped_at_epoch": len(hist),
            "early_stopped": len(hist) < epochs,
            "selected_by": "full-waveform validation composite (identical to REM-Net)",
            "metrics": hist[best_ep - 1] if best_ep > 0 else None})
        state = transition_state(state_path, "CHECKPOINT_FROZEN", fold=a.fold,
                                 arm=a.baseline, loss_arm="native",
                                 best_epoch=best_ep, test_accessed=False)
    else:
        frozen = json.loads((out / "best.json").read_text())
        best, best_ep = float(frozen["best_composite"]), int(frozen["best_epoch"])

    if not a.final_test:
        print(f"[done] {a.baseline} fold{a.fold:02d} validation-only: "
              f"best S={best:.4f} @ep{best_ep} -> {out}")
        return

    # The held-out subject is constructed only after the checkpoint is frozen.
    if state["stage"] == "CHECKPOINT_FROZEN":
        state = transition_state(state_path, "TEST_EVALUATING", fold=a.fold,
                                 arm=a.baseline, loss_arm="native",
                                 best_epoch=best_ep, test_accessed=True)
    if state["stage"] != "TEST_EVALUATING":
        raise RuntimeError(f"Test access denied in state {state['stage']}")
    checkpoint = torch.load(out / "checkpoints" / "best_composite.pt",
                            map_location=dev, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    te = Fold(a.fold, "test", "B1", scaler, allow_test=True, **fold_kw)
    lte = DataLoader(te, batch_size=BUDGET["batch_size"], shuffle=False, num_workers=0)
    test_metrics, arrays = evaluate(model, lte, dev, compute_shape200=False,
                                    return_outputs=True)
    atomic_npz(out / "test_predictions.npz", arrays)
    atomic_json(out / "test_metrics.json", {
        "baseline": a.baseline, "paper": BASELINES[a.baseline][2],
        "reproduction_lane": "common-protocol (adapted)",
        "fold": a.fold, "checkpoint_epoch": best_ep,
        "checkpoint_selected_by": "validation composite only",
        "test_opened_after_selection": True,
        "test_file_sha256": _split_sha(a.fold, "test"),
        "aggregation_within_fold": "per-window metrics averaged within held-out test subject",
        "event_metrics_valid": False,
        "event_metrics_note": "no event head in this baseline; event_f1_* is not applicable",
        "metrics": test_metrics})
    config = json.loads(config_path.read_text())
    config.update({"split_used": ["train", "val", "test"], "test_accessed": True,
                   "test_access_rule": "once, after validation checkpoint selection",
                   "test_file_sha256": _split_sha(a.fold, "test")})
    atomic_json(config_path, config)
    transition_state(state_path, "COMPLETE", fold=a.fold, arm=a.baseline,
                     loss_arm="native", best_epoch=best_ep, test_accessed=True,
                     test_metrics="test_metrics.json")
    print(f"[test] {a.baseline} fold{a.fold:02d}: PCC={test_metrics['pcc']:.4f} "
          f"R2={test_metrics['r2']:+.4f} -> {out / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
