"""train_anchor_head.py — Stage-1 standalone beat-anchor localization (route C).

Trains AnchorHeatmapNet (WSST -> per-sample anchor heatmap) ONLY, with no
waveform decoder and no morphology head, so anchor quality can be measured in
isolation. See docs/ASSEMBLY_DESIGN.md.

Example:
  python scripts/train_anchor_head.py --exp_tag q1_anchor_head_fold01 --fold_idx 1 \
    --epochs 40 --device cuda:0
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.mmecg_config import MMECGConfig  # noqa: E402
from src.data.mmecg_dataset import build_loso_wsst_loaders_h5  # noqa: E402
from src.models.remalign import AnchorHeatmapNet  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def centernet_focal(pred: torch.Tensor, gt: torch.Tensor, alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """Penalty-reduced focal loss for Gaussian heatmap targets (CenterNet)."""
    eps = 1e-6
    pred = pred.clamp(eps, 1.0 - eps)
    pos_mask = (gt >= 1.0 - 1e-4).float()
    neg_mask = 1.0 - pos_mask
    pos_loss = -((1.0 - pred) ** alpha) * torch.log(pred) * pos_mask
    neg_loss = -((1.0 - gt) ** beta) * (pred ** alpha) * torch.log(1.0 - pred) * neg_mask
    n_pos = pos_mask.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / n_pos


def peaks_from_heatmap(h1d: np.ndarray, min_dist: int = 40) -> np.ndarray:
    y = np.asarray(h1d, dtype=np.float32).reshape(-1)
    if y.size < 3:
        return np.empty((0,), dtype=np.int64)
    ymax = float(y.max())
    if ymax < 1e-6:
        return np.empty((0,), dtype=np.int64)
    thr = max(0.3, 0.5 * ymax)
    cand = np.flatnonzero((y[1:-1] >= y[:-2]) & (y[1:-1] >= y[2:]) & (y[1:-1] >= thr)) + 1
    peaks: list[int] = []
    for p in cand.tolist():
        if not peaks or p - peaks[-1] >= min_dist:
            peaks.append(int(p))
        elif y[p] > y[peaks[-1]]:
            peaks[-1] = int(p)
    return np.asarray(peaks, dtype=np.int64)


def match_counts(pred: np.ndarray, gt: np.ndarray, tol: int) -> tuple[int, int, int]:
    """Greedy nearest one-to-one matching within tol samples -> (tp, fp, fn)."""
    if gt.size == 0 and pred.size == 0:
        return 0, 0, 0
    if gt.size == 0:
        return 0, int(pred.size), 0
    if pred.size == 0:
        return 0, 0, int(gt.size)
    d = np.abs(gt[:, None] - pred[None, :])
    order = np.argsort(d.ravel())
    mg = np.zeros(gt.size, bool)
    mp = np.zeros(pred.size, bool)
    tp = 0
    for f in order:
        gi, pi = divmod(int(f), pred.size)
        if mg[gi] or mp[pi]:
            continue
        if d[gi, pi] <= tol:
            mg[gi] = mp[pi] = True
            tp += 1
    return tp, int((~mp).sum()), int((~mg).sum())


@torch.no_grad()
def val_f1(model, loader, device, tol: int = 20) -> float:
    model.eval()
    TP = FP = FN = 0
    for batch in loader:
        radar = batch["radar"].to(device)
        pred = model(radar).cpu().numpy()[:, 0]
        gt = batch["rpeak"].numpy()[:, 0]
        for i in range(pred.shape[0]):
            pp = peaks_from_heatmap(pred[i])
            gp = peaks_from_heatmap(gt[i])
            tp, fp, fn = match_counts(pp, gp, tol)
            TP += tp; FP += fp; FN += fn
    prec = TP / (TP + FP + 1e-8)
    rec = TP / (TP + FN + 1e-8)
    return float(2 * prec * rec / (prec + rec + 1e-8))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp_tag", required=True)
    ap.add_argument("--fold_idx", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--loso_h5_dir", default=None)
    ap.add_argument("--wsst_suffix", default="_wsst.h5")
    ap.add_argument("--balance_by", default="subject")
    ap.add_argument("--rpeak_sigma", type=float, default=5.0,
                    help="Gaussian R-target sigma in samples (narrower = sharper localization).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = MMECGConfig()
    loso_dir = args.loso_h5_dir or cfg.loso_h5_dir
    device = torch.device(args.device)

    train_loader, val_loader, _ = build_loso_wsst_loaders_h5(
        fold_idx=args.fold_idx, loso_dir=loso_dir, batch_size=args.batch_size,
        num_workers=cfg.num_workers, balanced_sampling=cfg.balanced_sampling,
        balance_by=args.balance_by, target_norm=cfg.target_norm, wsst_suffix=args.wsst_suffix,
        rpeak_sigma=args.rpeak_sigma,
    )

    model = AnchorHeatmapNet(in_channels=cfg.n_range_bins, hidden=cfg.baseline_hidden,
                             dropout=cfg.dropout, out_len=cfg.signal_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=cfg.weight_decay)

    run_dir = ROOT / cfg.exp_dir / args.exp_tag / f"fold_{args.fold_idx:02d}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "results").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({
        "exp_tag": args.exp_tag, "fold_idx": args.fold_idx, "model_arch": "anchor_heatmap",
        "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
        "loso_h5_dir": loso_dir, "wsst_suffix": args.wsst_suffix,
        "balance_by": args.balance_by, "rpeak_sigma_samples": args.rpeak_sigma, "seed": args.seed,
        "n_range_bins": cfg.n_range_bins, "signal_len": cfg.signal_len,
    }, indent=2))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"AnchorHeatmapNet params={n_params} | fold={args.fold_idx} | device={device}")
    best_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tot = 0.0; nb = 0
        for batch in train_loader:
            radar = batch["radar"].to(device)
            target = batch["rpeak"].to(device)
            pred = model(radar)
            loss = centernet_focal(pred, target)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tot += float(loss.item()); nb += 1
        f1 = val_f1(model, val_loader, device, tol=20)
        history.append({"epoch": epoch, "train_focal": tot / max(nb, 1), "val_f1_100ms": f1})
        print(f"Epoch {epoch:3d}/{args.epochs} | focal={tot/max(nb,1):.4f} | val_f1@100ms={f1:.4f} | {time.time()-t0:.1f}s")
        if f1 > best_f1:
            best_f1 = f1
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_f1_100ms": f1},
                       run_dir / "checkpoints" / "anchor_best.pt")
    (run_dir / "results" / "train_history.json").write_text(json.dumps(history, indent=2))
    print(f"Best val F1@100ms: {best_f1:.4f}")


if __name__ == "__main__":
    main()
