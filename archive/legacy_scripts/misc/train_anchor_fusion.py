"""train_anchor_fusion.py — route C stage-1b lever A: fused anchor localization.

Fuses WSST (strong detection) + raw rcg (fine timing) for beat-anchor
localization at full 1600 resolution. fold_01 only. See docs/ASSEMBLY_DESIGN.md.

Example:
  python scripts/train_anchor_fusion.py --exp_tag q1_anchor_fusion_fold01 --fold_idx 1 \
    --epochs 40 --rpeak_sigma 3 --device cuda:0
"""

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.mmecg_config import MMECGConfig  # noqa: E402
from src.data.mmecg_dataset import _gaussian_mask  # noqa: E402
from src.models.remalign import AnchorHeatmapFusionNet  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402
from scripts.train_anchor_head import centernet_focal, peaks_from_heatmap, match_counts  # noqa: E402
from scripts.evaluate_anchor_head import _metrics as anchor_metrics, _estimate_offset  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WIN = 1600


class FusionAnchorDataset(Dataset):
    def __init__(self, h5_path: Path, wsst_path: Path, sigma: float) -> None:
        self.h5_path = str(h5_path); self.wsst_path = str(wsst_path); self.sigma = float(sigma)
        with h5py.File(self.h5_path, "r") as h:
            self.n = int(h["ecg"].shape[0])
            self.rpeaks = [np.asarray(h["rpeak_indices"][i], dtype=np.int64) for i in range(self.n)]
        self._h = None; self._w = None

    def _handles(self):
        if self._h is None:
            self._h = h5py.File(self.h5_path, "r"); self._w = h5py.File(self.wsst_path, "r")
        return self._h, self._w

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        h, w = self._handles()
        rcg = np.asarray(h["rcg"][i], dtype=np.float32)
        rcg = (rcg - rcg.mean(1, keepdims=True)) / (rcg.std(1, keepdims=True) + 1e-6)
        wsst = np.asarray(w["wsst"][i], dtype=np.float32)
        target = _gaussian_mask(self.rpeaks[i], WIN, sigma=self.sigma)[None, :].astype(np.float32)
        return {"wsst": torch.from_numpy(wsst), "rcg": torch.from_numpy(rcg), "target": torch.from_numpy(target)}


@torch.no_grad()
def _val_f1(model, loader, device, tol=20):
    model.eval(); TP = FP = FN = 0
    for b in loader:
        pred = model(b["wsst"].to(device), b["rcg"].to(device)).cpu().numpy()[:, 0]
        gt = b["target"].numpy()[:, 0]
        for i in range(pred.shape[0]):
            tp, fp, fn = match_counts(peaks_from_heatmap(pred[i]), peaks_from_heatmap(gt[i]), tol)
            TP += tp; FP += fp; FN += fn
    p = TP / (TP + FP + 1e-8); r = TP / (TP + FN + 1e-8)
    return float(2 * p * r / (p + r + 1e-8))


@torch.no_grad()
def _build_items(model, h5_path, wsst_path, device, batch=32):
    items = []
    with h5py.File(h5_path, "r") as h, h5py.File(wsst_path, "r") as w, torch.no_grad():
        n = int(h["ecg"].shape[0])
        for lo in range(0, n, batch):
            idx = list(range(lo, min(lo + batch, n)))
            rcg = np.asarray(h["rcg"][idx], dtype=np.float32)
            rcg = (rcg - rcg.mean(2, keepdims=True)) / (rcg.std(2, keepdims=True) + 1e-6)
            wsst = torch.from_numpy(np.asarray(w["wsst"][idx], dtype=np.float32)).to(device)
            heat = model(wsst, torch.from_numpy(rcg).to(device)).cpu().numpy()[:, 0]
            for k, i in enumerate(idx):
                gt = np.sort(np.unique(np.asarray(h["rpeak_indices"][i], dtype=np.int64)))
                gt = gt[(gt >= 0) & (gt < WIN)].astype(np.float32)
                items.append((heat[k], gt, np.asarray(h["ecg"][i], dtype=np.float32).reshape(-1)))
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp_tag", required=True)
    ap.add_argument("--fold_idx", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--rpeak_sigma", type=float, default=3.0)
    ap.add_argument("--n_blocks", type=int, default=6)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--loso_h5_dir", default=None)
    ap.add_argument("--wsst_suffix", default="_wsst.h5")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = MMECGConfig()
    fold_dir = Path(args.loso_h5_dir or cfg.loso_h5_dir) / f"fold_{args.fold_idx:02d}"
    device = torch.device(args.device)

    def ds(split):
        return FusionAnchorDataset(fold_dir / f"{split}.h5", fold_dir / f"{split}{args.wsst_suffix}", args.rpeak_sigma)
    train_loader = DataLoader(ds("train"), batch_size=args.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(ds("val"), batch_size=args.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)

    model = AnchorHeatmapFusionNet(in_channels=cfg.n_range_bins, hidden=cfg.baseline_hidden,
                                   dropout=cfg.dropout, n_blocks=args.n_blocks, out_len=cfg.signal_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=cfg.weight_decay)

    run_dir = ROOT / cfg.exp_dir / args.exp_tag / f"fold_{args.fold_idx:02d}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "results").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({
        "exp_tag": args.exp_tag, "fold_idx": args.fold_idx, "model_arch": "anchor_heatmap_fusion",
        "input": "wsst+rcg", "epochs": args.epochs, "rpeak_sigma_samples": args.rpeak_sigma,
        "n_blocks": args.n_blocks, "seed": args.seed, "n_range_bins": cfg.n_range_bins,
    }, indent=2))

    print(f"AnchorHeatmapFusionNet params={sum(p.numel() for p in model.parameters())} | fold={args.fold_idx}")
    best = -1.0; history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); tot = 0.0; nb = 0
        for b in train_loader:
            pred = model(b["wsst"].to(device), b["rcg"].to(device))
            loss = centernet_focal(pred, b["target"].to(device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); opt.step()
            tot += float(loss.item()); nb += 1
        f1 = _val_f1(model, val_loader, device)
        history.append({"epoch": epoch, "train_focal": tot / max(nb, 1), "val_f1_100ms": f1})
        print(f"Epoch {epoch:3d}/{args.epochs} | focal={tot/max(nb,1):.4f} | val_f1@100ms={f1:.4f} | {time.time()-t0:.1f}s")
        if f1 > best:
            best = f1
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_f1_100ms": f1},
                       run_dir / "checkpoints" / "anchor_best.pt")
    (run_dir / "results" / "train_history.json").write_text(json.dumps(history, indent=2))
    print(f"Best val F1@100ms: {best:.4f}")

    ckpt = torch.load(run_dir / "checkpoints" / "anchor_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    test_items = _build_items(model, fold_dir / "test.h5", fold_dir / f"test{args.wsst_suffix}", device)
    val_items = _build_items(model, fold_dir / "val.h5", fold_dir / f"val{args.wsst_suffix}", device)
    offset = _estimate_offset(val_items, soft=True)
    variants = {
        "raw": anchor_metrics(test_items, soft=False, offset=0.0),
        "soft": anchor_metrics(test_items, soft=True, offset=0.0),
        "soft+offset": anchor_metrics(test_items, soft=True, offset=offset),
    }
    summary = {"exp_tag": args.exp_tag, "fold_idx": args.fold_idx, "input": "wsst+rcg",
               "num_test_windows": len(test_items), "pep_offset_ms_from_val": offset * (1000.0 / 200),
               "variants": variants}
    out = run_dir / "results_anchor"; out.mkdir(parents=True, exist_ok=True)
    (out / "anchor_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
