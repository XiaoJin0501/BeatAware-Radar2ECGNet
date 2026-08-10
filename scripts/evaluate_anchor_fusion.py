"""evaluate_anchor_fusion.py — enhanced eval for the fused WSST+rcg anchor head.

Reports F1@50/100/150, MAE, signed bias, missed/extra rate, RR error (crude +
matched), and predicted/oracle shape-given OLA PCC. See docs/ASSEMBLY_DESIGN.md.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.mmecg_config import MMECGConfig  # noqa: E402
from src.models.remalign import AnchorHeatmapFusionNet  # noqa: E402
from scripts.train_anchor_fusion import _build_items  # noqa: E402
from scripts.evaluate_anchor_head import _metrics, _estimate_offset  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp_tag", required=True)
    ap.add_argument("--fold_idx", type=int, required=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--loso_h5_dir", default=None)
    ap.add_argument("--wsst_suffix", default="_wsst.h5")
    ap.add_argument("--n_blocks", type=int, default=6)
    args = ap.parse_args()

    cfg = MMECGConfig()
    fold_dir = Path(args.loso_h5_dir or cfg.loso_h5_dir) / f"fold_{args.fold_idx:02d}"
    run_dir = ROOT / cfg.exp_dir / args.exp_tag / f"fold_{args.fold_idx:02d}"
    device = torch.device(args.device)

    model = AnchorHeatmapFusionNet(in_channels=cfg.n_range_bins, hidden=cfg.baseline_hidden,
                                   dropout=cfg.dropout, n_blocks=args.n_blocks, out_len=cfg.signal_len).to(device)
    ckpt = torch.load(run_dir / "checkpoints" / "anchor_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"]); model.eval()

    test_items = _build_items(model, fold_dir / "test.h5", fold_dir / f"test{args.wsst_suffix}", device)
    val_items = _build_items(model, fold_dir / "val.h5", fold_dir / f"val{args.wsst_suffix}", device)
    offset = _estimate_offset(val_items, soft=True)
    variants = {
        "raw": _metrics(test_items, soft=False, offset=0.0),
        "soft": _metrics(test_items, soft=True, offset=0.0),
        "soft+offset": _metrics(test_items, soft=True, offset=offset),
    }
    summary = {"exp_tag": args.exp_tag, "fold_idx": args.fold_idx, "input": "wsst+rcg",
               "num_test_windows": len(test_items), "pep_offset_ms_from_val": offset * (1000.0 / 200),
               "variants": variants}
    out = run_dir / "results_anchor"; out.mkdir(parents=True, exist_ok=True)
    (out / "anchor_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
