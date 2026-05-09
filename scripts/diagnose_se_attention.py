"""
diagnose_se_attention.py — D2: extract SE-attention weights from a trained model.

Loads a trained checkpoint (LOSO fold_01 best.pt by default), runs forward on
its test loader, and hooks the FMCWRangeEncoder's SE-attention vector
(B, 50) post-sigmoid. Aggregates the mean attention vector per
(subject_id, scene) so we can see whether the network concentrates on a few
range bins (and whether those bins agree with D1's correlation analysis).

Output:
  experiments_mmecg/diagnostics/D2_se_attention_heatmap.png
  experiments_mmecg/diagnostics/D2_se_attention_summary.csv

Usage:
  python scripts/diagnose_se_attention.py \
      --exp_tag mmecg_reg_loso_subject --protocol loso --fold_idx 1
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.mmecg_config import MMECGConfig
from src.data.mmecg_dataset import build_loso_loaders_h5, build_samplewise_loaders_h5
from src.models.BeatAwareNet.radar2ecgnet import BeatAwareRadar2ECGNet


STATE_NAMES = {0: "NB", 1: "IB", 2: "SP", 3: "PE"}
OUT_DIR = ROOT / "experiments_mmecg" / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_model(run_dir: Path, device: torch.device):
    cfg = MMECGConfig()
    saved = json.loads((run_dir / "config.json").read_text())
    for k, v in saved.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    model = BeatAwareRadar2ECGNet(
        input_type="fmcw",
        n_range_bins=cfg.n_range_bins,
        C=cfg.C, d_state=cfg.d_state, dropout=cfg.dropout,
        use_pam=cfg.use_pam, use_emd=cfg.use_emd,
        emd_max_delay=cfg.emd_max_delay,
        use_diffusion=getattr(cfg, "use_diffusion", False),
        diff_T=getattr(cfg, "diff_T", 100),
        diff_ddim_steps=getattr(cfg, "diff_ddim_steps", 20),
        diff_hidden=getattr(cfg, "diff_hidden", 256),
        diff_n_blocks=getattr(cfg, "diff_n_blocks", 8),
    ).to(device)

    ckpt = torch.load(run_dir / "checkpoints" / "best.pt",
                      map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model, cfg, ckpt


def _build_loader(protocol: str, fold_idx: int, cfg: MMECGConfig):
    kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        balanced_sampling=False,
        narrow_bandpass=cfg.narrow_bandpass,
    )
    if protocol == "samplewise":
        _, _, loader = build_samplewise_loaders_h5(sw_dir=cfg.samplewise_h5_dir, **kwargs)
    else:
        _, _, loader = build_loso_loaders_h5(fold_idx=fold_idx, loso_dir=cfg.loso_h5_dir, **kwargs)
    return loader


@torch.no_grad()
def _collect(model, loader, device):
    """Hook the SE attention output and collect per-sample (50,) vectors."""
    captured = {}

    enc = model.fmcw_enc

    def _hook_pre(_module, inputs):
        # inputs[0]: (B, 50) tensor passed to se_fc2
        # We instead hook the post-sigmoid output by hooking se_fc2 forward
        pass

    # The actual SE attention vector is sigmoid(se_fc2(relu(se_fc1(pool(x)))))
    # Easiest: monkey-patch the forward to capture
    original_forward = enc.forward
    capture_list = []

    def _forward_capture(x):
        B, R, L = x.shape
        x_filt = torch.nn.functional.gelu(enc.temporal_bn(enc.temporal_filter(x)))
        attn = enc.se_pool(x_filt).squeeze(-1)
        attn = torch.nn.functional.relu(enc.se_fc1(attn))
        attn = torch.sigmoid(enc.se_fc2(attn))   # (B, 50) — capture this
        capture_list.append(attn.detach().cpu().numpy())
        x_att = x_filt * attn.unsqueeze(-1)
        out = enc.proj_bn(enc.proj(x_att))
        return out

    enc.forward = _forward_capture

    subjects, scenes = [], []
    try:
        for batch in loader:
            radar = batch["radar"].to(device)
            _ = model(radar)
            subjects.extend(batch["subject"].tolist())
            scenes.extend(batch["state"].tolist())
    finally:
        enc.forward = original_forward

    attn_all = np.concatenate(capture_list, axis=0)   # (N, 50)
    return attn_all, np.array(subjects, dtype=int), np.array(scenes, dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_tag", default="mmecg_reg_loso_subject")
    parser.add_argument("--protocol", choices=["samplewise", "loso"], default="loso")
    parser.add_argument("--fold_idx", type=int, default=1)
    args = parser.parse_args()

    if args.protocol == "loso":
        run_label = f"fold_{args.fold_idx:02d}"
    else:
        run_label = "samplewise"
    run_dir = ROOT / "experiments_mmecg" / args.exp_tag / run_label
    if not run_dir.exists():
        sys.exit(f"[D2] run dir not found: {run_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[D2] loading {run_dir} on {device}")
    model, cfg, ckpt = _load_model(run_dir, device)
    print(f"[D2] best.pt epoch={ckpt.get('epoch')} val_pcc={ckpt.get('val_pcc'):.4f}")

    loader = _build_loader(args.protocol, args.fold_idx, cfg)
    print(f"[D2] running forward to capture SE attention…")
    attn, subj, scenes_int = _collect(model, loader, device)
    scenes = np.array([STATE_NAMES.get(int(s), "UNK") for s in scenes_int])
    N = len(attn)
    print(f"[D2] captured {N} attention vectors")

    # aggregate per (subject, scene)
    keys = list(zip(subj.tolist(), scenes.tolist()))
    groups = sorted(set(keys), key=lambda x: (x[0], x[1]))
    mat = np.zeros((len(groups), 50), dtype=np.float32)
    rows_csv = []
    for gi, (s, sc) in enumerate(groups):
        idxs = [i for i, k in enumerate(keys) if k == (s, sc)]
        mat[gi] = attn[idxs].mean(axis=0)
        top5 = np.argsort(mat[gi])[-5:][::-1].tolist()
        rows_csv.append({
            "subject_id": int(s),
            "scene": sc,
            "n_windows": len(idxs),
            "max_bin":   int(mat[gi].argmax()),
            "max_attn":  float(mat[gi].max()),
            "mean_attn": float(mat[gi].mean()),
            "std_attn":  float(mat[gi].std()),
            "top5_bins": str(top5),
        })

    # heatmap
    row_labels = [f"S{s} {sc}" for (s, sc) in groups]
    fig, ax = plt.subplots(figsize=(11, 0.32 * len(row_labels) + 1.2))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(0, 50, 5))
    ax.set_xlabel("Range bin (0..49)")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(
        f"D2: FMCWRangeEncoder SE-attention weights — {args.exp_tag}/{run_label}\n"
        f"(best.pt epoch {ckpt.get('epoch')}, val_pcc {ckpt.get('val_pcc'):.3f})",
        fontsize=10,
    )
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="sigmoid(SE) ∈ [0,1]")
    fig.tight_layout()
    out_png = OUT_DIR / f"D2_se_attention_heatmap_{args.exp_tag}_{run_label}.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[D2] saved {out_png}")

    df = pd.DataFrame(rows_csv)
    out_csv = OUT_DIR / f"D2_se_attention_summary_{args.exp_tag}_{run_label}.csv"
    df.to_csv(out_csv, index=False)
    print(f"[D2] saved {out_csv}")
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
