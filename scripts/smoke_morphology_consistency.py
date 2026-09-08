"""Train-only engineering check on fixed real batches; never a test benchmark."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.mmecg_dataset import MMECGWindowedWSSTH5Dataset
from src.losses.losses import TotalLoss
from src.losses.morphology_consistency import morphology_consistency_loss
from src.models.remalign.rem_align_net import REMAlignMTLNet
from src.utils.seeding import set_seed
from scripts.train_mmecg import _build_peak_gts_mmecg, _shape200_head_loss, _shape200_targets


def waveform_metrics(pred, gt):
    p, g = pred.detach().flatten(1), gt.detach().flatten(1)
    pc, gc = p - p.mean(1, keepdim=True), g - g.mean(1, keepdim=True)
    pcc = (pc * gc).sum(1) / (pc.norm(dim=1) * gc.norm(dim=1)).clamp_min(1e-8)
    mse = (p - g).square().mean(1)
    return {"pcc": pcc.mean().item(), "rmse": mse.sqrt().mean().item(),
            "mae": (p - g).abs().mean().item(),
            "r2": (1 - (p - g).square().sum(1) / gc.square().sum(1).clamp_min(1e-8)).mean().item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=80)
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    report = {"tier": "smoke", "publishable": False, "split": "train only",
              "seed": 42, "steps": args.steps, "device": str(device),
              "selection": "first occurrence per sorted subject, up to 8 subjects",
              "aggregation": "fixed-batch mean, not a population estimate",
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "code_sha256": {}, "datasets": {}}
    for name in ("scripts/smoke_morphology_consistency.py", "scripts/train_mmecg.py",
                 "src/losses/morphology_consistency.py", "src/data/mmecg_dataset.py"):
        report["code_sha256"][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    paths = {"MMECG": Path("/home/qhh2237/Datasets/MMECG/processed/loso/fold_01"),
             "CR_RVS": Path("/home/qhh2237/Datasets/CR_RVS/processed/q1_shape_kfold/fold_01")}
    for dataset, path in paths.items():
        ds = MMECGWindowedWSSTH5Dataset(path / "train.h5", path / "train_wsst.h5")
        indices = [int(np.flatnonzero(ds._subj == sid)[0]) for sid in sorted(set(ds._subj))[:8]]
        batch = default_collate([ds[i] for i in indices])
        x, gt = batch["radar"].to(device), batch["ecg"].to(device)
        peaks = _build_peak_gts_mmecg(batch, batch["rpeak"].to(device), device)
        row = {"indices": indices, "subjects": batch["subject"].tolist(),
               "input_shape": list(x.shape), "source": str(path),
               "annotation_coverage": batch["cycle_valid"].float().mean().item(),
               "source_files": {}, "arms": {}}
        for filename in ("train.h5", "train_wsst.h5"):
            stat = (path / filename).stat()
            row["source_files"][filename] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        for arm, weight, mode in (("B0", 0.0, "head"), ("B1", 0.05, "head"),
                                  ("B2", 0.05, "reference")):
            set_seed(42)
            model = REMAlignMTLNet(in_channels=x.shape[1], use_alignment=False).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            criterion = TotalLoss(alpha_stft=0.1, beta_peak=0.3).to(device)
            shape_gt = _shape200_targets(gt, batch["cycle_bounds"])
            def measure():
                model.eval()
                with torch.no_grad():
                    output, _ = model(x)
                    return {"waveform": waveform_metrics(output, gt),
                            "shape": waveform_metrics(output.shape200_pred, shape_gt)}
            before = measure()
            history = []
            started = time.monotonic()
            for step in range(args.steps):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                output, events = model(x)
                loss = criterion(output, gt, events, peaks, epoch=1)["total"]
                loss = loss + 0.8 * _shape200_head_loss(output.shape200_pred, gt,
                                                      batch["cycle_bounds"], pcc_weight=0.5)
                coupling, valid = morphology_consistency_loss(
                    output, output.shape200_pred, gt, batch["cycle_bounds"],
                    target_mode=mode, cycle_valid=batch["cycle_valid"],
                )
                total = loss + weight * coupling
                if not torch.isfinite(total):
                    raise RuntimeError(f"Non-finite loss: {dataset} {arm} step {step}")
                total.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                history.append({"step": step + 1, "total": total.item(),
                                "consistency": coupling.item(), "coverage": valid / len(indices),
                                "gradient_norm_before_clip": norm.item()})
            after = measure()
            row["arms"][arm] = {"before": before, "after": after, "history": history,
                                "seconds": time.monotonic() - started,
                                "parameters": sum(p.numel() for p in model.parameters())}
            print(json.dumps({"dataset": dataset, "arm": arm, "before": before,
                              "after": after}), flush=True)
        report["datasets"][dataset] = row
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Smoke complete: {args.output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
