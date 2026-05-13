"""
test_mmecg.py — MMECG 完整 4 级评估脚本（LOSO / Samplewise）

用法：
  # LOSO：单折
  python scripts/test_mmecg.py --exp_tag mmecg_v1 --fold_idx 1

  # LOSO：全部 11 折 + 汇总
  python scripts/test_mmecg.py --exp_tag mmecg_v1 --fold_idx -1

  # Samplewise
  python scripts/test_mmecg.py --exp_tag sw_v1 --protocol samplewise

输出（per fold / run）：
  experiments_mmecg/<exp_tag>/<run_label>/results/
    segment_metrics.csv    每行 = 1 个 8s segment，含全 4 级指标
    beat_metrics.csv       每行 = 1 个匹配 beat
    subject_summary.csv    按 (subject_id, scene) 聚合
    global_summary.json    全局 mean/median/std/IQR + QMR

LOSO 模式追加：
  experiments_mmecg/<exp_tag>/loso_summary/
    all_segments.csv       11 折 segment 合并
    subject_summary.csv
    global_summary.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.mmecg_config import MMECGConfig
from src.data.mmecg_dataset import (
    build_loso_calibration_loaders_h5,
    build_loso_loaders_h5,
    build_samplewise_loaders_h5,
)
from src.models.BeatAwareNet.radar2ecgnet import BeatAwareRadar2ECGNet
from src.utils.metrics import (
    compute_waveform_metrics_protocol,
    compute_qrst_peak_timing_errors,
    compute_relative_peak_timing_errors,
    compute_rr_interval_error,
    compute_t_wave_timing_error,
    compute_mdr,
    compute_fiducial_detection_f1,
    compute_clinical_interval_errors,
    summarize_subject_metrics,
    summarize_global_metrics,
)

STATE_NAMES = {0: "NB", 1: "IB", 2: "SP", 3: "PE"}
FS = 200
WIN_LEN = 1600


# =============================================================================
# 模型加载
# =============================================================================

def _load_model(cfg: MMECGConfig, run_dir: Path, device: torch.device):
    ckpt_path = run_dir / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # 尝试从保存的 config.json 恢复模型超参
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as jf:
            saved = json.load(jf)
        for k in ("C", "d_state", "dropout", "use_pam", "use_emd", "emd_max_delay",
                  "n_range_bins", "use_diffusion", "diff_T", "diff_ddim_steps",
                  "diff_hidden", "diff_n_blocks", "narrow_bandpass",
                  "use_output_lag_align", "output_lag_max_ms", "fs",
                  "topk_bins", "target_norm", "topk_method",
                  "fmcw_selector", "fmcw_topk", "fmcw_tau_init", "fmcw_tau_final"):
            if k in saved:
                setattr(cfg, k, saved[k])

    model = BeatAwareRadar2ECGNet(
        input_type="fmcw",
        n_range_bins=cfg.n_range_bins,
        C=cfg.C,
        d_state=cfg.d_state,
        dropout=cfg.dropout,
        use_pam=cfg.use_pam,
        use_emd=cfg.use_emd,
        emd_max_delay=cfg.emd_max_delay,
        use_diffusion=cfg.use_diffusion,
        diff_T=cfg.diff_T,
        diff_ddim_steps=cfg.diff_ddim_steps,
        diff_hidden=cfg.diff_hidden,
        diff_n_blocks=cfg.diff_n_blocks,
        use_output_lag_align=getattr(cfg, "use_output_lag_align", False),
        output_lag_max_samples=int(round(
            getattr(cfg, "output_lag_max_ms", 200.0) / 1000.0 * getattr(cfg, "fs", 200)
        )),
        fmcw_selector=getattr(cfg, "fmcw_selector", "se"),
        fmcw_topk=getattr(cfg, "fmcw_topk", 10),
        fmcw_tau=getattr(cfg, "fmcw_tau_final", 0.1),  # eval 用退火后温度
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    # strict=False: 兼容旧 ckpt（含已删除的 ConformerFusionBlock fusion.* 权重）
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if unexpected:
        print(f"[load_state_dict] dropped {len(unexpected)} unexpected keys "
              f"(e.g. {unexpected[:3]})")
    if missing:
        print(f"[load_state_dict] missing {len(missing)} keys (e.g. {missing[:3]})")
    model.eval()
    print(f"  Loaded checkpoint: epoch={ckpt.get('epoch','?')} "
          f"val_pcc={ckpt.get('val_pcc', float('nan')):.4f}")
    return model


# =============================================================================
# 推理：收集全部预测和 GT
# =============================================================================

def _run_inference(model, test_loader, device):
    """
    返回 lists（N 个 segment）:
      pred_list: [np.ndarray (1600,) float32]
      gt_list:   [np.ndarray (1600,) float32]
      r_idx_list, q_idx_list, s_idx_list, t_idx_list: [np.ndarray int32]
      subj_list:  [int]
      state_list: [int]
      delin_list: [bool]
    """
    pred_list, gt_list = [], []
    r_list, q_list, s_list, t_list = [], [], [], []
    subj_list, state_list, delin_list = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            radar  = batch["radar"].to(device)           # [B,50,1600]
            ecg_gt = batch["ecg"]                        # [B,1,1600] cpu
            subj   = batch["subject"].tolist()           # list[int]
            state  = batch["state"].tolist()             # list[int]

            ecg_pred, _ = model(radar)
            if not torch.isfinite(ecg_pred).all():
                # skip entire batch on NaN
                B = radar.shape[0]
                pred_list.extend([np.zeros(WIN_LEN, np.float32)] * B)
                gt_list.extend([ecg_gt[i, 0].numpy() for i in range(B)])
                r_list.extend([batch["r_idx"][i] for i in range(B)])
                q_list.extend([batch["q_idx"][i] for i in range(B)])
                s_list.extend([batch["s_idx"][i] for i in range(B)])
                t_list.extend([batch["t_idx"][i] for i in range(B)])
                subj_list.extend(subj)
                state_list.extend(state)
                delin_list.extend([False] * B)
                continue

            ecg_pred_np = ecg_pred.cpu().numpy()         # [B,1,1600]
            ecg_gt_np   = ecg_gt.numpy()

            for i in range(ecg_pred_np.shape[0]):
                pred_list.append(ecg_pred_np[i, 0])
                gt_list.append(ecg_gt_np[i, 0])
                r_list.append(batch["r_idx"][i])
                q_list.append(batch["q_idx"][i])
                s_list.append(batch["s_idx"][i])
                t_list.append(batch["t_idx"][i])
                subj_list.append(subj[i])
                state_list.append(state[i])
                delin_list.append(bool(batch["delin_valid"][i].item()))

    return (pred_list, gt_list, r_list, q_list, s_list, t_list,
            subj_list, state_list, delin_list)


# =============================================================================
# 单 segment 全 4 级评估
# =============================================================================

def _evaluate_segment(
    seg_id: int,
    pred_1d: np.ndarray,
    gt_1d:   np.ndarray,
    gt_r:    np.ndarray,
    gt_q:    np.ndarray,
    gt_s:    np.ndarray,
    gt_t:    np.ndarray,
    subject_id: int,
    state_code: int,
    delin_valid: bool,
) -> tuple[dict, list[dict]]:
    """
    返回 (segment_row dict, beat_rows list).
    beat_rows 每项对应一个匹配的 R 峰 beat。
    """
    row: dict = {
        "segment_id":          seg_id,
        "subject_id":          subject_id,
        "scene":               STATE_NAMES.get(state_code, "UNK"),
        "fs":                  FS,
        "segment_length_sec":  WIN_LEN / FS,
        "delineation_valid":   int(delin_valid),
    }

    # ── Level 1 ──────────────────────────────────────────────────────────────
    l1 = compute_waveform_metrics_protocol(pred_1d, gt_1d)
    row.update(l1)

    # ── Level 2 ──────────────────────────────────────────────────────────────
    qrst = compute_qrst_peak_timing_errors(
        pred_1d, gt_r, gt_q, gt_s, gt_t, fs=FS,
    )
    row.update(qrst["segment_summary"])

    # relative timing errors
    rel = compute_relative_peak_timing_errors(
        {"r": qrst["r_errors"], "q": qrst["q_errors"],
         "s": qrst["s_errors"], "t": qrst["t_errors"]},
        T_seg_samples=WIN_LEN, fs=FS,
    )
    row.update(rel)

    rr = compute_rr_interval_error(pred_1d, gt_r, fs=FS)
    row["rr_interval_error_ms_mean"] = rr["rr_interval_error_ms_mean"]
    row["ppi_error_ms_mean"]         = rr["ppi_error_ms_mean"]

    tw = compute_t_wave_timing_error(pred_1d, gt_t, fs=FS)
    row["t_wave_timing_error_ms_mean"] = tw["t_wave_timing_error_ms_mean"]

    mdr = compute_mdr(qrst["pred_r"], qrst["gt_r_v"], tolerance_samples=10)
    row.update(mdr)

    # QMR flag（该 segment 是否合格）
    n_gt_r   = qrst["segment_summary"]["num_r_gt"]
    n_pred_r = qrst["segment_summary"]["num_r_pred"]
    n_match  = qrst["segment_summary"]["num_matched_beats"]
    qualified = (n_gt_r >= 2) and (n_pred_r >= 2) and (n_match >= 1)
    row["qualified_flag"]         = int(qualified)
    row["segment_failed_pcc60"]   = int(row.get("pcc_raw", 0.0) < 0.60)

    # ── Level 3（需 delin_valid=True 保证 GT ECG delineation 可靠）────────────
    if delin_valid:
        try:
            fid = compute_fiducial_detection_f1(pred_1d, gt_1d, gt_r, fs=FS)
            row.update(fid)
        except Exception:
            pass
    else:
        # 填 NaN 占位，保持列结构一致
        for tol in ("150ms", "100ms", "50ms"):
            for fid_name in ("pon", "qon", "rpeak", "soff", "toff"):
                for metric in ("precision", "recall", "f1"):
                    row[f"{fid_name}_{metric}_{tol}"] = float("nan")
            row[f"average_f1_{tol}"] = float("nan")

    # ── Level 4 ──────────────────────────────────────────────────────────────
    if delin_valid:
        try:
            clin = compute_clinical_interval_errors(pred_1d, gt_1d, gt_r, fs=FS)
            row.update(clin)
        except Exception:
            pass
    else:
        for k in ("pr_interval_error_ms", "qrs_duration_error_ms",
                  "qt_interval_error_ms", "qtc_interval_error_ms"):
            row[k] = float("nan")

    # ── beat rows 补充 segment 元数据 ─────────────────────────────────────────
    beat_rows = []
    for br in qrst["beat_rows"]:
        br = dict(br)
        br["segment_id"] = seg_id
        br["subject_id"] = subject_id
        br["scene"]      = row["scene"]
        # 加上 RR 误差（从 rr["rr_errors"] 按 beat index 填，长度可能差 1）
        beat_rows.append(br)

    # 将 rr_errors 对应到 beat_rows（rr_errors 比 r_pairs 少 1）
    rr_errs = rr.get("rr_errors", [])
    for k, br in enumerate(beat_rows):
        br["rr_interval_error_ms"] = rr_errs[k] if k < len(rr_errs) else float("nan")

    return row, beat_rows


# =============================================================================
# 单次测试（LOSO 单折 or Samplewise）
# =============================================================================

def test_one_run(
    cfg: MMECGConfig,
    exp_tag: str,
    run_label: str,
    test_loader,
) -> pd.DataFrame:
    """
    运行一次完整测试，返回 segment_metrics DataFrame，同时写 4 个输出文件。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir    = Path(cfg.exp_dir) / exp_tag / run_label
    result_dir = run_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Testing: {run_label}  |  test_samples={len(test_loader.dataset)}")

    model = _load_model(cfg, run_dir, device)

    # ── 推理 ──────────────────────────────────────────────────────────────────
    (pred_list, gt_list, r_list, q_list, s_list, t_list,
     subj_list, state_list, delin_list) = _run_inference(model, test_loader, device)

    N = len(pred_list)
    print(f"  Collected {N} segments")

    # ── 逐 segment 评估 ────────────────────────────────────────────────────────
    all_seg_rows:  list[dict] = []
    all_beat_rows: list[dict] = []

    for i in range(N):
        seg_row, beat_rows = _evaluate_segment(
            seg_id      = i,
            pred_1d     = pred_list[i],
            gt_1d       = gt_list[i],
            gt_r        = r_list[i],
            gt_q        = q_list[i],
            gt_s        = s_list[i],
            gt_t        = t_list[i],
            subject_id  = subj_list[i],
            state_code  = state_list[i],
            delin_valid = delin_list[i],
        )
        all_seg_rows.append(seg_row)
        all_beat_rows.extend(beat_rows)

        if (i + 1) % 200 == 0:
            print(f"  Evaluated {i+1}/{N} segments ...")

    seg_df  = pd.DataFrame(all_seg_rows)
    beat_df = pd.DataFrame(all_beat_rows)

    # ── subject summary ────────────────────────────────────────────────────────
    subj_df = summarize_subject_metrics(all_seg_rows)

    # ── global summary ─────────────────────────────────────────────────────────
    global_dict = summarize_global_metrics(all_seg_rows)

    # ── 写文件 ─────────────────────────────────────────────────────────────────
    seg_df.to_csv(result_dir / "segment_metrics.csv",   index=False)
    beat_df.to_csv(result_dir / "beat_metrics.csv",     index=False)
    subj_df.to_csv(result_dir / "subject_summary.csv",  index=False)
    with open(result_dir / "global_summary.json", "w") as jf:
        json.dump(global_dict, jf, indent=2, default=_json_default)

    # ── 打印摘要 ───────────────────────────────────────────────────────────────
    pcc  = global_dict.get("pcc_raw_mean",  float("nan"))
    rmse = global_dict.get("rmse_norm_mean", float("nan"))
    r2   = global_dict.get("r2_mean",        float("nan"))
    qmr  = global_dict.get("qualified_monitoring_rate", float("nan"))
    rr_e = global_dict.get("rr_interval_error_ms_mean_mean", float("nan"))
    f1_  = global_dict.get("average_f1_150ms_mean", float("nan"))
    print(f"  PCC={pcc:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
    print(f"  QMR={qmr:.1f}%  RR_err={rr_e:.1f}ms  avg_F1_150ms={f1_:.4f}")
    print(f"  Results: {result_dir}")

    return seg_df


# =============================================================================
# 工具
# =============================================================================

def _json_default(obj):
    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return str(obj)


def _loso_summary(all_seg_dfs: list[pd.DataFrame], out_dir: Path) -> None:
    """合并 11 折 segment CSV，重新计算全局统计。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(all_seg_dfs, ignore_index=True)
    combined.to_csv(out_dir / "all_segments.csv", index=False)

    rows = combined.to_dict("records")
    subj_df     = summarize_subject_metrics(rows)
    global_dict = summarize_global_metrics(rows)

    subj_df.to_csv(out_dir / "subject_summary.csv", index=False)
    with open(out_dir / "global_summary.json", "w") as jf:
        json.dump(global_dict, jf, indent=2, default=_json_default)

    print(f"\n{'='*60}")
    print(f"LOSO Global Summary ({len(all_seg_dfs)} folds, {len(combined)} segments):")
    for k in ("pcc_raw_mean", "rmse_norm_mean", "r2_mean",
              "rr_interval_error_ms_mean_mean", "r_peak_error_ms_mean_mean",
              "average_f1_150ms_mean", "qrs_duration_error_ms_mean"):
        v = global_dict.get(k)
        if v is not None:
            print(f"  {k}: {v:.4f}")
    qmr = global_dict.get("qualified_monitoring_rate")
    if qmr is not None:
        print(f"  qualified_monitoring_rate: {qmr:.2f}%")
    print(f"  Saved: {out_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_tag",  type=str, required=True)
    parser.add_argument("--fold_idx", type=int, default=-1,
                        help="LOSO: 1-based (1~11); -1 = all folds")
    parser.add_argument("--protocol", type=str, default="loso",
                        choices=["loso", "loso_calib", "samplewise"])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--calib_ratio", type=float, default=0.4,
                        help="For protocol=loso_calib: fraction of held-out subject windows used as labeled calibration training.")
    parser.add_argument("--calib_val_ratio", type=float, default=0.1,
                        help="For protocol=loso_calib: fraction of held-out subject windows used as calibration validation.")
    parser.add_argument("--calib_n_train", type=int, default=None,
                        help="For protocol=loso_calib: fixed number of held-out subject windows used as labeled calibration training. Overrides calib_ratio when set.")
    parser.add_argument("--calib_n_val", type=int, default=None,
                        help="For protocol=loso_calib: fixed number of held-out subject windows used as calibration validation. Overrides calib_val_ratio when set.")
    parser.add_argument("--calib_seed", type=int, default=42,
                        help="For protocol=loso_calib: seed for calibration/eval split.")
    args = parser.parse_args()

    cfg = MMECGConfig()
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    print(f"MMECG test | exp_tag={args.exp_tag} | protocol={args.protocol}")

    # narrow_bandpass will be set per-run by _load_model reading config.json.
    # Pass current cfg value (default False) here; test_one_run re-loads
    # the saved config before building the model, but loaders need bandpass too.
    # We read it from the first available saved config.
    def _get_loader_overrides(exp_tag: str, run_label: str) -> dict:
        """Pull narrow_bandpass / topk_bins / target_norm out of saved config."""
        cfg_path = Path(cfg.exp_dir) / exp_tag / run_label / "config.json"
        if not cfg_path.exists():
            return {"narrow_bandpass": False}
        with open(cfg_path) as jf:
            saved = json.load(jf)
        tk = saved.get("topk_bins", 0)
        return {
            "narrow_bandpass": bool(saved.get("narrow_bandpass", False)),
            "topk_bins": tk if (tk and tk > 0) else None,
            "target_norm": saved.get("target_norm", "minmax"),
            "topk_method": saved.get("topk_method", "energy"),
        }

    if args.protocol == "samplewise":
        ovr = _get_loader_overrides(args.exp_tag, "samplewise")
        _, _, test_loader = build_samplewise_loaders_h5(
            sw_dir          = cfg.samplewise_h5_dir,
            batch_size      = cfg.batch_size,
            num_workers     = cfg.num_workers,
            balanced_sampling = False,
            **ovr,
        )
        test_one_run(cfg, args.exp_tag, run_label="samplewise",
                     test_loader=test_loader)
    else:
        folds = list(range(1, cfg.n_folds + 1)) if args.fold_idx == -1 else [args.fold_idx]
        print(f"LOSO folds: {folds}")

        all_seg_dfs = []
        for fold in folds:
            ovr = _get_loader_overrides(args.exp_tag, f"fold_{fold:02d}")
            if args.protocol == "loso_calib":
                _, _, test_loader = build_loso_calibration_loaders_h5(
                    fold_idx        = fold,
                    loso_dir        = cfg.loso_h5_dir,
                    calib_ratio     = args.calib_ratio,
                    calib_val_ratio = args.calib_val_ratio,
                    calib_n_train   = args.calib_n_train,
                    calib_n_val     = args.calib_n_val,
                    calib_seed      = args.calib_seed,
                    batch_size      = cfg.batch_size,
                    num_workers     = cfg.num_workers,
                    balanced_sampling = False,
                    **ovr,
                )
            else:
                _, _, test_loader = build_loso_loaders_h5(
                    fold_idx        = fold,
                    loso_dir        = cfg.loso_h5_dir,
                    batch_size      = cfg.batch_size,
                    num_workers     = cfg.num_workers,
                    balanced_sampling = False,
                    **ovr,
                )
            seg_df = test_one_run(
                cfg, args.exp_tag,
                run_label  = f"fold_{fold:02d}",
                test_loader = test_loader,
            )
            all_seg_dfs.append(seg_df)

        if len(all_seg_dfs) > 1:
            _loso_summary(
                all_seg_dfs,
                out_dir=Path(cfg.exp_dir) / args.exp_tag / "loso_summary",
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
