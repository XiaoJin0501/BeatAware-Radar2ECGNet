"""
metrics.py — ECG 重建质量评估指标

波形指标（逐样本，再取 batch 均值）：
  MAE   : Mean Absolute Error
  RMSE  : Root Mean Squared Error
  PCC   : Pearson Correlation Coefficient
  PRD   : Percent Root-mean-square Difference（ECG 领域标准）

峰值指标（在重建 ECG 上重新检测 R 峰，与 GT R 峰比较）：
  F1    : R 峰检测 F1 Score（容忍窗口 ±25ms = ±5 samples @200Hz）
"""

import numpy as np
import torch


# =============================================================================
# 波形指标（Tensor，batch 级别）
# =============================================================================

def compute_waveform_metrics(
    pred: torch.Tensor,   # (B, 1, L)
    gt:   torch.Tensor,   # (B, 1, L)
) -> dict[str, float]:
    """
    计算波形重建指标，返回 batch 均值。

    Returns
    -------
    dict with keys: mae, rmse, pcc, prd
    """
    pred_np = pred.detach().cpu().numpy().squeeze(1)   # (B, L)
    gt_np   = gt.detach().cpu().numpy().squeeze(1)

    B = pred_np.shape[0]
    mae_list, rmse_list, pcc_list, prd_list = [], [], [], []

    for i in range(B):
        p, g = pred_np[i], gt_np[i]

        mae  = float(np.mean(np.abs(p - g)))
        rmse = float(np.sqrt(np.mean((p - g) ** 2)))

        # PCC
        p_c = p - p.mean()
        g_c = g - g.mean()
        denom = np.sqrt((p_c ** 2).sum() * (g_c ** 2).sum()) + 1e-8
        pcc  = float(np.dot(p_c, g_c) / denom)

        # PRD = sqrt( sum((p-g)^2) / sum(g^2) ) * 100
        prd  = float(np.sqrt(np.sum((p - g) ** 2) / (np.sum(g ** 2) + 1e-8)) * 100)

        mae_list.append(mae)
        rmse_list.append(rmse)
        pcc_list.append(pcc)
        prd_list.append(prd)

    return {
        "mae":  float(np.mean(mae_list)),
        "rmse": float(np.mean(rmse_list)),
        "pcc":  float(np.mean(pcc_list)),
        "prd":  float(np.mean(prd_list)),
    }


# =============================================================================
# R 峰 F1（numpy，单样本）
# =============================================================================

_NK_AVAILABLE = None

def _check_neurokit() -> bool:
    global _NK_AVAILABLE
    if _NK_AVAILABLE is None:
        try:
            import neurokit2  # noqa: F401
            _NK_AVAILABLE = True
        except ImportError:
            _NK_AVAILABLE = False
    return _NK_AVAILABLE


def detect_rpeaks(ecg_1d: np.ndarray, fs: int = 200) -> np.ndarray:
    """
    用 NeuroKit2 在 numpy 1D ECG 上检测 R 峰索引。

    ecg_1d : (L,) float，值域 [0,1]（归一化 ECG）
    返回   : R 峰索引数组（可能为空）
    """
    if not _check_neurokit():
        return np.array([], dtype=np.int64)

    import neurokit2 as nk
    try:
        _, info = nk.ecg_peaks(ecg_1d, sampling_rate=fs, method="neurokit")
        peaks = info.get("ECG_R_Peaks", np.array([]))
        return np.asarray(peaks, dtype=np.int64)
    except Exception:
        return np.array([], dtype=np.int64)


def rpeak_f1(
    pred_peaks: np.ndarray,
    gt_peaks:   np.ndarray,
    tolerance:  int = 5,       # ±5 samples = ±25ms @200Hz
) -> float:
    """
    计算 R 峰检测 F1 Score。

    每个 gt 峰在 pred 中如果存在距离 ≤ tolerance 的配对，则视为 TP。
    未配对的 pred 峰视为 FP，未配对的 gt 峰视为 FN。

    Returns
    -------
    float : F1 ∈ [0, 1]
    """
    if len(gt_peaks) == 0 and len(pred_peaks) == 0:
        return 1.0
    if len(gt_peaks) == 0 or len(pred_peaks) == 0:
        return 0.0

    matched_gt   = np.zeros(len(gt_peaks),   dtype=bool)
    matched_pred = np.zeros(len(pred_peaks), dtype=bool)

    for i, gp in enumerate(gt_peaks):
        dists = np.abs(pred_peaks - gp)
        j = int(np.argmin(dists))
        if dists[j] <= tolerance and not matched_pred[j]:
            matched_gt[i]   = True
            matched_pred[j] = True

    tp = int(matched_gt.sum())
    fp = int((~matched_pred).sum())
    fn = int((~matched_gt).sum())

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return float(f1)


def compute_peak_metrics(
    pred: torch.Tensor,   # (B, 1, L)
    gt:   torch.Tensor,   # (B, 1, L)
    fs:   int = 200,
) -> dict[str, float]:
    """
    在 batch 中每条重建 ECG 上重新检测 R 峰，与 GT 比较，返回 F1 均值。

    注意：NeuroKit2 检测耗时较长，建议仅在 val/test 时调用，不在每个 train step 调用。
    """
    if not _check_neurokit():
        return {"rpeak_f1": float("nan")}

    pred_np = pred.detach().cpu().numpy().squeeze(1)
    gt_np   = gt.detach().cpu().numpy().squeeze(1)

    f1_list = []
    for i in range(pred_np.shape[0]):
        pred_peaks = detect_rpeaks(pred_np[i], fs=fs)
        gt_peaks   = detect_rpeaks(gt_np[i],   fs=fs)
        f1_list.append(rpeak_f1(pred_peaks, gt_peaks))

    return {"rpeak_f1": float(np.mean(f1_list))}


# =============================================================================
# 合并接口
# =============================================================================

def compute_all_metrics(
    pred: torch.Tensor,
    gt:   torch.Tensor,
    fs:   int = 200,
    compute_f1: bool = False,
) -> dict[str, float]:
    """
    一次性计算所有指标。

    compute_f1 : 是否计算 R 峰 F1（耗时，建议仅在 val/test epoch 末尾调用）
    """
    metrics = compute_waveform_metrics(pred, gt)
    if compute_f1:
        metrics.update(compute_peak_metrics(pred, gt, fs=fs))
    return metrics
