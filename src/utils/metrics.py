"""
metrics.py — ECG 重建质量评估指标

波形指标（逐样本，再取 batch 均值）：
  MAE   : Mean Absolute Error
  RMSE  : Root Mean Squared Error
  PCC   : Pearson Correlation Coefficient
  PRD   : Percent Root-mean-square Difference（ECG 领域标准）

峰值指标（在重建 ECG 上重新检测 R 峰，与 GT R 峰比较）：
  F1    : R 峰检测 F1 Score（容忍窗口 ±25ms = ±5 samples @200Hz）

高级指标（耗时，仅在 test.py 最终评估时调用）：
  DTW            : Dynamic Time Warping 距离（Sakoe-Chiba 窗口，归一化）
  RR Interval MAE: 平均 RR 间期绝对误差（ms），反映心率精度
  QRS Width MAE  : QRS 波群时限绝对误差（ms），反映心室除极精度
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
# 合并接口（快速，训练 val 可用）
# =============================================================================

def compute_all_metrics(
    pred: torch.Tensor,
    gt:   torch.Tensor,
    fs:   int = 200,
    compute_f1: bool = False,
) -> dict[str, float]:
    """
    计算基础指标（MAE/RMSE/PCC/PRD + 可选 F1）。

    compute_f1 : 是否计算 R 峰 F1（耗时，建议仅在 val/test epoch 末尾调用）
    """
    metrics = compute_waveform_metrics(pred, gt)
    if compute_f1:
        metrics.update(compute_peak_metrics(pred, gt, fs=fs))
    return metrics


# =============================================================================
# 高级指标（慢，仅 test.py 最终评估时调用）
# =============================================================================

def _dtw_distance_1d(
    x:      np.ndarray,
    y:      np.ndarray,
    window: int = 25,
) -> float:
    """
    Sakoe-Chiba 窗口约束的 DTW 距离（归一化为路径长度）。

    window : Sakoe-Chiba 带宽（samples），默认 25 = 125ms @200Hz
             限制搜索范围，将复杂度从 O(n²) 降至 O(n·window)
    """
    n = len(x)
    dtw = np.full((n + 1, n + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        j0 = max(1, i - window)
        j1 = min(n, i + window) + 1
        for j in range(j0, j1):
            cost = abs(x[i - 1] - y[j - 1])
            dtw[i, j] = cost + min(
                dtw[i - 1, j - 1],
                dtw[i - 1, j],
                dtw[i,     j - 1],
            )
    return float(dtw[n, n] / n)   # 归一化：除以路径长度


def compute_dtw_metric(
    pred:            torch.Tensor,    # (B, 1, L)
    gt:              torch.Tensor,    # (B, 1, L)
    window:          int = 25,
    max_samples:     int = 500,
) -> dict[str, float]:
    """
    计算 DTW 距离的 batch 均值。

    max_samples : 最多计算样本数（随机采样），防止大 batch 耗时过长。
                  设为 -1 表示计算全部样本（约 2 min / 2427 samples）。
    """
    pred_np = pred.detach().cpu().numpy().squeeze(1)   # (B, L)
    gt_np   = gt.detach().cpu().numpy().squeeze(1)

    n = len(pred_np)
    if max_samples > 0 and n > max_samples:
        idx = np.random.default_rng(42).choice(n, max_samples, replace=False)
        pred_np = pred_np[idx]
        gt_np   = gt_np[idx]

    dtw_vals = [
        _dtw_distance_1d(pred_np[i], gt_np[i], window=window)
        for i in range(len(pred_np))
    ]
    return {"dtw": float(np.mean(dtw_vals))}


def compute_rr_interval_mae(
    pred: torch.Tensor,   # (B, 1, L)
    gt:   torch.Tensor,   # (B, 1, L)
    fs:   int = 200,
) -> dict[str, float]:
    """
    平均 RR 间期绝对误差（ms）。

    逐样本计算 pred/GT 的平均 RR 间期（mean of np.diff(peaks) / fs * 1000），
    再取差的绝对值均值。跳过 R 峰 < 2 个的样本。
    """
    if not _check_neurokit():
        return {"rr_interval_mae": float("nan")}

    pred_np = pred.detach().cpu().numpy().squeeze(1)
    gt_np   = gt.detach().cpu().numpy().squeeze(1)

    errors = []
    for i in range(len(pred_np)):
        pred_peaks = detect_rpeaks(pred_np[i], fs)
        gt_peaks   = detect_rpeaks(gt_np[i],   fs)
        if len(pred_peaks) < 2 or len(gt_peaks) < 2:
            continue
        pred_rr = float(np.mean(np.diff(pred_peaks))) / fs * 1000   # ms
        gt_rr   = float(np.mean(np.diff(gt_peaks)))   / fs * 1000
        errors.append(abs(pred_rr - gt_rr))

    return {"rr_interval_mae": float(np.mean(errors)) if errors else float("nan")}


def _delineate_one(
    ecg_1d: np.ndarray,
    rpeaks: np.ndarray,
    fs:     int = 200,
) -> dict[str, np.ndarray]:
    """
    用 neurokit2 DWT delineation 对单条 ECG 提取所有临床波界（一次调用）。

    返回 dict（值为有效测量值的 ms 数组，可能为空）：
      qrs_widths   : R_Offsets - R_Onsets        （QRS 波群时限）
      qt_intervals : T_Offsets - R_Onsets        （QT 间期）
      pr_intervals : R_peaks   - P_Onsets        （PR 间期）

    DWT key 说明（neurokit2 0.2.x）：
      ECG_R_Onsets  = QRS 起始点（Q 波前）
      ECG_R_Offsets = QRS 终止点（J 点，S 波后）
      ECG_T_Offsets = T 波终点
      ECG_P_Onsets  = P 波起始点
    """
    empty = np.array([], dtype=float)
    result = {"qrs_widths": empty, "qt_intervals": empty, "pr_intervals": empty}

    if len(rpeaks) == 0:
        return result

    import neurokit2 as nk
    try:
        _, waves = nk.ecg_delineate(
            ecg_1d,
            {"ECG_R_Peaks": rpeaks},
            sampling_rate=fs,
            method="dwt",
        )

        def _get(key: str) -> np.ndarray:
            return np.asarray(
                waves.get(key, [np.nan] * len(rpeaks)), dtype=float
            )

        r_on  = _get("ECG_R_Onsets")    # QRS onset
        r_off = _get("ECG_R_Offsets")   # QRS offset (J-point)
        t_off = _get("ECG_T_Offsets")   # T wave end
        p_on  = _get("ECG_P_Onsets")    # P wave onset
        r     = rpeaks.astype(float)

        def _interval(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """(b - a) / fs * 1000 for valid pairs where b > a."""
            if len(a) != len(b):
                return empty
            mask = ~(np.isnan(a) | np.isnan(b)) & (b > a)
            return (b[mask] - a[mask]) / fs * 1000 if mask.any() else empty

        result["qrs_widths"]   = _interval(r_on, r_off)           # QRS ms
        result["qt_intervals"] = _interval(r_on, t_off)           # QT  ms
        result["pr_intervals"] = _interval(p_on, r) if len(p_on) == len(r) \
            else empty                                             # PR  ms

    except Exception:
        pass

    return result


def compute_interval_metrics(
    pred: torch.Tensor,   # (B, 1, L)
    gt:   torch.Tensor,   # (B, 1, L)
    fs:   int = 200,
) -> dict[str, float]:
    """
    计算 QRS / QT / PR 三个临床间期的 MAE（ms）。

    对每个样本做一次 DWT delineation（而非三次），跳过无法提取间期的样本。
    单位均为 ms：
      qrs_width_mae  : QRS 波群时限误差
      qt_interval_mae: QT 间期误差（心室复极）
      pr_interval_mae: PR 间期误差（房室传导）
    """
    if not _check_neurokit():
        return {
            "qrs_width_mae":   float("nan"),
            "qt_interval_mae": float("nan"),
            "pr_interval_mae": float("nan"),
        }

    pred_np = pred.detach().cpu().numpy().squeeze(1)
    gt_np   = gt.detach().cpu().numpy().squeeze(1)

    qrs_errs, qt_errs, pr_errs = [], [], []

    for i in range(len(pred_np)):
        pred_peaks = detect_rpeaks(pred_np[i], fs)
        gt_peaks   = detect_rpeaks(gt_np[i],   fs)

        pred_ivs = _delineate_one(pred_np[i], pred_peaks, fs)
        gt_ivs   = _delineate_one(gt_np[i],   gt_peaks,   fs)

        for errs, key in [
            (qrs_errs, "qrs_widths"),
            (qt_errs,  "qt_intervals"),
            (pr_errs,  "pr_intervals"),
        ]:
            p_arr = pred_ivs[key]
            g_arr = gt_ivs[key]
            if len(p_arr) > 0 and len(g_arr) > 0:
                errs.append(abs(float(np.mean(p_arr)) - float(np.mean(g_arr))))

    def _mean(lst: list) -> float:
        return float(np.mean(lst)) if lst else float("nan")

    return {
        "qrs_width_mae":   _mean(qrs_errs),
        "qt_interval_mae": _mean(qt_errs),
        "pr_interval_mae": _mean(pr_errs),
    }


def compute_advanced_metrics(
    pred:            torch.Tensor,
    gt:              torch.Tensor,
    fs:              int = 200,
    max_dtw_samples: int = 500,
) -> dict[str, float]:
    """
    一次性计算所有高级指标（耗时，仅在 test.py 最终评估时调用）：
      - DTW（Sakoe-Chiba 窗口 25 samples，归一化）
      - RR interval MAE（ms）
      - QRS width MAE / QT interval MAE / PR interval MAE（ms，DWT delineation）

    max_dtw_samples : DTW 最多计算样本数（-1 = 全部）。
    """
    metrics: dict[str, float] = {}
    metrics.update(compute_dtw_metric(pred, gt, max_samples=max_dtw_samples))
    metrics.update(compute_rr_interval_mae(pred, gt, fs=fs))
    metrics.update(compute_interval_metrics(pred, gt, fs=fs))
    return metrics
