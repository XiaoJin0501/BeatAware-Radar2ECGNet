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
        "mae":  float(np.nanmean(mae_list)),
        "rmse": float(np.nanmean(rmse_list)),
        "pcc":  float(np.nanmean(pcc_list)),
        "prd":  float(np.nanmean(prd_list)),
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


# =============================================================================
# ── 以下为按 evaluation_metrics_protocol.md 实现的 4 级评估函数 ─────────────────
# ── 训练循环继续使用上方的旧函数；以下函数仅在 test_mmecg.py 中调用 ────────────────
# =============================================================================

# ── 匹配工具（一对一最近邻） ──────────────────────────────────────────────────────

def _match_peaks(
    pred_peaks: np.ndarray,
    gt_peaks:   np.ndarray,
    tolerance:  int,
) -> list[tuple[int, int]]:
    """
    一对一最近邻匹配：在 tolerance samples 范围内将 pred 峰与 GT 峰配对。
    返回 matched (gt_idx, pred_idx) 对的列表。
    """
    if len(pred_peaks) == 0 or len(gt_peaks) == 0:
        return []
    matched_gt   = np.zeros(len(gt_peaks),   dtype=bool)
    matched_pred = np.zeros(len(pred_peaks), dtype=bool)
    pairs = []
    # 按距离从小到大贪心匹配
    dists = np.abs(gt_peaks[:, None].astype(int) - pred_peaks[None, :].astype(int))
    order = np.argsort(dists.ravel())
    for flat_idx in order:
        gi, pi = divmod(int(flat_idx), len(pred_peaks))
        if matched_gt[gi] or matched_pred[pi]:
            continue
        if dists[gi, pi] <= tolerance:
            pairs.append((gi, pi))
            matched_gt[gi]   = True
            matched_pred[pi] = True
    return pairs


def _detect_peaks_on_pred(pred_1d: np.ndarray, fs: int) -> np.ndarray:
    """在预测 ECG 上检测 R 峰，返回 int64 索引数组。"""
    return detect_rpeaks(pred_1d, fs)


def _delineate_peaks_on_pred(
    pred_1d: np.ndarray,
    pred_r:  np.ndarray,
    fs:      int,
) -> dict[str, np.ndarray]:
    """
    在预测 ECG 上做 DWT delineation，返回 Q/S/T 峰位置数组。
    -1 代表未检出。
    """
    empty = np.array([], dtype=np.int64)
    if len(pred_r) < 2:
        return {"q": empty, "s": empty, "t": empty}
    if not _check_neurokit():
        return {"q": empty, "s": empty, "t": empty}
    import neurokit2 as nk
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, waves = nk.ecg_delineate(pred_1d, pred_r.astype(int), sampling_rate=fs, method="dwt")

        def _parse(key):
            vals = waves.get(key, [np.nan] * len(pred_r))
            return np.array(
                [int(v) if (v is not None and not (isinstance(v, float) and np.isnan(v))) else -1
                 for v in vals],
                dtype=np.int64,
            )

        return {
            "q": _parse("ECG_Q_Peaks"),
            "s": _parse("ECG_S_Peaks"),
            "t": _parse("ECG_T_Peaks"),
        }
    except Exception:
        return {"q": empty, "s": empty, "t": empty}


# ── Level 1 ──────────────────────────────────────────────────────────────────

def compute_waveform_metrics_protocol(
    pred,   # (B, 1, L) tensor or (B, L) or (L,) ndarray
    gt,     # same shape as pred
) -> dict[str, float]:
    """
    Level 1 波形指标（协议命名版）：
      pcc_raw, rmse_norm, mae_norm, r2
    rmse_mV / mae_mV 暂置 NaN（无 mV 标定数据）。
    """
    if isinstance(pred, torch.Tensor):
        pred_np = pred.detach().cpu().numpy()
    else:
        pred_np = np.asarray(pred, dtype=np.float32)
    if isinstance(gt, torch.Tensor):
        gt_np = gt.detach().cpu().numpy()
    else:
        gt_np = np.asarray(gt, dtype=np.float32)
    if pred_np.ndim == 3:
        pred_np = pred_np.squeeze(1)
    if gt_np.ndim == 3:
        gt_np = gt_np.squeeze(1)
    if pred_np.ndim == 1:
        pred_np = pred_np[np.newaxis, :]
    if gt_np.ndim == 1:
        gt_np = gt_np[np.newaxis, :]

    pcc_list, rmse_list, mae_list, r2_list = [], [], [], []
    for i in range(len(pred_np)):
        p, g = pred_np[i], gt_np[i]
        # PCC
        pc = p - p.mean(); gc = g - g.mean()
        denom = np.sqrt((pc**2).sum() * (gc**2).sum()) + 1e-8
        pcc_list.append(float(np.dot(pc, gc) / denom))
        # RMSE / MAE
        diff = p - g
        rmse_list.append(float(np.sqrt(np.mean(diff**2))))
        mae_list.append(float(np.mean(np.abs(diff))))
        # R²
        ss_res = np.sum(diff**2)
        ss_tot = np.sum(gc**2) + 1e-8
        r2_list.append(float(1.0 - ss_res / ss_tot))

    return {
        "pcc_raw":   float(np.nanmean(pcc_list)),
        "rmse_norm": float(np.nanmean(rmse_list)),
        "mae_norm":  float(np.nanmean(mae_list)),
        "r2":        float(np.nanmean(r2_list)),
        "rmse_mV":   float("nan"),
        "mae_mV":    float("nan"),
    }


# ── Level 2 ──────────────────────────────────────────────────────────────────

def compute_qrst_peak_timing_errors(
    pred_1d:  np.ndarray,      # (L,) float，单条预测 ECG
    gt_r_arr: np.ndarray,      # int32，来自 H5
    gt_q_arr: np.ndarray,
    gt_s_arr: np.ndarray,
    gt_t_arr: np.ndarray,
    fs: int = 200,
    tolerance_samples: int = 10,   # ±50ms @ 200Hz
) -> dict:
    """
    Level 2：Q/R/S/T 峰位置误差（ms）。
    对预测 ECG 检测/delineate 峰，与 H5 存储的 GT 索引配对。

    返回:
      "segment_summary": {r_peak_error_ms_mean, q_peak_error_ms_mean,
                          s_peak_error_ms_mean, t_peak_error_ms_mean,
                          qrst_peak_error_ms_mean, n_matched,
                          num_r_gt, num_r_pred, num_matched_beats}
      "beat_rows": list of dicts（每个匹配 beat 的逐峰误差）
      per-peak lists: r_errors, q_errors, s_errors, t_errors（ms）
    """
    ms = 1000.0 / fs

    # ── 预测 ECG 峰检测 ────────────────────────────────────────────────────────
    pred_r = _detect_peaks_on_pred(pred_1d, fs)
    pred_delin = _delineate_peaks_on_pred(pred_1d, pred_r, fs)
    pred_q = pred_delin["q"]
    pred_s = pred_delin["s"]
    pred_t = pred_delin["t"]

    # GT 中过滤掉 -1（未检出）
    def _valid(arr): return arr[arr >= 0]
    gt_r_v = _valid(gt_r_arr)
    gt_q_v = _valid(gt_q_arr)
    gt_s_v = _valid(gt_s_arr)
    gt_t_v = _valid(gt_t_arr)
    pred_q_v = _valid(pred_q)
    pred_s_v = _valid(pred_s)
    pred_t_v = _valid(pred_t[pred_t >= 0] if len(pred_t) else np.array([], dtype=np.int64))

    # ── R 峰匹配 ───────────────────────────────────────────────────────────────
    r_pairs   = _match_peaks(pred_r, gt_r_v, tolerance_samples)
    r_errors  = [abs(int(gt_r_v[gi]) - int(pred_r[pi])) * ms for gi, pi in r_pairs]

    # ── Q/S/T 峰匹配 ──────────────────────────────────────────────────────────
    q_pairs  = _match_peaks(pred_q_v, gt_q_v, tolerance_samples)
    s_pairs  = _match_peaks(pred_s_v, gt_s_v, tolerance_samples)
    t_pairs  = _match_peaks(pred_t_v, gt_t_v, tolerance_samples)
    q_errors = [abs(int(gt_q_v[gi]) - int(pred_q_v[pi])) * ms for gi, pi in q_pairs]
    s_errors = [abs(int(gt_s_v[gi]) - int(pred_s_v[pi])) * ms for gi, pi in s_pairs]
    t_errors = [abs(int(gt_t_v[gi]) - int(pred_t_v[pi])) * ms for gi, pi in t_pairs]

    def _mean(lst): return float(np.mean(lst)) if lst else float("nan")

    all_errors = r_errors + q_errors + s_errors + t_errors

    # ── per-beat rows（R 峰匹配对驱动）──────────────────────────────────────────
    beat_rows = []
    for beat_id, (gi, pi) in enumerate(r_pairs):
        beat_rows.append({
            "beat_id":         beat_id,
            "r_peak_error_ms": abs(int(gt_r_v[gi]) - int(pred_r[pi])) * ms,
        })

    return {
        "segment_summary": {
            "r_peak_error_ms_mean":    _mean(r_errors),
            "q_peak_error_ms_mean":    _mean(q_errors),
            "s_peak_error_ms_mean":    _mean(s_errors),
            "t_peak_error_ms_mean":    _mean(t_errors),
            "qrst_peak_error_ms_mean": _mean(all_errors),
            "num_r_gt":                int(len(gt_r_v)),
            "num_r_pred":              int(len(pred_r)),
            "num_matched_beats":       int(len(r_pairs)),
        },
        "beat_rows":  beat_rows,
        "r_errors":   r_errors,
        "q_errors":   q_errors,
        "s_errors":   s_errors,
        "t_errors":   t_errors,
        "n_matched":  len(r_pairs),
        "pred_r":     pred_r,        # for compute_mdr
        "gt_r_v":     gt_r_v,        # filtered GT R peaks
    }


def compute_relative_peak_timing_errors(
    peak_errors_ms: dict[str, list[float]],
    T_seg_samples:  int,
    fs:             int = 200,
) -> dict[str, float]:
    """
    Level 2：Q/R/S/T 峰位置相对误差（%，归一化到 segment 长度）。
    peak_errors_ms: {"r": [...], "q": [...], "s": [...], "t": [...]}
    """
    T_ms = T_seg_samples / fs * 1000.0

    def _rel(lst):
        if not lst:
            return float("nan")
        return float(np.mean(lst)) / T_ms * 100.0

    r_rel = _rel(peak_errors_ms.get("r", []))
    q_rel = _rel(peak_errors_ms.get("q", []))
    s_rel = _rel(peak_errors_ms.get("s", []))
    t_rel = _rel(peak_errors_ms.get("t", []))
    all_v = [v for v in [r_rel, q_rel, s_rel, t_rel] if not np.isnan(v)]

    return {
        "r_peak_error_rel_percent":            r_rel,
        "q_peak_error_rel_percent":            q_rel,
        "s_peak_error_rel_percent":            s_rel,
        "t_peak_error_rel_percent":            t_rel,
        "qrst_peak_error_rel_percent_mean":    float(np.mean(all_v)) if all_v else float("nan"),
    }


def compute_rr_interval_error(
    pred_1d:  np.ndarray,   # (L,) float
    gt_r_arr: np.ndarray,   # int32，来自 H5
    fs: int = 200,
    tolerance_samples: int = 10,
) -> dict:
    """
    Level 2：逐 beat RR 间期误差（ms）。
    仅对配对成功的连续 R 峰计算。

    返回:
      rr_errors: list[float] ms
      rr_interval_error_ms_mean / ppi_error_ms_mean（radarODE-MTL 别名）
    """
    ms = 1000.0 / fs
    pred_r = _detect_peaks_on_pred(pred_1d, fs)
    gt_r_v = gt_r_arr[gt_r_arr >= 0]

    if len(pred_r) < 2 or len(gt_r_v) < 2:
        return {
            "rr_errors": [],
            "rr_interval_error_ms_mean": float("nan"),
            "ppi_error_ms_mean":         float("nan"),
        }

    pairs = _match_peaks(pred_r, gt_r_v, tolerance_samples)
    if len(pairs) < 2:
        return {
            "rr_errors": [],
            "rr_interval_error_ms_mean": float("nan"),
            "ppi_error_ms_mean":         float("nan"),
        }

    pairs_sorted = sorted(pairs, key=lambda x: x[0])
    rr_errors = []
    for i in range(len(pairs_sorted) - 1):
        g0, p0 = pairs_sorted[i]
        g1, p1 = pairs_sorted[i + 1]
        gt_rr   = (int(gt_r_v[g1])  - int(gt_r_v[g0]))  * ms
        pred_rr = (int(pred_r[p1])  - int(pred_r[p0]))   * ms
        rr_errors.append(abs(pred_rr - gt_rr))

    mean_err = float(np.mean(rr_errors)) if rr_errors else float("nan")
    return {
        "rr_errors":                 rr_errors,
        "rr_interval_error_ms_mean": mean_err,
        "ppi_error_ms_mean":         mean_err,
    }


def compute_t_wave_timing_error(
    pred_1d:  np.ndarray,   # (L,) float
    gt_t_arr: np.ndarray,   # int32（-1=missing）
    fs: int = 200,
    tolerance_samples: int = 30,   # ±150ms
) -> dict:
    """
    Level 2：T 波峰定位误差（ms）。对应 AirECG 报告的 T-wave timing error。
    """
    ms = 1000.0 / fs
    pred_r = _detect_peaks_on_pred(pred_1d, fs)
    pred_delin = _delineate_peaks_on_pred(pred_1d, pred_r, fs)
    pred_t = pred_delin["t"]
    pred_t_v = pred_t[pred_t >= 0] if len(pred_t) else np.array([], dtype=np.int64)
    gt_t_v   = gt_t_arr[gt_t_arr >= 0]

    pairs  = _match_peaks(pred_t_v, gt_t_v, tolerance_samples)
    errors = [abs(int(gt_t_v[gi]) - int(pred_t_v[pi])) * ms for gi, pi in pairs]
    mean_e = float(np.mean(errors)) if errors else float("nan")

    return {
        "t_errors":                  errors,
        "t_wave_timing_error_ms_mean": mean_e,
    }


def compute_qualified_monitoring_rate(segment_flags: list[bool]) -> dict[str, float]:
    """
    Level 2：合格监测率（QMR）。
    segment_flags: 每个 segment 是否合格（GT≥2 R峰 AND pred≥2 R峰 AND ≥1 matched RR）。
    """
    if not segment_flags:
        return {"qualified_monitoring_rate": float("nan")}
    qmr = float(np.mean(segment_flags)) * 100.0
    return {"qualified_monitoring_rate": qmr}


def compute_mdr(
    pred_r_peaks:      np.ndarray,
    gt_r_peaks:        np.ndarray,
    tolerance_samples: int = 10,
) -> dict[str, float]:
    """
    Level 2：R 峰漏检率（event-level MDR = 1 - Recall）。
    """
    if len(gt_r_peaks) == 0:
        return {"rpeak_mdr_event": float("nan")}
    pairs = _match_peaks(pred_r_peaks, gt_r_peaks, tolerance_samples)
    tp = len(pairs)
    fn = len(gt_r_peaks) - tp
    mdr = fn / (tp + fn) * 100.0
    return {"rpeak_mdr_event": float(mdr)}


# ── Level 3 ──────────────────────────────────────────────────────────────────

def _fiducial_f1(
    pred_pts: np.ndarray,
    gt_pts:   np.ndarray,
    tol:      int,
) -> dict[str, float]:
    """单类 fiducial 在给定 tolerance 下的 precision / recall / F1。"""
    if len(gt_pts) == 0 and len(pred_pts) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if len(gt_pts) == 0:
        return {"precision": 0.0, "recall": float("nan"), "f1": 0.0}
    if len(pred_pts) == 0:
        return {"precision": float("nan"), "recall": 0.0, "f1": 0.0}

    pairs = _match_peaks(pred_pts, gt_pts, tol)
    tp = len(pairs)
    fp = len(pred_pts) - tp
    fn = len(gt_pts)   - tp
    prec   = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1     = 2 * prec * recall / (prec + recall + 1e-8)
    return {"precision": float(prec), "recall": float(recall), "f1": float(f1)}


def compute_fiducial_detection_f1(
    pred_1d:      np.ndarray,   # (L,) float，预测 ECG
    gt_1d:        np.ndarray,   # (L,) float，GT ECG（用于 delineation）
    gt_r_peaks:   np.ndarray,   # int32（来自 H5）
    fs:           int = 200,
    tolerances_ms: tuple[int, ...] = (150, 100, 50),
) -> dict[str, float]:
    """
    Level 3：Pon/Qon/Rpeak/Soff/Toff 在多个 tolerance 下的 F1。
    对 GT ECG 做 DWT delineation 获取 onset/offset；
    对预测 ECG 同样 delineation 比较。

    命名严格按协议 Section 9：
      {fiducial}_{metric}_{tol}ms，如 pon_f1_150ms, average_f1_100ms
    """
    if not _check_neurokit():
        result = {}
        for tol in tolerances_ms:
            for fid in ["pon", "qon", "rpeak", "soff", "toff"]:
                for met in ["precision", "recall", "f1"]:
                    result[f"{fid}_{met}_{tol}ms"] = float("nan")
            result[f"average_f1_{tol}ms"] = float("nan")
        return result

    import neurokit2 as nk
    import warnings

    def _delineate_full(ecg_1d: np.ndarray, r_peaks: np.ndarray) -> dict[str, np.ndarray]:
        """返回所有 fiducial 点（过滤 NaN/-1）。"""
        empty = np.array([], dtype=np.int64)
        if len(r_peaks) < 2:
            return {k: empty for k in ["pon", "qon", "rpeak", "soff", "toff"]}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, waves = nk.ecg_delineate(
                    ecg_1d, r_peaks.astype(int), sampling_rate=fs, method="dwt"
                )

            def _get(key):
                vals = waves.get(key, [np.nan] * len(r_peaks))
                return np.array(
                    [int(v) for v in vals
                     if v is not None and not (isinstance(v, float) and np.isnan(v))],
                    dtype=np.int64,
                )

            return {
                "pon":    _get("ECG_P_Onsets"),
                "qon":    _get("ECG_R_Onsets"),    # QRS onset ≈ ECG_R_Onsets
                "rpeak":  r_peaks.astype(np.int64),
                "soff":   _get("ECG_R_Offsets"),   # QRS offset / J-point
                "toff":   _get("ECG_T_Offsets"),
            }
        except Exception:
            return {k: empty for k in ["pon", "qon", "rpeak", "soff", "toff"]}

    gt_r_v   = gt_r_peaks[gt_r_peaks >= 0].astype(np.int64)
    pred_r   = _detect_peaks_on_pred(pred_1d, fs).astype(np.int64)

    gt_fid   = _delineate_full(gt_1d,   gt_r_v)
    pred_fid = _delineate_full(pred_1d, pred_r)

    result: dict[str, float] = {}
    for tol_ms in tolerances_ms:
        tol_samp = int(tol_ms * fs / 1000)
        f1_vals  = []
        for fid in ["pon", "qon", "rpeak", "soff", "toff"]:
            m = _fiducial_f1(pred_fid[fid], gt_fid[fid], tol_samp)
            result[f"{fid}_precision_{tol_ms}ms"] = m["precision"]
            result[f"{fid}_recall_{tol_ms}ms"]    = m["recall"]
            result[f"{fid}_f1_{tol_ms}ms"]        = m["f1"]
            if not np.isnan(m["f1"]):
                f1_vals.append(m["f1"])
        result[f"average_f1_{tol_ms}ms"] = float(np.mean(f1_vals)) if f1_vals else float("nan")

    return result


# ── Level 4 ──────────────────────────────────────────────────────────────────

def compute_clinical_interval_errors(
    pred_1d:    np.ndarray,   # (L,) float，预测 ECG
    gt_1d:      np.ndarray,   # (L,) float，GT ECG
    gt_r_peaks: np.ndarray,   # int32（来自 H5）
    fs:         int = 200,
) -> dict[str, float]:
    """
    Level 4：临床间期误差（ms）。
      PR  = Qon  - Pon          （房室传导）
      QRS = Soff - Qon          （心室除极）
      QT  = Toff - Qon          （心室复极）
      QTc = QT(ms) / sqrt(RR(s)) [Bazett 校正]

    对 GT ECG 和预测 ECG 各做一次全 delineation，
    逐 beat 配对计算误差均值。
    """
    if not _check_neurokit():
        return {
            "pr_interval_error_ms":   float("nan"),
            "qrs_duration_error_ms":  float("nan"),
            "qt_interval_error_ms":   float("nan"),
            "qtc_interval_error_ms":  float("nan"),
        }

    import neurokit2 as nk
    import warnings

    gt_r_v  = gt_r_peaks[gt_r_peaks >= 0].astype(np.int64)
    pred_r  = _detect_peaks_on_pred(pred_1d, fs).astype(np.int64)

    def _full_delineate(ecg_1d, r_peaks):
        if len(r_peaks) < 2:
            return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, waves = nk.ecg_delineate(
                    ecg_1d, r_peaks.astype(int), sampling_rate=fs, method="dwt"
                )

            def _arr(key):
                vals = waves.get(key, [np.nan] * len(r_peaks))
                return np.array(
                    [float(v) if (v is not None and not (isinstance(v, float) and np.isnan(v)))
                     else np.nan for v in vals]
                )

            return {
                "pon":  _arr("ECG_P_Onsets"),
                "qon":  _arr("ECG_R_Onsets"),
                "soff": _arr("ECG_R_Offsets"),
                "toff": _arr("ECG_T_Offsets"),
                "r":    r_peaks.astype(float),
            }
        except Exception:
            return None

    gt_w   = _full_delineate(gt_1d,   gt_r_v)
    pred_w = _full_delineate(pred_1d, pred_r)

    if gt_w is None or pred_w is None:
        return {
            "pr_interval_error_ms":  float("nan"),
            "qrs_duration_error_ms": float("nan"),
            "qt_interval_error_ms":  float("nan"),
            "qtc_interval_error_ms": float("nan"),
        }

    def _interval(w, a_key, b_key):
        """(b - a) / fs * 1000 ms，过滤 NaN，只保留 b > a 的。"""
        a, b = w[a_key], w[b_key]
        n = min(len(a), len(b))
        vals = []
        for i in range(n):
            if not (np.isnan(a[i]) or np.isnan(b[i])) and b[i] > a[i]:
                vals.append((b[i] - a[i]) / fs * 1000.0)
        return np.array(vals)

    gt_pr   = _interval(gt_w,   "pon",  "qon")
    gt_qrs  = _interval(gt_w,   "qon",  "soff")
    gt_qt   = _interval(gt_w,   "qon",  "toff")
    pred_pr  = _interval(pred_w, "pon",  "qon")
    pred_qrs = _interval(pred_w, "qon",  "soff")
    pred_qt  = _interval(pred_w, "qon",  "toff")

    def _mae(a, b):
        n = min(len(a), len(b))
        if n == 0:
            return float("nan")
        return float(np.mean(np.abs(a[:n] - b[:n])))

    pr_err  = _mae(gt_pr,  pred_pr)
    qrs_err = _mae(gt_qrs, pred_qrs)
    qt_err  = _mae(gt_qt,  pred_qt)

    # QTc = QT(s) / sqrt(RR(s)) [Bazett]
    def _qtc_arr(qt_ms, r_peaks):
        if len(r_peaks) < 2 or len(qt_ms) == 0:
            return np.array([])
        rr_s = np.diff(r_peaks.astype(float)) / fs
        n = min(len(qt_ms), len(rr_s))
        qtc = qt_ms[:n] / 1000.0 / np.sqrt(rr_s[:n] + 1e-8) * 1000.0
        return qtc

    gt_qtc   = _qtc_arr(gt_qt,   gt_r_v)
    pred_qtc = _qtc_arr(pred_qt, pred_r)
    qtc_err  = _mae(gt_qtc, pred_qtc)

    return {
        "pr_interval_error_ms":  pr_err,
        "qrs_duration_error_ms": qrs_err,
        "qt_interval_error_ms":  qt_err,
        "qtc_interval_error_ms": qtc_err,
    }


# ── 汇总函数 ──────────────────────────────────────────────────────────────────

def summarize_subject_metrics(segment_rows: list[dict]) -> "pd.DataFrame":
    """
    按 (subject_id, scene) 分组，计算每组的 mean / median / std / IQR。
    返回 pd.DataFrame（每行 = 一个 subject × scene 组合）。
    """
    import pandas as pd

    df = pd.DataFrame(segment_rows)
    numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()
    for drop in ["segment_id", "fs", "segment_length_sec"]:
        if drop in numeric_cols:
            numeric_cols.remove(drop)

    rows = []
    for (subj, scene), grp in df.groupby(["subject_id", "scene"]):
        row: dict = {"subject_id": subj, "scene": scene, "n_segments": len(grp)}
        for col in numeric_cols:
            vals = grp[col].dropna().values
            if len(vals) == 0:
                row[f"{col}_mean"]   = float("nan")
                row[f"{col}_median"] = float("nan")
                row[f"{col}_std"]    = float("nan")
                row[f"{col}_iqr"]    = float("nan")
            else:
                row[f"{col}_mean"]   = float(np.mean(vals))
                row[f"{col}_median"] = float(np.median(vals))
                row[f"{col}_std"]    = float(np.std(vals))
                q75, q25 = np.percentile(vals, [75, 25])
                row[f"{col}_iqr"]    = float(q75 - q25)
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_global_metrics(segment_rows: list[dict]) -> dict:
    """
    全局 mean / median / std / IQR。
    返回 dict（可直接序列化为 JSON）。
    """
    import pandas as pd

    df = pd.DataFrame(segment_rows)
    numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()
    for drop in ["segment_id", "fs", "segment_length_sec", "subject_id"]:
        if drop in numeric_cols:
            numeric_cols.remove(drop)

    summary: dict = {
        "num_segments":       int(len(df)),
        "num_subjects":       int(df["subject_id"].nunique()) if "subject_id" in df else 0,
        "num_matched_beats":  int(df["num_matched_beats"].sum()) if "num_matched_beats" in df else 0,
    }
    if "qualified_flag" in df:
        summary["num_valid_segments"] = int(df["qualified_flag"].sum())
        summary["qualified_monitoring_rate"] = float(df["qualified_flag"].mean() * 100)
    if "segment_failed_pcc60" in df:
        summary["segment_failure_rate_pcc60"] = float(df["segment_failed_pcc60"].mean() * 100)

    for col in numeric_cols:
        vals = df[col].dropna().values
        if len(vals) == 0:
            summary[f"{col}_mean"]   = float("nan")
            summary[f"{col}_median"] = float("nan")
            summary[f"{col}_std"]    = float("nan")
            summary[f"{col}_iqr"]    = float("nan")
        else:
            summary[f"{col}_mean"]   = float(np.mean(vals))
            summary[f"{col}_median"] = float(np.median(vals))
            summary[f"{col}_std"]    = float(np.std(vals))
            q75, q25 = np.percentile(vals, [75, 25])
            summary[f"{col}_iqr"]    = float(q75 - q25)

    return summary
