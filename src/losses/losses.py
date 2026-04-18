"""
losses.py — BeatAware-Radar2ECGNet V2 损失函数

V2 改动（Phase A + Phase B）：
  1. STFT Loss: plain L1 → 谱收敛 (SC) + 对数幅度 (log-mag) 双项
  2. L_der：一阶 + 二阶导数 L1，强制 QRS 边界锐化
  3. L_peak：3 路 BCE（QRS / P / T 波），P/T 波 GT 无效时对应头置零
  4. L_interval：soft-argmax 提取 PR 间期（@200Hz），施加生理约束
     PR 正常范围 120-200ms；惩罚超出该范围的重构结果
  5. 自适应损失权重（同方差不确定性，Kendall & Gal 2018）：
       L_total = Σ_i [0.5 * exp(-log_var_i) * L_i + 0.5 * log_var_i]
     log_vars = nn.Parameter(zeros(4))，对应 [L_recon, L_peak, L_der, L_interval]

训练策略：前 warmup_epochs 个 epoch 只优化 L_recon（warm-up），之后解锁全部。

历史（V1）：
  L_total = L_time + alpha * L_freq + beta * L_peak（手动超参）

STFT 参数（@200Hz）：
    FFT_SIZES   = [128, 256, 512]
    HOP_SIZES   = [64,  128, 256]
    WIN_LENGTHS = [16,   32,  64]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 多分辨率 STFT：谱收敛 + 对数幅度
# =============================================================================

class MultiResolutionSTFTLoss(nn.Module):
    """
    多分辨率 STFT Loss（V2）。

    每组参数计算：
        SC   = ||gt_mag - pred_mag||_F / (||gt_mag||_F + ε)      谱收敛损失
        logM = mean |log(pred_mag+ε) - log(gt_mag+ε)|            对数幅度损失
        stft_i = (SC + logM) / 2

    最终取各分辨率均值。

    Parameters
    ----------
    fft_sizes   : list[int]  FFT 窗口大小列表
    hop_sizes   : list[int]  帧移列表
    win_lengths : list[int]  窗函数长度列表
    """

    def __init__(
        self,
        fft_sizes:   list = [128, 256, 512],
        hop_sizes:   list = [64,  128, 256],
        win_lengths: list = [16,   32,  64],
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.fft_sizes   = fft_sizes
        self.hop_sizes   = hop_sizes
        self.win_lengths = win_lengths

    @staticmethod
    def _stft_magnitude(
        x: torch.Tensor, n_fft: int, hop_length: int, win_length: int
    ) -> torch.Tensor:
        """x: (B, L) → (B, F, T)，F = n_fft//2 + 1"""
        window = torch.hann_window(win_length, device=x.device)
        S = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=False,
            return_complex=True,
        )
        return S.abs()   # (B, F, T)，非负，无需 abs()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        pred, gt : (B, 1, L)
        Returns  : scalar STFT Loss
        """
        pred_flat = pred.squeeze(1)   # (B, L)
        gt_flat   = gt.squeeze(1)

        parts = []
        for n_fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            pred_mag = self._stft_magnitude(pred_flat, n_fft, hop, win)
            gt_mag   = self._stft_magnitude(gt_flat,   n_fft, hop, win)

            # 谱收敛损失 (Spectral Convergence)
            sc = (
                torch.norm(gt_mag - pred_mag, p="fro", dim=(-2, -1))
                / (torch.norm(gt_mag, p="fro", dim=(-2, -1)) + 1e-8)
            ).mean()

            # 对数幅度损失 (Log-magnitude)
            logm = F.l1_loss(
                torch.log(pred_mag + 1e-7),
                torch.log(gt_mag   + 1e-7),
            )

            parts.append((sc + logm) * 0.5)

        return sum(parts) / len(self.fft_sizes)


# =============================================================================
# 导数损失 L_der
# =============================================================================

def derivative_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    一阶 + 二阶导数 L1 Loss。

    强制 QRS 波群高频转折点对齐，防止 Q/S 波在 MAE 平滑特性下消失。

    pred, gt : (B, 1, L)
    Returns  : scalar
    """
    d1_pred = torch.diff(pred, n=1, dim=-1)
    d1_gt   = torch.diff(gt,   n=1, dim=-1)
    d2_pred = torch.diff(pred, n=2, dim=-1)
    d2_gt   = torch.diff(gt,   n=2, dim=-1)
    return F.l1_loss(d1_pred, d1_gt) + F.l1_loss(d2_pred, d2_gt)


# =============================================================================
# 辅助：soft-argmax（可微分峰值位置提取）
# =============================================================================

def soft_argmax(mask: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """
    可微分的 argmax，通过 softmax 近似峰值位置（单位：采样点）。

    mask : (B, 1, L)
    Returns: (B,) — 峰值的期望位置（连续值，@200Hz 单位为采样点）
    """
    logits = mask.squeeze(1) / tau                                 # (B, L)
    w = F.softmax(logits, dim=-1)                                  # (B, L)
    pos = torch.arange(mask.shape[-1], device=mask.device, dtype=torch.float32)
    return (w * pos).sum(-1)                                       # (B,)


# =============================================================================
# TotalLoss（V2：4 任务自适应损失权重）
# =============================================================================

class TotalLoss(nn.Module):
    """
    BeatAware-Radar2ECGNet V2 总损失函数（Phase A + Phase B）。

    4 个任务自适应加权（同方差不确定性，Kendall & Gal 2018）：
        [0] L_recon    = L_time + alpha_stft * L_freq
        [1] L_peak     = BCE(QRS) + masked BCE(P) + masked BCE(T)
        [2] L_der      = L1(diff1) + L1(diff2)
        [3] L_interval = PR 间期软约束（soft-argmax）

    加权公式：L_total = Σ_i [0.5 * exp(-log_var_i) * L_i + 0.5 * log_var_i]

    log_vars = nn.Parameter(zeros(4))，须纳入 optimizer。
    初始 log_vars=0 → σ=1 → 各任务等权。

    训练策略：前 warmup_epochs epoch 只优化 L_recon，之后全部解锁。

    Parameters
    ----------
    alpha_stft    : float  STFT 在 L_recon 内的固定权重（默认 0.1）
    warmup_epochs : int    预热 epoch 数（默认 5）
    soft_argmax_tau : float  soft-argmax 温度参数（默认 0.1）
    """

    def __init__(
        self,
        alpha_stft:      float = 0.1,
        warmup_epochs:   int   = 5,
        soft_argmax_tau: float = 0.1,
        fft_sizes:       list  = [128, 256, 512],
        hop_sizes:       list  = [64,  128, 256],
        win_lengths:     list  = [16,   32,  64],
    ):
        super().__init__()
        self.alpha_stft      = alpha_stft
        self.warmup_epochs   = warmup_epochs
        self.soft_argmax_tau = soft_argmax_tau
        self.stft_loss = MultiResolutionSTFTLoss(fft_sizes, hop_sizes, win_lengths)

        # 4 个可学习 log σ²，对应 [recon, peak, der, interval]
        self.log_vars = nn.Parameter(torch.zeros(4))

    @staticmethod
    def _safe_bce(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """NaN 安全的 BCE，对预测值做 clamp。"""
        pred_safe = pred.nan_to_num(
            nan=0.5, posinf=1.0 - 1e-6, neginf=1e-6
        ).clamp(1e-6, 1.0 - 1e-6)
        return F.binary_cross_entropy(pred_safe, gt)

    def _peak_loss(
        self,
        peak_preds: tuple | None,
        peak_gts:   dict  | None,
    ) -> torch.Tensor:
        """
        3 路 BCE 峰值损失。

        peak_preds : (qrs_pred, p_pred, t_pred) 各 (B,1,L) | None
        peak_gts   : {
            'qrs': Tensor (B,1,L),
            'p':   Tensor (B,1,L) | None,
            't':   Tensor (B,1,L) | None,
            'p_valid': Tensor (B,) bool | None,
            't_valid': Tensor (B,) bool | None,
        } | None
        """
        if peak_preds is None or peak_gts is None:
            return torch.tensor(0.0)

        qrs_pred, p_pred, t_pred = peak_preds
        qrs_gt = peak_gts["qrs"]

        # QRS BCE（必选）
        l_qrs = self._safe_bce(qrs_pred, qrs_gt)

        # P 波 BCE（仅对有效样本，P/T 波 GT 文件存在且 delineation 成功）
        p_gt      = peak_gts.get("p")
        p_valid   = peak_gts.get("p_valid")
        if p_gt is not None and p_valid is not None and p_valid.any():
            vm = p_valid.view(-1, 1, 1)   # (B,1,1) broadcast mask
            l_p = self._safe_bce(p_pred[vm.expand_as(p_pred)].view(-1, 1, p_pred.shape[-1]),
                                 p_gt[vm.expand_as(p_gt)].view(-1, 1, p_gt.shape[-1]))
        else:
            l_p = qrs_pred.new_zeros(())

        # T 波 BCE（同上）
        t_gt      = peak_gts.get("t")
        t_valid   = peak_gts.get("t_valid")
        if t_gt is not None and t_valid is not None and t_valid.any():
            vm = t_valid.view(-1, 1, 1)
            l_t = self._safe_bce(t_pred[vm.expand_as(t_pred)].view(-1, 1, t_pred.shape[-1]),
                                 t_gt[vm.expand_as(t_gt)].view(-1, 1, t_gt.shape[-1]))
        else:
            l_t = qrs_pred.new_zeros(())

        return l_qrs + l_p + l_t

    def _interval_loss(
        self,
        peak_preds: tuple | None,
        peak_gts:   dict  | None,
    ) -> torch.Tensor:
        """
        PR 间期约束损失（soft-argmax，@200Hz，1 sample = 5ms）。

        正常 PR 间期：120-200ms（24-40 samples@200Hz）。
        仅对有有效 P 波 GT 的样本计算（否则 p_pred 梯度不稳定）。

        惩罚超出生理阈值：
            pr_penalty = ReLU(120 - pr_ms) + ReLU(pr_ms - 200)
        """
        if peak_preds is None:
            return torch.tensor(0.0)

        p_valid = peak_gts.get("p_valid") if peak_gts else None
        if p_valid is None or not p_valid.any():
            return torch.tensor(0.0)

        qrs_pred, p_pred, _ = peak_preds

        # 只对 p_valid=True 的样本计算
        qrs_pos = soft_argmax(qrs_pred[p_valid], tau=self.soft_argmax_tau)  # (B_valid,)
        p_pos   = soft_argmax(p_pred[p_valid],   tau=self.soft_argmax_tau)  # (B_valid,)

        pr_samples = qrs_pos - p_pos           # PR 间期（采样点，@200Hz）
        pr_ms      = pr_samples * 5.0          # 转换为毫秒

        # PR 在 [120, 200] ms 内无惩罚
        pr_penalty = F.relu(120.0 - pr_ms) + F.relu(pr_ms - 200.0)
        return pr_penalty.mean()

    def forward(
        self,
        ecg_pred:   torch.Tensor,
        ecg_gt:     torch.Tensor,
        peak_preds: tuple | None,
        peak_gts:   dict  | None,
        epoch:      int = 999,
    ) -> dict:
        """
        Parameters
        ----------
        ecg_pred   : (B, 1, L) — 重建 ECG，值域 [0, 1]
        ecg_gt     : (B, 1, L) — GT ECG
        peak_preds : (qrs_pred, p_pred, t_pred) 各 (B,1,L) | None
        peak_gts   : dict {
                         'qrs': (B,1,L),
                         'p':   (B,1,L) | None,
                         't':   (B,1,L) | None,
                         'p_valid': (B,) bool | None,
                         't_valid': (B,) bool | None,
                     } | None
        epoch      : int  当前 epoch（warm-up 判断）

        Returns
        -------
        dict:
            'total'     : L_total（用于 backward）
            'recon'     : L_recon
            'time'      : L_time
            'freq'      : L_freq
            'peak'      : L_peak（QRS + P + T）
            'der'       : L_der
            'interval'  : L_interval
            'log_vars'  : Tensor(4)，TensorBoard 权重轨迹
        """
        # ── 各子损失 ────────────────────────────────────────────────────────
        L_time  = F.l1_loss(ecg_pred, ecg_gt)
        L_freq  = self.stft_loss(ecg_pred, ecg_gt)
        L_recon = L_time + self.alpha_stft * L_freq

        L_der      = derivative_loss(ecg_pred, ecg_gt)
        L_peak     = self._peak_loss(peak_preds, peak_gts)
        L_interval = self._interval_loss(peak_preds, peak_gts)

        # ── 自适应加权 ─────────────────────────────────────────────────────
        losses = [L_recon, L_peak, L_der, L_interval]

        if epoch <= self.warmup_epochs:
            L_total = L_recon
        else:
            precision = torch.exp(-self.log_vars)   # (4,)
            L_total = sum(
                0.5 * p * l + 0.5 * lv
                for p, l, lv in zip(precision, losses, self.log_vars)
            )

        return {
            "total":    L_total,
            "recon":    L_recon.detach(),
            "time":     L_time.detach(),
            "freq":     L_freq.detach(),
            "peak":     L_peak.detach() if isinstance(L_peak, torch.Tensor) else torch.tensor(0.0),
            "der":      L_der.detach(),
            "interval": L_interval.detach() if isinstance(L_interval, torch.Tensor) else torch.tensor(0.0),
            "log_vars": self.log_vars.detach(),
        }
