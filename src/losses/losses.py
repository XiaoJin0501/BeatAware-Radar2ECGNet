"""
losses.py — BeatAware-Radar2ECGNet V2 损失函数

V2 改动（Phase A）：
  1. STFT Loss: plain L1 → 谱收敛 (SC) + 对数幅度 (log-mag) 双项
  2. 新增 L_der：一阶 + 二阶导数 L1，强制 QRS 边界锐化
  3. 自适应损失权重（同方差不确定性，Kendall & Gal 2018）：
       L_total = Σ_i [0.5 * exp(-log_var_i) * L_i + 0.5 * log_var_i]
     log_vars = nn.Parameter(zeros(3))，对应 [L_recon, L_peak, L_der]
     （Phase B 引入 L_interval 时扩展为 4 个）

Phase B 将新增：
  4. L_interval：soft-argmax 提取 PR/QT 间期，施加生理约束

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
# TotalLoss（V2：自适应损失权重）
# =============================================================================

class TotalLoss(nn.Module):
    """
    BeatAware-Radar2ECGNet V2 总损失函数。

    采用同方差不确定性（Kendall & Gal 2018）自动平衡 3 个任务：
        [0] L_recon = L_time + alpha_stft * L_freq
        [1] L_peak  = BCE(pred_mask, gt_mask)
        [2] L_der   = L1(diff1) + L1(diff2)

    加权公式（log_var = log σ²）：
        L_total = Σ_i [0.5 * exp(-log_var_i) * L_i + 0.5 * log_var_i]

    log_vars 是可学习参数，随模型一起优化（须纳入 optimizer）。
    初始 log_vars=0 对应 σ=1（各任务等权）。

    训练策略：前 warmup_epochs 个 epoch 只优化 L_recon（其余 L_i 乘以 0），
    之后解锁全部损失项，让 log_vars 自由学习。

    Parameters
    ----------
    alpha_stft    : float  L_freq 在 L_recon 内的固定权重（默认 0.1）
    warmup_epochs : int    预热 epoch 数（默认 5）
    fft_sizes, hop_sizes, win_lengths : list[int]  STFT 参数
    """

    def __init__(
        self,
        alpha_stft:    float = 0.1,
        warmup_epochs: int   = 5,
        fft_sizes:     list  = [128, 256, 512],
        hop_sizes:     list  = [64,  128, 256],
        win_lengths:   list  = [16,   32,  64],
    ):
        super().__init__()
        self.alpha_stft    = alpha_stft
        self.warmup_epochs = warmup_epochs
        self.stft_loss     = MultiResolutionSTFTLoss(fft_sizes, hop_sizes, win_lengths)

        # 3 个可学习 log σ²，对应 [recon, peak, der]
        # 初始化为 0 → σ=1 → 各任务等权
        self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(
        self,
        ecg_pred:  torch.Tensor,
        ecg_gt:    torch.Tensor,
        peak_pred: torch.Tensor | None,
        peak_gt:   torch.Tensor | None,
        epoch:     int = 999,
    ) -> dict:
        """
        Parameters
        ----------
        ecg_pred  : (B, 1, L) — 重建 ECG，值域 [0, 1]
        ecg_gt    : (B, 1, L) — GT ECG，值域 [0, 1]
        peak_pred : (B, 1, L) | None — PAM 峰值 Mask（use_pam=False 时为 None）
        peak_gt   : (B, 1, L) | None — GT 高斯软标签
        epoch     : int — 当前 epoch（用于 warm-up 判断）

        Returns
        -------
        dict:
            'total'    : L_total（用于 backward）
            'recon'    : L_recon（监控）
            'time'     : L_time（监控）
            'freq'     : L_freq（监控）
            'peak'     : L_peak（监控）
            'der'      : L_der（监控）
            'log_vars' : Tensor(3)，用于 TensorBoard 权重轨迹
        """
        # ── 各子损失 ────────────────────────────────────────────────────────
        L_time  = F.l1_loss(ecg_pred, ecg_gt)
        L_freq  = self.stft_loss(ecg_pred, ecg_gt)
        L_recon = L_time + self.alpha_stft * L_freq

        L_der = derivative_loss(ecg_pred, ecg_gt)

        if peak_pred is not None and peak_gt is not None:
            peak_pred_safe = peak_pred.nan_to_num(
                nan=0.5, posinf=1.0 - 1e-6, neginf=1e-6
            ).clamp(1e-6, 1.0 - 1e-6)
            L_peak = F.binary_cross_entropy(peak_pred_safe, peak_gt)
        else:
            L_peak = ecg_pred.new_zeros(())

        # ── 自适应加权 ─────────────────────────────────────────────────────
        losses = [L_recon, L_peak, L_der]

        if epoch <= self.warmup_epochs:
            # warm-up：只训练 L_recon，log_vars 冻结（不产生梯度影响）
            L_total = L_recon
        else:
            precision = torch.exp(-self.log_vars)   # [3]，= 1/σ²
            L_total = sum(
                0.5 * p * l + 0.5 * lv
                for p, l, lv in zip(precision, losses, self.log_vars)
            )

        return {
            "total":    L_total,
            "recon":    L_recon.detach(),
            "time":     L_time.detach(),
            "freq":     L_freq.detach(),
            "peak":     L_peak.detach(),
            "der":      L_der.detach(),
            "log_vars": self.log_vars.detach(),
        }
