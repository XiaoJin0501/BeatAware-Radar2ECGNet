"""
losses.py — BeatAwareRadar2ECGNet 损失函数

L_total = L_time + α·L_freq + β·L_peak

| 项      | 公式                                 | 默认权重 |
|---------|--------------------------------------|---------|
| L_time  | MAE(pred_ecg, gt_ecg)               | 1.0（基准，不调整）|
| L_freq  | 多分辨率 STFT Loss（在线计算）          | α：待实验确定 |
| L_peak  | BCE(pred_mask, gt_mask)             | β：待实验确定 |

STFT Loss 参数（@200Hz 目标采样率）：
    FFT_SIZES   = [128, 256, 512]   覆盖 QRS(~40ms) / P-T波(~150ms) / RR间期(~600ms)
    HOP_SIZES   = [64,  128, 256]
    WIN_LENGTHS = [16,   32,  64]

参考：Li et al., "Neural Speech Synthesis with Transformer Network" (ICASSP 2019)
     — STFT Loss 方案
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 多分辨率 STFT Loss
# =============================================================================

class MultiResolutionSTFTLoss(nn.Module):
    """
    多分辨率 STFT 幅度谱 L1 Loss。

    对 pred_ecg 和 gt_ecg 分别计算多组 STFT，取归一化 L1 之均值。

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
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths), (
            "fft_sizes, hop_sizes, win_lengths 长度必须一致"
        )
        self.fft_sizes   = fft_sizes
        self.hop_sizes   = hop_sizes
        self.win_lengths = win_lengths

    @staticmethod
    def _stft_magnitude(
        x: torch.Tensor, n_fft: int, hop_length: int, win_length: int
    ) -> torch.Tensor:
        """
        计算单分辨率 STFT 幅度谱。

        x : Tensor, (B, L)
        Returns: (B, F, T)，F = n_fft//2 + 1
        """
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
        return S.abs() + 1e-8   # (B, F, T)

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred, gt : Tensor, (B, 1, L)

        Returns
        -------
        Tensor : scalar STFT Loss
        """
        pred_flat = pred.squeeze(1)   # (B, L)
        gt_flat   = gt.squeeze(1)

        parts = []
        for n_fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            S_pred = self._stft_magnitude(pred_flat, n_fft, hop, win)
            S_gt   = self._stft_magnitude(gt_flat,   n_fft, hop, win)
            # 归一化 L1：消除幅度量纲差异
            parts.append((S_pred - S_gt).abs().sum() / (S_gt.sum() + 1e-8))  # S_gt already non-negative

        return sum(parts) / len(self.fft_sizes)


# =============================================================================
# TotalLoss
# =============================================================================

class TotalLoss(nn.Module):
    """
    BeatAwareRadar2ECGNet 总损失函数。

    L_total = L_time + alpha * L_freq + beta * L_peak

    注意：alpha 和 beta 均须经实验验证后确定，不预设固定值。
    默认值仅为起始参考（参考 BeatAware_R-M2Net 的经验值）。

    Parameters
    ----------
    alpha : float  L_freq 权重（建议起点 0.05，待消融实验确定）
    beta  : float  L_peak 权重（建议起点 1.0，待消融实验确定）
    fft_sizes, hop_sizes, win_lengths : list[int]
        多分辨率 STFT 参数（@200Hz 目标采样率）
    """

    def __init__(
        self,
        alpha: float = 0.05,
        beta:  float = 1.0,
        fft_sizes:   list = [128, 256, 512],
        hop_sizes:   list = [64,  128, 256],
        win_lengths: list = [16,   32,  64],
    ):
        super().__init__()
        self.alpha     = alpha
        self.beta      = beta
        self.stft_loss = MultiResolutionSTFTLoss(fft_sizes, hop_sizes, win_lengths)

    def forward(
        self,
        ecg_pred:  torch.Tensor,
        ecg_gt:    torch.Tensor,
        peak_pred: torch.Tensor | None,
        peak_gt:   torch.Tensor | None,
    ) -> dict:
        """
        Parameters
        ----------
        ecg_pred  : Tensor, (B, 1, L) — 模型输出重建 ECG，值域 [0, 1]
        ecg_gt    : Tensor, (B, 1, L) — GT ECG，值域 [0, 1]
        peak_pred : Tensor | None, (B, 1, L) — PAM 峰值 Mask 预测（use_pam=False 时为 None）
        peak_gt   : Tensor | None, (B, 1, L) — GT 高斯软标签（use_pam=False 时传 None）

        Returns
        -------
        dict with keys:
            'total' : L_total（用于 backward）
            'time'  : L_time（监控）
            'freq'  : L_freq（监控）
            'peak'  : L_peak（监控；use_pam=False 时为 0.0）
        """
        L_time = F.l1_loss(ecg_pred, ecg_gt)
        L_freq = self.stft_loss(ecg_pred, ecg_gt)

        if peak_pred is not None and peak_gt is not None:
            # nan_to_num 先处理 NaN/Inf（clamp 不会处理 NaN），再 clamp 到安全范围
            # NaN 来源：SSM 状态在长序列（L=1600）中数值爆炸 → Sigmoid(NaN)=NaN
            peak_pred_safe = peak_pred.nan_to_num(
                nan=0.5, posinf=1.0 - 1e-6, neginf=1e-6
            ).clamp(1e-6, 1 - 1e-6)
            L_peak  = F.binary_cross_entropy(peak_pred_safe, peak_gt)
            L_total = L_time + self.alpha * L_freq + self.beta * L_peak
        else:
            L_peak  = ecg_pred.new_zeros(())
            L_total = L_time + self.alpha * L_freq

        return {
            "total": L_total,
            "time":  L_time.detach(),
            "freq":  L_freq.detach(),
            "peak":  L_peak.detach(),
        }
