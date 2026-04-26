"""
losses.py — BeatAware-Radar2ECGNet 损失函数

L_total = L_recon + beta_peak * L_peak
  L_recon = L1(pred, gt) + alpha_stft * MultiResSTFT(pred, gt)
  L_peak  = BCE(qrs_pred, qrs_gt)   [QRS 峰值定位，固定权重]

设计原则：
  - 固定权重，不使用自适应 log_vars（避免目标函数被套利）
  - 仅监督 QRS 峰（P/T 波 GT 质量不稳定，引入噪声）
  - MultiResSTFT 使用 SC + log-mag 双项（V2 保留）

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
# TotalLoss（固定权重：L_recon + beta_peak * L_peak_QRS）
# =============================================================================

class TotalLoss(nn.Module):
    """
    BeatAware-Radar2ECGNet 总损失函数。

    L_total = L_recon + beta_peak * L_peak
      L_recon = L1(pred, gt) + alpha_stft * MultiResSTFT(pred, gt)
      L_peak  = BCE(qrs_pred, qrs_gt)   — 仅 QRS，固定权重

    Parameters
    ----------
    alpha_stft : float  STFT loss 在 L_recon 内的权重（默认 0.05）
    beta_peak  : float  L_peak 的权重（默认 1.0）
    """

    def __init__(
        self,
        alpha_stft: float = 0.05,
        beta_peak:  float = 1.0,
        fft_sizes:  list  = [128, 256, 512],
        hop_sizes:  list  = [64,  128, 256],
        win_lengths: list = [16,   32,  64],
    ):
        super().__init__()
        self.alpha_stft = alpha_stft
        self.beta_peak  = beta_peak
        self.stft_loss  = MultiResolutionSTFTLoss(fft_sizes, hop_sizes, win_lengths)

    @staticmethod
    def _safe_bce(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """NaN 安全的 BCE。"""
        pred_safe = pred.nan_to_num(
            nan=0.5, posinf=1.0 - 1e-6, neginf=1e-6
        ).clamp(1e-6, 1.0 - 1e-6)
        return F.binary_cross_entropy(pred_safe, gt)

    def _peak_loss(
        self,
        peak_preds: tuple | None,
        peak_gts:   dict  | None,
    ) -> torch.Tensor:
        """QRS-only BCE 峰值损失。"""
        if peak_preds is None or peak_gts is None:
            return torch.tensor(0.0)
        qrs_pred = peak_preds[0]          # (B, 1, L)
        qrs_gt   = peak_gts["qrs"]        # (B, 1, L)
        return self._safe_bce(qrs_pred, qrs_gt)

    def forward(
        self,
        ecg_pred:   torch.Tensor,
        ecg_gt:     torch.Tensor,
        peak_preds: tuple | None,
        peak_gts:   dict  | None,
        epoch:      int = 999,            # 保留参数签名兼容性，不再使用
    ) -> dict:
        """
        Returns dict:
            'total' : L_total（backward 用）
            'recon' : L_recon
            'time'  : L_time
            'freq'  : L_freq
            'peak'  : L_peak
        """
        L_time  = F.l1_loss(ecg_pred, ecg_gt)
        L_freq  = self.stft_loss(ecg_pred, ecg_gt)
        L_recon = L_time + self.alpha_stft * L_freq
        L_peak  = self._peak_loss(peak_preds, peak_gts)
        L_total = L_recon + self.beta_peak * L_peak

        return {
            "total": L_total,
            "recon": L_recon.detach(),
            "time":  L_time.detach(),
            "freq":  L_freq.detach(),
            "peak":  L_peak.detach() if isinstance(L_peak, torch.Tensor)
                     else torch.tensor(0.0),
        }
