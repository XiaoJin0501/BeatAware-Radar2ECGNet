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
# Lag-aware waveform loss：小范围时移鲁棒 PCC/L1
# =============================================================================

class LagAwareWaveformLoss(nn.Module):
    """
    在一个受限 lag 窗口内计算最佳对齐后的 PCC/L1 损失。

    设计用途：
      - 抵抗雷达机械响应与 ECG 电信号之间几十毫秒的残余错位；
      - 不改数据集，只在训练目标里给模型一个小范围的时序容忍度；
      - lag 窗口不应过大，避免模型通过任意平移来“作弊”。

    Positive lag means pred is compared as pred[lag:] vs gt[:-lag].
    """

    def __init__(self, max_lag_samples: int = 20, softmax_tau: float = 0.05):
        super().__init__()
        self.max_lag_samples = int(max_lag_samples)
        self.softmax_tau = float(softmax_tau)
        lags = torch.arange(
            -self.max_lag_samples, self.max_lag_samples + 1,
            dtype=torch.float32,
        )
        lag_cost = lags.abs() / max(float(self.max_lag_samples), 1.0)
        self.register_buffer("lag_cost", lag_cost)

    @staticmethod
    def _shifted_views(
        pred: torch.Tensor,
        gt: torch.Tensor,
        lag: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if lag > 0:
            return pred[..., lag:], gt[..., :-lag]
        if lag < 0:
            return pred[..., :lag], gt[..., -lag:]
        return pred, gt

    @staticmethod
    def _pcc_per_sample(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pred_c = pred - pred.mean(dim=-1, keepdim=True)
        gt_c = gt - gt.mean(dim=-1, keepdim=True)
        num = (pred_c * gt_c).sum(dim=-1)
        den = torch.sqrt(
            (pred_c.square().sum(dim=-1) * gt_c.square().sum(dim=-1)).clamp_min(1e-12)
        )
        pcc = num / den
        if pcc.ndim > 1:
            pcc = pcc.mean(dim=1)
        return pcc

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        pred, gt: (B, 1, L)

        Returns:
          lag_pcc_loss = mean(1 - best shifted PCC)
          lag_l1_loss  = L1 at the same best-PCC lag
        """
        pcc_parts = []
        l1_parts = []
        for lag in range(-self.max_lag_samples, self.max_lag_samples + 1):
            p, g = self._shifted_views(pred, gt, lag)
            pcc_parts.append(self._pcc_per_sample(p, g))
            l1_parts.append((p - g).abs().mean(dim=(-2, -1)))

        pcc_stack = torch.stack(pcc_parts, dim=0)  # (n_lags, B)
        l1_stack = torch.stack(l1_parts, dim=0)    # (n_lags, B)
        best_pcc, best_idx = pcc_stack.max(dim=0)
        best_l1 = l1_stack.gather(0, best_idx.unsqueeze(0)).squeeze(0)
        zero_pcc = pcc_stack[self.max_lag_samples]

        # Differentiable soft lag penalty. If high PCC mass moves away from
        # zero lag, this term increases and nudges the model back toward
        # synchronization instead of unconstrained shifted matching.
        weights = torch.softmax(pcc_stack / max(self.softmax_tau, 1e-6), dim=0)
        lag_penalty = (weights * self.lag_cost[:, None]).sum(dim=0)

        return {
            "lag_pcc": (1.0 - best_pcc).mean(),
            "lag_l1": best_l1.mean(),
            "zero_pcc": (1.0 - zero_pcc).mean(),
            "lag_penalty": lag_penalty.mean(),
        }


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
        use_lag_aware: bool = False,
        lag_max_samples: int = 20,
        lambda_lag_pcc: float = 0.2,
        lambda_lag_l1: float = 0.05,
        lambda_zero_pcc: float = 0.0,
        lambda_lag_penalty: float = 0.0,
        lag_softmax_tau: float = 0.05,
        fft_sizes:  list  = [128, 256, 512],
        hop_sizes:  list  = [64,  128, 256],
        win_lengths: list = [16,   32,  64],
    ):
        super().__init__()
        self.alpha_stft = alpha_stft
        self.beta_peak  = beta_peak
        self.stft_loss  = MultiResolutionSTFTLoss(fft_sizes, hop_sizes, win_lengths)
        self.use_lag_aware = use_lag_aware
        self.lambda_lag_pcc = lambda_lag_pcc
        self.lambda_lag_l1 = lambda_lag_l1
        self.lambda_zero_pcc = lambda_zero_pcc
        self.lambda_lag_penalty = lambda_lag_penalty
        self.lag_loss = (
            LagAwareWaveformLoss(lag_max_samples, softmax_tau=lag_softmax_tau)
            if use_lag_aware else None
        )

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

        zero = torch.tensor(0.0, device=ecg_pred.device)
        L_lag_pcc = zero
        L_lag_l1 = zero
        L_zero_pcc = zero
        L_lag_penalty = zero
        if self.lag_loss is not None:
            lag_parts = self.lag_loss(ecg_pred, ecg_gt)
            L_lag_pcc = lag_parts["lag_pcc"]
            L_lag_l1 = lag_parts["lag_l1"]
            L_zero_pcc = lag_parts["zero_pcc"]
            L_lag_penalty = lag_parts["lag_penalty"]

        L_total = (
            L_recon
            + self.beta_peak * L_peak
            + self.lambda_lag_pcc * L_lag_pcc
            + self.lambda_lag_l1 * L_lag_l1
            + self.lambda_zero_pcc * L_zero_pcc
            + self.lambda_lag_penalty * L_lag_penalty
        )

        return {
            "total": L_total,
            "recon": L_recon.detach(),
            "time":  L_time.detach(),
            "freq":  L_freq.detach(),
            "peak":  L_peak.detach() if isinstance(L_peak, torch.Tensor)
                     else torch.tensor(0.0),
            "lag_pcc": L_lag_pcc.detach(),
            "lag_l1":  L_lag_l1.detach(),
            "zero_pcc": L_zero_pcc.detach(),
            "lag_penalty": L_lag_penalty.detach(),
        }


# =============================================================================
# DiffusionLoss（扩散解码器专用：噪声预测 MSE + QRS-only BCE）
# =============================================================================

class DiffusionLoss(nn.Module):
    """
    条件扩散解码器损失函数。

    L_total = MSE(eps_pred, eps_true) + beta_peak * BCE(qrs_pred, qrs_gt)

    STFT 损失由扩散目标隐式覆盖，不再单独计算。
    返回 dict 键与 TotalLoss 兼容（recon=diff, freq=0, time=diff），
    以便训练日志不需要修改。

    Parameters
    ----------
    beta_peak : float  QRS 峰值 BCE 权重（默认 1.0）
    """

    def __init__(self, beta_peak: float = 1.0):
        super().__init__()
        self.beta_peak = beta_peak

    @staticmethod
    def _safe_bce(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pred_safe = pred.nan_to_num(
            nan=0.5, posinf=1.0 - 1e-6, neginf=1e-6
        ).clamp(1e-6, 1.0 - 1e-6)
        return F.binary_cross_entropy(pred_safe, gt)

    def forward(
        self,
        model_out:  tuple,           # (eps_pred (B,1,L), eps_true (B,1,L))
        ecg_gt:     torch.Tensor,    # unused but kept for API symmetry
        peak_preds: tuple | None,
        peak_gts:   dict  | None,
        epoch:      int = 999,
    ) -> dict:
        eps_pred, eps_true = model_out
        L_diff = F.mse_loss(eps_pred, eps_true)

        L_peak = torch.tensor(0.0, device=eps_pred.device)
        if peak_preds is not None and peak_gts is not None:
            qrs_pred = peak_preds[0]
            qrs_gt   = peak_gts["qrs"]
            L_peak   = self._safe_bce(qrs_pred, qrs_gt)

        L_total = L_diff + self.beta_peak * L_peak
        zero    = torch.tensor(0.0, device=eps_pred.device)
        return {
            "total": L_total,
            "recon": L_diff.detach(),
            "time":  L_diff.detach(),
            "freq":  zero,
            "peak":  L_peak.detach() if isinstance(L_peak, torch.Tensor) else zero,
            "lag_pcc": zero,
            "lag_l1":  zero,
            "zero_pcc": zero,
            "lag_penalty": zero,
        }
